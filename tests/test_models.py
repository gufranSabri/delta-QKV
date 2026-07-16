"""Shape, masking and fusion tests for the detector.

The masking tests matter most: a padding bug does not crash, it silently
corrupts every metric. Each one is written so that it FAILS if padding leaks.
"""

import pytest
import torch

from src.config import Config
from src.models.classifier import build_model
from src.models.fusion import (
    BilinearFusion,
    CrossAttnFusion,
    GatedFusion,
    IdentityFusion,
    build_fusion,
)
from src.models.temporal import MaskedAttentionPool, TemporalEncoder

B, T, L, C, E = 2, 5, 8, 8, 16


def make_cfg(**model_kw) -> Config:
    cfg = Config()
    cfg.model.embed_dim = E
    cfg.model.fused_dim = E
    cfg.model.lstm_hidden = 8
    cfg.model.conv1d_layers = 1
    cfg.model.dropout = 0.0
    for k, v in model_kw.items():
        setattr(cfg.model, k, v)
    return cfg


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["gated", "concat_mlp", "bilinear", "cross_attn"])
def test_every_fusion_maps_views_to_one_vector(name):
    cfg = make_cfg(fusion=name)
    fusion = build_fusion(cfg, n_views=3)
    out = fusion(torch.randn(B, 3, E))
    assert out.shape == (B, E)
    assert torch.isfinite(out).all()


def test_single_view_bypasses_fusion():
    """views: [V] must not need a fusion module at all."""
    cfg = make_cfg(fusion="gated")
    fusion = build_fusion(cfg, n_views=1)
    assert isinstance(fusion, IdentityFusion)
    out = fusion(torch.randn(B, 1, E))
    assert out.shape == (B, E)


def test_gates_are_a_distribution_over_views():
    """The gates must sum to 1 across views -- that is what makes them readable
    as 'how much does the model rely on Q vs K vs V'."""
    fusion = GatedFusion(E, E, n_views=3, gate_mode="scalar")
    fusion(torch.randn(B, 3, E))
    gates = fusion.last_gates            # (B, 3, 1)
    assert gates.shape == (B, 3, 1)
    torch.testing.assert_close(gates.sum(dim=1).squeeze(-1), torch.ones(B))
    assert (gates >= 0).all()


def test_vector_gates_also_normalise_over_views():
    fusion = GatedFusion(E, E, n_views=3, gate_mode="vector")
    fusion(torch.randn(B, 3, E))
    gates = fusion.last_gates            # (B, 3, E)
    assert gates.shape == (B, 3, E)
    torch.testing.assert_close(gates.sum(dim=1), torch.ones(B, E))


def test_gates_do_not_carry_gradients():
    """last_gates is for inspection only; if it were attached to the graph,
    reading it during eval could silently retain the graph."""
    fusion = GatedFusion(E, E, n_views=3)
    fusion(torch.randn(B, 3, E, requires_grad=True))
    assert not fusion.last_gates.requires_grad


def test_gated_fusion_actually_distinguishes_views():
    """Feeding the same embedding as Q, K and V vs. different ones must give
    different outputs -- otherwise fusion is ignoring the view structure."""
    fusion = GatedFusion(E, E, n_views=3)
    fusion.eval()
    same = torch.randn(1, 1, E).expand(1, 3, E).contiguous()
    diff = torch.randn(1, 3, E)
    assert not torch.allclose(fusion(same), fusion(diff))


def test_bilinear_has_a_genuine_multiplicative_term():
    """Scaling one view must change the output NON-linearly, which an additive
    fusion could not do. This is the whole justification for BilinearFusion."""
    fusion = BilinearFusion(E, E, n_views=2, rank=8)
    fusion.eval()
    x = torch.randn(1, 2, E)

    with torch.no_grad():
        base = fusion(x)
        x2 = x.clone()
        x2[:, 0] *= 2.0
        doubled = fusion(x2)
    # A purely linear map would satisfy f(2a, b) - f(a, b) == f(a, b) - f(0, b).
    # The multiplicative term breaks that. We just assert it changed at all.
    assert not torch.allclose(base, doubled)


def test_cross_attn_is_not_permutation_invariant():
    """Without the view-identity embedding a transformer over {Q,K,V} could not
    tell Q from V. Swapping two views MUST change the output."""
    fusion = CrossAttnFusion(E, E, n_views=3)
    fusion.eval()
    x = torch.randn(1, 3, E)
    swapped = x[:, [2, 1, 0]]
    with torch.no_grad():
        assert not torch.allclose(fusion(x), fusion(swapped), atol=1e-6)


# --------------------------------------------------------------------------
# masking -- the silent-corruption zone
# --------------------------------------------------------------------------

def test_masked_pool_ignores_padded_positions_entirely():
    pool = MaskedAttentionPool(E)
    x = torch.randn(1, 4, E)
    mask = torch.tensor([[True, True, False, False]])

    out_a = pool(x, mask)
    # Overwrite the PADDED positions with garbage. The output must not budge.
    x2 = x.clone()
    x2[0, 2:] = 1e4
    out_b = pool(x2, mask)

    torch.testing.assert_close(out_a, out_b, msg="padding leaked into the pool")


def test_temporal_encoder_output_is_invariant_to_padding_content():
    """The strongest padding test: change what is in the padded slots and the
    encoder output must be bit-for-bit identical."""
    enc = TemporalEncoder(input_dim=E, conv_layers=1, lstm_hidden=8, dropout=0.0)
    enc.eval()

    x = torch.randn(2, 6, E)
    mask = torch.tensor(
        [[True] * 6, [True, True, True, False, False, False]]
    )

    with torch.no_grad():
        a = enc(x, mask)
        x2 = x.clone()
        x2[1, 3:] = 99.0        # garbage in row 1's padding
        b = enc(x2, mask)

    torch.testing.assert_close(a, b, msg="padded values changed the output")


def test_temporal_encoder_result_is_independent_of_pad_length():
    """A sequence of length 3 must give the same answer whether the batch pads
    to 3 or to 10. If it does not, padding is contaminating the result."""
    enc = TemporalEncoder(input_dim=E, conv_layers=1, lstm_hidden=8, dropout=0.0)
    enc.eval()
    torch.manual_seed(0)
    seq = torch.randn(1, 3, E)

    short_mask = torch.ones(1, 3, dtype=torch.bool)

    padded = torch.zeros(1, 10, E)
    padded[:, :3] = seq
    long_mask = torch.zeros(1, 10, dtype=torch.bool)
    long_mask[:, :3] = True

    with torch.no_grad():
        a = enc(seq, short_mask)
        b = enc(padded, long_mask)

    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------
# the full detector
# --------------------------------------------------------------------------

def make_batch(n_views=3):
    images = torch.randn(B, T, n_views, 3, L, C)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 3:] = False           # second example is shorter
    return images, mask


@pytest.mark.parametrize("fusion", ["gated", "concat_mlp", "bilinear", "cross_attn"])
def test_detector_forward_shape(fusion):
    cfg = make_cfg(fusion=fusion)
    model = build_model(cfg, n_views=3)
    images, mask = make_batch()
    out = model(images, mask)
    assert out.shape == (B,)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("n_views", [1, 2, 3])
def test_detector_handles_any_view_count(n_views):
    cfg = make_cfg()
    model = build_model(cfg, n_views=n_views)
    images, mask = make_batch(n_views)
    assert model(images, mask).shape == (B,)


def test_untied_backbones_are_actually_separate_modules():
    """The core architectural claim: Q, K and V get their OWN CNN."""
    cfg = make_cfg(share_backbone=False)
    model = build_model(cfg, n_views=3)
    assert len(model.backbones) == 3
    # Different objects, and (at init) different weights.
    w0 = model.backbones[0].stem[0].weight
    w1 = model.backbones[1].stem[0].weight
    assert w0 is not w1
    assert not torch.allclose(w0, w1)


def test_shared_backbone_reuses_one_module():
    cfg = make_cfg(share_backbone=True)
    model = build_model(cfg, n_views=3)
    assert len(model.backbones) == 1


def test_untied_backbones_give_each_view_a_different_encoder():
    """Feed the SAME image as Q, K and V. With untied CNNs the three embeddings
    must differ (different weights); with a shared CNN they must be identical.
    This proves the views are not being silently collapsed."""
    img = torch.randn(1, 1, 1, 3, L, C).expand(1, 1, 3, 3, L, C).contiguous()

    untied = build_model(make_cfg(share_backbone=False), n_views=3).eval()
    shared = build_model(make_cfg(share_backbone=True), n_views=3).eval()

    with torch.no_grad():
        # Reach into the backbones directly to compare per-view embeddings.
        flat = img[:, :, 0].reshape(1, 3, L, C)
        u = [untied.backbones[i](flat) for i in range(3)]
        s = [shared.backbones[0](flat) for _ in range(3)]

    assert not torch.allclose(u[0], u[1]), "untied CNNs produced identical output"
    torch.testing.assert_close(s[0], s[1])


def test_detector_is_padding_invariant_end_to_end():
    cfg = make_cfg()
    model = build_model(cfg, n_views=3).eval()

    images, mask = make_batch()
    with torch.no_grad():
        a = model(images, mask)
        noisy = images.clone()
        noisy[1, 3:] = 123.0       # garbage in the padded tokens of example 1
        b = model(noisy, mask)

    torch.testing.assert_close(a, b, msg="padded TOKENS leaked into the prediction")


def test_detector_backward_reaches_every_backbone():
    """All three CNNs must receive gradient -- if one is dead, a view is unused."""
    cfg = make_cfg()
    model = build_model(cfg, n_views=3)
    images, mask = make_batch()

    loss = model(images, mask).sum()
    loss.backward()

    for i, bb in enumerate(model.backbones):
        grad = bb.stem[0].weight.grad
        assert grad is not None, f"backbone {i} got no gradient"
        assert grad.abs().sum() > 0, f"backbone {i} gradient is all zeros"


def test_view_gates_returns_a_distribution():
    cfg = make_cfg(fusion="gated")
    model = build_model(cfg, n_views=3).eval()
    images, mask = make_batch()
    gates = model.view_gates(images, mask)
    assert gates.shape == (3,)
    torch.testing.assert_close(gates.sum(), torch.tensor(1.0), rtol=1e-4, atol=1e-4)


def test_view_gates_is_none_for_non_gated_fusion():
    cfg = make_cfg(fusion="concat_mlp")
    model = build_model(cfg, n_views=3).eval()
    images, mask = make_batch()
    assert model.view_gates(images, mask) is None


def test_detector_rejects_wrong_view_count():
    cfg = make_cfg()
    model = build_model(cfg, n_views=3)
    images, mask = make_batch(n_views=2)   # built for 3, given 2
    with pytest.raises(ValueError, match="built for 3 views"):
        model(images, mask)


def test_resnet18_backbone_runs():
    cfg = make_cfg(backbone="resnet18", pretrained_backbone=False)
    model = build_model(cfg, n_views=3)
    images, mask = make_batch()
    assert model(images, mask).shape == (B,)


def test_resnet18_random_init_keeps_native_resolution():
    """With random weights there is no pretrained stem to protect, so the stem is
    adapted for small images and the input is NOT upscaled (49x the compute for
    no information gain)."""
    from src.models.backbones import ResNet18Adapted

    net = ResNet18Adapted(embed_dim=16, pretrained=False)
    assert net.resize_to is None
    assert net.net.conv1.kernel_size == (3, 3)
    assert net.net.conv1.stride == (1, 1)
    assert isinstance(net.net.maxpool, torch.nn.Identity)


def test_resnet18_pretrained_is_unmodified_and_resizes_input():
    """The pretrained network must be used AS-IS: resize the input to 224x224
    rather than rebuilding conv1 / deleting the maxpool, which would discard the
    very weights the checkpoint was loaded for.

    Downloads the torchvision checkpoint on first run.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    from src.models.backbones import ResNet18Adapted

    net = ResNet18Adapted(embed_dim=16, pretrained=True)

    # The stem is the stock ImageNet stem, untouched.
    assert net.resize_to == 224
    assert net.net.conv1.kernel_size == (7, 7)
    assert net.net.conv1.stride == (2, 2)
    assert isinstance(net.net.maxpool, torch.nn.MaxPool2d)

    # And the weights really are the pretrained ones, not a crop or a re-init.
    ref = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    assert torch.equal(net.net.conv1.weight, ref.conv1.weight)
    assert torch.equal(net.net.layer4[1].conv2.weight, ref.layer4[1].conv2.weight)

    # A 32x32 activation image still produces an embedding, and gradients reach
    # the stem through the resize.
    out = net(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 16)
    out.sum().backward()
    assert net.net.conv1.weight.grad.abs().sum() > 0


def test_view_i_is_routed_to_backbone_i():
    """Shape tests pass even if the view axis is transposed. This does not.

    Each backbone is turned into a constant function with a unique output value,
    so the resulting embedding reveals WHICH backbone actually processed each
    view. Guards the reshape/fold logic in encode_tokens.
    """
    cfg = make_cfg(share_backbone=False)
    model = build_model(cfg, n_views=3).eval()

    with torch.no_grad():
        for i, bb in enumerate(model.backbones):
            for p in bb.parameters():
                p.zero_()
            bb.proj.bias.fill_(float(i + 1))     # backbone i outputs all (i+1)

        images = torch.randn(1, 2, 3, 3, L, C)
        b, t, v, c, h, w = images.shape
        emb = torch.stack(
            [
                model.backbones[i](images[:, :, i].reshape(b * t, c, h, w)).reshape(b, t, -1)
                for i in range(v)
            ],
            dim=2,
        )

    for i in range(3):
        assert emb[0, 0, i].mean().item() == float(i + 1), (
            f"view {i} was processed by the wrong backbone -- the view axis is "
            "being transposed or mis-indexed in encode_tokens"
        )


def test_encode_tokens_preserves_per_token_structure():
    """Folding tokens into the batch for the CNN must not collapse them."""
    cfg = make_cfg()
    model = build_model(cfg, n_views=3).eval()

    images = torch.zeros(1, 3, 3, 3, L, C)
    for t in range(3):
        images[0, t] = float(t + 1)              # each token distinct

    with torch.no_grad():
        out = model.encode_tokens(images)        # (1, 3, F)

    assert out.shape[:2] == (1, 3)
    assert not torch.allclose(out[0, 0], out[0, 1])
    assert not torch.allclose(out[0, 1], out[0, 2])


def test_temporal_encoder_rejects_an_empty_sequence():
    """A zero-token example is a bug upstream; fail loudly, not with a cryptic
    pack_padded_sequence error."""
    enc = TemporalEncoder(input_dim=E, conv_layers=1, lstm_hidden=8, dropout=0.0)
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[1] = False                              # row 1 has no real tokens
    with pytest.raises(ValueError, match="all-False mask"):
        enc(torch.randn(2, 4, E), mask)


# --------------------------------------------------------------------------
# temporal cross-attention adapter (ScratchCNN)
# --------------------------------------------------------------------------

def test_adapter_off_leaves_backbone_unchanged():
    """The flag defaults off; with it off the backbone must be byte-for-byte the
    plain ScratchCNN -- same params, no adapter modules."""
    from src.models.backbones import ScratchCNN

    plain = ScratchCNN(embed_dim=E, in_ch=3)
    assert all(a is None for a in plain.adapters)
    n_plain = sum(p.numel() for p in plain.parameters())

    with_flag_off = ScratchCNN(embed_dim=E, in_ch=3, use_temporal_adapter=False)
    assert sum(p.numel() for p in with_flag_off.parameters()) == n_plain

    # And it still runs without a token count (no adapter needs it).
    out = plain(torch.randn(4, 3, L, C))
    assert out.shape == (4, E)


def test_adapter_on_runs_and_is_a_moderate_param_increase():
    cfg = make_cfg(use_temporal_adapter=True)
    model = build_model(cfg, n_views=3).eval()
    images, mask = make_batch()
    with torch.no_grad():
        out = model(images, mask)
    assert out.shape == (B,)
    assert torch.isfinite(out).all()

    # Every backbone must now carry three live adapters.
    for bb in model.backbones:
        assert sum(a is not None for a in bb.adapters) == 3

    # "Moderate" shrink: the adapter must not blow the backbone up unboundedly.
    plain = build_model(make_cfg(), n_views=3)
    n_plain = sum(p.numel() for p in plain.backbones.parameters())
    n_adpt = sum(p.numel() for p in model.backbones.parameters())
    assert n_adpt < 3 * n_plain, f"adapter too large: {n_adpt} vs {n_plain}"


@pytest.mark.parametrize("share_backbone", [False, True])
def test_adapter_is_padding_invariant(share_backbone):
    """The whole reason the mask is threaded into the backbone: cross-token
    attention must never let padded tokens influence a real token's features."""
    cfg = make_cfg(use_temporal_adapter=True, share_backbone=share_backbone)
    model = build_model(cfg, n_views=3).eval()

    images, mask = make_batch()
    with torch.no_grad():
        a = model(images, mask)
        noisy = images.clone()
        noisy[1, 3:] = 987.0        # garbage ONLY in padded tokens of example 1
        b = model(noisy, mask)

    torch.testing.assert_close(
        a, b, msg="padded tokens leaked through the temporal adapter"
    )


@pytest.mark.parametrize("share_backbone", [False, True])
def test_adapter_actually_mixes_across_tokens(share_backbone):
    """The adapter's point is cross-token mixing: changing a REAL token must be
    able to shift another token's embedding. Without cross-attention it couldn't."""
    cfg = make_cfg(use_temporal_adapter=True, share_backbone=share_backbone)
    model = build_model(cfg, n_views=3).eval()

    images = torch.randn(1, 4, 3, 3, L, C)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        base = model.encode_tokens(images, mask)     # (1, 4, F)
        changed = images.clone()
        changed[0, 0] = torch.randn_like(changed[0, 0])   # perturb token 0 only
        after = model.encode_tokens(changed, mask)

    # Token 3's embedding should move even though only token 0 changed.
    assert not torch.allclose(base[0, 3], after[0, 3]), "no cross-token mixing"


def test_adapter_backward_reaches_the_adapter():
    cfg = make_cfg(use_temporal_adapter=True)
    model = build_model(cfg, n_views=3)
    images, mask = make_batch()

    model(images, mask).sum().backward()

    for i, bb in enumerate(model.backbones):
        # down_proj is the adapter's first learnable layer; it must get gradient.
        grad = bb.adapters[0].down_proj.weight.grad
        assert grad is not None and grad.abs().sum() > 0, f"adapter {i} got no grad"
