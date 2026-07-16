"""Unit tests for the pure image-construction math.

These use small hand-verifiable tensors and never load a model.
"""

import pytest
import torch

from src.extract.tensor_ops import (
    add_delta_channels,
    add_transform_channels,
    build_view_image,
    pool_feature_axis,
    pool_layer_axis,
)


# --------------------------------------------------------------------------
# pool_feature_axis
# --------------------------------------------------------------------------

def test_pool_max_picks_chunk_maxima():
    # D=6 -> 3 columns, so chunks are [0:2], [2:4], [4:6].
    raw = torch.tensor([[[1.0, 5.0, 3.0, 2.0, -1.0, -7.0]]])  # (T=1, L=1, D=6)
    out = pool_feature_axis(raw, n_cols=3, mode="max")
    assert out.shape == (1, 1, 3)
    # max(1,5)=5   max(3,2)=3   max(-1,-7)=-1
    assert torch.equal(out[0, 0], torch.tensor([5.0, 3.0, -1.0]))


def test_pool_mean_and_l2():
    raw = torch.tensor([[[3.0, 4.0, 0.0, 0.0]]])  # (1, 1, 4) -> 2 cols
    mean = pool_feature_axis(raw, n_cols=2, mode="mean")
    assert torch.equal(mean[0, 0], torch.tensor([3.5, 0.0]))
    l2 = pool_feature_axis(raw, n_cols=2, mode="l2")
    # ||(3,4)|| = 5
    assert torch.allclose(l2[0, 0], torch.tensor([5.0, 0.0]))


def test_pool_rejects_indivisible_dim():
    raw = torch.zeros(1, 1, 10)
    with pytest.raises(ValueError, match="not divisible"):
        pool_feature_axis(raw, n_cols=3)


def test_pool_rejects_bad_mode():
    with pytest.raises(ValueError, match="pool mode"):
        pool_feature_axis(torch.zeros(1, 1, 4), n_cols=2, mode="median")


def test_pool_is_contiguous_chunks_not_strided():
    # Guards against accidentally reshaping to (chunk, n_cols) instead of
    # (n_cols, chunk), which would interleave dims rather than chunk them.
    raw = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    out = pool_feature_axis(raw, n_cols=2, mode="max")
    # contiguous chunks: [0..3] -> 3, [4..7] -> 7.  strided would give 6, 7.
    assert torch.equal(out[0, 0], torch.tensor([3.0, 7.0]))


# --------------------------------------------------------------------------
# add_delta_channels -- the boundary behaviour is the whole point
# --------------------------------------------------------------------------

@pytest.fixture
def pooled_ramp():
    """(T=2, L=4, C=3), where pooled[t, l, c] = 10*l + c + 100*t.

    Layer-adjacent differences are therefore exactly +/-10 everywhere, which
    makes every expected delta trivially checkable by hand.
    """
    t_idx = torch.arange(2).view(2, 1, 1)
    l_idx = torch.arange(4).view(1, 4, 1)
    c_idx = torch.arange(3).view(1, 1, 3)
    return (100 * t_idx + 10 * l_idx + c_idx).float()


def test_channel0_is_the_pooled_input_untouched(pooled_ramp):
    img = add_delta_channels(pooled_ramp, boundary_mode="zero")
    assert img.shape == (2, 4, 3, 3)
    assert torch.equal(img[..., 0], pooled_ramp)


def test_interior_deltas_have_correct_sign_and_magnitude(pooled_ramp):
    img = add_delta_channels(pooled_ramp, boundary_mode="zero")
    # Interior layers 1 and 2 (of L=4). Values increase by 10 per layer, so:
    #   delta-to-prev = pooled[l] - pooled[l-1] = +10
    #   delta-to-next = pooled[l] - pooled[l+1] = -10
    for layer in (1, 2):
        assert torch.all(img[:, layer, :, 1] == 10.0), "delta-to-prev should be +10"
        assert torch.all(img[:, layer, :, 2] == -10.0), "delta-to-next should be -10"

    # Explicitly: channel 1 at l=2 equals pooled[2] - pooled[1].
    assert torch.equal(img[:, 2, :, 1], pooled_ramp[:, 2] - pooled_ramp[:, 1])
    # And channel 2 at l=2 equals pooled[2] - pooled[3].
    assert torch.equal(img[:, 2, :, 2], pooled_ramp[:, 2] - pooled_ramp[:, 3])


def test_deltas_are_signed_not_absolute(pooled_ramp):
    """A descending ramp must flip the sign; abs() would not."""
    descending = pooled_ramp.flip(dims=[1])
    img = add_delta_channels(descending, boundary_mode="zero")
    # Now values DECREASE by 10 per layer, so prev-delta is -10 (was +10).
    assert torch.all(img[:, 1:3, :, 1] == -10.0)
    assert torch.all(img[:, 1:3, :, 2] == +10.0)


def test_zero_boundary_zeroes_exactly_the_two_impossible_deltas(pooled_ramp):
    img = add_delta_channels(pooled_ramp, boundary_mode="zero")
    # Layer 0 has no previous layer.
    assert torch.all(img[:, 0, :, 1] == 0.0)
    # Last layer (L-1 == 3) has no next layer.
    assert torch.all(img[:, 3, :, 2] == 0.0)
    # ...and nothing else got zeroed: layer 0 still has a valid forward delta,
    # and the last layer still has a valid backward delta.
    assert torch.all(img[:, 0, :, 2] == -10.0)
    assert torch.all(img[:, 3, :, 1] == +10.0)


def test_replicate_boundary_copies_nearest_valid_delta(pooled_ramp):
    img = add_delta_channels(pooled_ramp, boundary_mode="replicate")
    # Layer 0's backward delta is copied from layer 1's (+10), not zeroed.
    assert torch.equal(img[:, 0, :, 1], img[:, 1, :, 1])
    assert torch.all(img[:, 0, :, 1] == 10.0)
    # Last layer's forward delta is copied from layer L-2's (-10).
    assert torch.equal(img[:, 3, :, 2], img[:, 2, :, 2])
    assert torch.all(img[:, 3, :, 2] == -10.0)


def test_wrap_boundary_references_the_opposite_end(pooled_ramp):
    """The spec's original request: at l=0, ch1 references l=L-1; at l=L-1,
    ch2 references l=0. Kept as an ablation option, not the default."""
    img = add_delta_channels(pooled_ramp, boundary_mode="wrap")
    L = 4

    # At l=0: delta-to-prev wraps to the LAST layer.
    expected = pooled_ramp[:, 0] - pooled_ramp[:, L - 1]
    assert torch.equal(img[:, 0, :, 1], expected)
    assert torch.all(img[:, 0, :, 1] == -30.0)  # 0 - 30

    # At l=L-1: delta-to-next wraps to layer 0.
    expected = pooled_ramp[:, L - 1] - pooled_ramp[:, 0]
    assert torch.equal(img[:, L - 1, :, 2], expected)
    assert torch.all(img[:, L - 1, :, 2] == +30.0)  # 30 - 0

    # Wrap must not disturb the interior.
    assert torch.all(img[:, 1:3, :, 1] == 10.0)
    assert torch.all(img[:, 1:3, :, 2] == -10.0)


def test_wrap_boundary_magnitude_dwarfs_interior():
    """Documents WHY zero is the default: the wrapped rows are outliers.

    With L=4 and a linear ramp the wrap delta is 3x the interior delta; for a
    real 32-layer model the embedding-vs-final-layer gap is far larger still.
    """
    t_idx = torch.zeros(1, 1, 1)
    pooled = (10 * torch.arange(8).view(1, 8, 1) + t_idx).float()  # L=8
    img = add_delta_channels(pooled, boundary_mode="wrap")
    interior = img[:, 1:7, :, 1].abs().max()
    wrapped = img[:, 0, :, 1].abs().max()
    assert wrapped > interior * 5, "wrap row should be a large-magnitude outlier"


def test_all_boundary_modes_agree_on_the_interior(pooled_ramp):
    imgs = {m: add_delta_channels(pooled_ramp, m) for m in ("zero", "replicate", "wrap")}
    for mode, img in imgs.items():
        assert torch.equal(img[:, 1:3, :, 1], imgs["zero"][:, 1:3, :, 1]), mode
        assert torch.equal(img[:, 1:3, :, 2], imgs["zero"][:, 1:3, :, 2]), mode


def test_add_delta_rejects_single_layer():
    with pytest.raises(ValueError, match="at least 2 layers"):
        add_delta_channels(torch.zeros(2, 1, 3))


def test_add_delta_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"expected \(T, L, C\)"):
        add_delta_channels(torch.zeros(2, 3))


# --------------------------------------------------------------------------
# build_view_image -- pooling and deltas composed
# --------------------------------------------------------------------------

def test_build_view_image_end_to_end_shape_and_squareness():
    T, L, D = 5, 4, 12
    raw = torch.randn(T, L, D)
    img = build_view_image(raw, n_cols=L)  # n_cols == L gives a square image
    assert img.shape == (T, L, L, 3)


def test_build_view_image_matches_manual_composition():
    raw = torch.randn(3, 4, 8)
    img = build_view_image(raw, n_cols=4, pool_mode="max", boundary_mode="zero")
    manual = add_delta_channels(
        pool_feature_axis(raw.float(), n_cols=4, mode="max"), "zero"
    )
    assert torch.equal(img, manual)


def test_build_view_image_handles_gqa_narrow_views():
    """Q and K/V have different D under GQA but must yield the SAME image size.

    Llama-3-8B: D_q = 4096 (32 heads x 128), D_kv = 1024 (8 kv-heads x 128).
    Both pool to 32 columns -- only the chunk width differs (128 vs 32).
    """
    L = 32
    q = torch.randn(2, L, 4096)
    k = torch.randn(2, L, 1024)
    img_q = build_view_image(q, n_cols=L)
    img_k = build_view_image(k, n_cols=L)
    assert img_q.shape == img_k.shape == (2, 32, 32, 3)


def test_build_view_image_upcasts_fp16_input():
    raw = torch.randn(2, 4, 8, dtype=torch.float16)
    img = build_view_image(raw, n_cols=4)
    assert img.dtype == torch.float32


# --------------------------------------------------------------------------
# add_transform_channels -- (raw, DWT, FFT) along the layer axis
# --------------------------------------------------------------------------

def test_transform_channels_shape_and_raw_channel_is_identity():
    pooled = torch.randn(3, 8, 5)
    img = add_transform_channels(pooled)
    assert img.shape == (3, 8, 5, 3)
    # Channel 0 must be the raw pooled activation, untouched.
    assert torch.equal(img[..., 0], pooled)


def test_transform_channels_are_nonnegative_magnitudes():
    pooled = torch.randn(2, 16, 4)
    img = add_transform_channels(pooled)
    # DWT and FFT channels are magnitudes -> non-negative everywhere.
    assert (img[..., 1] >= 0).all()
    assert (img[..., 2] >= 0).all()


def test_transform_channels_flag_a_localised_layer_jump():
    """A signal flat except for a jump at one layer should light up the DWT
    detail channel LOCALLY at that layer, not uniformly."""
    T, L, C = 1, 16, 1
    pooled = torch.zeros(T, L, C)
    pooled[0, 8, 0] = 10.0            # a single-layer spike
    img = add_transform_channels(pooled)
    dwt = img[0, :, 0, 1]
    # The largest DWT response should sit at/near the jump (layers 7-8), not far.
    peak = int(dwt.argmax())
    assert 7 <= peak <= 8


def test_build_view_image_transforms_matches_add_transform():
    raw = torch.randn(3, 6, 12)
    img = build_view_image(raw, n_cols=6, extraction_type="transforms", pool_mode="max")
    manual = add_transform_channels(pool_feature_axis(raw.float(), n_cols=6, mode="max"))
    assert torch.allclose(img, manual)


def test_build_view_image_rejects_unknown_extraction_type():
    with pytest.raises(ValueError, match="extraction_type"):
        build_view_image(torch.randn(2, 4, 8), n_cols=4, extraction_type="nonsense")


# --------------------------------------------------------------------------
# pool_layer_axis -- only needed for cross-LLM training
# --------------------------------------------------------------------------

def test_pool_layer_axis_is_noop_when_sizes_match():
    img = torch.randn(3, 8, 8, 3)
    assert torch.equal(pool_layer_axis(img, 8), img)


def test_pool_layer_axis_downsamples_and_preserves_other_axes():
    # Qwen (L=28) -> common L_eff=24, say.
    img = torch.randn(3, 28, 32, 3)
    out = pool_layer_axis(img, 24)
    assert out.shape == (3, 24, 32, 3)


def test_pool_layer_axis_takes_maxima_over_layer_groups():
    # L=4 -> 2: adaptive max pool halves it, so out[0]=max(l0,l1), out[1]=max(l2,l3).
    img = torch.zeros(1, 4, 1, 3)
    img[0, :, 0, 0] = torch.tensor([1.0, 7.0, 2.0, 3.0])
    out = pool_layer_axis(img, 2)
    assert out[0, 0, 0, 0] == 7.0
    assert out[0, 1, 0, 0] == 3.0


def test_pool_layer_axis_refuses_to_upsample():
    with pytest.raises(ValueError, match="cannot pool layer axis up"):
        pool_layer_axis(torch.randn(1, 8, 8, 3), 16)
