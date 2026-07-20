"""Grad-CAM / Eigen-CAM over a trained detector's per-image CNN backbones.

Each example feeds one image per CNN stream into the model -- one per VIEW
(Q/K/V) under `model.channels=default`, or one per EXTRACTION CHANNEL
(raw/DWT1/DWT2, or raw/dprev/dnext, each stacking all views) under `same`;
see `_stream_titles` for the exact mapping, which also respects
`model.include` (only the kept streams are ever fed to a backbone, so only
those get a heatmap). This renders one heatmap per actual image fed to the
backbones, showing where that backbone's last conv block is looking when the
model makes its prediction -- overlaid on the image itself, in
(layer x pooled-column) space.

Grad-CAM needs a backward pass (weights = gradient of the target scalar w.r.t.
each channel, averaged spatially); Eigen-CAM needs only a forward pass (the
first principal component of the activation map), which also makes it usable
for a look at what the network responds to independent of any particular
target. Both are computed per token and then averaged over the real
(non-padded) tokens of the example, since the CNN treats every token
independently and only the temporal encoder downstream mixes them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import Config
from src.data.dataset import collate
from src.models.classifier import build_model
from src.data.dataset import n_images
from src.utils.logger import get_logger
from src.utils.seed import pick_device, seed_everything

logger = get_logger(__name__)


def _last_conv_block(backbone: torch.nn.Module) -> torch.nn.Module:
    """The layer whose output we hook: last residual block before pooling."""
    name = backbone.__class__.__name__
    if name == "ScratchCNN":
        return backbone.blocks[-1]
    if name == "ResNet18Adapted":
        return backbone.net.layer4
    raise ValueError(f"no known last-conv-block for backbone {name!r}")


class _Recorder:
    """Grabs a layer's forward activations and (if needed) its gradient."""

    def __init__(self, layer: torch.nn.Module, need_grad: bool):
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._need_grad = need_grad
        self._fwd = layer.register_forward_hook(self._on_forward)
        self._bwd = layer.register_full_backward_hook(self._on_backward) if need_grad else None

    def _on_forward(self, module, inp, out):
        self.activations = out
        if self._need_grad:
            out.retain_grad()

    def _on_backward(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove(self):
        self._fwd.remove()
        if self._bwd is not None:
            self._bwd.remove()


def _cam_from_activations(activations: torch.Tensor, gradients: torch.Tensor | None) -> torch.Tensor:
    """(N, E, H, W) activations [+ (N, E, H, W) gradients] -> (N, H, W) heatmaps in [0, 1].

    Grad-CAM (gradients given): channel weights = spatially-averaged gradient of
    the target w.r.t. that channel; the map is the ReLU'd weighted sum.
    Eigen-CAM (gradients None): the map is each activation's projection onto the
    first principal component of the channel dimension, computed once over all N
    token maps together so every token is expressed in the same basis.
    """
    if gradients is not None:
        weights = gradients.mean(dim=(2, 3), keepdim=True)   # (N, E, 1, 1)
        cam = (weights * activations).sum(dim=1)              # (N, H, W)
        cam = F.relu(cam)
    else:
        n, e, h, w = activations.shape
        flat = activations.permute(0, 2, 3, 1).reshape(-1, e)    # (N*H*W, E)
        flat = flat - flat.mean(dim=0, keepdim=True)
        # Top right-singular vector = first principal component direction.
        _, _, v = torch.linalg.svd(flat, full_matrices=False)
        pc1 = v[0]                                                # (E,)
        cam = (flat @ pc1).reshape(n, h, w)
        cam = cam.abs()

    # Per-token min-max normalisation so every token's map uses the full range.
    flat = cam.reshape(cam.shape[0], -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    flat = (flat - lo) / (hi - lo).clamp(min=1e-8)
    return flat.reshape(cam.shape)


def _set_eval_but_cudnn_rnn_trainable(model: torch.nn.Module) -> None:
    """cuDNN refuses to run an LSTM's backward pass in eval mode ("cudnn RNN
    backward can only be called in training mode") -- it only keeps the
    buffers backward needs when training=True. Grad-CAM needs that backward
    pass, so the LSTM has to be flipped to train(); everything else that would
    make training-mode nondeterministic (Dropout, BatchNorm running stats) is
    forced back to eval-style behaviour explicitly.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.LSTM):
            module.train()


def _load_example(cfg: Config, ckpt: dict, dataset_name: str, idx: int):
    """One example, collated into a batch of size 1 (so masking matches training)."""
    from src.train import load_source

    stats = ckpt["stats"]
    source = load_source(cfg, dataset_name, cfg.llm.alias)
    source.stats = stats
    item = source[idx]
    batch = collate([item])
    return batch, source


def run_cam(
    cfg: Config,
    checkpoint: str | Path,
    dataset_name: str | None = None,
    idx: int = 0,
    method: str = "gradcam",
    out: str | None = None,
    max_tokens_shown: int | None = 20,
) -> Path:
    """Render a Grad-CAM/Eigen-CAM figure for one example to a PNG.

    One row per image actually fed to a backbone (see module docstring for
    what `model.channels`/`model.include` make that set), one column per
    GENERATED TOKEN of the response -- i.e. every token in the sentence gets
    its own heatmap, nothing is averaged away. `max_tokens_shown` caps how
    many token columns get rendered (keeps the figure legible on long
    responses); pass None to always render every real token.
    """
    if method not in ("gradcam", "eigencam"):
        raise ValueError(f"method must be 'gradcam' or 'eigencam', got {method!r}")

    seed_everything(cfg.train.seed)
    device = pick_device()

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Mirror test.py: the checkpoint's shape-determining fields win over the config.
    cfg.extract.views = ckpt["views"]
    cfg.model.channels = ckpt.get("channels", "default")
    cfg.model.include = ckpt.get("include", None)
    cfg.model.stream2.enable = ckpt.get("stream2_enable", False)
    cfg.model.stream2.include = ckpt.get("stream2_include", None)

    if cfg.model.stream2.enable:
        raise NotImplementedError(
            "CAM only covers stream 1 (the per-token image stream); "
            "this checkpoint has stream2 enabled."
        )

    name = dataset_name or cfg.dataset.name
    batch, source = _load_example(cfg, ckpt, name, idx)
    images, labels, mask, origins = batch
    images = images.to(device)          # (1, T, V, 3, L, C)
    mask = mask.to(device)              # (1, T)

    n_streams = n_images(cfg.model.channels, len(cfg.extract.views), cfg.model.include)
    model = build_model(cfg, n_views=n_streams).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    need_grad = method == "gradcam"
    if need_grad:
        _set_eval_but_cudnn_rnn_trainable(model)

    backbones = model.backbones if not model.share_backbone else [model.backbones[0]] * n_streams
    recorders = [_Recorder(_last_conv_block(bb), need_grad) for bb in backbones]

    with torch.set_grad_enabled(need_grad):
        logit = model(images, mask)              # (1,)
        prob = torch.sigmoid(logit).item()
        if need_grad:
            model.zero_grad(set_to_none=True)
            logit.sum().backward()

    real_tokens = int(mask[0].sum().item())
    n_shown = real_tokens if max_tokens_shown is None else min(real_tokens, max_tokens_shown)
    if n_shown < real_tokens:
        logger.warning(
            "response has %d tokens; showing the first %d (pass max_tokens_shown=None for all)",
            real_tokens, n_shown,
        )

    cams = []            # one (T_shown, h, w) map per image stream -- no token averaging
    for rec in recorders:
        acts = rec.activations                    # (T, E, h, w) -- see note below
        grads = rec.gradients if need_grad else None
        cam = _cam_from_activations(acts, grads)   # (T, h, w)
        # encode_tokens flattens (B*T) into the batch per view/stream, B=1 here,
        # so this axis IS the token axis, in generation order.
        cams.append(cam[:n_shown].detach().cpu().numpy())

    for rec in recorders:
        rec.remove()

    titles = _stream_titles(
        cfg.extract.views, cfg.extract.extraction_type, cfg.model.channels, cfg.model.include,
    )
    dest = Path(out) if out else ckpt_path.parent / f"cam_{method}_{name}_{idx:05d}.png"
    _render(
        cams=cams,
        images=images[0, :n_shown].cpu().numpy(),   # (T_shown, n_streams, 3, L, C)
        titles=titles,
        method=method,
        label=int(labels.item()),
        prob=prob,
        origin=origins[0],
        idx=idx,
        dest=dest,
    )
    logger.info("wrote %s (label=%d, p(hallucinated)=%.3f)", dest, int(labels.item()), prob)
    return dest


#: The 3 extraction channels, by extraction_type (src/extract/tensor_ops.py).
_CHANNEL_NAMES = {
    "delta": ["raw", "dprev", "dnext"],
    "transforms": ["raw", "DWT1", "DWT2"],
}


def _stream_titles(
    views: list[str],
    extraction_type: str,
    channels_mode: str,
    include: list[int] | None,
) -> list[str]:
    """One label per image actually fed to the backbones, in `regroup_channels`
    order (src/data/dataset.py) -- i.e. what `model.channels` turns the (view,
    extraction-channel) axes into, THEN filtered by `model.include` exactly the
    way QKVImageDataset._finish applies it, so labels line up with cams 1:1.

    `default`  -> one image PER VIEW (each still carrying all 3 extraction
                  channels) -- labelled Q, K, V.
    `same`     -> one image PER EXTRACTION CHANNEL, each stacking every view --
                  labelled by channel semantics (raw/DWT1/DWT2 or raw/dprev/dnext).
    `first_only` -> a single image: the raw channel of every view, stacked.
    """
    if channels_mode == "default":
        titles = list(views)
    elif channels_mode == "same":
        names = _CHANNEL_NAMES.get(extraction_type, ["raw", "ch1", "ch2"])
        titles = names[:3]
    elif channels_mode == "first_only":
        titles = [f"raw ({'/'.join(views)} stacked)"]
    else:
        raise ValueError(f"unknown model.channels mode {channels_mode!r}")

    if include is not None:
        titles = [titles[i] for i in include]
    return titles


def _render(
    cams: list[np.ndarray],
    images: np.ndarray,
    titles: list[str],
    method: str,
    label: int,
    prob: float,
    origin: str,
    idx: int,
    dest: Path,
) -> None:
    """Grid layout: one ROW per image stream, one COLUMN per generated token.

    cams[s]:   (T, h, w) for stream s.
    images:    (T, n_streams, 3, L, C) -- the actual model input, un-averaged.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_streams = len(cams)
    n_tokens = cams[0].shape[0]

    fig, axes = plt.subplots(
        n_streams, n_tokens,
        figsize=(1.6 * n_tokens, 1.6 * n_streams),
        squeeze=False,
    )
    for s in range(n_streams):
        for t in range(n_tokens):
            ax = axes[s][t]
            bg = images[t, s, 0]           # channel 0 (raw) of stream s, token t
            cam = cams[s][t]
            if cam.shape != bg.shape:
                cam_t = torch.from_numpy(cam)[None, None].float()
                cam = F.interpolate(
                    cam_t, size=bg.shape, mode="bilinear", align_corners=False
                )[0, 0].numpy()

            ax.imshow(bg, cmap="gray", aspect="auto")
            ax.imshow(cam, cmap="jet", alpha=0.5, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if s == 0:
                ax.set_title(f"t{t}", fontsize=8)
            if t == 0:
                ax.set_ylabel(titles[s], fontsize=9)

    fig.suptitle(
        f"{method}  |  {origin} example {idx}  |  "
        f"label={label} (1=hallucinated)  p(hallucinated)={prob:.3f}  |  "
        f"{n_tokens} of the response's generated tokens, one column each",
        fontsize=11,
    )
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=130)
    plt.close(fig)
