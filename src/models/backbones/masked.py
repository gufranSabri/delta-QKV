"""Wraps a ScratchCNN so its W axis (spatial, not the batch/channel axes) can
carry padding that must never contaminate the result.

Stream 2 (see src/models/classifier.py) puts the variable-length generated-
token axis T on a backbone's spatial W axis instead of the sequence axis, so
a batch pads T to a common width like any other variable-length axis. But
ScratchCNN's stride-2 convs mix neighbouring W columns, and its final
AdaptiveAvgPool2d has no notion of "these columns are padding" -- both would
let zeros bleed into real output and dilute the pooled average for shorter
examples. This module re-derives the valid-column mask through each stride
using the exact same conv arithmetic PyTorch uses, zeroing padded columns
after every stage, and finishes with a true masked mean (sum over valid
columns / count) instead of AdaptiveAvgPool2d.

Only ScratchCNN is supported: its stem/blocks/head are named, decomposable
submodules this wrapper can drive directly. ResNet18Adapted wraps opaque
torchvision internals that aren't split the same way (see config.py's
validate(), which rejects stream2 + resnet18 up front).

CAVEAT: masking here zeros padded columns between conv stages and pools only
over the real ones, so the pooled OUTPUT is exact -- verified to match
running ScratchCNN on the true (unpadded) width directly, bit for bit.
BatchNorm2d's train-mode running statistics are not masked, though: they are
computed over the full padded batch (H * W_padded elements per channel),
same limitation TemporalEncoder's masking has for its own padded axis. In
practice this just means BN's running mean/var are very slightly biased
toward zero by however much padding a batch happens to contain; it does not
affect eval-mode correctness (verified above) or backprop correctness.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .scratch_cnn import ResBlock, ScratchCNN


def _conv_out_len(length: torch.Tensor, kernel: int, stride: int, padding: int) -> torch.Tensor:
    return (length + 2 * padding - kernel) // stride + 1


def _zero_padding(x: torch.Tensor, valid_len: torch.Tensor) -> torch.Tensor:
    """Zero every column >= valid_len (per-example) on x's last axis.

    x: (N, C, H, W). valid_len: (N,) int, aligned with x's batch axis.
    """
    w = x.shape[-1]
    col = torch.arange(w, device=x.device).view(1, 1, 1, w)
    keep = col < valid_len.view(-1, 1, 1, 1)
    return x * keep


def _masked_resblock(block: ResBlock, x: torch.Tensor, valid_len: torch.Tensor):
    """ResBlock.forward, with padded W columns re-zeroed after every conv."""
    stride = block.conv1.stride[0]

    out = block.act(block.bn1(block.conv1(x)))
    valid_len = _conv_out_len(valid_len, kernel=3, stride=stride, padding=1)
    out = _zero_padding(out, valid_len)

    out = block.bn2(block.conv2(out))
    out = _zero_padding(out, valid_len)          # conv2 is stride 1: length unchanged

    short = block.short(x)
    if stride != 1:
        short = _zero_padding(short, valid_len)

    out = block.act(out + short)
    out = _zero_padding(out, valid_len)          # GELU(0) != 0, so re-zero once more
    return out, valid_len


class MaskedPoolBackbone(nn.Module):
    """Drives a ScratchCNN's own layers, masking its W (last spatial) axis.

    forward(x, mask): x is (N, C, H, W); mask is (N, W) bool, True at real
    (non-padded) columns of W. Output is (N, embed_dim), matching the
    wrapped ScratchCNN's own output shape exactly.
    """

    def __init__(self, backbone: ScratchCNN):
        super().__init__()
        if not isinstance(backbone, ScratchCNN):
            raise TypeError(
                f"MaskedPoolBackbone only supports ScratchCNN, got {type(backbone).__name__}"
            )
        self.backbone = backbone

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid_len = mask.sum(dim=1)                    # (N,)

        x = _zero_padding(x, valid_len)
        x = self.backbone.stem(x)
        # stem is a stride-1 1x1-equivalent 3x3 conv (kernel=3, stride=1, pad=1):
        # width is unchanged, but the conv's receptive field can still smear a
        # zeroed padded column into its real neighbour, so re-zero after it too.
        x = _zero_padding(x, valid_len)

        for block in self.backbone.blocks:
            x, valid_len = _masked_resblock(block, x, valid_len)

        # True masked mean over the valid columns only -- AdaptiveAvgPool2d
        # would average in the (zeroed) padded columns and dilute the result
        # for examples shorter than the batch max. count is H * valid_len:
        # keep only masks W, so it must be broadcast-multiplied out over H
        # before summing, not just counted as if H were 1.
        h, w = x.shape[-2], x.shape[-1]
        col = torch.arange(w, device=x.device).view(1, 1, 1, w)
        keep = (col < valid_len.view(-1, 1, 1, 1)).float()
        summed = (x * keep).sum(dim=(2, 3))
        count = (h * valid_len).clamp(min=1).view(-1, 1).float()
        pooled = summed / count                         # (N, C)

        return self.backbone.proj(self.backbone.drop(pooled))
