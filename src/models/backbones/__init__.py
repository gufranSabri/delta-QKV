"""Per-view CNN backbones: one token image -> one embedding vector.

Each backbone consumes a 3-channel (raw, delta-prev, delta-next) L x C image for
a SINGLE view (Q, K or V) and emits an E-dim embedding. The view axis is folded
into the batch by the caller, so a backbone never knows which view it is looking
at -- that is the fusion module's job.
"""

from __future__ import annotations

from .resnet18 import IMAGENET_SIZE, ResNet18Adapted
from .scratch_cnn import IN_CHANNELS, ResBlock, ScratchCNN

__all__ = [
    "IN_CHANNELS",
    "IMAGENET_SIZE",
    "ResBlock",
    "ScratchCNN",
    "ResNet18Adapted",
    "build_backbone",
]


def build_backbone(cfg):
    from src.data.dataset import n_channels

    name = cfg.model.backbone
    in_ch = n_channels(cfg.model.channels, len(cfg.extract.views))

    if name == "scratch_cnn":
        return ScratchCNN(
            embed_dim=cfg.model.embed_dim, dropout=cfg.model.dropout, in_ch=in_ch
        )
    if name == "resnet18":
        return ResNet18Adapted(
            embed_dim=cfg.model.embed_dim,
            pretrained=cfg.model.pretrained_backbone,
            dropout=cfg.model.dropout,
            in_ch=in_ch,
        )
    raise ValueError(f"unknown backbone {name!r}")
