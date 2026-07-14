"""Per-view CNN backbones: one token image -> one embedding vector.

Each backbone consumes a 3-channel (raw, delta-prev, delta-next) L x C image for
a SINGLE view (Q, K or V) and emits an E-dim embedding. The view axis is folded
into the batch by the caller, so a backbone never knows which view it is looking
at -- that is the fusion module's job.
"""

from __future__ import annotations

import torch
import torch.nn as nn

IN_CHANNELS = 3  # raw, delta-to-prev, delta-to-next


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

        # Projection shortcut when shape changes, identity otherwise.
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.short = nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.short(x))


class ScratchCNN(nn.Module):
    """A small ResNet-style stack built for 32x32 activation maps.

    Trained from scratch on purpose: activation images share essentially no
    low-level statistics with natural photographs, so ImageNet's edge/colour
    filters are a weak prior here. ~250k params, which is cheap enough that
    running three untied copies (one per view) costs nothing meaningful.
    """

    def __init__(self, embed_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResBlock(32, 32),                 # L x C
            ResBlock(32, 64, stride=2),       # L/2 x C/2
            ResBlock(64, 128, stride=2),      # L/4 x C/4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, x):                     # x: (N, 3, L, C)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)           # (N, 128)
        return self.proj(self.drop(x))        # (N, E)


class ResNet18Adapted(nn.Module):
    """torchvision ResNet-18, adapted CIFAR-style for small inputs.

    A stock ResNet-18 opens with a 7x7 stride-2 conv and a stride-2 maxpool,
    which between them shrink a 32x32 input to 8x8 before the first residual
    block -- far too aggressive. We swap in a 3x3 stride-1 conv and drop the
    maxpool, the standard small-image adaptation.

    Because we keep views SEPARATE (3 input channels each), the pretrained conv1
    weights are actually shape-compatible and can be retained. Had we stacked
    Q/K/V into one 9-channel image, conv1 would have had to be discarded.
    """

    def __init__(self, embed_dim: int = 128, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)

        old_conv1 = net.conv1
        net.conv1 = nn.Conv2d(IN_CHANNELS, 64, 3, stride=1, padding=1, bias=False)
        if pretrained:
            # Reuse the pretrained RGB filters: average the 7x7 kernel down to
            # 3x3 by taking its centre crop. Better than a random init, since the
            # 3-channel structure lines up.
            with torch.no_grad():
                net.conv1.weight.copy_(old_conv1.weight[:, :, 2:5, 2:5])

        net.maxpool = nn.Identity()
        net.fc = nn.Identity()
        self.net = net
        self.proj = nn.Linear(512, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, x):                     # (N, 3, L, C)
        feats = self.net(x)                   # (N, 512)
        return self.proj(self.drop(feats))    # (N, E)


def build_backbone(cfg) -> nn.Module:
    name = cfg.model.backbone
    if name == "scratch_cnn":
        return ScratchCNN(embed_dim=cfg.model.embed_dim, dropout=cfg.model.dropout)
    if name == "resnet18":
        return ResNet18Adapted(
            embed_dim=cfg.model.embed_dim,
            pretrained=cfg.model.pretrained_backbone,
            dropout=cfg.model.dropout,
        )
    raise ValueError(f"unknown backbone {name!r}")
