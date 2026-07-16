"""ScratchCNN: a small ResNet-style backbone trained from scratch."""

from __future__ import annotations

import torch.nn as nn

#: Channels per image in the DEFAULT layout (the three extraction channels:
#: raw + two derived). Under model.channels=first_only/same the views are stacked
#: onto the channel axis instead, so the count is len(views) -- which for the full
#: Q/K/V set is also 3. Both backbones take the count explicitly rather than
#: leaning on that coincidence (the hidden-state source has just one view).
IN_CHANNELS = 3


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

    def __init__(self, embed_dim: int = 128, dropout: float = 0.0, in_ch: int = IN_CHANNELS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
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
