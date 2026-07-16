"""Per-view CNN backbones: one token image -> one embedding vector.

Each backbone consumes a 3-channel (raw, delta-prev, delta-next) L x C image for
a SINGLE view (Q, K or V) and emits an E-dim embedding. The view axis is folded
into the batch by the caller, so a backbone never knows which view it is looking
at -- that is the fusion module's job.
"""

from __future__ import annotations

import torch
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



import torch
import torch.nn as nn
import math

class PatchEmbed(nn.Module):
    """Splits an image into patches and projects them into the transformer's dimension."""
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_ch: int = 3, embed_dim: int = 128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Projection is mathematically equivalent to a non-overlapping Conv2D
        self.proj = nn.Conv2d(
            in_ch, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        # input x: (B, in_ch, H, W)
        x = self.proj(x)                  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class ScratchViT(nn.Module):
    """A small ViT replacement for ScratchCNN.
    Designed for 32x32 activation maps. Keeps the parameter count lightweight
    (~250k-350k params depending on depth/heads) so it's cheap to train from scratch.
    """
    def __init__(
        self, 
        embed_dim: int = 128, 
        dropout: float = 0.0, 
        in_ch: int = 3, # Assuming IN_CHANNELS matches your pipeline's default
        img_size: int = 32,
        patch_size: int = 4,
        depth: int = 4,         # Number of Transformer blocks
        num_heads: int = 4,     # Multi-head attention heads (embed_dim 128 must be divisible by this)
        mlp_ratio: float = 4.0, # Expansion ratio in MLP blocks
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_ch=in_ch, 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # 2. Learnable Classification Token (CLS)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3. Positional Encoding (Learnable, 1D)
        # We need num_patches + 1 positions because of the prepended CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True # Post-LN can be unstable; pre-LN (norm_first) is safer from scratch
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 5. Output Normalization and Projection
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim) # Linear layer matching your original architecture
        
        self._init_weights()

    def _init_weights(self):
        # Truncated normal initialization for tokens and pos embeddings
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Input shape: (B, in_ch, H, W) e.g., (N, 3, 32, 32)
        B = x.shape[0]
        
        # Patchify & Project -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        
        # Prepend CLS token to the patch sequence -> (B, num_patches + 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Positional Encoding
        x = self.pos_drop(x + self.pos_embed)
        
        # Pass through Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # Extract the CLS token's final state -> (B, embed_dim)
        cls_out = x[:, 0]
        
        # Final projection to match original CNN signature output (B, E)
        return self.proj(cls_out)


#: What a torchvision ImageNet model expects. Resizing to this keeps the
#: pretrained stem (7x7 stride-2 conv + stride-2 maxpool) operating at the scale
#: its filters were actually learned at.
IMAGENET_SIZE = 224


class ResNet18Adapted(nn.Module):
    """torchvision ResNet-18.

    Two ways to feed a 32x32 activation image to an ImageNet ResNet:

    - `pretrained=True`  -> UPSCALE THE INPUT to 224x224 and leave the network
      completely untouched. A stock ResNet-18 opens with a 7x7 stride-2 conv and
      a stride-2 maxpool; at 32x32 those would crush the image to 8x8 before the
      first residual block, so the usual fix is to replace conv1 and delete the
      maxpool. But that *throws away the pretrained stem* -- exactly the weights
      you loaded the checkpoint for. Resizing instead means every pretrained
      filter sees inputs at the scale it was trained on, and the model is used
      as-is, which is also what makes "pretrained" an honest label in the
      ablation table.

    - `pretrained=False` -> keep the network at native resolution and adapt the
      stem CIFAR-style (3x3 stride-1 conv, no maxpool). With random weights there
      is nothing to preserve, and upscaling 32x32 to 224x224 would be ~49x the
      compute for no information gain.

    Because we keep views SEPARATE (3 input channels each), the pretrained conv1
    is shape-compatible and needs no surgery. Had we stacked Q/K/V into one
    9-channel image, conv1 would have had to be discarded either way.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        pretrained: bool = True,
        dropout: float = 0.0,
        in_ch: int = IN_CHANNELS,
    ):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        if pretrained and in_ch != 3:
            # conv1's pretrained kernel is (64, 3, 7, 7). Any other channel count
            # would force us to discard or reshape it, which silently defeats the
            # point of loading pretrained weights -- fail loudly instead.
            raise ValueError(
                f"pretrained resnet18 needs exactly 3 input channels, got {in_ch}. "
                "Use pretrained_backbone=false, or a channels mode that yields 3."
            )

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)

        self.resize_to = IMAGENET_SIZE if pretrained else None

        if not pretrained:
            # Random init: no pretrained stem to protect, so adapt it to the
            # native (small) image size rather than paying to upscale.
            net.conv1 = nn.Conv2d(in_ch, 64, 3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()

        net.fc = nn.Identity()
        self.net = net
        self.proj = nn.Linear(512, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, x):                     # (N, 3, L, C)
        if self.resize_to is not None:
            # Bilinear, aligned corners off -- the standard torchvision resize.
            # `antialias=True` matters when DOWNsampling; we only ever upsample
            # here, but it is set for correctness if L or C ever exceeds 224.
            x = nn.functional.interpolate(
                x,
                size=(self.resize_to, self.resize_to),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        feats = self.net(x)                   # (N, 512)
        return self.proj(self.drop(feats))    # (N, E)


def build_backbone(cfg) -> nn.Module:
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
    if name == "scratch_vit":
        return ScratchViT(
            embed_dim=cfg.model.embed_dim,
            dropout=cfg.model.dropout,
            in_ch=in_ch,
            img_size=32,          # Activation images are always 32x32
            patch_size=4,         # 4x4 patches -> 8x8 tokens
            depth=4,              # Number of Transformer blocks
            num_heads=4,          # Multi-head attention heads (embed_dim must be divisible by this)
            mlp_ratio=4.0,        # Expansion ratio in MLP blocks
        )

    raise ValueError(f"unknown backbone {name!r}")
