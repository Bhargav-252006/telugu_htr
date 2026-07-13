"""
src/models/cnn_encoder.py

Shared visual encoder used by both the CTC and AR models.

Architecture:
    Input [B, 1, 64, 512]
      │
      ▼
    Conv1x1(1→3)              expand grayscale for ResNet
      │
      ▼
    ResNet-18 (pretrained, backbone only — no avgpool, no fc)
    Maxpool stride:
        If high_res_temporal: stride=(2, 1)
        Else: stride=(2, 2)
    Layer3 and Layer4: stride=(1, 1) to preserve width resolution.
      → feature map [B, 512, 1, 64]   (H collapsed, W preserved)
      │
      ▼
    Squeeze height dim         [B, 512, 64]
      │
      ▼
    Conv1D(512→d_model)       channel reduction over sequence
      │
      ▼
    Positional encoding (sinusoidal)
      │
      ▼
    Encoder memory [B, 64, d_model]
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for 1-D sequences."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                  # [1, L, D]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, D]"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Stride-patched ResNet-18 builder
# ---------------------------------------------------------------------------

def _patch_resnet18_strides(model: nn.Module, high_res_temporal: bool = False) -> nn.Module:
    """
    Modify ResNet-18 to preserve width resolution for CTC/AR.
    By default (high_res_temporal=False), stride downsamples width by 8.
    If high_res_temporal=True, maxpool stride becomes (2, 1), downsampling width by 4.
    """
    # Patch maxpool
    if high_res_temporal:
        model.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)
    else:
        model.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)

    # Remove width-stride from layer3 and layer4: compress height but keep width
    for layer_name in ("layer3", "layer4"):
        layer = getattr(model, layer_name)
        first_block = layer[0]

        # Reconstruct conv1 with stride=(2,1) — setting .stride is a no-op on Conv2d!
        if hasattr(first_block, "conv1"):
            old = first_block.conv1
            new_conv = nn.Conv2d(
                old.in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=(2, 1),
                padding=old.padding, bias=(old.bias is not None),
            )
            new_conv.weight = old.weight
            if old.bias is not None:
                new_conv.bias = old.bias
            first_block.conv1 = new_conv

        # Reconstruct the downsample projection (skip connection) with stride=(2,1)
        if first_block.downsample is not None:
            old_ds = first_block.downsample[0]
            new_ds = nn.Conv2d(
                old_ds.in_channels, old_ds.out_channels,
                kernel_size=old_ds.kernel_size, stride=(2, 1),
                padding=old_ds.padding, bias=(old_ds.bias is not None),
            )
            new_ds.weight = old_ds.weight
            if old_ds.bias is not None:
                new_ds.bias = old_ds.bias
            first_block.downsample[0] = new_ds

    return model


# ---------------------------------------------------------------------------
# CNN Encoder
# ---------------------------------------------------------------------------

class ResNetEncoder(nn.Module):
    """
    ResNet-18 visual encoder producing a sequence of feature vectors.

    Parameters
    ----------
    d_model           : output feature dimension (default 256)
    pretrained        : load ImageNet weights (default True)
    dropout           : dropout on positional encoding
    high_res_temporal : use lower stride in maxpool to preserve more temporal resolution
    """

    def __init__(
        self,
        d_model: int     = 256,
        pretrained: bool = True,
        dropout: float   = 0.1,
        high_res_temporal: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.high_res_temporal = high_res_temporal
        self.width_downsample = 4 if high_res_temporal else 8

        # ---- ResNet-18 backbone with patched strides ----
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet  = tvm.resnet18(weights=weights)
        resnet  = _patch_resnet18_strides(resnet, high_res_temporal=high_res_temporal)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        # After patch: output is [B, 512, 1, 64] for input [B, 3, 64, 512]

        # ---- Channel & Height reduction: 512 → d_model via 2D conv ----
        # Takes [B, 512, 2, W'] and outputs [B, d_model, 1, W'] (for H=64)
        self.proj = nn.Sequential(
            nn.Conv2d(512, d_model, kernel_size=(2, 1), bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

        # ---- Positional encoding ----
        self.pos_enc = PositionalEncoding(d_model, max_len=512, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, 1, H, W]  grayscale image tensor in [-1, 1]

        Returns
        -------
        memory : [B, S, d_model]
                 S = W/8 = 64  for W=512, using patched strides.
                 (height is fully collapsed to 1 then squeezed)
        """
        # 1-ch → 3-ch (replicate grayscale across RGB channels for pretrained ResNet)
        x = x.repeat(1, 3, 1, 1)                  # [B, 3, H, W]

        # ResNet backbone
        feat = self.backbone(x)                   # [B, 512, 2, W'] (for H=64)

        # Spatial compression & channel reduction
        feat = self.proj(feat)                    # [B, d_model, H', W']

        # Collapse any remaining height dimension robustly
        feat = feat.mean(dim=2)                   # [B, d_model, W']

        feat = feat.permute(0, 2, 1)              # [B, W', d_model]

        # Positional encoding
        memory = self.pos_enc(feat)               # [B, S, d_model]
        return memory

    def get_output_len(self, input_width: int) -> int:
        """Compute sequence length S for a given input width W (useful for CTC)."""
        dummy = torch.zeros(1, 1, 64, input_width)
        with torch.no_grad():
            out = self.forward(dummy)
        return out.size(1)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ResNetEncoder shape test ===")
    enc = ResNetEncoder(d_model=256, pretrained=False)
    dummy = torch.randn(2, 1, 64, 512)
    out   = enc(dummy)
    print(f"Input  : {list(dummy.shape)}")   # [2, 1, 64, 512]
    print(f"Output : {list(out.shape)}")     # [2, 64, 256]  ← S=64, not 32
    seq_len = enc.get_output_len(512)
    print(f"Sequence length for W=512: {seq_len}")
