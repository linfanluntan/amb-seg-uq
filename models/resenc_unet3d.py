"""
3D Residual Encoder U-Net following the nnU-Net V2 ResEnc architecture.

This is a production-grade 3D segmentation backbone with:
- Residual connections in the encoder (pre-activation bottleneck blocks)
- Strided convolutions for downsampling (not max pooling)
- Transposed convolutions for upsampling
- Deep supervision
- Instance normalization
- Configurable depth/features auto-computed from patch size (nnU-Net style)
- Supports both standard softmax and evidential (Dirichlet) output heads

References:
    Isensee et al., "nnU-Net: A self-configuring method for deep learning-based
    biomedical image segmentation," Nature Methods 18(2):203-211, 2021.

    Isensee et al., "nnU-Net Revisited: A call for rigorous validation in 3D
    medical image segmentation," MICCAI 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import numpy as np


class ConvINReLU3D(nn.Module):
    """Conv3d -> InstanceNorm3d -> LeakyReLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride,
                       padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock3D(nn.Module):
    """
    Pre-activation residual block for 3D encoder.

    Two 3x3x3 convolutions with instance norm and LeakyReLU,
    plus a residual skip connection (with 1x1x1 projection if channels change).
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.conv1 = ConvINReLU3D(in_ch, out_ch, kernel_size=3,
                                   stride=stride, dropout=dropout)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )
        self.relu = nn.LeakyReLU(0.01, inplace=True)

        # Residual projection
        self.skip = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + identity)
        return out


class StackedResBlocks3D(nn.Module):
    """Stack of N residual blocks at one resolution level."""
    def __init__(self, in_ch: int, out_ch: int, n_blocks: int = 2,
                 stride: int = 1, dropout: float = 0.0):
        super().__init__()
        blocks = [ResidualBlock3D(in_ch, out_ch, stride=stride, dropout=dropout)]
        for _ in range(n_blocks - 1):
            blocks.append(ResidualBlock3D(out_ch, out_ch, stride=1, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ResEncUNet3D(nn.Module):
    """
    3D Residual Encoder U-Net (nnU-Net V2 style).

    Architecture:
    - Encoder: stacked residual blocks with strided convolutions for downsampling
    - Decoder: transposed convolutions + skip concatenation + conv blocks
    - Output: standard softmax logits or evidential Dirichlet parameters
    - Deep supervision at each decoder level

    The model automatically determines pool kernel sizes based on input patch
    dimensions following nnU-Net conventions.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        base_features: int = 32,
        max_features: int = 320,
        n_stages: int = 5,
        blocks_per_stage: int = 2,
        dropout: float = 0.0,
        deep_supervision: bool = True,
        evidential: bool = False,
        evidential_activation: str = "softplus",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.evidential = evidential
        self.deep_supervision = deep_supervision
        self.n_stages = n_stages

        # Compute feature maps per stage (nnU-Net: double at each stage, cap at max)
        features = []
        f = base_features
        for s in range(n_stages):
            features.append(min(f, max_features))
            f *= 2
        self.features = features

        # Compute pool/stride kernels per stage from patch size
        # nnU-Net convention: halve each spatial dim that is >= 8
        self.pool_kernels = self._compute_pool_kernels(patch_size, n_stages)

        # ========== Encoder ==========
        self.encoder_stages = nn.ModuleList()
        in_ch = in_channels
        for s in range(n_stages):
            stride = self.pool_kernels[s] if s > 0 else [1, 1, 1]
            self.encoder_stages.append(
                StackedResBlocks3D(
                    in_ch, features[s],
                    n_blocks=blocks_per_stage,
                    stride=tuple(stride) if s > 0 else 1,
                    dropout=dropout,
                )
            )
            in_ch = features[s]

        # ========== Decoder ==========
        self.decoder_upconvs = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()

        for s in range(n_stages - 2, -1, -1):
            # Transposed convolution for upsampling
            pool_k = self.pool_kernels[s + 1]
            self.decoder_upconvs.append(
                nn.ConvTranspose3d(
                    features[s + 1], features[s],
                    kernel_size=pool_k, stride=pool_k, bias=False,
                )
            )
            # Decoder conv block (after concatenation with skip)
            self.decoder_stages.append(
                StackedResBlocks3D(
                    features[s] * 2, features[s],
                    n_blocks=blocks_per_stage,
                    stride=1, dropout=dropout,
                )
            )

        # ========== Output heads ==========
        if evidential:
            self.output_conv = nn.Conv3d(features[0], num_classes, 1)
            if evidential_activation == "softplus":
                self.evidence_act = nn.Softplus()
            elif evidential_activation == "relu":
                self.evidence_act = nn.ReLU()
            elif evidential_activation == "exp":
                self.evidence_act = lambda x: torch.exp(torch.clamp(x, max=10))
            else:
                self.evidence_act = nn.Softplus()
        else:
            self.output_conv = nn.Conv3d(features[0], num_classes, 1)

        # Deep supervision heads (one per decoder level except the last)
        if deep_supervision:
            self.ds_heads = nn.ModuleList()
            for s in range(n_stages - 2, 0, -1):
                self.ds_heads.append(nn.Conv3d(features[s], num_classes, 1))

        self._init_weights()

    def _compute_pool_kernels(
        self, patch_size: Tuple[int, ...], n_stages: int
    ) -> List[List[int]]:
        """
        Compute per-stage pool kernel sizes from patch dimensions.
        nnU-Net convention: stride 2 along axes that are still >= some minimum (8).
        Stage 0 has no pooling (stride=1).
        """
        kernels = [[1, 1, 1]]  # Stage 0: no downsampling
        current_size = list(patch_size)
        for s in range(1, n_stages):
            k = []
            for d in range(3):
                if current_size[d] >= 8:
                    k.append(2)
                    current_size[d] //= 2
                else:
                    k.append(1)
            kernels.append(k)
        return kernels

    def _init_weights(self):
        """Kaiming initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode="fan_out",
                                        nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C_in, D, H, W) input volume

        Returns:
            dict with keys:
                'logits': (B, num_classes, D, H, W)
                'prob': (B, num_classes, D, H, W) softmax or Dirichlet mean
                'alpha', 'evidence': only if evidential=True
                'deep_supervision': list of logits at lower resolutions
        """
        # Encoder: collect skip features at each stage
        skips = []
        out = x
        for s, enc in enumerate(self.encoder_stages):
            out = enc(out)
            skips.append(out)

        # Decoder: bottom-up with skip connections
        decoder_outputs = []
        for i, (upconv, dec) in enumerate(
            zip(self.decoder_upconvs, self.decoder_stages)
        ):
            # The skip index: from second-to-last stage going up
            skip_idx = self.n_stages - 2 - i
            up = upconv(out)
            skip = skips[skip_idx]

            # Handle spatial size mismatch (can happen with odd dimensions)
            if up.shape[2:] != skip.shape[2:]:
                up = F.interpolate(up, size=skip.shape[2:], mode="trilinear",
                                    align_corners=False)

            out = torch.cat([skip, up], dim=1)
            out = dec(out)
            decoder_outputs.append(out)

        # Final output
        result = {}
        if self.evidential:
            raw = self.output_conv(out)
            evidence = self.evidence_act(raw)
            alpha = evidence + 1.0
            S = alpha.sum(dim=1, keepdim=True)
            result["logits"] = raw
            result["evidence"] = evidence
            result["alpha"] = alpha
            result["prob"] = alpha / S
        else:
            logits = self.output_conv(out)
            result["logits"] = logits
            result["prob"] = F.softmax(logits, dim=1)

        # Deep supervision
        if self.deep_supervision and self.training:
            ds_list = []
            for i, ds_head in enumerate(self.ds_heads):
                feat = decoder_outputs[i]
                ds_logits = ds_head(feat)
                ds_list.append(ds_logits)
            result["deep_supervision"] = ds_list

        return result

    def enable_dropout(self):
        """Enable dropout at test time for MC Dropout inference."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout3d, nn.Dropout)):
                m.train()

    def get_output_size(self, input_size: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute output spatial size given an input spatial size."""
        # For a valid U-Net the output size equals the input size
        return input_size
