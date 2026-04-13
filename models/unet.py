"""
2D and 3D U-Net architectures for medical image segmentation.
Supports standard softmax output, MC Dropout, and Evidential heads.

Reference:
    Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
    Segmentation," MICCAI 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ConvBlock(nn.Module):
    """Double convolution block with optional dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
        norm: str = "instance",
        activation: str = "leakyrelu",
        dim: int = 2,
    ):
        super().__init__()
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        if norm == "instance":
            Norm = nn.InstanceNorm2d if dim == 2 else nn.InstanceNorm3d
        elif norm == "batch":
            Norm = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
        elif norm == "group":
            Norm = lambda c: nn.GroupNorm(min(32, c), c)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        Drop = nn.Dropout2d if dim == 2 else nn.Dropout3d

        if activation == "leakyrelu":
            Act = lambda: nn.LeakyReLU(0.01, inplace=True)
        elif activation == "relu":
            Act = lambda: nn.ReLU(inplace=True)
        else:
            Act = lambda: nn.LeakyReLU(0.01, inplace=True)

        layers = [
            Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            Norm(out_channels),
            Act(),
        ]
        if dropout_rate > 0:
            layers.append(Drop(p=dropout_rate))

        layers += [
            Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            Norm(out_channels),
            Act(),
        ]
        if dropout_rate > 0:
            layers.append(Drop(p=dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    Standard U-Net with configurable depth, normalization, and output heads.

    Supports:
    - Standard softmax output (num_classes channels)
    - Evidential output (num_classes Dirichlet concentration parameters)
    - MC Dropout (dropout kept active at test time)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        features: List[int] = None,
        dropout_rate: float = 0.0,
        norm: str = "instance",
        activation: str = "leakyrelu",
        dim: int = 2,
        deep_supervision: bool = False,
        evidential: bool = False,
        evidential_activation: str = "softplus",
    ):
        """
        Args:
            in_channels: Number of input channels.
            num_classes: Number of output classes.
            features: List of feature channels at each encoder level.
            dropout_rate: Dropout rate (0 = no dropout).
            dim: 2 for 2D, 3 for 3D.
            evidential: If True, output Dirichlet concentrations instead of logits.
            evidential_activation: Activation for evidential output ('softplus', 'relu', 'exp').
        """
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256, 512]

        self.num_classes = num_classes
        self.dim = dim
        self.evidential = evidential
        self.deep_supervision = deep_supervision

        Pool = nn.MaxPool2d if dim == 2 else nn.MaxPool3d
        Up = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = in_channels
        for feat in features[:-1]:
            self.encoders.append(
                ConvBlock(in_ch, feat, dropout_rate, norm, activation, dim)
            )
            self.pools.append(Pool(kernel_size=2, stride=2))
            in_ch = feat

        # Bottleneck
        self.bottleneck = ConvBlock(
            features[-2], features[-1], dropout_rate, norm, activation, dim
        )

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(len(features) - 2, -1, -1):
            self.upconvs.append(
                Up(features[i + 1], features[i], kernel_size=2, stride=2)
            )
            self.decoders.append(
                ConvBlock(features[i] * 2, features[i], dropout_rate, norm, activation, dim)
            )

        # Output head
        Conv1x1 = nn.Conv2d if dim == 2 else nn.Conv3d
        if evidential:
            # Output: Dirichlet concentration parameters (alpha)
            self.output_conv = Conv1x1(features[0], num_classes, kernel_size=1)
            if evidential_activation == "softplus":
                self.evidence_act = nn.Softplus()
            elif evidential_activation == "relu":
                self.evidence_act = nn.ReLU()
            elif evidential_activation == "exp":
                self.evidence_act = lambda x: torch.exp(torch.clamp(x, max=10))
            else:
                self.evidence_act = nn.Softplus()
        else:
            self.output_conv = Conv1x1(features[0], num_classes, kernel_size=1)

        # Deep supervision heads
        if deep_supervision:
            self.ds_heads = nn.ModuleList()
            for feat in features[:-1]:
                self.ds_heads.append(Conv1x1(feat, num_classes, kernel_size=1))

    def forward(
        self, x: torch.Tensor
    ) -> dict:
        """
        Forward pass.

        Returns:
            dict with keys:
                'logits': (B, C, H, W) raw logits or Dirichlet concentrations
                'alpha': (B, C, H, W) Dirichlet alpha parameters (if evidential)
                'evidence': (B, C, H, W) evidence values (if evidential)
                'features': list of encoder features (optional)
        """
        # Encoder path
        skip_connections = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        decoder_features = []
        for i, (upconv, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = skip_connections[-(i + 1)]

            # Handle size mismatch
            if x.shape != skip.shape:
                diff = [s - x_ for s, x_ in zip(skip.shape[2:], x.shape[2:])]
                padding = []
                for d in reversed(diff):
                    padding.extend([d // 2, d - d // 2])
                x = F.pad(x, padding)

            x = torch.cat([skip, x], dim=1)
            x = dec(x)
            decoder_features.append(x)

        # Output
        output = {}

        if self.evidential:
            raw = self.output_conv(x)
            evidence = self.evidence_act(raw)
            alpha = evidence + 1.0  # Dirichlet concentration: alpha = evidence + 1
            output["logits"] = raw
            output["evidence"] = evidence
            output["alpha"] = alpha
            # Mean prediction (expected class probabilities)
            S = alpha.sum(dim=1, keepdim=True)
            output["prob"] = alpha / S
        else:
            logits = self.output_conv(x)
            output["logits"] = logits
            output["prob"] = F.softmax(logits, dim=1)

        # Deep supervision outputs
        if self.deep_supervision and self.training:
            ds_outputs = []
            for i, feat in enumerate(decoder_features[:-1]):
                ds_out = self.ds_heads[len(self.ds_heads) - 1 - i](feat)
                # Upsample to original resolution
                ds_out = F.interpolate(ds_out, size=output["logits"].shape[2:], mode="bilinear" if self.dim == 2 else "trilinear", align_corners=False)
                ds_outputs.append(ds_out)
            output["deep_supervision"] = ds_outputs

        return output

    def enable_dropout(self):
        """Enable dropout at test time for MC Dropout."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()
