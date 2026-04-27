from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    for groups in reversed(range(1, max_groups + 1)):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            _group_norm(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + x)


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels),
            ConvNormAct(out_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
        )
        self.refine = nn.Sequential(
            ConvNormAct(out_channels, out_channels),
            ConvNormAct(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.project(x))


class HairMapDAE(nn.Module):
    """2D denoising autoencoder for packed hair maps [mask, dx, dy, depth]."""

    def __init__(
        self,
        *,
        in_channels: int = 4,
        out_channels: int = 4,
        encoder_channels: Sequence[int] = (32, 64, 128, 256),
        latent_channels: int = 128,
        bottleneck_blocks: int = 2,
    ) -> None:
        super().__init__()
        if not encoder_channels:
            raise ValueError("encoder_channels must be a non-empty sequence")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.encoder_channels = tuple(int(c) for c in encoder_channels)
        self.latent_channels = int(latent_channels)
        self.bottleneck_blocks = int(bottleneck_blocks)

        stages = []
        prev_channels = self.in_channels
        for channels in self.encoder_channels:
            stages.append(EncoderStage(prev_channels, channels))
            prev_channels = channels
        self.encoder = nn.Sequential(*stages)

        self.to_latent = nn.Sequential(
            ConvNormAct(self.encoder_channels[-1], self.latent_channels),
            ConvNormAct(self.latent_channels, self.latent_channels),
        )

        bottleneck_layers = [ResidualBlock(self.latent_channels) for _ in range(self.bottleneck_blocks)]
        self.bottleneck = nn.Sequential(*bottleneck_layers) if bottleneck_layers else nn.Identity()

        decoder_stages = []
        prev_channels = self.latent_channels
        for channels in reversed(self.encoder_channels):
            decoder_stages.append(DecoderStage(prev_channels, channels))
            prev_channels = channels
        self.decoder = nn.Sequential(*decoder_stages)

        self.output_head = nn.Sequential(
            ConvNormAct(prev_channels, prev_channels),
            nn.Conv2d(prev_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
        )

    @classmethod
    def from_config(cls, config) -> "HairMapDAE":
        dae_cfg = getattr(config, 'dae', None)
        model_cfg = getattr(dae_cfg, 'model', None)
        if model_cfg is None:
            raise ValueError("config.dae.model must be defined for HairMapDAE")
        return cls(
            in_channels=int(getattr(model_cfg, 'in_channels', 4)),
            out_channels=int(getattr(model_cfg, 'out_channels', 4)),
            encoder_channels=tuple(getattr(model_cfg, 'encoder_channels', (32, 64, 128, 256))),
            latent_channels=int(getattr(model_cfg, 'latent_channels', 128)),
            bottleneck_blocks=int(getattr(model_cfg, 'bottleneck_blocks', 2)),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        latent = self.to_latent(features)
        return self.bottleneck(latent)

    def decode(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.decoder(latent)
        raw = self.output_head(features)
        mask_logits = raw[:, :1]
        orient_raw = raw[:, 1:3]
        depth_raw = raw[:, 3:4]

        mask = torch.sigmoid(mask_logits)
        orient = torch.tanh(orient_raw)
        depth = torch.sigmoid(depth_raw)
        packed = torch.cat([mask, orient * mask, depth * mask], dim=1)
        return {
            'raw': raw,
            'mask_logits': mask_logits,
            'mask': mask,
            'orientation': orient,
            'depth': depth,
            'reconstruction': packed,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(x)
        decoded = self.decode(latent)
        decoded['latent'] = latent
        return decoded
