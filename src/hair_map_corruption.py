from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.synthetic_hair_map_generator import SyntheticHairMapBatch, SyntheticHairMapGenerator


@dataclass
class CorruptionOutput:
    corrupted_map: torch.Tensor
    pre_render_map: torch.Tensor
    applied_families: List[List[str]]


class HairMapCorruptor(nn.Module):
    """Configurable corruption pipeline for packed hair maps."""

    FAMILY_ORDER = (
        'support',
        'strand_dropout',
        'blur_resample',
        'orientation_jitter',
        'orientation_sign_flip',
        'orientation_confidence',
        'depth_holes',
        'depth_drift',
        'misregistration',
        'partial_occlusion',
        'channel_inconsistency',
    )

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.cfg = getattr(getattr(config, 'dae', None), 'corruption', None)
        if self.cfg is None:
            raise ValueError("config.dae.corruption must be defined")

    def forward(
        self,
        bundle: SyntheticHairMapBatch,
        generator: SyntheticHairMapGenerator,
        *,
        phase: str = 'train',
        families_per_sample: Optional[Sequence[Sequence[str]]] = None,
    ) -> CorruptionOutput:
        batch_size = bundle.clean_map.shape[0]
        corrupted = bundle.clean_map.clone()
        applied_families: List[List[str]] = []

        if families_per_sample is None:
            selections = [self._select_families() if phase == 'train' else self._select_families() for _ in range(batch_size)]
        else:
            if len(families_per_sample) != batch_size:
                raise ValueError(
                    f"families_per_sample must have length {batch_size}, got {len(families_per_sample)}."
                )
            selections = [self._validate_family_names(sample_families) for sample_families in families_per_sample]

        rerender_mask = bundle.strand_mask
        if rerender_mask is not None:
            rerender_mask = rerender_mask.clone()
            for batch_idx, families in enumerate(selections):
                if 'strand_dropout' not in families:
                    continue
                rerender_mask[batch_idx] = self._apply_strand_dropout_mask(rerender_mask[batch_idx])

        if rerender_mask is not None and bundle.strand_mask is not None and not torch.equal(rerender_mask, bundle.strand_mask):
            corrupted = generator.rerender_with_strand_mask(bundle, rerender_mask)

        pre_render_map = corrupted.clone()
        for batch_idx, families in enumerate(selections):
            sample = corrupted[batch_idx]
            active_families = []
            for family in families:
                if family == 'strand_dropout':
                    active_families.append(family)
                    continue
                sample = self._apply_family(sample, family)
                active_families.append(family)
            sample = self._apply_final_pose_jitter(sample)
            corrupted[batch_idx] = sample
            applied_families.append(active_families)

        return CorruptionOutput(
            corrupted_map=corrupted,
            pre_render_map=pre_render_map,
            applied_families=applied_families,
        )

    def available_families(self, *, include_disabled: bool = True) -> List[str]:
        family_names = []
        for name in self.FAMILY_ORDER:
            family_cfg = getattr(self.cfg, name, None)
            if family_cfg is None:
                continue
            weight = float(getattr(family_cfg, 'weight', 0.0))
            if include_disabled or weight > 0.0:
                family_names.append(name)
        return family_names

    def _select_families(self) -> List[str]:
        min_ops = int(getattr(self.cfg, 'min_ops_per_sample', 2))
        max_ops = int(getattr(self.cfg, 'max_ops_per_sample', 4))
        num_ops = int(torch.randint(min_ops, max_ops + 1, (1,)).item())

        weights = []
        family_names = []
        for name in self.FAMILY_ORDER:
            family_cfg = getattr(self.cfg, name, None)
            if family_cfg is None:
                continue
            weight = float(getattr(family_cfg, 'weight', 0.0))
            if weight <= 0:
                continue
            family_names.append(name)
            weights.append(weight)

        if not family_names:
            return []

        probs = torch.tensor(weights, dtype=torch.float32)
        probs = probs / probs.sum().clamp_min(1e-8)
        order = torch.multinomial(probs, num_samples=min(num_ops, len(family_names)), replacement=False)
        return [family_names[idx] for idx in order.tolist()]

    def _validate_family_names(self, family_names: Sequence[str]) -> List[str]:
        available = set(self.available_families(include_disabled=True))
        cleaned = []
        for name in family_names:
            if name not in available:
                raise ValueError(f"Unknown corruption family `{name}`. Available families: {sorted(available)}")
            cleaned.append(str(name))
        return cleaned

    def _apply_family(self, sample: torch.Tensor, family: str) -> torch.Tensor:
        if family == 'support':
            return self._apply_support(sample)
        if family == 'blur_resample':
            return self._apply_blur_resample(sample)
        if family == 'orientation_jitter':
            return self._apply_orientation_jitter(sample)
        if family == 'orientation_sign_flip':
            return self._apply_orientation_sign_flip(sample)
        if family == 'orientation_confidence':
            return self._apply_orientation_confidence(sample)
        if family == 'depth_holes':
            return self._apply_depth_holes(sample)
        if family == 'depth_drift':
            return self._apply_depth_drift(sample)
        if family == 'misregistration':
            return self._apply_misregistration(sample)
        if family == 'partial_occlusion':
            return self._apply_partial_occlusion(sample)
        if family == 'channel_inconsistency':
            return self._apply_channel_inconsistency(sample)
        return sample

    def _apply_strand_dropout_mask(self, strand_mask: torch.Tensor) -> torch.Tensor:
        cfg = getattr(self.cfg, 'strand_dropout')
        keep_ratio = float(torch.empty(1).uniform_(
            float(getattr(cfg, 'keep_ratio_min', 0.45)),
            float(getattr(cfg, 'keep_ratio_max', 0.85)),
        ).item())
        random_keep = torch.rand_like(strand_mask.float()) < keep_ratio
        dropped = strand_mask.bool() & random_keep
        if dropped.any():
            return dropped
        return strand_mask

    def _apply_support(self, sample: torch.Tensor) -> torch.Tensor:
        cfg = getattr(self.cfg, 'support')
        choice = self._random_choice(('morphology', 'holes', 'boundary_trim'))
        mask = sample[:1].clone()

        if choice == 'morphology':
            kernel = int(torch.randint(
                int(getattr(cfg, 'kernel_min', 3)),
                int(getattr(cfg, 'kernel_max', 9)) + 1,
                (1,),
            ).item())
            kernel = kernel + 1 if kernel % 2 == 0 else kernel
            pad = kernel // 2
            if bool(torch.randint(0, 2, (1,)).item()):
                mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=pad)
            else:
                mask = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel, stride=1, padding=pad)
        elif choice == 'holes':
            mask = self._apply_rect_holes(
                mask,
                max_holes=int(getattr(cfg, 'max_holes', 5)),
                size_min=float(getattr(cfg, 'hole_size_min', 0.03)),
                size_max=float(getattr(cfg, 'hole_size_max', 0.12)),
            )
        else:
            mask = self._apply_boundary_trim(
                mask,
                fraction_min=float(getattr(cfg, 'trim_fraction_min', 0.05)),
                fraction_max=float(getattr(cfg, 'trim_fraction_max', 0.18)),
            )

        mask = self._ensure_nonempty_mask(mask, sample[:1])
        sample = sample.clone()
        sample[:1] = mask
        sample[1:3] = sample[1:3] * mask
        sample[3:4] = sample[3:4] * mask
        return sample

    def _apply_blur_resample(self, sample: torch.Tensor) -> torch.Tensor:
        cfg = getattr(self.cfg, 'blur_resample')
        scale = float(torch.empty(1).uniform_(
            float(getattr(cfg, 'scale_min', 0.45)),
            float(getattr(cfg, 'scale_max', 0.85)),
        ).item())
        h, w = sample.shape[-2:]
        down_h = max(16, int(round(h * scale)))
        down_w = max(16, int(round(w * scale)))
        down = F.interpolate(sample.unsqueeze(0), size=(down_h, down_w), mode='bilinear', align_corners=False)
        up = F.interpolate(down, size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
        up[:1] = up[:1].clamp(0.0, 1.0)
        up[3:4] = up[3:4].clamp(0.0, 1.0)
        return up

    def _apply_orientation_jitter(self, sample: torch.Tensor) -> torch.Tensor:
        cfg = getattr(self.cfg, 'orientation_jitter')
        max_angle_deg = float(getattr(cfg, 'max_angle_deg', 18.0))
        angle_field = self._smooth_noise_field(
            sample.shape[-2:],
            amplitude=max_angle_deg * torch.pi / 180.0,
            low_res=int(getattr(cfg, 'low_res', 32)),
            device=sample.device,
            dtype=sample.dtype,
        )
        cos_theta = torch.cos(angle_field)
        sin_theta = torch.sin(angle_field)
        dx = sample[1:2]
        dy = sample[2:3]
        rotated_x = dx * cos_theta - dy * sin_theta
        rotated_y = dx * sin_theta + dy * cos_theta
        out = sample.clone()
        out[1:2] = rotated_x * out[:1]
        out[2:3] = rotated_y * out[:1]
        return out

    def _apply_orientation_sign_flip(self, sample: torch.Tensor) -> torch.Tensor:
        cfg = getattr(self.cfg, 'orientation_sign_flip')
        out = sample.clone()
        support_mask = out[:1] > 0.5
        if not bool(support_mask.any()):
            return out

        _, height, width = out.shape
        max_regions = int(getattr(cfg, 'max_regions', 4))
        size_min = float(getattr(cfg, 'size_min', 0.03))
        size_max = float(getattr(cfg, 'size_max', 0.10))
        num_regions = int(torch.randint(1, max(2, max_regions + 1), (1,)).item())

        for _ in range(num_regions):
            region_h = max(1, int(round(height * float(torch.empty(1).uniform_(size_min, size_max).item()))))
            region_w = max(1, int(round(width * float(torch.empty(1).uniform_(size_min, size_max).item()))))
            top = int(torch.randint(0, max(1, height - region_h + 1), (1,)).item())
            left = int(torch.randint(0, max(1, width - region_w + 1), (1,)).item())

            region_support = support_mask[:, top:top + region_h, left:left + region_w]
            if not bool(region_support.any()):
                continue

            region_sign = region_support.to(dtype=out.dtype)
            out[1:3, top:top + region_h, left:left + region_w] = (
                -out[1:3, top:top + region_h, left:left + region_w] * region_sign
                + out[1:3, top:top + region_h, left:left + region_w] * (1.0 - region_sign)
            )

        out[1:3] = out[1:3] * out[:1]
        return out

    def _apply_orientation_confidence(self, sample: torch.Tensor) -> torch.Tensor:
        cfg = getattr(self.cfg, 'orientation_confidence')
        attenuation = self._smooth_noise_field(
            sample.shape[-2:],
            amplitude=1.0,
            low_res=int(getattr(cfg, 'low_res', 24)),
            device=sample.device,
            dtype=sample.dtype,
            positive_only=True,
        )
        attenuation = attenuation.clamp(
            float(getattr(cfg, 'attenuation_min', 0.2)),
            float(getattr(cfg, 'attenuation_max', 0.8)),
        )
        out = sample.clone()
        out[1:3] = out[1:3] * attenuation * out[:1]
        return out

    def _apply_depth_holes(self, sample: torch.Tensor) -> torch.Tensor:
        cfg = getattr(self.cfg, 'depth_holes')
        out = sample.clone()
        depth_mask = self._apply_rect_holes(
            out[:1].clone(),
            max_holes=int(getattr(cfg, 'max_holes', 4)),
            size_min=float(getattr(cfg, 'hole_size_min', 0.02)),
            size_max=float(getattr(cfg, 'hole_size_max', 0.10)),
        )
        removed = (out[:1] > 0.5) & (depth_mask < 0.5)
        out[3:4][removed] = 0.0
        return out

    def _apply_depth_drift(self, sample: torch.Tensor) -> torch.Tensor:
        cfg = getattr(self.cfg, 'depth_drift')
        drift = self._smooth_noise_field(
            sample.shape[-2:],
            amplitude=float(getattr(cfg, 'max_abs_bias', 0.12)),
            low_res=int(getattr(cfg, 'low_res', 24)),
            device=sample.device,
            dtype=sample.dtype,
        )
        out = sample.clone()
        out[3:4] = (out[3:4] + drift * out[:1]).clamp(0.0, 1.0)
        return out

    def _apply_misregistration(self, sample: torch.Tensor) -> torch.Tensor:
        cfg = getattr(self.cfg, 'misregistration')
        shift_px = int(getattr(cfg, 'max_shift_px', 8))
        shift_x = int(torch.randint(-shift_px, shift_px + 1, (1,)).item())
        shift_y = int(torch.randint(-shift_px, shift_px + 1, (1,)).item())
        out = sample.clone()
        shifted = self._translate(out[1:4], shift_x=shift_x, shift_y=shift_y)
        out[1:4] = shifted
        out[3:4] = out[3:4].clamp(0.0, 1.0)
        return out

    def _apply_partial_occlusion(self, sample: torch.Tensor) -> torch.Tensor:
        cfg = getattr(self.cfg, 'partial_occlusion')
        mask = sample[:1].clone()
        occluded = self._apply_rect_holes(
            mask,
            max_holes=int(getattr(cfg, 'max_regions', 2)),
            size_min=float(getattr(cfg, 'size_min', 0.08)),
            size_max=float(getattr(cfg, 'size_max', 0.25)),
        )
        occluded = self._ensure_nonempty_mask(occluded, sample[:1])
        out = sample.clone()
        out[:1] = occluded
        out[1:3] = out[1:3] * occluded
        out[3:4] = out[3:4] * occluded
        return out

    def _apply_channel_inconsistency(self, sample: torch.Tensor) -> torch.Tensor:
        choice = self._random_choice(('orientation_only', 'depth_only'))
        if choice == 'orientation_only':
            return self._apply_orientation_confidence(sample)
        return self._apply_depth_drift(sample)

    def _apply_final_pose_jitter(self, sample: torch.Tensor) -> torch.Tensor:
        cfg = getattr(self.cfg, 'final_pose_jitter', None)
        if cfg is None or not bool(getattr(cfg, 'enabled', False)):
            return sample

        apply_probability = float(getattr(cfg, 'apply_probability', 1.0))
        if apply_probability <= 0.0:
            return sample
        if apply_probability < 1.0 and float(torch.rand(1).item()) > apply_probability:
            return sample

        max_rotation_deg = float(getattr(cfg, 'max_rotation_deg', 0.0))
        max_translation_px = float(getattr(cfg, 'max_translation_px', 0.0))
        if max_rotation_deg <= 0.0 and max_translation_px <= 0.0:
            return sample

        angle_deg = float(torch.empty(1).uniform_(-max_rotation_deg, max_rotation_deg).item())
        angle_rad = angle_deg * torch.pi / 180.0
        shift_x_px = float(torch.empty(1).uniform_(-max_translation_px, max_translation_px).item())
        shift_y_px = float(torch.empty(1).uniform_(-max_translation_px, max_translation_px).item())

        _, height, width = sample.shape
        cos_theta = float(torch.cos(torch.tensor(angle_rad)).item())
        sin_theta = float(torch.sin(torch.tensor(angle_rad)).item())
        tx = 2.0 * shift_x_px / max(1.0, float(width))
        ty = 2.0 * shift_y_px / max(1.0, float(height))

        theta = sample.new_tensor([
            [cos_theta, -sin_theta, tx],
            [sin_theta, cos_theta, ty],
        ]).unsqueeze(0)
        grid = F.affine_grid(theta, size=(1, sample.shape[0], height, width), align_corners=False)
        warped = F.grid_sample(
            sample.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        ).squeeze(0)

        dx = warped[1:2]
        dy = warped[2:3]
        rotated_x = dx * cos_theta - dy * sin_theta
        rotated_y = dx * sin_theta + dy * cos_theta

        out = warped.clone()
        final_mask = self._ensure_nonempty_mask(out[:1].clamp(0.0, 1.0), sample[:1])
        out[:1] = final_mask
        out[1:2] = rotated_x * final_mask
        out[2:3] = rotated_y * final_mask
        out[3:4] = out[3:4].clamp(0.0, 1.0) * final_mask
        return out

    def _apply_rect_holes(self, mask: torch.Tensor, *, max_holes: int, size_min: float, size_max: float) -> torch.Tensor:
        out = mask.clone()
        _, h, w = out.shape
        num_holes = int(torch.randint(1, max(2, max_holes + 1), (1,)).item())
        for _ in range(num_holes):
            hole_h = max(1, int(round(h * float(torch.empty(1).uniform_(size_min, size_max).item()))))
            hole_w = max(1, int(round(w * float(torch.empty(1).uniform_(size_min, size_max).item()))))
            top = int(torch.randint(0, max(1, h - hole_h + 1), (1,)).item())
            left = int(torch.randint(0, max(1, w - hole_w + 1), (1,)).item())
            out[:, top:top + hole_h, left:left + hole_w] = 0.0
        return out

    def _apply_boundary_trim(self, mask: torch.Tensor, *, fraction_min: float, fraction_max: float) -> torch.Tensor:
        out = mask.clone()
        _, h, w = out.shape
        fraction = float(torch.empty(1).uniform_(fraction_min, fraction_max).item())
        side = self._random_choice(('left', 'right', 'top', 'bottom'))
        if side in {'left', 'right'}:
            width = max(1, int(round(w * fraction)))
            if side == 'left':
                out[:, :, :width] = 0.0
            else:
                out[:, :, w - width:] = 0.0
        else:
            height = max(1, int(round(h * fraction)))
            if side == 'top':
                out[:, :height, :] = 0.0
            else:
                out[:, h - height:, :] = 0.0
        return out

    def _ensure_nonempty_mask(self, new_mask: torch.Tensor, fallback_mask: torch.Tensor) -> torch.Tensor:
        if float(new_mask.sum()) > 1e-6:
            return new_mask.clamp(0.0, 1.0)
        return fallback_mask.clamp(0.0, 1.0)

    def _translate(self, tensor: torch.Tensor, *, shift_x: int, shift_y: int) -> torch.Tensor:
        if shift_x == 0 and shift_y == 0:
            return tensor
        out = torch.roll(tensor, shifts=(shift_y, shift_x), dims=(-2, -1))
        if shift_y > 0:
            out[..., :shift_y, :] = 0.0
        elif shift_y < 0:
            out[..., shift_y:, :] = 0.0
        if shift_x > 0:
            out[..., :, :shift_x] = 0.0
        elif shift_x < 0:
            out[..., :, shift_x:] = 0.0
        return out

    def _smooth_noise_field(
        self,
        hw,
        *,
        amplitude: float,
        low_res: int,
        device,
        dtype,
        positive_only: bool = False,
    ) -> torch.Tensor:
        h, w = int(hw[0]), int(hw[1])
        low_h = max(2, h // max(1, low_res))
        low_w = max(2, w // max(1, low_res))
        noise = torch.randn(1, 1, low_h, low_w, device=device, dtype=dtype)
        noise = F.interpolate(noise, size=(h, w), mode='bilinear', align_corners=False)
        if positive_only:
            noise = noise.sigmoid()
        else:
            noise = noise.tanh() * amplitude
        return noise.squeeze(0)

    @staticmethod
    def _random_choice(choices):
        index = int(torch.randint(0, len(choices), (1,)).item())
        return choices[index]
