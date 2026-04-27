from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.FLAME.FLAME import FLAME
from src.helpers import FLAMEHairStrandAttachment, HairStrandRasterizer, HairTemplateManager
from src.smirk_encoder import SmirkEncoder
import src.utils.utils as utils


@dataclass
class SyntheticHairMapBatch:
    clean_map: torch.Tensor
    visibility_mask: torch.Tensor
    depth_map: torch.Tensor
    strands: torch.Tensor
    strand_mask: Optional[torch.Tensor]
    flame_vertices: torch.Tensor
    flame_faces: torch.Tensor
    cam_params: torch.Tensor
    template_paths: list[str]
    coarse_map: Optional[torch.Tensor] = None
    coarse_visibility_mask: Optional[torch.Tensor] = None
    coarse_depth_map: Optional[torch.Tensor] = None


class SyntheticHairMapGenerator(nn.Module):
    """Online synthetic pair generator that mirrors the current step-2 render path."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.force_closed_mouth = bool(getattr(self.config.train, 'force_closed_mouth', False))
        self.zero_flame_expression = bool(getattr(self.config.train, 'zero_flame_expression', False))

        self.smirk_face_encoder = SmirkEncoder(
            n_exp=self.config.arch.num_expression,
            n_shape=self.config.arch.num_shape,
        )
        self._load_smirk_encoder_weights(getattr(self.config, 'checkpoint_smirk', None))
        utils.freeze_module(self.smirk_face_encoder, 'smirk face encoder')
        self.smirk_face_encoder.eval()

        self.flame = FLAME(
            n_exp=self.config.arch.num_expression,
            n_shape=self.config.arch.num_shape,
        )
        for param in self.flame.parameters():
            param.requires_grad_(False)
        self.flame.eval()

        hair20k_cfg = getattr(self.config.dataset, 'Hair20k', None)
        blendshape_override = None
        coarse_dim = 10
        template_dir_override = None
        load_roots_override = True
        aug_override = None
        root_cache_cfg = None
        if hair20k_cfg is not None:
            blendshape_override = getattr(hair20k_cfg, 'blenshape_path', None) or getattr(hair20k_cfg, 'blendshape_path', None)
            coarse_dim = int(getattr(hair20k_cfg, 'coarse_dim', coarse_dim))
            template_dir_override = getattr(hair20k_cfg, 'Hair20k_path', None)
            load_roots_override = getattr(hair20k_cfg, 'load_roots', True)
            aug_override = getattr(hair20k_cfg, 'use_augmentation', None)
            root_cache_cfg = {
                'enabled': getattr(hair20k_cfg, 'root_cache_enabled', False),
                'root_scale': getattr(hair20k_cfg, 'root_scale', 2.5),
                'workers': getattr(hair20k_cfg, 'root_cache_workers', 16),
                'knn_k': getattr(hair20k_cfg, 'root_cache_knn_k', 8),
                'force_rebuild': getattr(hair20k_cfg, 'root_cache_force_rebuild', False),
                'lazy_init': getattr(hair20k_cfg, 'root_cache_lazy_init', None),
                'lock_timeout_sec': getattr(hair20k_cfg, 'root_cache_lock_timeout_sec', None),
                'lock_poll_sec': getattr(hair20k_cfg, 'root_cache_lock_poll_sec', None),
                'stale_lock_sec': getattr(hair20k_cfg, 'root_cache_stale_lock_sec', None),
            }

        perm_cfg = getattr(self.config, 'perm', None)
        scalp_bounds = getattr(perm_cfg, 'scalp_bounds', getattr(self.config.arch, 'scalp_bounds', None))
        strand_basis_path = blendshape_override or getattr(
            self.config.arch,
            'strand_basis_path',
            'assets/blend-shapes/strands-blend-shapes.npz',
        )
        hair_template_aug_cfg = getattr(self.config.train, 'hair_template_augmentation', None)
        fine_detail_cfg = self._extract_nested_cfg(hair_template_aug_cfg, 'fine_detail')
        self.load_template_roots = bool(load_roots_override)
        self.hair_attachment = FLAMEHairStrandAttachment(
            flame_model=self.flame,
            scale_to_perm=getattr(self.config.arch, 'perm_scale', 100.0),
            translation_to_perm=tuple(getattr(self.config.arch, 'perm_translation', (0.0, 0.0, 0.0))),
            rotation_to_perm_euler_deg=tuple(getattr(self.config.arch, 'perm_rotation_euler_deg', (0.0, 0.0, 0.0))),
            perm_head_mesh_path=getattr(self.config.arch, 'perm_head_mesh_path', None),
            scalp_bounds=scalp_bounds,
            mask_threshold=getattr(self.config.arch, 'mask_threshold', 0.5),
            max_mask_samples=getattr(self.config.arch, 'max_mask_samples', None),
            strand_basis_path=strand_basis_path,
            coarse_dim=coarse_dim,
            fine_detail_cfg=fine_detail_cfg,
            enable_pre_render_culling=getattr(self.config.arch, 'enable_pre_render_culling', False),
        )
        for param in self.hair_attachment.parameters():
            param.requires_grad_(False)
        self.hair_attachment.eval()

        rasterizer_cull_cfg = getattr(self.config.arch, 'rasterizer_strand_culling', None)
        self.hair_rasterizer = HairStrandRasterizer(
            image_size=getattr(self.config.arch, 'hair_render_size', getattr(self.config.dataset, 'resolution', 512)),
            background_color=(0.0, 0.0, 0.0),
            head_scale=getattr(self.config.arch, 'pre_render_head_scale', 0.96),
            enable_strand_culling=bool(
                getattr(
                    rasterizer_cull_cfg,
                    'enabled',
                    getattr(self.config.arch, 'enable_rasterizer_strand_culling', False),
                )
            ),
            strand_culling_root_segments=getattr(
                rasterizer_cull_cfg,
                'root_segments',
                getattr(self.config.arch, 'rasterizer_strand_culling_root_segments', 4),
            ),
            strand_culling_min_occluded_pixels=getattr(
                rasterizer_cull_cfg,
                'min_occluded_pixels',
                getattr(self.config.arch, 'rasterizer_strand_culling_min_occluded_pixels', 1),
            ),
            strand_culling_min_occluded_fraction=getattr(
                rasterizer_cull_cfg,
                'min_occluded_fraction',
                getattr(self.config.arch, 'rasterizer_strand_culling_min_occluded_fraction', 0.0),
            ),
        )

        aug_cfg = self._make_aug_cfg(hair_template_aug_cfg, aug_override)
        template_dir = template_dir_override or getattr(self.config.dataset, 'hair_template_dir', None)
        self.hair_template_manager = HairTemplateManager(
            template_dir=template_dir,
            aug_cfg=aug_cfg,
            load_roots=self.load_template_roots,
            root_cache_cfg=root_cache_cfg,
            scalp_bounds=scalp_bounds,
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        include_coarse_map: bool = False,
        hair_templates: Optional[Dict[str, object]] = None,
        enable_fine_detail_aug: Optional[bool] = None,
        fine_detail_gain_override: Optional[float] = None,
        fine_detail_noise_std_override: Optional[float] = None,
        force_fine_detail_apply: Optional[bool] = None,
    ) -> Optional[SyntheticHairMapBatch]:
        with torch.no_grad():
            _, flame_params, _, cam_params = self._predict_flame_params(batch)
            if hair_templates is None:
                hair_templates = self.hair_template_manager.sample(batch['img'].shape[0], batch['img'].device)
            if hair_templates is None:
                return None

            attachment_out = self.hair_attachment(
                flame_params=flame_params,
                hair_textures=hair_templates['texture'],
                roots=hair_templates.get('roots') if self.load_template_roots else None,
                scalp_masks=hair_templates['mask'],
                return_low_frequency=include_coarse_map,
                enable_fine_detail_aug=enable_fine_detail_aug,
                fine_detail_gain_override=fine_detail_gain_override,
                fine_detail_noise_std_override=fine_detail_noise_std_override,
                force_fine_detail_apply=force_fine_detail_apply,
                return_flame_mesh=False,
                debug_dump=False,
                debug_dump_hit_faces=False,
            )
            strand_mask = attachment_out.metadata.get('strand_visibility')
            raster_out = self._rasterize_strands(
                flame_vertices=attachment_out.metadata['flame_vertices'],
                flame_faces=attachment_out.metadata['flame_faces'],
                strands=attachment_out.full_resolution,
                cam_params=cam_params,
                strand_mask=strand_mask,
            )
            coarse_raster_out = None
            if include_coarse_map:
                if attachment_out.low_frequency is None:
                    raise RuntimeError("Coarse Hair20k render requested but no low-frequency strands were returned.")
                coarse_raster_out = self._rasterize_strands(
                    flame_vertices=attachment_out.metadata['flame_vertices'],
                    flame_faces=attachment_out.metadata['flame_faces'],
                    strands=attachment_out.low_frequency,
                    cam_params=cam_params,
                    strand_mask=strand_mask,
                )
            return SyntheticHairMapBatch(
                clean_map=raster_out.image,
                visibility_mask=raster_out.visibility_mask,
                depth_map=raster_out.depth,
                strands=attachment_out.full_resolution,
                strand_mask=strand_mask,
                flame_vertices=attachment_out.metadata['flame_vertices'],
                flame_faces=attachment_out.metadata['flame_faces'],
                cam_params=cam_params,
                template_paths=list(hair_templates['paths']),
                coarse_map=None if coarse_raster_out is None else coarse_raster_out.image,
                coarse_visibility_mask=None if coarse_raster_out is None else coarse_raster_out.visibility_mask,
                coarse_depth_map=None if coarse_raster_out is None else coarse_raster_out.depth,
            )

    def rerender_with_strand_mask(
        self,
        bundle: SyntheticHairMapBatch,
        strand_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        with torch.no_grad():
            rerender = self._rasterize_strands(
                flame_vertices=bundle.flame_vertices,
                flame_faces=bundle.flame_faces,
                strands=bundle.strands,
                cam_params=bundle.cam_params,
                strand_mask=strand_mask,
            )
            return rerender.image

    def _rasterize_strands(
        self,
        *,
        flame_vertices: torch.Tensor,
        flame_faces: torch.Tensor,
        strands: torch.Tensor,
        cam_params: torch.Tensor,
        strand_mask: Optional[torch.Tensor],
    ):
        return self.hair_rasterizer.forward(
            flame_vertices=flame_vertices,
            flame_faces=flame_faces,
            strands=strands,
            cam_params=cam_params,
            strand_mask=strand_mask,
        )

    def _load_smirk_encoder_weights(self, checkpoint_path: Optional[str]) -> None:
        if not checkpoint_path:
            raise ValueError("config.checkpoint_smirk must be provided to initialize the smirk encoder.")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint_smirk not found at {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location='cpu')
        encoder_state = {
            key.replace('smirk_encoder.', '', 1): value
            for key, value in state_dict.items()
            if key.startswith('smirk_encoder')
        }
        missing, unexpected = self.smirk_face_encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            print(f"[SyntheticHairMapGenerator] Missing smirk encoder keys: {missing}")
        if unexpected:
            print(f"[SyntheticHairMapGenerator] Unexpected smirk encoder keys: {unexpected}")

    def _apply_flame_debug_overrides(self, flame_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not (self.force_closed_mouth or self.zero_flame_expression):
            return flame_params

        overridden = dict(flame_params)
        if self.zero_flame_expression and overridden.get('expression_params') is not None:
            overridden['expression_params'] = torch.zeros_like(overridden['expression_params'])
        if self.force_closed_mouth and overridden.get('jaw_params') is not None:
            jaw_params = overridden['jaw_params'].clone()
            jaw_params[..., 0] = 0.0
            overridden['jaw_params'] = jaw_params
        return overridden

    def _predict_flame_params(self, batch, *, return_raw: bool = False):
        smirk_img = batch.get('smirk_img', batch['img'])
        raw_flame_params = self.smirk_face_encoder(smirk_img)
        flame_params = self._apply_flame_debug_overrides(raw_flame_params)

        crop_cam_params = raw_flame_params.get(
            'cam',
            torch.zeros(
                raw_flame_params['pose_params'].shape[0],
                3,
                device=raw_flame_params['pose_params'].device,
                dtype=batch['img'].dtype,
            ),
        )
        crop_transform = batch.get('smirk_crop_transform')
        cam_params = self._remap_smirk_camera_to_full(
            crop_cam_params,
            crop_transform,
            smirk_img.shape[-2],
            smirk_img.shape[-1],
            batch['img'].shape[-2],
            batch['img'].shape[-1],
        )
        if return_raw:
            return smirk_img, flame_params, raw_flame_params, crop_cam_params, cam_params
        return smirk_img, flame_params, crop_cam_params, cam_params

    def _crop_ndc_to_full_ndc(self, points_ndc, crop_transform, crop_h, crop_w, full_h, full_w):
        crop_transform = crop_transform.to(points_ndc.device, dtype=points_ndc.dtype)
        inv_transform = torch.linalg.inv(crop_transform)

        crop_x = (points_ndc[..., 0] + 1.0) * (float(crop_w) * 0.5)
        crop_y = (points_ndc[..., 1] + 1.0) * (float(crop_h) * 0.5)
        crop_points = torch.stack([crop_x, crop_y, torch.ones_like(crop_x)], dim=-1)

        full_points = torch.einsum('bij,bnj->bni', inv_transform, crop_points)
        full_xy = full_points[..., :2] / full_points[..., 2:].clamp_min(1e-8)

        full_x = full_xy[..., 0] / float(full_w) * 2.0 - 1.0
        full_y = full_xy[..., 1] / float(full_h) * 2.0 - 1.0
        return torch.stack([full_x, full_y], dim=-1)

    def _remap_smirk_camera_to_full(self, crop_cam, crop_transform, crop_h, crop_w, full_h, full_w):
        if crop_transform is None:
            return crop_cam

        center = torch.stack(
            [crop_cam[:, 0] * crop_cam[:, 1], crop_cam[:, 0] * crop_cam[:, 2]],
            dim=-1,
        )
        x_plus = torch.stack(
            [crop_cam[:, 0] * (1.0 + crop_cam[:, 1]), crop_cam[:, 0] * crop_cam[:, 2]],
            dim=-1,
        )
        y_plus = torch.stack(
            [crop_cam[:, 0] * crop_cam[:, 1], crop_cam[:, 0] * (1.0 + crop_cam[:, 2])],
            dim=-1,
        )
        crop_points = torch.stack([center, x_plus, y_plus], dim=1)
        full_points = self._crop_ndc_to_full_ndc(crop_points, crop_transform, crop_h, crop_w, full_h, full_w)

        scale_x = full_points[:, 1, 0] - full_points[:, 0, 0]
        scale_y = full_points[:, 2, 1] - full_points[:, 0, 1]
        scale = 0.5 * (scale_x + scale_y)
        scale = torch.where(scale.abs() < 1e-8, torch.full_like(scale, 1e-8), scale)

        tx = full_points[:, 0, 0] / scale
        ty = full_points[:, 0, 1] / scale
        return torch.stack([scale, tx, ty], dim=-1)

    def _make_aug_cfg(self, base_cfg, use_aug_override):
        if base_cfg is None:
            cfg = {}
        elif isinstance(base_cfg, dict):
            cfg = dict(base_cfg)
        else:
            keys = ('enabled', 'scale_min', 'scale_max', 'noise_std', 'value_range', 'apply_probability')
            cfg = {k: getattr(base_cfg, k) for k in keys if hasattr(base_cfg, k)}
        if use_aug_override is not None:
            cfg['enabled'] = bool(use_aug_override)
        return cfg if cfg else None

    def _extract_nested_cfg(self, base_cfg, key):
        if base_cfg is None:
            return None
        if isinstance(base_cfg, dict):
            return base_cfg.get(key)
        return getattr(base_cfg, key, None)
