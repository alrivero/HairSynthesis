"""Manage loading and sampling of hair template NPZ files."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch


class HairTemplateManager:
    """Utility that loads, optionally augments, and samples hair templates."""

    def __init__(
        self,
        template_dir: Optional[str],
        aug_cfg: Optional[object] = None,
        *,
        load_roots: bool = True,
    ) -> None:
        self.template_dir = os.path.expanduser(template_dir) if template_dir else None
        self.template_paths: List[str] = self._discover_paths(self.template_dir)
        self.load_roots = bool(load_roots)

        enabled = self._cfg_get(aug_cfg, 'enabled', False)
        scale_min = self._cfg_get(aug_cfg, 'scale_min', 0.8)
        scale_max = self._cfg_get(aug_cfg, 'scale_max', 1.2)
        noise_std = self._cfg_get(aug_cfg, 'noise_std', 0.05)
        value_range = self._cfg_get(aug_cfg, 'value_range', (-200.0, 200.0))
        apply_prob = self._cfg_get(aug_cfg, 'apply_probability', 1.0)

        self.aug_enabled = bool(enabled)
        self.scale_range = (float(scale_min), float(scale_max))
        self.noise_std = float(noise_std)
        self.value_range = (float(value_range[0]), float(value_range[1])) if isinstance(value_range, (list, tuple)) else (-200.0, 200.0)
        self.aug_probability = float(apply_prob)
        self.aug_probability = min(max(self.aug_probability, 0.0), 1.0)

        self.latest_sample: Optional[Dict[str, object]] = None
        self._template_queue: List[str] = []
        self._bad_template_paths: Set[str] = set()

    def sample(self, batch_size: int, device: torch.device) -> Optional[Dict[str, object]]:
        if not self.template_paths or batch_size <= 0:
            return None

        textures: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        roots: Optional[List[torch.Tensor]] = [] if self.load_roots else None
        used_paths: List[str] = []
        selected_this_batch: Set[str] = set()

        while len(textures) < batch_size:
            path = self._next_template_path(exclude=selected_this_batch)
            if path is None:
                break

            loaded = self._read_template(path)
            if loaded is None:
                self._mark_bad_template(path)
                continue

            tex, mask, root = loaded
            textures.append(tex)
            masks.append(mask)
            if self.load_roots and root is not None:
                roots.append(root)
            used_paths.append(path)
            selected_this_batch.add(path)

        if not textures:
            return None

        texture_tensor = torch.stack(textures, dim=0).to(device)
        mask_tensor = torch.stack(masks, dim=0).to(device)
        roots_tensor = None
        if self.load_roots and roots:
            roots_tensor = self._pack_roots(roots).to(device)

        sample = {
            'texture': texture_tensor,
            'mask': mask_tensor,
            'paths': used_paths,
        }
        if roots_tensor is not None:
            sample['roots'] = roots_tensor

        if self.aug_enabled:
            sample['texture'] = self._augment_textures(sample['texture'])

        self.latest_sample = sample
        return sample

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _next_template_path(self, exclude: Optional[Set[str]] = None) -> Optional[str]:
        if not self._template_queue:
            self._refill_template_queue(exclude=exclude)

        if not self._template_queue:
            return None

        return self._template_queue.pop()

    def _refill_template_queue(self, exclude: Optional[Set[str]] = None) -> None:
        paths = [
            path for path in self.template_paths
            if path not in self._bad_template_paths
        ]
        if not paths:
            self._template_queue = []
            return

        random.shuffle(paths)
        if exclude:
            excluded_paths = [path for path in paths if path in exclude]
            fresh_paths = [path for path in paths if path not in exclude]
            paths = excluded_paths + fresh_paths
        self._template_queue = paths

    def _mark_bad_template(self, path: str) -> None:
        self._bad_template_paths.add(path)
        if self._template_queue:
            self._template_queue = [
                queued_path for queued_path in self._template_queue
                if queued_path != path
            ]

    def _discover_paths(self, template_dir: Optional[str]) -> List[str]:
        if not template_dir or not os.path.isdir(template_dir):
            if template_dir:
                print(f"[HairTemplateManager] Template dir '{template_dir}' not found.")
            return []
        paths = sorted(str(p) for p in Path(template_dir).glob('*.npz'))
        if not paths:
            print(f"[HairTemplateManager] No templates found in '{template_dir}'.")
        else:
            print(f"[HairTemplateManager] Found {len(paths)} templates in '{template_dir}'.")
        return paths

    def _read_template(self, path: str):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found")
            if os.path.getsize(path) == 0:
                raise ValueError("Empty template file")
            with np.load(path, allow_pickle=True) as data:
                texture = torch.from_numpy(data['texture']).float()
                mask = torch.from_numpy(data['mask'].astype(np.float32))
                roots = None
                if self.load_roots:
                    roots = torch.from_numpy(data['roots']).float()
            return texture, mask, roots
        except Exception as exc:
            print(f"[HairTemplateManager] Failed to load template {path}: {exc}")
            return None

    def _pack_roots(self, roots: List[torch.Tensor]) -> torch.Tensor:
        max_len = max(root.shape[0] for root in roots)
        feat_dim = roots[0].shape[1]
        packed = torch.full(
            (len(roots), max_len, feat_dim),
            float('nan'),
            dtype=roots[0].dtype,
        )
        for idx, root in enumerate(roots):
            count = root.shape[0]
            packed[idx, :count] = root
        return packed

    def _augment_textures(self, textures: torch.Tensor) -> torch.Tensor:
        if self.aug_probability <= 0:
            return textures

        scale_low, scale_high = self.scale_range
        scales = torch.empty(textures.shape[0], 1, 1, 1, device=textures.device).uniform_(scale_low, scale_high)
        augmented = textures * scales
        if self.noise_std > 0:
            augmented = augmented + torch.randn_like(augmented) * self.noise_std
        min_val, max_val = self.value_range
        augmented = torch.clamp(augmented, min_val, max_val)

        if self.aug_probability < 1.0:
            mask = (torch.rand(textures.shape[0], 1, 1, 1, device=textures.device) < self.aug_probability).float()
            textures = augmented * mask + textures * (1.0 - mask)
            return textures
        return augmented

    @staticmethod
    def _cfg_get(cfg, key, default):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)
