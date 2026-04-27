"""Manage loading, caching, and sampling of hair template NPZ files."""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch


class HairTemplateManager:
    """Utility that loads, optionally augments, and samples hair templates."""

    ROOT_CACHE_VERSION = 1
    DEFAULT_ROOT_SCALE = 2.5
    DEFAULT_ROOT_CACHE_WORKERS = 16
    DEFAULT_ROOT_CACHE_K = 8
    DEFAULT_ROOT_CACHE_LOCK_TIMEOUT_SEC = 600.0
    DEFAULT_ROOT_CACHE_LOCK_POLL_SEC = 0.1
    DEFAULT_ROOT_CACHE_STALE_LOCK_SEC = 1800.0

    def __init__(
        self,
        template_dir: Optional[str],
        aug_cfg: Optional[object] = None,
        *,
        load_roots: bool = True,
        root_cache_cfg: Optional[object] = None,
        scalp_bounds: Optional[Sequence[float]] = None,
    ) -> None:
        self.template_dir = os.path.expanduser(template_dir) if template_dir else None
        self.template_paths: List[str] = self._discover_paths(self.template_dir)
        self.load_roots = bool(load_roots)
        self.scalp_bounds = self._parse_scalp_bounds(scalp_bounds)

        enabled = self._cfg_get(aug_cfg, 'enabled', False)
        scale_min = self._cfg_get(aug_cfg, 'scale_min', 0.8)
        scale_max = self._cfg_get(aug_cfg, 'scale_max', 1.2)
        noise_std = self._cfg_get(aug_cfg, 'noise_std', 0.05)
        value_range = self._cfg_get(aug_cfg, 'value_range', (-200.0, 200.0))
        apply_prob = self._cfg_get(aug_cfg, 'apply_probability', 1.0)

        self.aug_enabled = bool(enabled)
        self.scale_range = (float(scale_min), float(scale_max))
        self.noise_std = float(noise_std)
        self.value_range = (
            (float(value_range[0]), float(value_range[1]))
            if isinstance(value_range, (list, tuple))
            else (-200.0, 200.0)
        )
        self.aug_probability = float(apply_prob)
        self.aug_probability = min(max(self.aug_probability, 0.0), 1.0)

        root_cache_enabled = bool(self._cfg_get(root_cache_cfg, 'enabled', False))
        self.root_cache_enabled = self.load_roots and root_cache_enabled
        self.root_scale = max(float(self._cfg_get(root_cache_cfg, 'root_scale', self.DEFAULT_ROOT_SCALE)), 1e-4)
        self.root_cache_workers = max(
            1,
            int(self._cfg_get(root_cache_cfg, 'workers', self.DEFAULT_ROOT_CACHE_WORKERS)),
        )
        self.root_cache_knn_k = max(
            1,
            int(self._cfg_get(root_cache_cfg, 'knn_k', self.DEFAULT_ROOT_CACHE_K)),
        )
        self.root_cache_force_rebuild = bool(self._cfg_get(root_cache_cfg, 'force_rebuild', False))
        world_size_env = os.environ.get('WORLD_SIZE')
        default_lazy_init = False
        if world_size_env is not None:
            try:
                default_lazy_init = int(world_size_env) > 1
            except ValueError:
                default_lazy_init = False
        self.root_cache_lazy_init = bool(self._cfg_get(root_cache_cfg, 'lazy_init', default_lazy_init))
        self.root_cache_lock_timeout_sec = max(
            1.0,
            float(self._cfg_get(root_cache_cfg, 'lock_timeout_sec', self.DEFAULT_ROOT_CACHE_LOCK_TIMEOUT_SEC)),
        )
        self.root_cache_lock_poll_sec = max(
            0.01,
            float(self._cfg_get(root_cache_cfg, 'lock_poll_sec', self.DEFAULT_ROOT_CACHE_LOCK_POLL_SEC)),
        )
        self.root_cache_stale_lock_sec = max(
            self.root_cache_lock_poll_sec,
            float(self._cfg_get(root_cache_cfg, 'stale_lock_sec', self.DEFAULT_ROOT_CACHE_STALE_LOCK_SEC)),
        )
        self.root_cache_dir = self._build_cache_dir() if self.root_cache_enabled else None

        self.latest_sample: Optional[Dict[str, object]] = None
        self._template_queue: List[str] = []
        self._bad_template_paths: Set[str] = set()
        self._root_cache_map: Dict[str, str] = {}

        if root_cache_enabled and not self.load_roots:
            print("[HairTemplateManager] Root cache requested but load_roots=False; skipping root cache.")
        if self.root_cache_enabled:
            if self.scalp_bounds is None:
                raise ValueError("scalp_bounds must be provided when Hair20k root cache is enabled.")
            self._prepare_root_cache()

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
                    roots_np = self._load_roots_array(path, data)
                    if roots_np is not None:
                        roots = torch.from_numpy(roots_np).float()
            return texture, mask, roots
        except Exception as exc:
            print(f"[HairTemplateManager] Failed to load template {path}: {exc}")
            return None

    def _load_roots_array(self, path: str, data: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
        if self.root_cache_enabled:
            cache_path = self._root_cache_map.get(path)
            if cache_path is None:
                cache_path = self._ensure_root_cache_entry(path)
            if cache_path:
                try:
                    with np.load(cache_path, allow_pickle=False) as cache_data:
                        return np.asarray(cache_data['roots'], dtype=np.float32)
                except Exception as exc:
                    print(f"[HairTemplateManager] Failed to load cached roots {cache_path}: {exc}")
                    self._root_cache_map.pop(path, None)
                    refreshed_cache_path = self._ensure_root_cache_entry(path)
                    if refreshed_cache_path:
                        with np.load(refreshed_cache_path, allow_pickle=False) as cache_data:
                            return np.asarray(cache_data['roots'], dtype=np.float32)

        if 'roots' not in data:
            return None
        return np.asarray(data['roots'], dtype=np.float32)

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
            mask = (
                torch.rand(textures.shape[0], 1, 1, 1, device=textures.device) < self.aug_probability
            ).float()
            textures = augmented * mask + textures * (1.0 - mask)
            return textures
        return augmented

    # ------------------------------------------------------------------
    # Root cache helpers
    # ------------------------------------------------------------------
    def _prepare_root_cache(self) -> None:
        if self.root_cache_dir is None or not self.template_paths:
            return

        self.root_cache_dir.mkdir(parents=True, exist_ok=True)

        if self.root_cache_lazy_init:
            print(
                f"[HairTemplateManager] Root cache lazy init enabled for {len(self.template_paths)} templates "
                f"(scale={self.root_scale:.4f}, k={self.root_cache_knn_k})."
            )
            return

        pending: List[Tuple[str, Path]] = []
        ready_count = 0
        for template_path in self.template_paths:
            cache_path = self._cache_path_for_template(template_path)
            if not self.root_cache_force_rebuild and self._is_cache_valid(cache_path, template_path):
                self._root_cache_map[template_path] = str(cache_path)
                ready_count += 1
            else:
                pending.append((template_path, cache_path))

        if not pending:
            print(
                f"[HairTemplateManager] Root cache ready for all {len(self.template_paths)} templates "
                f"(scale={self.root_scale:.4f}, k={self.root_cache_knn_k})."
            )
            return

        print(
            f"[HairTemplateManager] Preparing root cache for {len(pending)} templates "
            f"with {self.root_cache_workers} workers "
            f"(scale={self.root_scale:.4f}, k={self.root_cache_knn_k})."
        )
        success_count = 0
        failure_count = 0
        with ThreadPoolExecutor(max_workers=self.root_cache_workers) as executor:
            future_map = {
                executor.submit(self._build_root_cache_entry, template_path, cache_path): (template_path, cache_path)
                for template_path, cache_path in pending
            }
            for idx, future in enumerate(as_completed(future_map), start=1):
                template_path, cache_path = future_map[future]
                ok, error = future.result()
                if ok:
                    self._root_cache_map[template_path] = str(cache_path)
                    success_count += 1
                else:
                    failure_count += 1
                    print(f"[HairTemplateManager] Root cache failed for {template_path}: {error}")

                if idx % 250 == 0 or idx == len(pending):
                    print(
                        f"[HairTemplateManager] Root cache progress: {idx}/{len(pending)} "
                        f"(ready={ready_count + success_count}, failed={failure_count})"
                    )

        if failure_count:
            print(
                f"[HairTemplateManager] Root cache completed with {failure_count} failures. "
                "Failed templates will fall back to original roots."
            )
        else:
            print(f"[HairTemplateManager] Root cache completed for {len(self._root_cache_map)} templates.")

    def _build_root_cache_entry(self, template_path: str, cache_path: Path) -> Tuple[bool, Optional[str]]:
        try:
            source_stat = os.stat(template_path)
            with np.load(template_path, allow_pickle=True) as data:
                if 'roots' not in data:
                    raise KeyError("Template does not contain roots.")
                if 'mask' not in data:
                    raise KeyError("Template does not contain mask.")
                roots = np.asarray(data['roots'], dtype=np.float32)
                mask = np.asarray(data['mask'])

            cached_roots = self._build_cached_roots(roots, mask, template_path)
            self._write_cache_file(
                cache_path=cache_path,
                roots=cached_roots,
                source_size=source_stat.st_size,
                source_mtime_ns=source_stat.st_mtime_ns,
            )
            return True, None
        except Exception as exc:
            return False, str(exc)

    def _build_cached_roots(
        self,
        roots: np.ndarray,
        mask: np.ndarray,
        template_path: str,
    ) -> np.ndarray:
        base_roots = self._prepare_roots_array(roots)
        base_count = base_roots.shape[0]
        if base_count == 0:
            raise ValueError("Template has no valid roots.")

        target_count = max(1, int(round(base_count * self.root_scale)))
        if target_count == base_count:
            return base_roots.astype(np.float32, copy=True)

        mask_bool = self._prepare_mask(mask)
        norm_uv = self._normalize_uv(base_roots[:, :2]).astype(np.float32, copy=False)

        if target_count < base_count:
            return self._downsample_roots(base_roots, norm_uv, target_count)

        seed = self._seed_from_template(template_path)
        rng = np.random.default_rng(seed)
        extra_count = target_count - base_count
        neighbor_idx, neighbor_dist = self._build_neighbor_index(norm_uv)
        extra_roots = self._generate_extra_roots(
            base_roots=base_roots,
            norm_uv=norm_uv,
            mask_bool=mask_bool,
            extra_count=extra_count,
            neighbor_idx=neighbor_idx,
            neighbor_dist=neighbor_dist,
            rng=rng,
        )
        if extra_roots.shape[0] != extra_count:
            raise RuntimeError(
                f"Expected {extra_count} generated roots, received {extra_roots.shape[0]}."
            )
        return np.concatenate([base_roots, extra_roots], axis=0).astype(np.float32, copy=False)

    def _prepare_roots_array(self, roots: np.ndarray) -> np.ndarray:
        roots = np.asarray(roots, dtype=np.float32)
        if roots.ndim == 3:
            roots = roots.reshape(-1, roots.shape[-1])
        if roots.ndim != 2 or roots.shape[1] < 2:
            raise ValueError(f"Expected roots with shape (N, C), got {tuple(roots.shape)}")

        valid_mask = np.isfinite(roots[:, :2]).all(axis=1)
        if roots.shape[1] > 2:
            valid_mask &= np.isfinite(roots[:, 2:]).all(axis=1)
        roots = roots[valid_mask]
        if roots.size == 0:
            return np.zeros((0, roots.shape[1] if roots.ndim == 2 else 3), dtype=np.float32)
        return roots

    def _prepare_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 2:
            raise ValueError(f"Expected mask with shape (H, W) or (1, H, W), got {tuple(mask.shape)}")
        mask_bool = mask.astype(np.float32) > 0.0
        if not np.any(mask_bool):
            mask_bool = np.ones_like(mask_bool, dtype=bool)
        return mask_bool

    def _downsample_roots(
        self,
        base_roots: np.ndarray,
        norm_uv: np.ndarray,
        target_count: int,
    ) -> np.ndarray:
        sort_idx = self._morton_sort_indices(norm_uv)
        keep_positions = np.linspace(0, len(sort_idx) - 1, num=target_count).round().astype(np.int64)
        keep_idx = sort_idx[keep_positions]
        return base_roots[keep_idx].astype(np.float32, copy=False)

    def _generate_extra_roots(
        self,
        *,
        base_roots: np.ndarray,
        norm_uv: np.ndarray,
        mask_bool: np.ndarray,
        extra_count: int,
        neighbor_idx: np.ndarray,
        neighbor_dist: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        feat_dim = base_roots.shape[1]
        generated = np.zeros((extra_count, feat_dim), dtype=np.float32)
        local_scale = np.median(neighbor_dist, axis=1).astype(np.float32, copy=False)
        valid_scale = np.isfinite(local_scale) & (local_scale > 0)
        fallback_scale = float(np.median(local_scale[valid_scale])) if np.any(valid_scale) else (1.0 / 256.0)
        local_scale[~valid_scale] = fallback_scale

        accepted = 0
        attempts = 0
        max_attempts = max(extra_count * 16, 1024)
        num_roots = base_roots.shape[0]
        num_neighbors = neighbor_idx.shape[1]

        while accepted < extra_count and attempts < max_attempts:
            remaining = extra_count - accepted
            batch = min(max(remaining * 2, 256), 4096)
            anchor_idx = rng.integers(0, num_roots, size=batch, endpoint=False)
            neighbor_choice = rng.integers(0, num_neighbors, size=batch, endpoint=False)
            neighbor_sel = neighbor_idx[anchor_idx, neighbor_choice]
            invalid_neighbor = neighbor_sel < 0
            if invalid_neighbor.any():
                neighbor_sel[invalid_neighbor] = anchor_idx[invalid_neighbor]

            interp = rng.uniform(0.2, 0.8, size=batch).astype(np.float32, copy=False)
            anchor_uv = norm_uv[anchor_idx]
            neighbor_uv = norm_uv[neighbor_sel]
            cand_uv = anchor_uv * (1.0 - interp[:, None]) + neighbor_uv * interp[:, None]

            jitter_scale = (local_scale[anchor_idx] * 0.35).astype(np.float32, copy=False)
            jitter = rng.normal(0.0, 1.0, size=(batch, 2)).astype(np.float32, copy=False)
            cand_uv = np.clip(cand_uv + jitter * jitter_scale[:, None], 0.0, 1.0)

            valid = self._mask_lookup(mask_bool, cand_uv)
            if not np.any(valid):
                attempts += batch
                continue

            valid_idx = np.flatnonzero(valid)[:remaining]
            count = int(valid_idx.shape[0])
            if count <= 0:
                attempts += batch
                continue

            chosen_anchor = anchor_idx[valid_idx]
            chosen_neighbor = neighbor_sel[valid_idx]
            chosen_interp = interp[valid_idx]
            selected_uv = cand_uv[valid_idx]

            generated_chunk = np.zeros((count, feat_dim), dtype=np.float32)
            generated_chunk[:, :2] = self._denormalize_uv(selected_uv)
            if feat_dim > 2:
                tail_anchor = base_roots[chosen_anchor, 2:]
                tail_neighbor = base_roots[chosen_neighbor, 2:]
                generated_chunk[:, 2:] = (
                    tail_anchor * (1.0 - chosen_interp[:, None]) + tail_neighbor * chosen_interp[:, None]
                )

            generated[accepted:accepted + count] = generated_chunk
            accepted += count
            attempts += batch

        if accepted < extra_count:
            remaining = extra_count - accepted
            generated[accepted:] = self._fallback_extra_roots(
                base_roots=base_roots,
                mask_bool=mask_bool,
                count=remaining,
                rng=rng,
            )

        return generated

    def _fallback_extra_roots(
        self,
        *,
        base_roots: np.ndarray,
        mask_bool: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        feat_dim = base_roots.shape[1]
        result = np.zeros((count, feat_dim), dtype=np.float32)
        norm_uv = self._normalize_uv(base_roots[:, :2]).astype(np.float32, copy=False)
        chosen = rng.integers(0, base_roots.shape[0], size=count, endpoint=False)
        base_uv = norm_uv[chosen]
        jitter = rng.normal(0.0, 1.0 / 512.0, size=(count, 2)).astype(np.float32, copy=False)
        cand_uv = np.clip(base_uv + jitter, 0.0, 1.0)
        valid = self._mask_lookup(mask_bool, cand_uv)
        if not np.all(valid):
            cand_uv[~valid] = base_uv[~valid]
        result[:, :2] = self._denormalize_uv(cand_uv)
        if feat_dim > 2:
            result[:, 2:] = base_roots[chosen, 2:]
        return result

    def _build_neighbor_index(self, norm_uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        num_roots = norm_uv.shape[0]
        if num_roots <= 1:
            return np.full((num_roots, 1), -1, dtype=np.int32), np.full((num_roots, 1), np.inf, dtype=np.float32)

        sort_idx = self._morton_sort_indices(norm_uv)
        sorted_uv = norm_uv[sort_idx]
        max_window = min(num_roots - 1, max(self.root_cache_knn_k * 4, 32))
        candidate_count = max_window * 2

        candidate_idx = np.full((num_roots, candidate_count), -1, dtype=np.int32)
        candidate_dist = np.full((num_roots, candidate_count), np.inf, dtype=np.float32)

        slot = 0
        for offset in range(1, max_window + 1):
            dist = np.linalg.norm(sorted_uv[offset:] - sorted_uv[:-offset], axis=1).astype(np.float32, copy=False)

            candidate_idx[:-offset, slot] = np.arange(offset, num_roots, dtype=np.int32)
            candidate_dist[:-offset, slot] = dist
            slot += 1

            candidate_idx[offset:, slot] = np.arange(0, num_roots - offset, dtype=np.int32)
            candidate_dist[offset:, slot] = dist
            slot += 1

        k = min(self.root_cache_knn_k, num_roots - 1)
        best_pos = np.argpartition(candidate_dist, kth=k - 1, axis=1)[:, :k]
        best_dist = np.take_along_axis(candidate_dist, best_pos, axis=1)
        order = np.argsort(best_dist, axis=1)
        best_pos = np.take_along_axis(best_pos, order, axis=1)
        best_dist = np.take_along_axis(best_dist, order, axis=1)
        best_neighbors_sorted = np.take_along_axis(candidate_idx, best_pos, axis=1)
        best_neighbors_orig = sort_idx[best_neighbors_sorted]

        neighbor_idx = np.empty((num_roots, k), dtype=np.int32)
        neighbor_dist = np.empty((num_roots, k), dtype=np.float32)
        neighbor_idx[sort_idx] = best_neighbors_orig
        neighbor_dist[sort_idx] = best_dist
        return neighbor_idx, neighbor_dist

    def _morton_sort_indices(self, norm_uv: np.ndarray) -> np.ndarray:
        bins = 65535
        quantized = np.clip(np.round(norm_uv * bins), 0, bins).astype(np.uint32)
        morton = self._part1by1(quantized[:, 0]) | (self._part1by1(quantized[:, 1]) << 1)
        return np.argsort(morton, kind='mergesort')

    @staticmethod
    def _part1by1(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.uint64) & 0x0000FFFF
        values = (values | (values << 8)) & 0x00FF00FF
        values = (values | (values << 4)) & 0x0F0F0F0F
        values = (values | (values << 2)) & 0x33333333
        values = (values | (values << 1)) & 0x55555555
        return values

    def _normalize_uv(self, uv: np.ndarray) -> np.ndarray:
        if self.scalp_bounds is None:
            raise ValueError("scalp_bounds are required for UV normalization.")
        u_min, u_max, v_min, v_max = self.scalp_bounds
        denom_u = max(u_max - u_min, 1e-6)
        denom_v = max(v_max - v_min, 1e-6)
        norm = uv.astype(np.float32, copy=True)
        norm[:, 0] = (norm[:, 0] - u_min) / denom_u
        norm[:, 1] = (norm[:, 1] - v_min) / denom_v
        return np.clip(norm, 0.0, 1.0)

    def _denormalize_uv(self, uv: np.ndarray) -> np.ndarray:
        if self.scalp_bounds is None:
            raise ValueError("scalp_bounds are required for UV denormalization.")
        u_min, u_max, v_min, v_max = self.scalp_bounds
        denorm = uv.astype(np.float32, copy=True)
        denorm[:, 0] = denorm[:, 0] * (u_max - u_min) + u_min
        denorm[:, 1] = denorm[:, 1] * (v_max - v_min) + v_min
        return denorm

    def _mask_lookup(self, mask_bool: np.ndarray, uv_norm: np.ndarray) -> np.ndarray:
        height, width = mask_bool.shape
        x = np.clip(np.round(uv_norm[:, 0] * (width - 1)).astype(np.int64), 0, width - 1)
        y = np.clip(np.round(uv_norm[:, 1] * (height - 1)).astype(np.int64), 0, height - 1)
        return mask_bool[y, x]

    def _build_cache_dir(self) -> Optional[Path]:
        if not self.template_dir:
            return None
        cache_name = (
            f"v{self.ROOT_CACHE_VERSION}_"
            f"s{self._format_cache_float(self.root_scale)}_"
            f"k{self.root_cache_knn_k}"
        )
        return Path(self.template_dir) / ".root_cache" / cache_name

    def _cache_path_for_template(self, template_path: str) -> Path:
        if self.root_cache_dir is None:
            raise ValueError("Root cache directory is not initialized.")
        return self.root_cache_dir / Path(template_path).name

    def _lock_path_for_cache(self, cache_path: Path) -> Path:
        return cache_path.with_suffix(f"{cache_path.suffix}.lock")

    def _ensure_root_cache_entry(self, template_path: str) -> Optional[str]:
        if not self.root_cache_enabled:
            return None

        cache_path = self._cache_path_for_template(template_path)
        if not self.root_cache_force_rebuild and self._is_cache_valid(cache_path, template_path):
            cache_path_str = str(cache_path)
            self._root_cache_map[template_path] = cache_path_str
            return cache_path_str

        lock_path = self._lock_path_for_cache(cache_path)
        if not self._acquire_cache_lock(lock_path):
            if not self.root_cache_force_rebuild and self._is_cache_valid(cache_path, template_path):
                cache_path_str = str(cache_path)
                self._root_cache_map[template_path] = cache_path_str
                return cache_path_str
            return None

        try:
            if not self.root_cache_force_rebuild and self._is_cache_valid(cache_path, template_path):
                cache_path_str = str(cache_path)
                self._root_cache_map[template_path] = cache_path_str
                return cache_path_str

            ok, error = self._build_root_cache_entry(template_path, cache_path)
            if not ok:
                print(f"[HairTemplateManager] Root cache failed for {template_path}: {error}")
                return None

            cache_path_str = str(cache_path)
            self._root_cache_map[template_path] = cache_path_str
            return cache_path_str
        finally:
            self._release_cache_lock(lock_path)

    def _acquire_cache_lock(self, lock_path: Path) -> bool:
        deadline = time.monotonic() + self.root_cache_lock_timeout_sec
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                    handle.write(f"{os.getpid()}\n{time.time():.6f}\n")
                return True
            except FileExistsError:
                if self._lock_is_stale(lock_path):
                    try:
                        os.remove(lock_path)
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    print(f"[HairTemplateManager] Timed out waiting for cache lock {lock_path}.")
                    return False
                time.sleep(self.root_cache_lock_poll_sec)

    def _release_cache_lock(self, lock_path: Path) -> None:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass

    def _lock_is_stale(self, lock_path: Path) -> bool:
        try:
            age_sec = time.time() - lock_path.stat().st_mtime
        except FileNotFoundError:
            return False
        return age_sec >= self.root_cache_stale_lock_sec

    def _is_cache_valid(self, cache_path: Path, template_path: str) -> bool:
        if not cache_path.exists():
            return False

        try:
            source_stat = os.stat(template_path)
            with np.load(cache_path, allow_pickle=False) as cache_data:
                version = int(np.asarray(cache_data['cache_version']).item())
                source_size = int(np.asarray(cache_data['source_size']).item())
                source_mtime_ns = int(np.asarray(cache_data['source_mtime_ns']).item())
                root_scale = float(np.asarray(cache_data['root_scale']).item())
                knn_k = int(np.asarray(cache_data['knn_k']).item())
                roots = np.asarray(cache_data['roots'])
        except Exception:
            return False

        if version != self.ROOT_CACHE_VERSION:
            return False
        if source_size != source_stat.st_size or source_mtime_ns != source_stat.st_mtime_ns:
            return False
        if not np.isclose(root_scale, self.root_scale, atol=1e-6):
            return False
        if knn_k != self.root_cache_knn_k:
            return False
        if roots.ndim != 2 or roots.shape[0] == 0 or roots.shape[1] < 2:
            return False
        return True

    def _write_cache_file(
        self,
        *,
        cache_path: Path,
        roots: np.ndarray,
        source_size: int,
        source_mtime_ns: int,
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(cache_path.parent),
            prefix=f"{cache_path.stem}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, 'wb') as handle:
                np.savez_compressed(
                    handle,
                    roots=roots.astype(np.float32, copy=False),
                    cache_version=np.int32(self.ROOT_CACHE_VERSION),
                    source_size=np.int64(source_size),
                    source_mtime_ns=np.int64(source_mtime_ns),
                    root_scale=np.float32(self.root_scale),
                    knn_k=np.int32(self.root_cache_knn_k),
                )
            os.replace(tmp_path, cache_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def _seed_from_template(self, template_path: str) -> int:
        seed_input = (
            f"{template_path}|{self.root_scale:.8f}|{self.root_cache_knn_k}|v{self.ROOT_CACHE_VERSION}"
        ).encode('utf-8')
        digest = hashlib.blake2b(seed_input, digest_size=8).digest()
        return int.from_bytes(digest, byteorder='little', signed=False)

    @staticmethod
    def _parse_scalp_bounds(scalp_bounds: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
        if scalp_bounds is None:
            return None
        if len(scalp_bounds) != 4:
            raise ValueError("scalp_bounds must contain exactly four values.")
        return tuple(float(value) for value in scalp_bounds)

    @staticmethod
    def _format_cache_float(value: float) -> str:
        text = f"{float(value):.4f}".rstrip('0').rstrip('.')
        return text.replace('.', 'p')

    @staticmethod
    def _cfg_get(cfg, key, default):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            value = cfg.get(key, default)
        else:
            value = getattr(cfg, key, default)
        if value is None:
            return default
        return value
