from __future__ import annotations

from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def depth2vis(depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Convert normalized depth maps into masked RGB visualizations."""
    depth_np = depth.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    batch_size, _, height, width = depth_np.shape
    depth_vis = []
    for batch_idx in range(batch_size):
        depth_b = depth_np[batch_idx, 0]
        mask_b = mask_np[batch_idx, 0]
        valid_mask = mask_b > 0.5
        valid_depth = depth_b[valid_mask]

        if valid_depth.size == 0:
            norm_masked_depth = np.zeros_like(depth_b, dtype=np.float32)
        else:
            valid_min = np.nanmin(valid_depth)
            valid_max = np.nanmax(valid_depth)
            depth_range = valid_max - valid_min

            if not np.isfinite(depth_range) or depth_range <= 1e-8:
                norm_masked_depth = np.zeros_like(depth_b, dtype=np.float32)
            else:
                norm_masked_depth = (depth_b - valid_min) / depth_range
                norm_masked_depth = np.clip(norm_masked_depth, 0.0, 1.0)

        cmap = plt.get_cmap('jet')
        depth_rgb = cmap(norm_masked_depth)[..., 0:3]
        depth_rgb = depth_rgb * mask_b[..., None]
        depth_vis.append(torch.from_numpy(depth_rgb).permute(2, 0, 1))

    return torch.stack(depth_vis, dim=0)


def strand2vis(strand: torch.Tensor) -> torch.Tensor:
    """Visualize packed orientation maps with the same HSV convention as training logs."""
    strand_map_batch = strand.detach().cpu().numpy()
    batch_size, _, height, width = strand_map_batch.shape

    strand_vis = []
    for batch_idx in range(batch_size):
        strand_map = strand_map_batch[batch_idx]
        green_c = strand_map[1]
        blue_c = strand_map[2]

        theta = np.arctan2(-blue_c, green_c)
        theta = (theta + 2 * np.pi) % (2 * np.pi)
        theta = (2 * np.pi) - theta

        h = theta / (2 * np.pi)
        s = np.ones_like(h)
        v = np.ones_like(h)

        hsv = np.zeros((height, width, 3), dtype=np.uint8)
        hsv[:, :, 0] = h * 179
        hsv[:, :, 1] = s * 255
        hsv[:, :, 2] = v * 255

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255.0
        mask = np.clip((strand_map[0] - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
        rgb = rgb * mask[..., None]
        strand_vis.append(torch.from_numpy(rgb).permute(2, 0, 1))

    return torch.stack(strand_vis, dim=0)


def packed_map_stage_views(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = packed[:, :1]
    orientation = strand2vis(torch.cat([mask, packed[:, 1:3]], dim=1))
    depth = depth2vis(packed[:, 3:4], mask)
    mask_vis = mask.repeat(1, 3, 1, 1)
    return orientation, depth, mask_vis


def save_packed_map_comparison(
    stage_maps: Sequence[tuple[str, torch.Tensor]],
    save_path: str,
) -> None:
    """Save a grid with one sample per row and stage triplets left-to-right."""
    if not stage_maps:
        raise ValueError("stage_maps must contain at least one stage.")

    stage_views = {name: packed_map_stage_views(packed) for name, packed in stage_maps}
    batch_size = stage_maps[0][1].shape[0]
    row_tensors = []
    for sample_idx in range(batch_size):
        sample_views = []
        for name, _packed in stage_maps:
            orientation_vis, depth_vis, mask_vis = stage_views[name]
            sample_views.extend([
                orientation_vis[sample_idx:sample_idx + 1],
                depth_vis[sample_idx:sample_idx + 1],
                mask_vis[sample_idx:sample_idx + 1],
            ])
        row_tensors.append(torch.cat(sample_views, dim=0))

    grid = make_grid(torch.cat(row_tensors, dim=0), nrow=len(stage_maps) * 3)
    save_image(grid, save_path)
