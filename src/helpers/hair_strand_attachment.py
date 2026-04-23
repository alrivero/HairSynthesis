"""Utilities for binding PERM hair strands to posed FLAME meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.FLAME.FLAME import FLAME
from src.FLAME.lbs import batch_rodrigues
from pytorch3d.structures import Meshes


@dataclass
class HairAttachmentOutput:
    """Structured return type for glued strands."""

    full_resolution: torch.Tensor
    low_frequency: Optional[torch.Tensor]
    metadata: Dict[str, torch.Tensor]
    flame_mesh: Optional[torch.Tensor] = None


class FLAMEHairStrandAttachment(nn.Module):
    """Glue decoded PERM strands onto FLAME meshes in arbitrary poses."""

    def __init__(
        self,
        flame_model: FLAME,
        *,
        scale_to_perm: float = 100.0,
        translation_to_perm: Optional[Tuple[float, float, float]] = None,
        rotation_to_perm_euler_deg: Optional[Tuple[float, float, float]] = None,
        perm_head_mesh_path: Optional[str] = None,
        scalp_bounds: Optional[Tuple[float, float, float, float]] = None,
        mask_threshold: float = 0.5,
        max_mask_samples: Optional[int] = None,
        strand_basis_path: str = "assets/blend-shapes/strands-blend-shapes.npz",
        ray_chunk_size: int = 128,
        device: Optional[torch.device] = None,
        enable_pre_render_culling: bool = False,
    ) -> None:
        super().__init__()
        self.flame_model = flame_model
        self.scale_to_perm = float(scale_to_perm)
        if translation_to_perm is None:
            translation_to_perm = (0.0, 0.0, 0.0)
        if len(translation_to_perm) != 3:
            raise ValueError("translation_to_perm must have exactly 3 values")
        self.translation_to_perm = tuple(float(v) for v in translation_to_perm)
        if rotation_to_perm_euler_deg is None:
            rotation_to_perm_euler_deg = (0.0, 0.0, 0.0)
        if len(rotation_to_perm_euler_deg) != 3:
            raise ValueError("rotation_to_perm_euler_deg must have exactly 3 values")
        self.rotation_to_perm_euler_deg = tuple(float(v) for v in rotation_to_perm_euler_deg)
        self.perm_head_mesh_path = perm_head_mesh_path
        self._perm_head_mesh_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.scalp_bounds = scalp_bounds or (0.0, 1.0, 0.0, 1.0)
        self.mask_threshold = float(mask_threshold)
        self.max_mask_samples = max_mask_samples
        self.low_rank_dim = 10
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ray_chunk_size = max(1, int(ray_chunk_size))
        self.mesh_separation = 1000.0
        self.enable_pre_render_culling = bool(enable_pre_render_culling)

        self.register_buffer('faces_tensor', self.flame_model.faces_tensor.clone())
        self.register_buffer('vertex_lbs_weights', self.flame_model.lbs_weights.clone())

        strand_npz = np.load(strand_basis_path)
        mean = torch.from_numpy(strand_npz['mean_shape']).float()
        basis = torch.from_numpy(strand_npz['blend_shapes']).float()
        self.num_coeff = basis.shape[0]
        self.num_strand_points = basis.shape[1]
        self.register_buffer('strand_mean', mean)
        self.register_buffer('strand_basis', basis)
        self.register_buffer('strand_basis_flat', basis.view(self.num_coeff, -1))

    @torch.no_grad()
    def forward(
        self,
        flame_params: Dict[str, torch.Tensor],
        hair_textures: torch.Tensor,
        roots: Optional[torch.Tensor] = None,
        scalp_masks: Optional[torch.Tensor] = None,
        *,
        return_low_frequency: bool = False,
        return_flame_mesh: bool = False,
        debug_dump: bool = False,
        debug_dump_hit_faces: bool = False,
    ) -> HairAttachmentOutput:
        """Decode, align, and glue PERM strands to posed FLAME meshes."""
        # 1) Decode FLAME meshes for each element in the batch.
        flame_outputs = self._decode_flame_meshes(flame_params)

        # 2) Align FLAME canonical meshes to PERM space and store transforms.
        canonical_vertices = flame_outputs.get('canonical_vertices', flame_outputs['vertices'])
        joint_transforms = flame_outputs.get('joint_transforms')
        perm_space_vertices, alignment_meta = self._to_perm_space(
            canonical_vertices,
            flame_params,
            joint_transforms=joint_transforms,
            apply_global_rotation=False,
        )

        # 3) Prepare sampling roots in PERM coordinates (either provided or derived from masks).
        perm_roots = self._prepare_roots(
            roots,
            scalp_masks,
            perm_space_vertices,
            debug_dump_hit_faces=debug_dump_hit_faces,
        )

        # 4) Decode the full-resolution hair strands in PERM space.
        hair_bundle = self._decode_hair_textures(hair_textures, perm_roots, return_low_frequency)

        # 5) Optionally cull strands directly in canonical PERM space before posing them back.
        if self.enable_pre_render_culling:
            visibility = self._cull_in_perm_space(
                hair_bundle['full'],
                perm_space_vertices,
                hair_bundle['root_mask'],
                scale=0.96,
            )
        else:
            visibility = hair_bundle['root_mask'].to(hair_bundle['full'].dtype)
        hair_bundle['full'] = hair_bundle['full'] * visibility.unsqueeze(-1).unsqueeze(-1)
        if return_low_frequency and 'low' in hair_bundle:
            hair_bundle['low'] = hair_bundle['low'] * visibility.unsqueeze(-1).unsqueeze(-1)
        hair_bundle['visibility'] = visibility

        # 6) Project the decoded strands back to the original FLAME pose/scale.
        glued_full = self._project_to_flame_space(
            hair_bundle['full'], alignment_meta, hair_bundle.get('root_weights')
        )
        glued_low = None
        if return_low_frequency and 'low' in hair_bundle:
            glued_low = self._project_to_flame_space(
                hair_bundle['low'], alignment_meta, hair_bundle.get('root_weights')
            )

        if debug_dump:
            self._dump_debug_geometry(flame_outputs['vertices'], glued_full, visibility)
        return HairAttachmentOutput(
            full_resolution=glued_full,
            low_frequency=glued_low,
            metadata={
                'flame_vertices': flame_outputs['vertices'],
                'flame_faces': self.faces_tensor,
                'perm_aligned_vertices': perm_space_vertices,
                'strand_visibility': visibility,
                **alignment_meta,
            },
            flame_mesh=flame_outputs['vertices'] if return_flame_mesh else None,
        )

    # ---------------------------------------------------------------------
    # Helper methods (skeleton implementations)
    # ---------------------------------------------------------------------
    def _decode_flame_meshes(self, flame_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run the frozen FLAME model and gather mesh outputs."""
        with torch.no_grad():
            outputs = self.flame_model(flame_params)
        return outputs

    def _to_perm_space(
        self,
        vertices: torch.Tensor,
        flame_params: Dict[str, torch.Tensor],
        *,
        joint_transforms: Optional[torch.Tensor] = None,
        apply_global_rotation: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Map FLAME vertices to the canonical PERM frame and record undo transforms."""
        if vertices.ndim != 3 or vertices.shape[-1] != 3:
            raise ValueError("vertices must have shape (B, V, 3)")

        device = vertices.device
        dtype = vertices.dtype
        batch, *_ = vertices.shape

        if apply_global_rotation:
            pose_params = flame_params.get('pose_params', None)
            if pose_params is not None:
                global_pose = pose_params[:, :3]
                rotation = batch_rodrigues(global_pose.reshape(-1, 3)).view(batch, 3, 3)
            else:
                rotation = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1)
            inv_rotation = rotation.transpose(1, 2)
            unposed = torch.einsum('bij,bvj->bvi', inv_rotation, vertices)
        else:
            rotation = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1)
            inv_rotation = rotation.transpose(1, 2)
            unposed = vertices

        # Scale into PERM's metric space, center the mesh, apply a fixed Euler rotation, then offset.
        scale = torch.full((batch, 1, 1), self.scale_to_perm, device=device, dtype=dtype)
        scaled = unposed * scale

        centroid_perm = scaled.mean(dim=1, keepdim=True)
        perm_rotation = self._rotation_matrix_from_euler_xyz(
            self.rotation_to_perm_euler_deg,
            device=device,
            dtype=dtype,
        ).view(1, 3, 3).expand(batch, -1, -1)
        perm_inv_rotation = perm_rotation.transpose(1, 2)
        perm_translation = torch.tensor(
            self.translation_to_perm,
            device=device,
            dtype=dtype,
        ).view(1, 1, 3).expand(batch, -1, -1)
        centered = scaled - centroid_perm
        rotated = torch.einsum('bij,bvj->bvi', perm_rotation, centered)
        perm_vertices = rotated + perm_translation

        alignment_meta: Dict[str, torch.Tensor] = {
            'rotation': rotation,
            'inv_rotation': inv_rotation,
            'scale': scale,
            'perm_centroid': centroid_perm,
            'perm_rotation': perm_rotation,
            'perm_inv_rotation': perm_inv_rotation,
            'perm_translation': perm_translation,
        }
        if joint_transforms is not None:
            alignment_meta['joint_transforms'] = joint_transforms
        return perm_vertices, alignment_meta

    def _prepare_roots(
        self,
        roots: Optional[torch.Tensor],
        scalp_masks: Optional[torch.Tensor],
        perm_vertices: torch.Tensor,
        *,
        debug_dump_hit_faces: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Ensure we have PERM-space root samples for each strand/texel."""
        device = perm_vertices.device

        if roots is not None:
            uv_coords, valid_mask = self._extract_uv_from_roots(roots)
        else:
            if scalp_masks is None:
                raise ValueError("Either roots or scalp_masks must be provided.")
            uv_coords, valid_mask = self._sample_uv_from_masks(scalp_masks)

        positions, face_indices, barycentric = self._uv_to_surface_points(
            uv_coords, valid_mask, perm_vertices
        )
        if debug_dump_hit_faces:
            self._dump_uv_hit_face_debug(uv_coords, valid_mask, perm_vertices, face_indices, roots=roots)
        valid_counts = valid_mask.sum(dim=1)
        skin_weights = self._compute_skin_weights(face_indices, barycentric, valid_mask)
        if skin_weights is not None:
            skin_weights = skin_weights.to(device)

        return {
            'positions': positions.to(device),
            'valid_mask': valid_mask.to(device),
            'uv_coords': uv_coords.to(device),
            'valid_counts': valid_counts,
            'face_indices': face_indices,
            'barycentric': barycentric,
            'skin_weights': skin_weights,
        }

    def _decode_hair_textures(
        self,
        hair_textures: torch.Tensor,
        perm_roots: Dict[str, torch.Tensor],
        return_low_frequency: bool,
    ) -> Dict[str, torch.Tensor]:
        """Decode 64-D textures into strand geometry (full + optional low-rank)."""
        if hair_textures.ndim != 4:
            raise ValueError("hair_textures must have shape (B, C, H, W)")
        if hair_textures.shape[1] != self.num_coeff:
            raise ValueError(f"Expected {self.num_coeff} channels, got {hair_textures.shape[1]}")

        uv_coords = perm_roots['uv_coords']
        valid_mask = perm_roots['valid_mask']
        positions = perm_roots['positions']

        norm_uv = self._normalize_uv(uv_coords).clamp(0.0, 1.0)
        grid = norm_uv.unsqueeze(1)  # (B, 1, N, 2)
        grid = grid * 2.0 - 1.0

        coeffs = F.grid_sample(
            hair_textures,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        coeffs = coeffs.squeeze(2).permute(0, 2, 1)  # (B, N, 64)
        coeffs = coeffs * valid_mask.unsqueeze(-1)

        full_strands = self._decode_coefficients(coeffs, positions)
        mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1).to(full_strands.dtype)
        full_strands = full_strands * mask_expanded

        bundle: Dict[str, torch.Tensor] = {
            'full': full_strands,
            'coefficients': coeffs,
            'root_mask': valid_mask,
        }
        if perm_roots.get('skin_weights') is not None:
            bundle['root_weights'] = perm_roots['skin_weights']

        if return_low_frequency:
            low_coeffs = coeffs.clone()
            low_coeffs[..., self.low_rank_dim :] = 0.0
            low_strands = self._decode_coefficients(low_coeffs, positions)
            low_strands = low_strands * mask_expanded
            bundle['low'] = low_strands

        return bundle

    def _project_to_flame_space(
        self,
        strands: torch.Tensor,
        alignment_meta: Dict[str, torch.Tensor],
        root_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply inverse alignment plus FLAME skinning to return to mesh space."""
        if strands.ndim != 4:
            raise ValueError("strands must have shape (B, N, P, 3)")

        batch = strands.shape[0]
        scale = alignment_meta['scale'].view(batch, 1, 1, 1)
        perm_centroid = alignment_meta['perm_centroid'].view(batch, 1, 1, 3)
        perm_translation = alignment_meta.get('perm_translation')
        if perm_translation is None:
            perm_translation = torch.zeros_like(perm_centroid)
        else:
            perm_translation = perm_translation.view(batch, 1, 1, 3)
        perm_inv_rotation = alignment_meta.get('perm_inv_rotation')
        if perm_inv_rotation is None:
            perm_inv_rotation = torch.eye(3, device=strands.device, dtype=strands.dtype).view(1, 3, 3).expand(batch, -1, -1)

        centered = strands - perm_translation
        rotated_back = torch.einsum(
            'bij,bnkj->bnki',
            perm_inv_rotation.to(device=strands.device, dtype=strands.dtype),
            centered,
        )
        canonical = (rotated_back + perm_centroid) / scale

        joint_transforms = alignment_meta.get('joint_transforms')
        if root_weights is not None and joint_transforms is not None:
            if joint_transforms.device != canonical.device:
                joint_transforms = joint_transforms.to(canonical.device)
            world = self._apply_skinning(canonical, root_weights, joint_transforms)
        else:
            rotation = alignment_meta['rotation']
            reshaped = canonical.view(batch, -1, 3)
            world = torch.einsum('bij,bvj->bvi', rotation, reshaped)
            world = world.view_as(strands)
        return world

    def _apply_skinning(
        self,
        canonical: torch.Tensor,
        root_weights: torch.Tensor,
        joint_transforms: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FLAME joint transforms using root skinning weights."""
        if root_weights is None or joint_transforms is None:
            return canonical
        R = joint_transforms[..., :3, :3]
        t = joint_transforms[..., :3, 3]

        batch, max_roots, num_points, _ = canonical.shape
        weights = root_weights
        if weights.shape[1] != max_roots:
            max_roots = min(weights.shape[1], max_roots)
            canonical = canonical[:, :max_roots]
            weights = weights[:, :max_roots]
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, N, J, 1, 1)

        rotated = torch.einsum('bjkl,bnpl->bnjpk', R, canonical)
        weighted_rot = rotated * weights
        translation = t[:, None, :, None, :].expand(-1, max_roots, -1, num_points, -1)
        weighted_t = translation * weights.expand(-1, -1, -1, num_points, -1)
        world = weighted_rot.sum(dim=2) + weighted_t.sum(dim=2)
        return world

    def _cull_in_perm_space(
        self,
        strands: torch.Tensor,
        perm_vertices: torch.Tensor,
        strand_mask: torch.Tensor,
        *,
        scale: float = 0.96,
    ) -> torch.Tensor:
        """Cull strands whose samples fall inside the canonical FLAME mesh in PERM space."""
        if strands.ndim != 4:
            raise ValueError("strands must have shape (B, N, P, 3)")

        batch, max_roots, num_points, _ = strands.shape
        device = strands.device
        verts = self._shrink_vertices(perm_vertices.to(device), scale=scale).float()
        faces = self.faces_tensor.to(device).long()
        faces = faces.unsqueeze(0).expand(batch, -1, -1)

        meshes = Meshes(verts=list(torch.unbind(verts, dim=0)), faces=list(torch.unbind(faces, dim=0)))
        points = strands.reshape(batch, -1, 3).float()
        inside = meshes.contains_points(points).view(batch, max_roots, num_points)

        strand_inside = inside.any(dim=-1)
        visibility = (~strand_inside) & strand_mask.bool()
        return visibility.to(strands.dtype)

    def _dump_debug_geometry(
        self,
        flame_vertices: torch.Tensor,
        strands: torch.Tensor,
        visibility: torch.Tensor,
        *,
        head_path: str = "test_head.ply",
        strand_path: str = "test_strands.ply",
    ) -> None:
        """Write the first batch element's head mesh and visible strands to disk."""
        if strands.shape[0] == 0 or flame_vertices.shape[0] == 0:
            return

        head = flame_vertices[0].detach().cpu()
        faces = self.faces_tensor.detach().cpu().long()
        self._write_mesh_ply(Path(head_path), head.numpy(), faces.numpy())

        strand_sample = strands[0].detach()
        mask = visibility[0]
        if mask.dtype != torch.bool:
            mask = mask > 0
        if mask.any():
            strand_sample = strand_sample[mask]
        strand_points = strand_sample.reshape(-1, 3)
        finite = torch.isfinite(strand_points).all(dim=-1)
        strand_points = strand_points[finite]
        if strand_points.numel() == 0:
            return
        self._write_point_ply(Path(strand_path), strand_points.cpu().numpy())

    def _write_mesh_ply(self, path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
        path = Path(path)
        with path.open('w', encoding='ascii') as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {vertices.shape[0]}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write(f"element face {faces.shape[0]}\n")
            handle.write("property list uchar int vertex_indices\n")
            handle.write("end_header\n")
            for v in vertices:
                handle.write(f"{v[0]} {v[1]} {v[2]}\n")
            for tri in faces:
                handle.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")

    def _write_face_subset_ply(
        self,
        path: Path,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        face_indices: torch.Tensor,
    ) -> None:
        valid_face_ids = torch.unique(face_indices[face_indices >= 0])
        if valid_face_ids.numel() == 0:
            return

        subset_faces = faces[valid_face_ids]
        used_vertices, inverse = torch.unique(subset_faces.reshape(-1), sorted=True, return_inverse=True)
        subset_vertices = vertices[used_vertices]
        remapped_faces = inverse.view(-1, 3)
        self._write_mesh_ply(path, subset_vertices.numpy(), remapped_faces.numpy())

    def _write_point_ply(self, path: Path, points: np.ndarray) -> None:
        path = Path(path)
        with path.open('w', encoding='ascii') as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {points.shape[0]}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write("end_header\n")
            for p in points:
                handle.write(f"{p[0]} {p[1]} {p[2]}\n")

    def _write_trusted_perm_root_points(
        self,
        path: Path,
        roots: torch.Tensor,
        perm_vertices: torch.Tensor,
    ) -> None:
        uvw, valid_mask = self._extract_uvw_from_roots(roots)
        if uvw.shape[0] == 0:
            return

        uvw0 = uvw[0]
        valid0 = valid_mask[0]
        if not valid0.any():
            return

        points = self._uvw_to_cartesian_trusting_w(uvw0[valid0], perm_vertices.to(device=uvw.device, dtype=uvw.dtype))
        finite = torch.isfinite(points).all(dim=-1)
        points = points[finite]
        if points.numel() == 0:
            return
        self._write_point_ply(path, points.detach().cpu().numpy())

    def _shrink_vertices(self, vertices: torch.Tensor, scale: float) -> torch.Tensor:
        if scale == 1.0:
            return vertices
        centroid = vertices.mean(dim=1, keepdim=True)
        return (vertices - centroid) * scale + centroid

    def _batch_offsets(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if batch_size == 1:
            return torch.zeros(1, 3, dtype=dtype, device=device)
        offsets = torch.zeros(batch_size, 3, dtype=dtype, device=device)
        stride = torch.arange(batch_size, dtype=dtype, device=device) * self.mesh_separation
        offsets[:, 0] = stride
        return offsets

    def _rotation_matrix_from_euler_xyz(
        self,
        euler_deg: Tuple[float, float, float],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        angles = torch.tensor(euler_deg, device=device, dtype=dtype) * (torch.pi / 180.0)
        cx, cy, cz = torch.cos(angles)
        sx, sy, sz = torch.sin(angles)

        rx = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
            device=device,
            dtype=dtype,
        )
        ry = torch.tensor(
            [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
            device=device,
            dtype=dtype,
        )
        rz = torch.tensor(
            [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        return rz @ ry @ rx

    def _decode_coefficients(
        self,
        coeffs: torch.Tensor,
        root_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Convert PCA coefficients into strand geometry and translate to root anchors."""
        batch, max_roots, _ = coeffs.shape
        geom = torch.matmul(coeffs, self.strand_basis_flat)  # (B, N, P*3)
        geom = geom.view(batch, max_roots, self.num_strand_points, 3)
        geom = geom + self.strand_mean.view(1, 1, self.num_strand_points, 3)
        anchored = geom + root_positions.unsqueeze(2)
        return anchored

    # ------------------------------------------------------------------
    # UV sampling utilities
    # ------------------------------------------------------------------
    def _sample_uv_from_masks(
        self,
        scalp_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample UV texel coordinates following provided scalp masks."""
        if scalp_masks.dim() != 4:
            raise ValueError("scalp_masks must have shape (B, 1, H, W)")

        batch, _, height, width = scalp_masks.shape
        device = scalp_masks.device

        u_coords = (torch.arange(width, device=device, dtype=scalp_masks.dtype) + 0.5) / width
        v_coords = (torch.arange(height, device=device, dtype=scalp_masks.dtype) + 0.5) / height
        uu, vv = torch.meshgrid(u_coords, v_coords, indexing='xy')
        uv_grid = torch.stack([uu, vv], dim=-1)  # (W, H, 2)
        uv_grid = uv_grid.permute(1, 0, 2).reshape(-1, 2)  # (H*W, 2) matching raster order

        mask_flat = scalp_masks.view(batch, -1).clamp_min(0.0)
        counts = torch.ceil(mask_flat.sum(dim=1)).long()
        counts[counts == 0] = mask_flat.shape[1]
        if self.max_mask_samples is not None:
            counts = torch.clamp(counts, max=self.max_mask_samples)
        max_count = int(counts.max().item())

        uv_samples = torch.zeros(batch, max_count, 2, device=device, dtype=scalp_masks.dtype)
        valid_mask = torch.zeros(batch, max_count, dtype=torch.bool, device=device)

        for b in range(batch):
            n = min(int(counts[b].item()), max_count)
            weights = mask_flat[b]
            if float(weights.sum().item()) <= 1e-8:
                weights = torch.ones_like(weights)
            idx = torch.multinomial(weights, n, replacement=True)
            uv_samples[b, :n] = uv_grid[idx]
            valid_mask[b, :n] = True

        # Rescale UVs to scalp bounds.
        u_min, u_max, v_min, v_max = self.scalp_bounds
        uv_samples[..., 0] = uv_samples[..., 0] * (u_max - u_min) + u_min
        uv_samples[..., 1] = uv_samples[..., 1] * (v_max - v_min) + v_min

        return uv_samples, valid_mask

    def _extract_uv_from_roots(
        self,
        roots: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse stored root tensors (uvw) into UV coordinates and validity masks."""
        if roots.dim() == 4:
            batch = roots.shape[0]
            uv = roots[..., :2].reshape(batch, -1, 2)
        elif roots.dim() == 3:
            batch = roots.shape[0]
            uv = roots[..., :2]
        else:
            raise ValueError("roots must have shape (B, N, 3) or (B, H, W, 3)")

        valid_mask = torch.isfinite(uv).all(dim=-1)
        invalid_mask = ~valid_mask
        if invalid_mask.any():
            uv = torch.where(valid_mask.unsqueeze(-1), uv, torch.zeros_like(uv))
        return uv, valid_mask

    def _uv_to_surface_points(
        self,
        uv_coords: torch.Tensor,
        valid_mask: torch.Tensor,
        perm_vertices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project raw PERM spherical UVs back to the scalp surface via ray-mesh intersections."""
        faces = self.faces_tensor.to(device=perm_vertices.device, dtype=torch.long)
        return self._uv_to_surface_points_with_faces(uv_coords, valid_mask, perm_vertices, faces)

    def _uv_to_surface_points_with_faces(
        self,
        uv_coords: torch.Tensor,
        valid_mask: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project raw PERM spherical UVs back to a batched surface via ray-mesh intersections."""
        batch, max_count, _ = uv_coords.shape
        device = vertices.device
        dtype = vertices.dtype
        results = torch.zeros(batch, max_count, 3, device=device, dtype=dtype)
        face_indices = torch.full((batch, max_count), -1, dtype=torch.long, device=device)
        barycentric = torch.zeros(batch, max_count, 3, device=device, dtype=dtype)

        batch_ids, root_ids = torch.nonzero(valid_mask, as_tuple=True)
        if batch_ids.numel() == 0:
            return results, face_indices, barycentric

        uv = uv_coords[batch_ids, root_ids]
        directions = self._uv_to_directions(uv).to(device=device, dtype=dtype)
        directions = F.normalize(directions, dim=-1)

        batch_offsets = self._batch_offsets(batch, dtype=dtype, device=device)
        ray_offsets = batch_offsets[batch_ids]
        origins = ray_offsets.clone()

        vertex_offsets = batch_offsets[:, None, :]
        vertices_offset = vertices + vertex_offsets
        vertex_base = (
            torch.arange(batch, device=device, dtype=torch.long) * vertices.shape[1]
        )[:, None, None]
        batched_faces = faces.unsqueeze(0) + vertex_base
        triangles = vertices_offset.reshape(-1, 3)[batched_faces.reshape(-1, 3)]
        face_ids = torch.arange(faces.shape[0], device=device, dtype=torch.long).repeat(batch)

        hits, hit_mask, hit_faces, hit_bary = self._raycast_mesh(
            origins, directions, triangles, face_ids
        )

        if (~hit_mask).any():
            radii = torch.linalg.norm(vertices, dim=-1).amax(dim=1).clamp(min=1e-6)
            fallback = directions[~hit_mask] * radii[batch_ids[~hit_mask]].unsqueeze(-1)
            hits[~hit_mask] = fallback + ray_offsets[~hit_mask]

        hits = hits - ray_offsets
        results[batch_ids, root_ids] = hits
        face_indices[batch_ids, root_ids] = hit_faces
        barycentric[batch_ids, root_ids] = hit_bary

        return results, face_indices, barycentric

    def _compute_skin_weights(
        self,
        face_indices: torch.Tensor,
        barycentric: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if face_indices is None or barycentric is None:
            return None
        faces = self.faces_tensor.to(face_indices.device)
        batch, max_roots = face_indices.shape
        safe_idx = face_indices.clone()
        safe_idx[safe_idx < 0] = 0
        tri_vertices = faces[safe_idx.view(-1)].view(batch, max_roots, 3)
        vert_weights = self.vertex_lbs_weights.to(face_indices.device)[tri_vertices]  # (B,N,3,J)
        bary = barycentric.unsqueeze(-1)
        weights = (vert_weights * bary).sum(dim=2)
        if valid_mask is not None:
            weights = weights * valid_mask.unsqueeze(-1)
        weights = torch.where(face_indices.unsqueeze(-1) >= 0, weights, torch.zeros_like(weights))
        return weights

    def _dump_uv_hit_face_debug(
        self,
        uv_coords: torch.Tensor,
        valid_mask: torch.Tensor,
        perm_vertices: torch.Tensor,
        face_indices: torch.Tensor,
        *,
        roots: Optional[torch.Tensor] = None,
    ) -> None:
        """Dump hit-face submeshes for the first aligned mesh and optional original PERM mesh."""
        if uv_coords.shape[0] == 0 or perm_vertices.shape[0] == 0:
            return

        current_faces = self.faces_tensor.detach().cpu().long()
        self._write_face_subset_ply(
            Path("test_hit_faces_current.ply"),
            perm_vertices[0].detach().cpu(),
            current_faces,
            face_indices[0].detach().cpu(),
        )

        perm_mesh = self._load_perm_head_mesh()
        if perm_mesh is None:
            return

        perm_vertices_ref, perm_faces_ref = perm_mesh
        perm_vertices_ref = perm_vertices_ref.to(device=uv_coords.device, dtype=perm_vertices.dtype).unsqueeze(0)
        perm_faces_ref = perm_faces_ref.to(device=uv_coords.device, dtype=torch.long)
        _, perm_face_indices, _ = self._uv_to_surface_points_with_faces(
            uv_coords[:1],
            valid_mask[:1],
            perm_vertices_ref,
            perm_faces_ref,
        )
        self._write_face_subset_ply(
            Path("test_hit_faces_original_perm.ply"),
            perm_vertices_ref[0].detach().cpu(),
            perm_faces_ref.detach().cpu(),
            perm_face_indices[0].detach().cpu(),
        )
        if roots is not None:
            self._write_trusted_perm_root_points(
                Path("test_perm_roots_trust_w.ply"),
                roots,
                perm_vertices_ref[0],
            )

    def _raycast_mesh(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        triangles: torch.Tensor,
        face_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Intersect rays and triangles using chunked GPU-friendly Möller–Trumbore."""
        v0 = triangles[:, 0]
        edge1 = triangles[:, 1] - v0
        edge2 = triangles[:, 2] - v0

        total_rays = directions.shape[0]
        hits = torch.zeros_like(directions)
        hit_mask = torch.zeros(total_rays, dtype=torch.bool, device=directions.device)
        hit_faces = torch.full((total_rays,), -1, dtype=torch.long, device=directions.device)
        hit_bary = torch.zeros(total_rays, 3, dtype=directions.dtype, device=directions.device)

        chunk_size = min(self.ray_chunk_size, max(1, total_rays))
        for start in range(0, total_rays, chunk_size):
            end = min(start + chunk_size, total_rays)
            chunk_hits, chunk_mask, chunk_faces, chunk_bary = self._raycast_chunk(
                origins[start:end],
                directions[start:end],
                v0,
                edge1,
                edge2,
                face_ids,
            )
            hits[start:end] = chunk_hits
            hit_mask[start:end] = chunk_mask
            hit_faces[start:end] = chunk_faces
            hit_bary[start:end] = chunk_bary

        return hits, hit_mask, hit_faces, hit_bary

    def _raycast_chunk(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        v0: torch.Tensor,
        edge1: torch.Tensor,
        edge2: torch.Tensor,
        face_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a chunk of rays against the full mesh."""
        eps = 1e-6
        dir_exp = directions.unsqueeze(1)
        edge2_exp = edge2.unsqueeze(0)
        p_vec = torch.cross(dir_exp, edge2_exp, dim=-1)

        edge1_exp = edge1.unsqueeze(0)
        det = (edge1_exp * p_vec).sum(dim=-1)
        det_mask = torch.abs(det) > eps

        inv_det = torch.zeros_like(det)
        inv_det[det_mask] = 1.0 / det[det_mask]

        v0_exp = v0.unsqueeze(0)
        origin_exp = origins.unsqueeze(1)
        t_vec = origin_exp - v0_exp

        u = ((t_vec * p_vec).sum(dim=-1)) * inv_det
        u_mask = (u >= 0.0) & (u <= 1.0)

        q_vec = torch.cross(t_vec, edge1_exp, dim=-1)
        v = ((dir_exp * q_vec).sum(dim=-1)) * inv_det
        v_mask = (v >= 0.0) & ((u + v) <= 1.0)

        t = ((edge2_exp * q_vec).sum(dim=-1)) * inv_det
        t_mask = t > eps

        valid = det_mask & u_mask & v_mask & t_mask
        t_valid = torch.where(valid, t, torch.full_like(t, float('inf')))

        best_t, _ = torch.min(t_valid, dim=1)
        has_hit = torch.isfinite(best_t)

        chunk_hits = origins.clone()
        chunk_faces = torch.full((origins.shape[0],), -1, dtype=torch.long, device=origins.device)
        chunk_bary = torch.zeros(origins.shape[0], 3, dtype=origins.dtype, device=origins.device)
        if has_hit.any():
            chunk_hits[has_hit] = origins[has_hit] + directions[has_hit] * best_t[has_hit].unsqueeze(-1)
            best_indices = torch.argmin(t_valid, dim=1)
            valid_indices = best_indices[has_hit]
            chunk_faces[has_hit] = face_ids[valid_indices]
            gathered_u = torch.gather(u, 1, best_indices.unsqueeze(-1)).squeeze(-1)
            gathered_v = torch.gather(v, 1, best_indices.unsqueeze(-1)).squeeze(-1)
            bary = torch.stack([1.0 - gathered_u - gathered_v, gathered_u, gathered_v], dim=-1)
            chunk_bary[has_hit] = bary[has_hit]

        return chunk_hits, has_hit, chunk_faces, chunk_bary

    def _uv_to_directions(self, uv: torch.Tensor) -> torch.Tensor:
        """Convert raw PERM spherical UV samples to unit directions on the scalp sphere."""
        eps = 1e-4
        uv_clamped = uv.clamp(min=eps, max=1.0 - eps)
        uv_pi = uv_clamped * torch.pi
        cot_u = 1.0 / torch.tan(uv_pi[:, 0])
        cot_v = 1.0 / torch.tan(uv_pi[:, 1])

        h = 2.0 / (cot_u ** 2 + cot_v ** 2 + 1.0)
        directions = torch.zeros(uv.shape[0], 3, dtype=uv.dtype, device=uv.device)
        directions[:, 0] = h * cot_u
        directions[:, 1] = h - 1.0
        directions[:, 2] = h * cot_v

        norms = torch.linalg.norm(directions, dim=-1, keepdim=True).clamp(min=1e-6)
        return directions / norms

    def _fallback_surface_points(
        self,
        perm_vertices: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback for rays that miss the mesh: project onto bounding sphere."""
        radius = torch.linalg.norm(perm_vertices, dim=-1).amax().clamp(min=1e-6)
        return directions * radius

    def _normalize_uv(self, uv: torch.Tensor) -> torch.Tensor:
        """Normalize UV coordinates from scalp bounds to [0, 1]."""
        u_min, u_max, v_min, v_max = self.scalp_bounds
        denom_u = max(u_max - u_min, 1e-6)
        denom_v = max(v_max - v_min, 1e-6)
        norm = uv.clone()
        norm[..., 0] = (uv[..., 0] - u_min) / denom_u
        norm[..., 1] = (uv[..., 1] - v_min) / denom_v
        return norm

    def _extract_uvw_from_roots(
        self,
        roots: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if roots.dim() == 4:
            batch = roots.shape[0]
            uvw = roots[..., :3].reshape(batch, -1, 3)
        elif roots.dim() == 3:
            batch = roots.shape[0]
            uvw = roots[..., :3]
        else:
            raise ValueError("roots must have shape (B, N, 3) or (B, H, W, 3)")

        valid_mask = torch.isfinite(uvw).all(dim=-1)
        if (~valid_mask).any():
            uvw = torch.where(valid_mask.unsqueeze(-1), uvw, torch.zeros_like(uvw))
        return uvw, valid_mask

    def _uvw_to_cartesian_trusting_w(
        self,
        uvw: torch.Tensor,
        perm_vertices: torch.Tensor,
    ) -> torch.Tensor:
        directions = self._uv_to_directions(uvw[..., :2]).to(device=uvw.device, dtype=uvw.dtype)
        centroid = (perm_vertices.amin(dim=0) + perm_vertices.amax(dim=0)) * 0.5
        centroid[1] = 0.0
        return directions * uvw[..., 2:].to(device=uvw.device, dtype=uvw.dtype) + centroid

    def _load_perm_head_mesh(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.perm_head_mesh_path is None:
            return None
        if self._perm_head_mesh_cache is not None:
            return self._perm_head_mesh_cache

        path = Path(self.perm_head_mesh_path)
        suffix = path.suffix.lower()
        if suffix == '.obj':
            mesh = self._load_obj_mesh(path)
        elif suffix == '.ply':
            mesh = self._load_ply_mesh(path)
        else:
            raise ValueError(f"Unsupported perm head mesh format: {path.suffix}")

        self._perm_head_mesh_cache = mesh
        return mesh

    def _load_obj_mesh(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        vertices = []
        faces = []
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                if line.startswith('v '):
                    parts = line.strip().split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.strip().split()[1:]
                    if len(parts) != 3:
                        continue
                    face = []
                    for part in parts:
                        face.append(int(part.split('/')[0]) - 1)
                    faces.append(face)

        return torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)

    def _load_ply_mesh(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        with path.open('r', encoding='utf-8') as handle:
            header = []
            while True:
                line = handle.readline()
                if line == '':
                    raise ValueError(f"Unexpected end of file while reading PLY header: {path}")
                header.append(line.strip())
                if line.strip() == 'end_header':
                    break

            vertex_count = 0
            face_count = 0
            for line in header:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('element face'):
                    face_count = int(line.split()[-1])

            vertices = []
            for _ in range(vertex_count):
                parts = handle.readline().strip().split()
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])

            faces = []
            for _ in range(face_count):
                parts = handle.readline().strip().split()
                if not parts:
                    continue
                if int(parts[0]) != 3:
                    continue
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])

        return torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)
