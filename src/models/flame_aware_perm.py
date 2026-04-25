"""FLAME-aware PERM decoder that avoids PERM's fixed-head root parameterization."""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    import dnnlib
    import legacy
except ModuleNotFoundError as exc:
    if exc.name != 'click':
        raise
    sys.modules.setdefault('click', types.ModuleType('click'))
    import dnnlib
    import legacy

from src.FLAME.FLAME import FLAME


class FlameAwarePermDecoder(nn.Module):
    """Decode PERM latents on a per-image FLAME scalp instead of a fixed head mesh."""

    def __init__(
        self,
        flame_model: FLAME,
        *,
        model_path: str,
        scalp_bounds: Tuple[float, float, float, float],
        scale_to_perm: float = 100.0,
        translation_to_perm: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_to_perm_euler_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scalp_mask_path: str = "assets/FLAME_masks/FLAME_masks.pkl",
        scalp_mask_key: str = "scalp",
        root_grid_resolution: int = 64,
        guide_mask_threshold: float = 0.35,
        ray_chunk_size: int = 128,
        use_guide_mask: bool = True,
        num_render_strands: Optional[int] = None,
        max_root_count: Optional[int] = None,
        latent_space: str = "broadcast_w",
    ) -> None:
        super().__init__()
        object.__setattr__(self, 'flame_model', flame_model)
        self.scale_to_perm = float(scale_to_perm)
        self.translation_to_perm = tuple(float(v) for v in translation_to_perm)
        self.rotation_to_perm_euler_deg = tuple(float(v) for v in rotation_to_perm_euler_deg)
        self.scalp_bounds = tuple(float(v) for v in scalp_bounds)
        self.root_grid_resolution = max(2, int(root_grid_resolution))
        self.guide_mask_threshold = float(guide_mask_threshold)
        self.ray_chunk_size = max(1, int(ray_chunk_size))
        self.use_guide_mask = bool(use_guide_mask)
        root_cap = num_render_strands
        if root_cap is None:
            root_cap = max_root_count
        self.num_render_strands = None if root_cap is None else max(1, int(root_cap))
        self.latent_space = str(latent_space)

        self.register_buffer('faces_tensor', self.flame_model.faces_tensor.clone())
        self.register_buffer('vertex_lbs_weights', self.flame_model.lbs_weights.clone())
        self.register_buffer('scalp_face_mask', self._load_scalp_face_mask(scalp_mask_path, scalp_mask_key))

        self._load_perm_networks(model_path)
        self._build_root_support()

    @property
    def theta_dim(self) -> int:
        return int(self.G_raw.w_dim)

    @property
    def beta_dim(self) -> int:
        return int(self.G_res.w_dim)

    @property
    def theta_num_ws(self) -> int:
        return int(self.G_raw.backbone.mapping.num_ws)

    @property
    def beta_num_ws(self) -> int:
        return int(self.G_res.num_ws)

    def theta_avg(self) -> torch.Tensor:
        return self.G_raw.backbone.mapping.w_avg.detach().clone()

    def beta_avg(self) -> torch.Tensor:
        return self.G_res.encoder.w_avg.detach().clone()

    def forward(
        self,
        *,
        flame_params: Dict[str, torch.Tensor],
        theta: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        truncation_psi: float = 1.0,
        truncation_cutoff: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            flame_outputs = self.flame_model(flame_params)
            canonical_vertices = flame_outputs['canonical_vertices']
            posed_vertices = flame_outputs['vertices']
            joint_transforms = flame_outputs['joint_transforms']
            perm_vertices, alignment_meta = self._to_perm_space(
                canonical_vertices,
                joint_transforms=joint_transforms,
            )
            root_bundle = self._sample_roots(perm_vertices)

        theta_ws = self._prepare_theta_ws(theta, perm_vertices.shape[0], perm_vertices.device)
        beta_ws = self._prepare_beta_ws(beta, perm_vertices.shape[0], perm_vertices.device)

        guide = self.G_raw.synthesis(theta_ws, noise_mode='const')
        low_rank = self.G_superres(
            {'image_raw': guide['image'], 'image_mask': guide['image_mask']}
        )['image']
        high_rank = self.G_res.synthesis(beta_ws, noise_mode='const')['image']
        coeff_texture = torch.cat([low_rank, high_rank], dim=1)

        coeff = self._sample_texture(
            coeff_texture,
            root_bundle['uv_norm'],
            mode='nearest',
        )
        guide_mask = self._sample_texture(
            guide['image_mask'],
            root_bundle['uv_norm'],
            mode='bilinear',
        ).squeeze(-1)

        strands_perm = self._decode_coefficients(coeff, root_bundle['positions'])
        strand_mask = root_bundle['valid_mask']
        if self.use_guide_mask:
            strand_mask = strand_mask & (guide_mask > self.guide_mask_threshold)

        strands_world = self._project_to_flame_space(
            strands_perm,
            alignment_meta,
            root_bundle['root_weights'],
        )

        return {
            'strands': strands_world,
            'strand_mask': strand_mask,
            'guide_mask': guide_mask,
            'coeff_texture': coeff_texture,
            'guide_image': guide['image'],
            'guide_mask_image': guide['image_mask'],
            'flame_vertices': posed_vertices,
            'flame_faces': self.faces_tensor,
            'perm_vertices': perm_vertices,
            'root_positions_perm': root_bundle['positions'],
            'root_uv_norm': root_bundle['uv_norm'],
            'root_uv_raw': root_bundle['uv_raw'],
        }

    def _load_perm_networks(self, model_path: str) -> None:
        network_pkl_raw = Path(model_path) / 'stylegan2-raw-texture.pkl'
        network_pkl_superres = Path(model_path) / 'unet-superres.pkl'
        network_pkl_res = Path(model_path) / 'vae-res-texture.pkl'

        if not network_pkl_raw.exists():
            raise FileNotFoundError(f"PERM raw texture checkpoint not found: {network_pkl_raw}")
        if not network_pkl_superres.exists():
            raise FileNotFoundError(f"PERM super-resolution checkpoint not found: {network_pkl_superres}")
        if not network_pkl_res.exists():
            raise FileNotFoundError(f"PERM residual texture checkpoint not found: {network_pkl_res}")

        with dnnlib.util.open_url(str(network_pkl_raw)) as f:
            self.G_raw = legacy.load_network_pkl(f)['G_ema'].eval().requires_grad_(False)
        with dnnlib.util.open_url(str(network_pkl_superres)) as f:
            self.G_superres = legacy.load_network_pkl(f)['G'].eval().requires_grad_(False)
        with dnnlib.util.open_url(str(network_pkl_res)) as f:
            self.G_res = legacy.load_network_pkl(f)['G'].eval().requires_grad_(False)

    def _load_scalp_face_mask(self, scalp_mask_path: str, scalp_mask_key: str) -> torch.Tensor:
        with open(scalp_mask_path, 'rb') as f:
            flame_masks = pickle.load(f, encoding='latin1')

        mask_vertices = torch.as_tensor(flame_masks[scalp_mask_key], dtype=torch.long)
        vertex_mask = torch.zeros(self.flame_model.v_template.shape[0], dtype=torch.bool)
        vertex_mask[mask_vertices] = True
        return vertex_mask[self.flame_model.faces_tensor].all(dim=1)

    def _build_root_support(self) -> None:
        device = self.faces_tensor.device
        dtype = self.flame_model.v_template.dtype
        res = self.root_grid_resolution

        u_coords = (torch.arange(res, dtype=dtype, device=device) + 0.5) / res
        v_coords = (torch.arange(res, dtype=dtype, device=device) + 0.5) / res
        uu, vv = torch.meshgrid(u_coords, v_coords, indexing='xy')
        uv_norm = torch.stack([uu, vv], dim=-1).reshape(-1, 2)
        uv_raw = self._rescale_uv(uv_norm, inverse=True)
        directions = self._uv_to_directions(uv_raw)

        template_vertices = self.flame_model.v_template.unsqueeze(0).to(device=device, dtype=dtype)
        perm_template, _ = self._to_perm_space(template_vertices, joint_transforms=None)
        origin = self._mesh_origin_from_bounds(perm_template)[0]
        hits, hit_mask, _, _ = self._raycast_single_mesh(
            origins=origin.expand(directions.shape[0], -1),
            directions=directions,
            vertices=perm_template[0],
            scalp_only=True,
        )
        del hits

        uv_norm = uv_norm[hit_mask]
        uv_raw = uv_raw[hit_mask]
        directions = directions[hit_mask]

        if self.num_render_strands is not None and uv_norm.shape[0] > self.num_render_strands:
            keep = torch.linspace(
                0,
                uv_norm.shape[0] - 1,
                steps=self.num_render_strands,
                device=device,
            ).round().long()
            uv_norm = uv_norm[keep]
            uv_raw = uv_raw[keep]
            directions = directions[keep]

        self.register_buffer('root_uv_norm', uv_norm)
        self.register_buffer('root_uv_raw', uv_raw)
        self.register_buffer('root_directions', directions)

    def _prepare_theta_ws(
        self,
        theta: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if theta is None:
            theta = self.theta_avg().to(device=device).unsqueeze(0).expand(batch_size, -1)
        return self._expand_latent(theta, self.theta_num_ws)

    def _prepare_beta_ws(
        self,
        beta: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if beta is None:
            beta = self.beta_avg().to(device=device).unsqueeze(0).expand(batch_size, -1)
        return self._expand_latent(beta, self.beta_num_ws)

    def _expand_latent(self, latent: torch.Tensor, num_ws: int) -> torch.Tensor:
        if latent.ndim == 3:
            return latent
        if latent.ndim != 2:
            raise ValueError(f"Expected latent shape (B, C) or (B, L, C), got {tuple(latent.shape)}")
        if self.latent_space != 'broadcast_w':
            raise NotImplementedError(
                f"Unsupported PERM latent space '{self.latent_space}'. Only 'broadcast_w' is implemented."
            )
        return latent.unsqueeze(1).repeat(1, num_ws, 1)

    def _sample_roots(self, perm_vertices: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = perm_vertices.shape[0]
        num_roots = self.root_uv_norm.shape[0]
        device = perm_vertices.device
        dtype = perm_vertices.dtype

        positions = torch.zeros(batch_size, num_roots, 3, device=device, dtype=dtype)
        valid_mask = torch.zeros(batch_size, num_roots, device=device, dtype=torch.bool)
        face_indices = torch.full((batch_size, num_roots), -1, device=device, dtype=torch.long)
        barycentric = torch.zeros(batch_size, num_roots, 3, device=device, dtype=dtype)

        directions = self.root_directions.to(device=device, dtype=dtype)
        for batch_idx in range(batch_size):
            origin = self._mesh_origin_from_bounds(perm_vertices[batch_idx:batch_idx + 1])[0]
            hits, hit_mask, hit_faces, hit_bary = self._raycast_single_mesh(
                origins=origin.expand(num_roots, -1),
                directions=directions,
                vertices=perm_vertices[batch_idx],
                scalp_only=True,
            )
            positions[batch_idx] = hits
            valid_mask[batch_idx] = hit_mask
            face_indices[batch_idx] = hit_faces
            barycentric[batch_idx] = hit_bary

        root_weights = self._compute_skin_weights(face_indices, barycentric, valid_mask)
        return {
            'positions': positions,
            'valid_mask': valid_mask,
            'face_indices': face_indices,
            'barycentric': barycentric,
            'root_weights': root_weights,
            'uv_norm': self.root_uv_norm.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype),
            'uv_raw': self.root_uv_raw.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype),
        }

    def _sample_texture(
        self,
        image: torch.Tensor,
        uv_norm: torch.Tensor,
        *,
        mode: str,
    ) -> torch.Tensor:
        grid = uv_norm * 2.0 - 1.0
        sampled = F.grid_sample(
            image,
            grid.unsqueeze(2),
            mode=mode,
            padding_mode='border',
            align_corners=True,
        )
        return sampled.squeeze(-1).permute(0, 2, 1).contiguous()

    def _decode_coefficients(
        self,
        coeff: torch.Tensor,
        root_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_roots, _ = coeff.shape
        decoded = self.G_res.strand_codec.decode(coeff.reshape(batch_size * num_roots, -1))
        decoded = decoded.reshape(batch_size, num_roots, -1, 3)
        decoded = F.pad(decoded, (0, 0, 1, 0), mode='constant', value=0.0)
        return decoded + root_positions.unsqueeze(2)

    def _to_perm_space(
        self,
        vertices: torch.Tensor,
        *,
        joint_transforms: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if vertices.ndim != 3 or vertices.shape[-1] != 3:
            raise ValueError("vertices must have shape (B, V, 3)")

        device = vertices.device
        dtype = vertices.dtype
        batch_size = vertices.shape[0]

        scale = torch.full((batch_size, 1, 1), self.scale_to_perm, device=device, dtype=dtype)
        scaled = vertices * scale
        perm_centroid = scaled.mean(dim=1, keepdim=True)
        perm_rotation = self._rotation_matrix_from_euler_xyz(
            self.rotation_to_perm_euler_deg,
            device=device,
            dtype=dtype,
        ).view(1, 3, 3).expand(batch_size, -1, -1)
        perm_inv_rotation = perm_rotation.transpose(1, 2)
        perm_translation = torch.tensor(
            self.translation_to_perm,
            device=device,
            dtype=dtype,
        ).view(1, 1, 3).expand(batch_size, -1, -1)

        centered = scaled - perm_centroid
        rotated = torch.einsum('bij,bvj->bvi', perm_rotation, centered)
        perm_vertices = rotated + perm_translation

        alignment_meta = {
            'scale': scale,
            'perm_centroid': perm_centroid,
            'perm_inv_rotation': perm_inv_rotation,
            'perm_translation': perm_translation,
        }
        if joint_transforms is not None:
            alignment_meta['joint_transforms'] = joint_transforms

        return perm_vertices, alignment_meta

    def _project_to_flame_space(
        self,
        strands: torch.Tensor,
        alignment_meta: Dict[str, torch.Tensor],
        root_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = strands.shape[0]
        scale = alignment_meta['scale'].view(batch_size, 1, 1, 1)
        perm_centroid = alignment_meta['perm_centroid'].view(batch_size, 1, 1, 3)
        perm_translation = alignment_meta['perm_translation'].view(batch_size, 1, 1, 3)
        perm_inv_rotation = alignment_meta['perm_inv_rotation']

        centered = strands - perm_translation
        rotated_back = torch.einsum('bij,bnkj->bnki', perm_inv_rotation, centered)
        canonical = (rotated_back + perm_centroid) / scale

        joint_transforms = alignment_meta.get('joint_transforms')
        if root_weights is None or joint_transforms is None:
            return canonical
        return self._apply_skinning(canonical, root_weights, joint_transforms)

    def _apply_skinning(
        self,
        canonical: torch.Tensor,
        root_weights: torch.Tensor,
        joint_transforms: torch.Tensor,
    ) -> torch.Tensor:
        rotation = joint_transforms[..., :3, :3]
        translation = joint_transforms[..., :3, 3]

        batch_size, num_roots, num_points, _ = canonical.shape
        weights = root_weights[:, :num_roots].unsqueeze(-1).unsqueeze(-1)
        rotated = torch.einsum('bjkl,bnpl->bnjpk', rotation, canonical)
        weighted_rot = rotated * weights
        translated = translation[:, None, :, None, :].expand(-1, num_roots, -1, num_points, -1)
        weighted_t = translated * weights.expand(-1, -1, -1, num_points, -1)
        return weighted_rot.sum(dim=2) + weighted_t.sum(dim=2)

    def _compute_skin_weights(
        self,
        face_indices: torch.Tensor,
        barycentric: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        safe_idx = face_indices.clamp(min=0)
        tri_vertices = self.faces_tensor.to(face_indices.device)[safe_idx.view(-1)].view(*safe_idx.shape, 3)
        vert_weights = self.vertex_lbs_weights.to(face_indices.device)[tri_vertices]
        weights = (vert_weights * barycentric.unsqueeze(-1)).sum(dim=2)
        weights = torch.where(valid_mask.unsqueeze(-1), weights, torch.zeros_like(weights))
        return weights

    def _raycast_single_mesh(
        self,
        *,
        origins: torch.Tensor,
        directions: torch.Tensor,
        vertices: torch.Tensor,
        scalp_only: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        faces = self.faces_tensor.to(device=vertices.device, dtype=torch.long)
        if scalp_only:
            face_ids = torch.nonzero(self.scalp_face_mask.to(vertices.device), as_tuple=False).squeeze(1)
            faces = faces[face_ids]
        else:
            face_ids = torch.arange(faces.shape[0], device=vertices.device, dtype=torch.long)

        triangles = vertices[faces]
        v0 = triangles[:, 0]
        edge1 = triangles[:, 1] - v0
        edge2 = triangles[:, 2] - v0

        hits = torch.zeros_like(directions)
        hit_mask = torch.zeros(directions.shape[0], dtype=torch.bool, device=vertices.device)
        hit_faces = torch.full((directions.shape[0],), -1, dtype=torch.long, device=vertices.device)
        hit_bary = torch.zeros(directions.shape[0], 3, dtype=vertices.dtype, device=vertices.device)

        chunk_size = min(self.ray_chunk_size, max(1, directions.shape[0]))
        for start in range(0, directions.shape[0], chunk_size):
            end = min(start + chunk_size, directions.shape[0])
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
        t_vec = origins.unsqueeze(1) - v0_exp
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

        hits = origins.clone()
        hit_faces = torch.full((origins.shape[0],), -1, dtype=torch.long, device=origins.device)
        hit_bary = torch.zeros(origins.shape[0], 3, dtype=origins.dtype, device=origins.device)

        if has_hit.any():
            hits[has_hit] = origins[has_hit] + directions[has_hit] * best_t[has_hit].unsqueeze(-1)
            best_idx = torch.argmin(t_valid, dim=1)
            valid_idx = best_idx[has_hit]
            hit_faces[has_hit] = face_ids[valid_idx]
            gather_u = torch.gather(u, 1, best_idx.unsqueeze(-1)).squeeze(-1)
            gather_v = torch.gather(v, 1, best_idx.unsqueeze(-1)).squeeze(-1)
            bary = torch.stack([1.0 - gather_u - gather_v, gather_u, gather_v], dim=-1)
            hit_bary[has_hit] = bary[has_hit]

        return hits, has_hit, hit_faces, hit_bary

    def _mesh_origin_from_bounds(self, vertices: torch.Tensor) -> torch.Tensor:
        origin = (vertices.amin(dim=1) + vertices.amax(dim=1)) * 0.5
        origin[:, 1] = 0.0
        return origin

    def _uv_to_directions(self, uv_raw: torch.Tensor) -> torch.Tensor:
        eps = 1e-4
        uv = uv_raw.clamp(min=eps, max=1.0 - eps) * torch.pi
        cot_u = 1.0 / torch.tan(uv[:, 0])
        cot_v = 1.0 / torch.tan(uv[:, 1])
        h = 2.0 / (cot_u.square() + cot_v.square() + 1.0)
        directions = torch.zeros(uv.shape[0], 3, device=uv.device, dtype=uv.dtype)
        directions[:, 0] = h * cot_u
        directions[:, 1] = h - 1.0
        directions[:, 2] = h * cot_v
        return F.normalize(directions, dim=-1, eps=1e-6)

    def _rescale_uv(self, uv: torch.Tensor, *, inverse: bool) -> torch.Tensor:
        u_min, u_max, v_min, v_max = self.scalp_bounds
        out = uv.clone()
        if inverse:
            out[..., 0] = uv[..., 0] * (u_max - u_min) + u_min
            out[..., 1] = uv[..., 1] * (v_max - v_min) + v_min
        else:
            out[..., 0] = (uv[..., 0] - u_min) / max(u_max - u_min, 1e-6)
            out[..., 1] = (uv[..., 1] - v_min) / max(v_max - v_min, 1e-6)
        return out.clamp(0.0, 1.0)

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
