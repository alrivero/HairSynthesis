"""Utilities to decode FLAME parameters into canonical meshes for PERM."""

from __future__ import annotations

import pickle
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from pytorch3d.io import load_obj

from external.perm.src.hair.hair_models import Perm
from src.FLAME.FLAME import FLAME
from src.FLAME.lbs import lbs, batch_rodrigues


@dataclass
class CanonicalMeshBatch:
    """Container for FLAME canonical meshes and pose bookkeeping."""

    vertices: torch.Tensor
    """Canonical FLAME vertices decoded without the rigid pose (B, V, 3)."""

    root_pose: torch.Tensor
    """Original axis-angle root pose that was stripped before decoding (B, 3)."""

    neck_pose: torch.Tensor
    jaw_pose: torch.Tensor
    eye_pose: torch.Tensor
    eyelid_params: Optional[torch.Tensor]
    root_joint: torch.Tensor

    def to_perm_space(self, scale: float = 100.0) -> 'PermMeshBatch':
        """Scale canonical meshes by ``scale`` to match PERM coordinates."""

        scaled_vertices = self.vertices * scale
        return PermMeshBatch(
            vertices=scaled_vertices,
            root_pose=self.root_pose,
            neck_pose=self.neck_pose,
            jaw_pose=self.jaw_pose,
            eye_pose=self.eye_pose,
            eyelid_params=self.eyelid_params,
            root_joint=self.root_joint * scale,
            scale=scale,
        )


@dataclass
class PermMeshBatch(CanonicalMeshBatch):
    """Canonical meshes scaled into PERM's unit system."""

    scale: float = 100.0


@dataclass
class ScalpSamples:
    """Batched scalp sampling result tied to canonical FLAME faces."""

    points: torch.Tensor
    canonical_points: torch.Tensor
    face_indices: torch.Tensor
    barycentric_coords: torch.Tensor
    cache_key: str


class FlameAwarePerm(Perm):
    """PERM wrapper that allows separate canonical vs placement roots."""

    def superresolution_with_placement(
        self,
        canonical_roots: torch.Tensor,
        placement_roots: torch.Tensor,
        img: dict,
        beta: torch.Tensor,
    ):
        if beta.ndim == 2:
            beta = beta.unsqueeze(1).repeat(1, self.G_res.num_ws, 1)
        low_rank_coeff = self.G_superres(img)['image']
        high_rank_coeff = self.G_res.synthesis(beta, noise_mode='const')['image']
        image = torch.cat([low_rank_coeff, high_rank_coeff], dim=1)

        coords = self.hair_roots.cartesian_to_spherical(canonical_roots)[..., :2]
        coords = self.hair_roots.rescale(coords)
        strands = self.G_res.sample(image, coords, mode='nearest')
        strands.position = strands.position + placement_roots.unsqueeze(2)

        return {'image': image, 'strands': strands}

    def forward_with_roots(
        self,
        canonical_roots: torch.Tensor,
        placement_roots: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        trunc: float = 1.0,
        trunc_cutoff: Optional[int] = None,
        random_seed: int = 0,
    ):
        batch_size = canonical_roots.shape[0]
        device = canonical_roots.device
        if placement_roots is None:
            placement_roots = canonical_roots
        theta = theta if theta is not None else torch.from_numpy(np.random.RandomState(random_seed).randn(batch_size, self.G_raw.z_dim)).to(device)
        beta = beta if beta is not None else torch.from_numpy(np.random.RandomState(random_seed).randn(batch_size, self.G_res.w_dim)).to(device)

        out_guide = self.guide_strands(theta, trunc=trunc, trunc_cutoff=trunc_cutoff)
        out = self.superresolution_with_placement(
            canonical_roots=canonical_roots,
            placement_roots=placement_roots,
            img={'image_raw': out_guide['image'], 'image_mask': out_guide['image_mask']},
            beta=beta,
        )
        out.update(guide_strands=out_guide['strands'], theta=theta, beta=beta)
        return out


class FlamePermBridge(nn.Module):
    """Decode FLAME parameters into meshes aligned with PERM coordinates."""

    def __init__(
        self,
        n_shape: int = 300,
        n_expression: int = 50,
        flame_model_path: str = 'assets/FLAME2020/generic_model.pkl',
        flame_lmk_embedding_path: str = 'assets/landmark_embedding.npy',
        canonical_mesh_path: str = 'assets/head_template.obj',
        scalp_mask_path: str = 'assets/FLAME_masks/FLAME_masks.pkl',
        scalp_mask_key: str = 'scalp',
        perm_model_path: Optional[str] = None,
        perm_head_mesh: Optional[str] = None,
        perm_scalp_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        super().__init__()

        self.n_shape = n_shape
        self.n_expression = n_expression

        flame = FLAME(
            flame_model_path=flame_model_path,
            flame_lmk_embedding_path=flame_lmk_embedding_path,
            n_shape=n_shape,
            n_exp=n_expression,
        )
        flame.eval()
        self.flame = flame

        template_vertices, template_faces = self._load_canonical_mesh(canonical_mesh_path)
        scalp_face_indices, scalp_face_probs = self._build_scalp_sampling(
            template_vertices, template_faces, scalp_mask_path, scalp_mask_key
        )

        self.register_buffer('template_vertices', template_vertices)
        self.register_buffer('template_faces', template_faces)
        self.register_buffer('scalp_face_indices', scalp_face_indices)
        self.register_buffer('scalp_face_probs', scalp_face_probs)

        self._sampling_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        self._perm_config: Optional[Dict[str, object]] = None
        self._perm: Optional[FlameAwarePerm] = None
        if perm_model_path is not None:
            if perm_head_mesh is None:
                perm_head_mesh = canonical_mesh_path
            if perm_scalp_bounds is None:
                raise ValueError('perm_scalp_bounds must be provided when loading PERM.')
            self._perm_config = {
                'model_path': perm_model_path,
                'head_mesh': perm_head_mesh,
                'scalp_bounds': perm_scalp_bounds,
            }
            self._init_perm()

    def _ensure_device(self, device: torch.device) -> None:
        # Move FLAME + cached buffers (and PERM if loaded) onto the requested device.
        current_param = next(self.parameters(), None)
        current_device = current_param.device if current_param is not None else None
        if current_device != device:
            self.to(device)

    @torch.no_grad()
    def forward(
        self,
        flame_params: Dict[str, torch.Tensor],
        perm_params: Dict[str, torch.Tensor],
        num_strands: int,
        cache_key: Optional[str] = None,
        refresh_cache: bool = False,
        scale: float = 100.0,
    ) -> Tuple[Dict[str, object], ScalpSamples]:
        """End-to-end pipeline: FLAME params + PERM params -> posed strands."""

        canonical = self._decode_flame(flame_params)
        perm_output, samples = self.decode_perms_strands(
            flame_vertices=canonical.vertices,
            perm_params=perm_params,
            num_strands=num_strands,
            cache_key=cache_key,
            refresh_cache=refresh_cache,
            scale=scale,
        )
        posed_output = self._apply_flame_pose(perm_output, canonical, scale=scale)
        posed_output['root_pose'] = canonical.root_pose
        posed_output['root_joint'] = canonical.root_joint
        return posed_output, samples

    @torch.no_grad()
    def _decode_flame(self, flame_params: Dict[str, torch.Tensor]) -> CanonicalMeshBatch:
        """Decode FLAME tensors into canonical meshes without the rigid pose."""

        # Unpack FLAME parameters produced by the encoder (already on GPU).
        shape_params = flame_params['shape_params']
        expression_params = flame_params['expression_params']
        pose_params = flame_params['pose_params']
        jaw_params = flame_params['jaw_params']
        eyelid_params = flame_params.get('eyelid_params', None)
        neck_pose_params = flame_params.get('neck_pose_params', None)
        eye_pose_params = flame_params.get('eye_pose_params', None)

        device = shape_params.device
        self._ensure_device(device)

        # Sanity-check we were given the full PCA lengths expected by the loaded FLAME model.
        if shape_params.shape[1] != self.n_shape or expression_params.shape[1] != self.n_expression:
            raise ValueError('Shape/expression dimensions must match FLAME configuration.')

        batch_size = shape_params.shape[0]

        # If extra pose components were omitted, fall back to the learnable defaults bundled with FLAME.
        if neck_pose_params is None:
            neck_pose_params = self.flame.neck_pose.expand(batch_size, -1).to(device)
        if eye_pose_params is None:
            eye_pose_params = self.flame.eye_pose.expand(batch_size, -1).to(device)

        # Concatenate shape + expression into the betas tensor FLAME expects.
        betas = torch.cat([shape_params, expression_params], dim=1)

        # Remove the global rigid rotation before skinning but keep a copy for later re-application.
        pose_no_rigid = torch.zeros_like(pose_params)
        root_pose = pose_params.clone()

        # Broadcast the template mesh so LBS can deform it per batch element.
        template_vertices = self.flame.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        # Run linear blend skinning with the zeroed root pose and remaining joint rotations.
        verts, joints = lbs(
            betas=betas,
            pose=torch.cat([pose_no_rigid, neck_pose_params, jaw_params, eye_pose_params], dim=1),
            v_template=template_vertices,
            shapedirs=self.flame.shapedirs,
            posedirs=self.flame.posedirs,
            J_regressor=self.flame.J_regressor,
            parents=self.flame.parents,
            lbs_weights=self.flame.lbs_weights,
            dtype=self.flame.dtype,
        )

        # Apply eyelid offsets if they were predicted; this matches FLAME.forward behaviour.
        if eyelid_params is not None:
            verts = verts + self.flame.r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]
            verts = verts + self.flame.l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]

        return CanonicalMeshBatch(
            vertices=verts,
            root_pose=root_pose,
            neck_pose=neck_pose_params,
            jaw_pose=jaw_params,
            eye_pose=eye_pose_params,
            eyelid_params=eyelid_params,
            root_joint=joints[:, 0],
        )

    @torch.no_grad()
    def decode_perm_space(self, flame_params: Dict[str, torch.Tensor], scale: float = 100.0) -> PermMeshBatch:
        """Convenience wrapper that decodes FLAME params and scales vertices for PERM."""

        canonical = self._decode_flame(flame_params)
        return canonical.to_perm_space(scale=scale)

    @torch.no_grad()
    def decode_perms_strands(
        self,
        flame_vertices: torch.Tensor,
        perm_params: Dict[str, torch.Tensor],
        num_strands: int,
        cache_key: Optional[str] = None,
        refresh_cache: bool = False,
        scale: float = 100.0,
    ) -> Tuple[Dict[str, object], ScalpSamples]:
        """Sample scalp roots on FLAME meshes and decode strands with PERM."""

        self._ensure_device(flame_vertices.device)

        perm_model = self._get_perm()
        perm_device = next(perm_model.parameters()).device

        samples = self.sample_scalp_points(
            vertices=flame_vertices,
            num_points=num_strands,
            cache_key=cache_key,
            refresh_cache=refresh_cache,
        )

        placement_roots = (samples.points * scale).to(device=perm_device, dtype=torch.float32)

        canonical_points = samples.canonical_points.to(perm_device, dtype=torch.float32)
        batch_size = placement_roots.shape[0]
        canonical_roots = canonical_points.unsqueeze(0).expand(batch_size, -1, -1) * scale

        theta = perm_params.get('theta')
        if theta is not None:
            theta = theta.to(perm_device)
        beta = perm_params.get('beta')
        if beta is not None:
            beta = beta.to(perm_device)
        trunc = perm_params.get('trunc', 1.0)
        trunc_cutoff = perm_params.get('trunc_cutoff', None)
        random_seed = perm_params.get('random_seed', 0)

        perm_output = perm_model.forward_with_roots(
            canonical_roots=canonical_roots,
            placement_roots=placement_roots,
            theta=theta,
            beta=beta,
            trunc=trunc,
            trunc_cutoff=trunc_cutoff,
            random_seed=random_seed,
        )
        perm_output['placement_roots'] = placement_roots
        perm_output['canonical_roots'] = canonical_roots

        return perm_output, samples

    def sample_scalp_points(
        self,
        vertices: torch.Tensor,
        num_points: int,
        cache_key: Optional[str] = None,
        refresh_cache: bool = False,
    ) -> ScalpSamples:
        """Sample scalp points on each mesh using cached canonical barycentric templates."""

        cache_key = cache_key or f'{num_points}'
        face_ids, bary = self._get_sampling_template(
            num_points=num_points,
            cache_key=cache_key,
            refresh_cache=refresh_cache,
            device=vertices.device,
        )

        global_face_ids = self.scalp_face_indices[face_ids]
        selected_faces = self.template_faces[global_face_ids]

        template_triangles = self.template_vertices[selected_faces]
        canonical_points = (template_triangles * bary.view(num_points, 3, 1)).sum(dim=1)

        batch_size = vertices.shape[0]
        tri_indices = selected_faces.reshape(-1)
        tri_vertices = vertices[:, tri_indices, :].reshape(batch_size, num_points, 3, 3)
        weights = bary.view(1, num_points, 3, 1)
        points = (tri_vertices * weights).sum(dim=2)

        return ScalpSamples(
            points=points,
            canonical_points=canonical_points,
            face_indices=global_face_ids,
            barycentric_coords=bary,
            cache_key=cache_key,
        )

    def _init_perm(self) -> None:
        if self._perm_config is None:
            raise RuntimeError('PERM configuration was not provided.')
        if self._perm is None:
            self._perm = FlameAwarePerm(**self._perm_config).eval().requires_grad_(False)

    def _get_perm(self) -> FlameAwarePerm:
        if self._perm is None:
            self._init_perm()
        assert self._perm is not None
        return self._perm

    def _load_canonical_mesh(self, mesh_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        verts, faces, _ = load_obj(mesh_path)
        faces = faces.verts_idx.to(dtype=torch.long)
        return verts, faces

    def _build_scalp_sampling(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        mask_path: str,
        mask_key: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with open(mask_path, 'rb') as f:
            flame_masks = pickle.load(f, encoding='latin1')

        mask_vertices = torch.as_tensor(flame_masks[mask_key], dtype=torch.long)
        vertex_mask = torch.zeros(vertices.shape[0], dtype=torch.bool)
        vertex_mask[mask_vertices] = True
        scalp_face_mask = vertex_mask[faces].all(dim=1)
        scalp_face_indices = torch.nonzero(scalp_face_mask, as_tuple=False).squeeze(1)

        scalp_faces = faces[scalp_face_indices]
        tri_vertices = vertices[scalp_faces]
        edges1 = tri_vertices[:, 1] - tri_vertices[:, 0]
        edges2 = tri_vertices[:, 2] - tri_vertices[:, 0]
        areas = 0.5 * torch.norm(torch.cross(edges1, edges2, dim=1), dim=1)
        probs = areas / torch.clamp(areas.sum(), min=1e-8)

        return scalp_face_indices, probs

    def _sample_barycentric(self, num_points: int, device: torch.device) -> torch.Tensor:
        u = torch.rand(num_points, device=device)
        v = torch.rand(num_points, device=device)
        sqrt_u = torch.sqrt(u)
        b0 = 1 - sqrt_u
        b1 = sqrt_u * (1 - v)
        b2 = sqrt_u * v
        return torch.stack([b0, b1, b2], dim=1)

    def _get_sampling_template(
        self,
        num_points: int,
        cache_key: str,
        refresh_cache: bool,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = self._sampling_cache.get(cache_key)
        if cached is None or refresh_cache:
            face_ids = torch.multinomial(self.scalp_face_probs, num_points, replacement=True)
            bary = self._sample_barycentric(num_points, device=self.scalp_face_probs.device)
            cached = (face_ids, bary)
            self._sampling_cache[cache_key] = cached

        face_ids, bary = cached
        return face_ids.to(device), bary.to(device)

    def _apply_flame_pose(
        self,
        perm_output: Dict[str, object],
        canonical: CanonicalMeshBatch,
        scale: float,
    ) -> Dict[str, object]:
        """Rotate PERM outputs into the posed FLAME frame."""

        rot_mats = batch_rodrigues(canonical.root_pose.view(-1, 3)).view(-1, 3, 3)
        pivots = canonical.root_joint

        def transform_positions(pos: torch.Tensor) -> torch.Tensor:
            if pos is None:
                return None
            pts = pos / scale
            centered = pts - pivots[:, None, None, :]
            rotated = torch.einsum('bij,b...j->b...i', rot_mats, centered)
            return rotated + pivots[:, None, None, :]

        for key in ['strands', 'guide_strands']:
            strands = perm_output.get(key)
            if strands is not None and strands.position is not None:
                strands.position = transform_positions(strands.position)
        if 'placement_roots' in perm_output:
            perm_output['placement_roots'] = transform_positions(perm_output['placement_roots'].unsqueeze(2)).squeeze(2)
        if 'canonical_roots' in perm_output:
            perm_output['canonical_roots'] = perm_output['canonical_roots'] / scale

        return perm_output
