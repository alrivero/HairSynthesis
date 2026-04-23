"""Lightweight strand rasterizer that composites hair over a posed FLAME mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer import rasterize_meshes
from pytorch3d.structures import Meshes

from src.renderer.util import batch_orth_proj


@dataclass
class HairRenderOutput:
    """Container for debugging or downstream supervision."""

    image: torch.Tensor              # (B, 4, H, W) packed [mask, dx, dy, depth_norm]
    visibility_mask: torch.Tensor    # (B, 1, H, W) binary mask of rendered hair
    depth: torch.Tensor              # (B, 1, H, W) raw z-buffer of the visible hair


class HairStrandRasterizer:
    """Project per-strand geometry into images with mesh-based occlusion."""

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 512,
        *,
        background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        head_scale: float = 0.96,
        sigma: float = 0.0,
        faces_per_pixel: int = 1,
        stroke_width: float = 0.004,
        occlusion_epsilon: float = 1e-4,
        depth_bias: float = 10.0,
        device: Optional[torch.device] = None,
    ) -> None:
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = (int(image_size[0]), int(image_size[1]))
        self.background_color = torch.tensor(background_color, dtype=torch.float32)
        self.head_scale = float(head_scale)
        self.blur_radius = float(sigma)
        self.faces_per_pixel = int(faces_per_pixel)
        self.stroke_width = float(stroke_width)
        self.occlusion_epsilon = float(occlusion_epsilon)
        self.depth_bias = float(depth_bias)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def forward(
        self,
        *,
        flame_vertices: torch.Tensor,
        flame_faces: torch.Tensor,
        strands: torch.Tensor,
        cam_params: torch.Tensor,
        strand_colors: Optional[torch.Tensor] = None,
        stroke_width: Optional[float] = None,
    ) -> HairRenderOutput:
        """Render hair strands into HairSynthesis-style 2D features.

        Args:
            flame_vertices: (B, V, 3) posed FLAME vertices in world space.
            flame_faces: (F, 3) or (B, F, 3) triangle indices.
            strands: (B, N, P, 3) strand samples already glued to FLAME space.
            cam_params: (B, 3) orthographic parameters [scale, tx, ty].
            strand_colors: optional (B, N, P, C) per-point direction features;
                when omitted, image-plane dx/dy are derived from the rendered strands.
            stroke_width: width of the billboarded strand segments in clip units.
        """
        batch_size = flame_vertices.shape[0]
        device = flame_vertices.device
        stroke = float(stroke_width) if stroke_width is not None else self.stroke_width

        faces = self._expand_faces(flame_faces, batch_size, device)

        scaled_flame_vertices = self._scale_vertices(flame_vertices, self.head_scale)
        projected_vertices = self._project_vertices(scaled_flame_vertices, cam_params)
        raster_vertices = self._to_raster_space(projected_vertices)
        verts_list = list(torch.unbind(raster_vertices.contiguous(), dim=0))
        faces_batched = faces.reshape(faces.shape[0], -1, 3)
        faces_list = [fb.contiguous().long() for fb in torch.unbind(faces_batched, dim=0)]
        head_meshes = Meshes(verts=verts_list, faces=faces_list)
        head_meshes = self._ensure_triangle_mesh(head_meshes, label='head')

        raster_strands = self._prepare_strands(strands, cam_params)
        strand_feature_map = self._prepare_features(raster_strands, strand_colors)
        hair_verts_list, hair_faces_list, hair_feature_list = self._build_strand_mesh(
            raster_strands,
            strand_feature_map,
            stroke,
        )
        hair_meshes = Meshes(verts=hair_verts_list, faces=hair_faces_list)
        hair_meshes = self._ensure_triangle_mesh(hair_meshes, label='hair')

        head_fragments = self._rasterize(head_meshes, blur=0.0)
        hair_fragments = self._rasterize(hair_meshes, blur=self.blur_radius)

        head_depth, head_mask = self._extract_depth(head_fragments)
        hair_depth, hair_mask = self._extract_depth(hair_fragments)

        visible_mask = hair_mask & (
            (~head_mask) | (hair_depth <= (head_depth - self.occlusion_epsilon))
        )

        hair_features = self._sample_hair_attributes(
            hair_meshes,
            hair_fragments,
            hair_feature_list,
            feat_dim=2,
        )
        visible_dxdy = self._apply_visibility_mask(hair_features, visible_mask)
        visible_dxdy = self._rotate_dxdy_to_hairstep(visible_dxdy)
        depth_norm = self._normalize_visible_depth(hair_depth, visible_mask)

        depth_map = torch.where(
            visible_mask,
            hair_depth,
            torch.zeros_like(hair_depth),
        )
        packed = torch.cat(
            [
                visible_mask.unsqueeze(1).float(),
                visible_dxdy,
                depth_norm.unsqueeze(1),
            ],
            dim=1,
        )

        return HairRenderOutput(
            image=packed,
            visibility_mask=visible_mask.unsqueeze(1).float(),
            depth=depth_map.unsqueeze(1),
        )

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------
    def _project_vertices(self, vertices: torch.Tensor, cam_params: torch.Tensor) -> torch.Tensor:
        """Project vertices into the same intermediate space used by the legacy renderer."""
        proj = batch_orth_proj(vertices, cam_params)  # (B, V, 3)
        proj[:, :, 1:] = -proj[:, :, 1:]
        proj[:, :, 2] = proj[:, :, 2] + self.depth_bias
        return proj

    def _prepare_strands(self, strands: torch.Tensor, cam_params: torch.Tensor) -> torch.Tensor:
        batch, num_strands, num_points, _ = strands.shape
        flattened = strands.reshape(batch, -1, 3)
        projected = self._project_vertices(flattened, cam_params)
        raster = self._to_raster_space(projected)
        return raster.reshape(batch, num_strands, num_points, 3)

    def _scale_vertices(self, vertices: torch.Tensor, scale: float) -> torch.Tensor:
        if scale == 1.0:
            return vertices
        center = vertices.mean(dim=1, keepdim=True)
        return (vertices - center) * scale + center

    def _to_raster_space(self, vertices: torch.Tensor) -> torch.Tensor:
        fixed = vertices.clone()
        fixed[..., :2] = -fixed[..., :2]
        h, w = self.image_size
        if h > w:
            fixed[..., 1] = fixed[..., 1] * h / max(w, 1)
        elif w > h:
            fixed[..., 0] = fixed[..., 0] * w / max(h, 1)
        return fixed

    def _expand_faces(self, faces: torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
        if faces.dim() == 2:
            faces = faces.unsqueeze(0)
        if faces.shape[0] == 1 and batch > 1:
            faces = faces.expand(batch, -1, -1)
        faces = faces.to(device).long()
        if faces.shape[-1] != 3:
            faces = faces.reshape(faces.shape[0], -1, 3)
        return faces

    # ------------------------------------------------------------------
    # Strand processing
    # ------------------------------------------------------------------
    def _prepare_features(
        self,
        strands: torch.Tensor,
        strand_colors: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if strand_colors is None:
            diffs = strands[:, :, 1:, :2] - strands[:, :, :-1, :2]
            last = diffs[:, :, -1:, :]
            features = torch.cat([diffs, last], dim=2)
            features = F.normalize(features, dim=-1, eps=1e-6)
        else:
            if strand_colors.shape[-1] < 2:
                raise ValueError("strand_colors must have at least 2 channels when provided")
            features = strand_colors[..., :2]
        return features.clamp(-1.0, 1.0)

    def _build_strand_mesh(
        self,
        strands: torch.Tensor,
        features: torch.Tensor,
        stroke_width: float,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        verts_list: List[torch.Tensor] = []
        faces_list: List[torch.Tensor] = []
        feature_list: List[torch.Tensor] = []

        batch, num_strands, num_points, _ = strands.shape
        for b in range(batch):
            start = strands[b, :, :-1, :]
            end = strands[b, :, 1:, :]
            seg_features = features[b, :, :-1, :]

            dir2d = end[..., :2] - start[..., :2]
            seg_len = torch.linalg.norm(dir2d, dim=-1)
            valid = seg_len > 1e-6

            if not valid.any():
                verts_list.append(strands.new_zeros((0, 3)))
                faces_list.append(torch.zeros((0, 3), dtype=torch.long, device=strands.device))
                feature_list.append(strands.new_zeros((0, 2)))
                continue

            start_valid = start[valid]
            end_valid = end[valid]
            seg_features_valid = seg_features[valid]
            dir_valid = dir2d[valid]

            perp = torch.stack([-dir_valid[:, 1], dir_valid[:, 0]], dim=-1)
            perp = F.normalize(perp, dim=-1, eps=1e-6) * (stroke_width * 0.5)
            offset = torch.cat(
                [perp, torch.zeros(perp.shape[0], 1, device=perp.device, dtype=perp.dtype)],
                dim=-1,
            )

            v0 = start_valid - offset
            v1 = start_valid + offset
            v2 = end_valid + offset
            v3 = end_valid - offset

            quad = torch.stack([v0, v1, v2, v3], dim=1).reshape(-1, 3).contiguous()
            verts_list.append(quad)

            feature_rep = seg_features_valid.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 2)
            feature_list.append(feature_rep.clamp(-1.0, 1.0))

            num_segments = start_valid.shape[0]
            idx = torch.arange(num_segments, device=quad.device)[:, None] * 4
            tris = torch.stack(
                [
                    torch.stack([idx[:, 0] + 0, idx[:, 0] + 1, idx[:, 0] + 2], dim=-1),
                    torch.stack([idx[:, 0] + 0, idx[:, 0] + 2, idx[:, 0] + 3], dim=-1),
                ],
                dim=1,
            )
            faces = tris.view(-1, 3).contiguous().long()
            faces_list.append(faces)

        faces_list = [
            (faces if faces.numel() == 0 else faces.reshape(-1, 3).contiguous().long())
            for faces in faces_list
        ]
        return verts_list, faces_list, feature_list

    # ------------------------------------------------------------------
    # Rasterization & compositing
    # ------------------------------------------------------------------
    def _rasterize(self, meshes: Meshes, blur: float) -> Fragments:
        pix_to_face, zbuf, bary, dists = rasterize_meshes(
            meshes,
            image_size=list(self.image_size),
            blur_radius=blur,
            faces_per_pixel=self.faces_per_pixel,
            bin_size=None,
            max_faces_per_bin=None,
            perspective_correct=False,
        )
        return Fragments(
            pix_to_face=pix_to_face,
            zbuf=zbuf,
            bary_coords=bary,
            dists=dists,
        )

    def _extract_depth(self, fragments: Fragments) -> Tuple[torch.Tensor, torch.Tensor]:
        zbuf = fragments.zbuf[..., 0]
        mask = fragments.pix_to_face[..., 0] >= 0
        depth = torch.where(mask, zbuf, torch.full_like(zbuf, float('inf')))
        return depth, mask

    def _sample_hair_attributes(
        self,
        meshes: Meshes,
        fragments: Fragments,
        attr_list: List[torch.Tensor],
        feat_dim: int,
    ) -> torch.Tensor:
        if meshes.isempty() or len(attr_list) == 0:
            batch = len(meshes)
            height, width = fragments.pix_to_face.shape[1:3]
            return torch.zeros(
                batch,
                feat_dim,
                height,
                width,
                device=fragments.pix_to_face.device,
                dtype=torch.float32,
            )

        packed_attrs = torch.cat(attr_list, dim=0)
        if packed_attrs.numel() == 0:
            batch = len(meshes)
            height, width = fragments.pix_to_face.shape[1:3]
            return torch.zeros(
                batch,
                feat_dim,
                height,
                width,
                device=fragments.pix_to_face.device,
                dtype=torch.float32,
            )

        face_vertex_attrs = packed_attrs[meshes.faces_packed()]
        sampled = self._interpolate_face_vertex_attributes(face_vertex_attrs, fragments)
        attrs = sampled[..., 0, :].permute(0, 3, 1, 2).contiguous()
        return attrs

    def _apply_visibility_mask(self, attrs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return attrs * mask.unsqueeze(1).float()

    def _rotate_dxdy_to_hairstep(self, dxdy: torch.Tensor) -> torch.Tensor:
        """Rotate image-plane directions 90 degrees counter-clockwise to match HairStep."""
        rotated = torch.stack([-dxdy[:, 1], dxdy[:, 0]], dim=1)
        return rotated.clamp(-1.0, 1.0)

    def _normalize_visible_depth(self, depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = depth.shape[0]
        valid_batch = mask.view(batch, -1).any(dim=1)

        min_depth = torch.where(mask, depth, torch.full_like(depth, float('inf'))).amin(dim=(1, 2))
        max_depth = torch.where(mask, depth, torch.full_like(depth, float('-inf'))).amax(dim=(1, 2))
        min_depth = torch.where(valid_batch, min_depth, torch.zeros_like(min_depth))
        max_depth = torch.where(valid_batch, max_depth, torch.ones_like(max_depth))

        denom = (max_depth - min_depth).clamp(min=1e-6)
        # Match HairStep depth convention: closer pixels should have larger values.
        normalized = (max_depth[:, None, None] - depth) / denom[:, None, None]
        normalized = torch.where(mask, normalized, torch.zeros_like(normalized))
        normalized = torch.where(valid_batch[:, None, None], normalized, torch.zeros_like(normalized))
        return normalized.clamp(0.0, 1.0)

    def _ensure_triangle_mesh(self, meshes: Meshes, label: str) -> Meshes:
        verts = meshes.verts_list()
        faces = meshes.faces_list()
        textures = meshes.textures
        fixed = False

        new_faces: List[torch.Tensor] = []
        for idx, face in enumerate(faces):
            if face.numel() == 0:
                new_faces.append(face.reshape(0, 3))
                continue
            if face.dim() != 2 or face.shape[1] != 3:
                reshaped = face.reshape(-1, 3).contiguous().long()
                print(f"[HairStrandRasterizer] reshaped {label} mesh {idx} faces {face.shape} -> {reshaped.shape}")
                face = reshaped
                fixed = True
            new_faces.append(face.contiguous().long())

        if fixed:
            meshes = Meshes(verts=verts, faces=new_faces, textures=textures)
        return meshes

    def _interpolate_face_vertex_attributes(
        self,
        face_vertex_attrs: torch.Tensor,
        fragments: Fragments,
    ) -> torch.Tensor:
        pix_to_face = fragments.pix_to_face.clone()
        bary = fragments.bary_coords

        mask = pix_to_face < 0
        pix_to_face[mask] = 0

        n, h, w, k, _ = bary.shape
        feat_dim = face_vertex_attrs.shape[-1]
        idx = pix_to_face.reshape(n * h * w * k, 1, 1).expand(n * h * w * k, 3, feat_dim)
        pixel_face_vals = face_vertex_attrs.gather(0, idx).reshape(n, h, w, k, 3, feat_dim)
        pixel_vals = (bary[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0
        return pixel_vals
