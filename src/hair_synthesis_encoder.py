import torch
from torch import nn
import numpy as np

from external.HairStep.lib.model.img2hairstep.UNet import Model as StrandModel
from external.HairStep.lib.model.img2hairstep.hourglass import Model as DepthModel
from src.utils.torchutils import torch_nanminmax


class HairSynthesisEncoder(nn.Module):
    """Wrapper around HairStep's strand and optional depth encoders."""

    def __init__(
        self,
        img2strand_ckpt=None,
        img2depth_ckpt=None,
        config=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.encoder_mode = getattr(self.config.arch, 'encoder_mode', 'hairstep_maps')

        self.strand_encoder = StrandModel()
        if img2strand_ckpt is not None:
            self.strand_encoder.load_state_dict(torch.load(img2strand_ckpt))

        if self.encoder_mode == 'perm_latent':
            perm_latent_dim = int(getattr(self.config.arch, 'perm_latent_dim', 512))
            perm_condition_dim = int(getattr(self.config.arch, 'perm_theta_condition_dim', 256))
            perm_cfg = getattr(self.config, 'perm', None)
            self.theta_pool = nn.AdaptiveAvgPool2d(1)
            self.fused_pool = nn.AdaptiveAvgPool2d(1)
            self.theta_head = nn.Sequential(
                nn.Linear(256, 256),
                nn.LeakyReLU(inplace=True),
                nn.Linear(256, perm_latent_dim),
            )
            self.theta_condition = nn.Sequential(
                nn.Linear(perm_latent_dim, perm_condition_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(perm_condition_dim, perm_condition_dim),
            )
            self.beta_head = nn.Sequential(
                nn.Linear(256 + 16 + perm_condition_dim, 256),
                nn.LeakyReLU(inplace=True),
                nn.Linear(256, perm_latent_dim),
            )
            theta_base, beta_base = self._load_perm_init_params(
                getattr(perm_cfg, 'init_params_path', None),
            )
            self.register_buffer('theta_base', theta_base)
            self.register_buffer('beta_base', beta_base)
            self._initialize_perm_latent_heads()
        elif config.arch.depth_branch:
            self.depth_encoder = DepthModel()
            if img2depth_ckpt is not None:
                depth_state = torch.load(img2depth_ckpt)
                # HairStep trained DepthModel under DataParallel, so strip the module prefix.
                if "module." in list(depth_state.keys())[0]:
                    depth_state = {k.replace("module.", ""): v for k, v in depth_state.items()}
                self.depth_encoder.load_state_dict(depth_state)

    def forward(self, img, hair_mask, body_mask):
        """Extract strand features and optional depth.

        Args:
            img (torch.Tensor): Input image of shape (B, C, H, W).
            hair_mask (torch.Tensor): Hair mask of shape (B, 1, H, W).
            body_mask (torch.Tensor): Body mask of shape (B, 1, H, W).

        Returns:
            Dict[str, torch.Tensor]:
                `strand_params`: (B, 3, H, W)
                `depth_params`: (B, 1, H, W) when depth_branch is enabled
        """
        img = img * hair_mask

        if self.encoder_mode == 'perm_latent':
            deep_feat, fused_feat = self._extract_backbone_features(img)
            deep_pooled = self.theta_pool(deep_feat).flatten(1)
            fused_pooled = self.fused_pool(fused_feat).flatten(1)

            theta_residual = self.theta_head(deep_pooled)
            theta = self._compose_perm_latent(theta_residual, self.theta_base)
            theta_condition_input = theta.mean(dim=1) if theta.ndim == 3 else theta
            theta_condition = self.theta_condition(theta_condition_input).detach()
            beta_residual = self.beta_head(torch.cat([deep_pooled, fused_pooled, theta_condition], dim=1))
            beta = self._compose_perm_latent(beta_residual, self.beta_base)

            return {
                'theta': theta,
                'beta': beta,
                'theta_residual': theta_residual,
                'beta_residual': beta_residual,
            }

        body = body_mask * (1 - hair_mask)
        strand_pred = self.strand_encoder(img)
        strand_pred = strand_pred * 2.0 - 1.0

        if self.config.arch.encoder_norm_strand_magnitude:
            strand_pred = strand_pred * hair_mask
            x = strand_pred[:, 0:1, :, :]
            y = strand_pred[:, 1:2, :, :]

            magnitude = torch.sqrt(x * x + y * y + 1e-10)
            x_norm = x / magnitude
            y_norm = y / magnitude
            strand_pred = torch.cat([x_norm, y_norm], dim=1)
        else:
            strand_pred = strand_pred.clamp(0.0, 1.0)

        strand_pred = torch.cat([hair_mask + body * 0.5, strand_pred * hair_mask], dim=1)

        if self.config.arch.depth_branch:
            depth_pred = self.depth_encoder(img)
            depth_pred_norm = self._normalize_depth_prediction(depth_pred, hair_mask)

            return {
                'strand_params': strand_pred,
                'depth_params': depth_pred_norm,
            }

        return {
            'strand_params': strand_pred,
        }

    def _extract_backbone_features(self, img):
        body = self.strand_encoder.body
        r1 = body.C1(img)
        r2 = body.C2(body.D1(r1))
        r3 = body.C3(body.D2(r2))
        r4 = body.C4(body.D3(r3))
        deep = body.C5(body.D4(r4))

        o1 = body.C6(body.U1(deep, r4))
        o2 = body.C7(body.U2(o1, r3))
        o3 = body.C8(body.U3(o2, r2))
        fused = body.C9(body.U4(o3, r1))
        return deep, fused

    def _initialize_perm_latent_heads(self) -> None:
        # Start new PERM latent regressors at exactly zero while keeping the HairStep backbone warm-started.
        for head in (self.theta_head, self.theta_condition, self.beta_head):
            final_linear = head[-1]
            if not isinstance(final_linear, nn.Linear):
                raise TypeError("Expected perm latent heads to end with nn.Linear layers.")
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def _load_perm_init_params(self, init_params_path):
        if not init_params_path:
            return None, None

        perm_params = np.load(init_params_path)
        if 'theta' not in perm_params or 'beta' not in perm_params:
            raise KeyError(
                f"PERM init params at {init_params_path} must contain 'theta' and 'beta' arrays."
            )

        theta = self._normalize_perm_base_array(perm_params['theta'], name='theta')
        beta = self._normalize_perm_base_array(perm_params['beta'], name='beta')
        return theta, beta

    def _normalize_perm_base_array(self, array, *, name):
        tensor = torch.as_tensor(array, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim not in {2, 3}:
            raise ValueError(
                f"PERM init param '{name}' must be rank 1, 2, or 3; got shape {tuple(tensor.shape)}."
            )
        return tensor.contiguous()

    def _compose_perm_latent(self, residual, base):
        if base is None:
            return residual

        if base.ndim == residual.ndim:
            if base.shape[0] == 1 and residual.shape[0] != 1:
                base = base.expand(residual.shape[0], *base.shape[1:])
            elif base.shape[0] != residual.shape[0]:
                raise ValueError(
                    f"PERM base latent batch {base.shape[0]} does not match residual batch {residual.shape[0]}."
                )
            return base.to(device=residual.device, dtype=residual.dtype) + residual

        if base.ndim == 3 and residual.ndim == 2:
            if base.shape[0] == 1 and residual.shape[0] != 1:
                base = base.expand(residual.shape[0], -1, -1)
            elif base.shape[0] != residual.shape[0]:
                raise ValueError(
                    f"PERM base latent batch {base.shape[0]} does not match residual batch {residual.shape[0]}."
                )
            return base.to(device=residual.device, dtype=residual.dtype) + residual.unsqueeze(1)

        raise ValueError(
            f"Unsupported PERM latent composition shapes: residual={tuple(residual.shape)}, "
            f"base={tuple(base.shape)}."
        )

    def _normalize_depth_prediction(self, depth_pred, hair_mask, eps=1e-8):
        """Normalize depth over the valid hair support and avoid NaNs on empty or flat masks."""
        hair_mask = hair_mask.float().clamp(0.0, 1.0)
        valid_hair = hair_mask > 1e-6
        depth_on_hair = torch.where(
            valid_hair,
            depth_pred,
            torch.full_like(depth_pred, float('nan')),
        )

        max_val = torch_nanminmax(depth_on_hair, 'max', dim=(1, 2, 3), keepdim=True)
        min_val = torch_nanminmax(depth_on_hair, 'min', dim=(1, 2, 3), keepdim=True)
        depth_range = max_val - min_val

        has_valid_range = (
            torch.isfinite(min_val)
            & torch.isfinite(max_val)
            & torch.isfinite(depth_range)
            & (depth_range > eps)
        )

        safe_min = torch.where(has_valid_range, min_val, torch.zeros_like(min_val))
        safe_range = torch.where(has_valid_range, depth_range, torch.ones_like(depth_range))
        normalized = (depth_pred - safe_min) / safe_range
        normalized = torch.where(has_valid_range, normalized, torch.zeros_like(normalized))
        return normalized.mul(hair_mask).clamp(0.0, 1.0)
