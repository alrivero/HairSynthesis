from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from src.base_trainer import BaseHairTrainer
from src.hair_map_corruption import HairMapCorruptor
from src.hair_map_visualization import save_packed_map_comparison
from src.models.hair_map_dae import HairMapDAE
from src.synthetic_hair_map_generator import SyntheticHairMapGenerator


class HairMapDAETrainer(BaseHairTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.hair_encoder = HairMapDAE.from_config(config)
        self.synthetic_generator = SyntheticHairMapGenerator(config)
        self.synthetic_generator.eval()
        for param in self.synthetic_generator.parameters():
            param.requires_grad_(False)
        self.corruptor = HairMapCorruptor(config)
        self.current_loss: Optional[Dict[str, float]] = None

    def configure_optimizers(self, n_steps, epoch_idx=None):
        self.n_steps = max(1, int(n_steps))
        lr = float(
            self._cfg_get(
                self.config.train,
                'lr_encoder',
                self._cfg_get(self.config.train, 'lr_dae', self._cfg_get(self.config.train, 'lr', 1e-4)),
            )
        )
        optimizer_cfg = getattr(self.config.train, 'optimizer', None)
        betas = tuple(getattr(optimizer_cfg, 'betas', (0.9, 0.999)))
        weight_decay = float(getattr(optimizer_cfg, 'weight_decay', 0.0))
        if hasattr(self, 'encoder_optimizer'):
            for param_group in self.encoder_optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.encoder_optimizer = torch.optim.Adam(
                self._hair_encoder_module().parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.encoder_optimizer,
            T_max=self.n_steps,
            eta_min=0.01 * lr,
        )

    def scheduler_step(self):
        if hasattr(self, 'encoder_scheduler'):
            self.encoder_scheduler.step()

    def save_model(self, state_dict, save_path, *, epoch_idx=None):
        checkpoint = {
            'dae_model': self._hair_encoder_module().state_dict(),
            'epoch_idx': epoch_idx,
        }
        if hasattr(self, 'encoder_optimizer'):
            checkpoint['optimizer'] = self.encoder_optimizer.state_dict()
        if hasattr(self, 'encoder_scheduler'):
            checkpoint['scheduler'] = self.encoder_scheduler.state_dict()
        torch.save(checkpoint, save_path)

    def load_model(self, resume, device='cuda'):
        checkpoint = torch.load(resume, map_location=device)
        if 'dae_model' in checkpoint:
            self._hair_encoder_module().load_state_dict(checkpoint['dae_model'], strict=False)
            if hasattr(self, 'encoder_optimizer') and checkpoint.get('optimizer') is not None:
                self.encoder_optimizer.load_state_dict(checkpoint['optimizer'])
            if hasattr(self, 'encoder_scheduler') and checkpoint.get('scheduler') is not None:
                self.encoder_scheduler.load_state_dict(checkpoint['scheduler'])
            return checkpoint.get('epoch_idx')
        self._hair_encoder_module().load_state_dict(checkpoint, strict=False)
        return None

    def step(self, batch, batch_idx, phase='train', epoch_idx=None):
        if phase == 'train':
            self.train()
            torch.set_grad_enabled(True)
        else:
            self.eval()
            torch.set_grad_enabled(False)
        self.synthetic_generator.eval()

        with torch.no_grad():
            bundle = self.synthetic_generator(batch)
            if bundle is None:
                zero_losses = {
                    'mask_loss': 0.0,
                    'orientation_loss': 0.0,
                    'depth_loss': 0.0,
                    'mask_iou': 0.0,
                    'latent_abs_mean': 0.0,
                    'skipped_synthesis': 1.0,
                    'total_loss': 0.0,
                }
                self.current_loss = zero_losses
                return {}
            corruption = self.corruptor(bundle, self.synthetic_generator, phase=phase)

        outputs = self.hair_encoder(corruption.corrupted_map)
        clean_map = bundle.clean_map
        clean_mask = clean_map[:, :1]
        clean_orient = clean_map[:, 1:3]
        clean_depth = clean_map[:, 3:4]

        mask_loss = F.binary_cross_entropy_with_logits(outputs['mask_logits'], clean_mask)
        orientation_loss = self._masked_cosine_angular_loss(outputs['orientation'], clean_orient, clean_mask)
        depth_loss = self._masked_l1_loss(outputs['depth'], clean_depth, clean_mask)
        mask_iou = self._mask_iou(outputs['mask'], clean_mask)
        latent_abs_mean = outputs['latent'].detach().abs().mean()

        total_loss = (
            mask_loss * float(self.config.train.loss_weights['mask_loss'])
            + orientation_loss * float(self.config.train.loss_weights['orientation_loss'])
            + depth_loss * float(self.config.train.loss_weights['depth_loss'])
        )

        if phase == 'train':
            self.optimizers_zero_grad()
            total_loss.backward()
            self._clip_gradients(clip_encoder=True, clip_generator=False)
            self.optimizers_step(step_encoder=True, step_fuse_generator=False)
            self.scheduler_step()

        losses = {
            'mask_loss': float(mask_loss.detach().item()),
            'orientation_loss': float(orientation_loss.detach().item()),
            'depth_loss': float(depth_loss.detach().item()),
            'mask_iou': float(mask_iou.detach().item()),
            'latent_abs_mean': float(latent_abs_mean.item()),
            'total_loss': float(total_loss.detach().item()),
        }
        self.current_loss = losses
        self.logging(batch_idx, losses, phase)

        return {
            'clean_map': clean_map.detach().cpu(),
            'corrupted_map': corruption.corrupted_map.detach().cpu(),
            'pre_render_map': corruption.pre_render_map.detach().cpu(),
            'reconstructed_map': outputs['reconstruction'].detach().cpu(),
            'mask_logits': outputs['mask_logits'].detach().cpu(),
            'mask_prob': outputs['mask'].detach().cpu(),
            'orientation': outputs['orientation'].detach().cpu(),
            'depth': outputs['depth'].detach().cpu(),
            'applied_families': corruption.applied_families,
        }

    def _masked_cosine_angular_loss(self, predicted, target, mask, eps=1e-6):
        if mask.shape[-2:] != predicted.shape[-2:]:
            mask = F.interpolate(mask.float(), size=predicted.shape[-2:], mode='bilinear', align_corners=False)
        valid_target = (torch.linalg.norm(target, dim=1, keepdim=True) > eps).float()
        valid_mask = mask.float() * valid_target
        denom = valid_mask.sum().clamp_min(eps)

        predicted_dir = F.normalize(predicted, dim=1, eps=eps)
        target_dir = F.normalize(target, dim=1, eps=eps)
        cosine = (predicted_dir * target_dir).sum(dim=1, keepdim=True)
        loss = (1.0 - cosine) * valid_mask
        return loss.sum() / denom

    def _masked_l1_loss(self, predicted, target, mask, eps=1e-6):
        if mask.shape[-2:] != predicted.shape[-2:]:
            mask = F.interpolate(mask.float(), size=predicted.shape[-2:], mode='bilinear', align_corners=False)
        denom = mask.float().sum().clamp_min(eps)
        return (torch.abs(predicted - target) * mask.float()).sum() / denom

    def _mask_iou(self, predicted_mask, target_mask, threshold=0.5, eps=1e-6):
        pred = (predicted_mask >= threshold).float()
        target = (target_mask >= threshold).float()
        intersection = (pred * target).sum()
        union = ((pred + target) > 0).float().sum().clamp_min(eps)
        return intersection / union

    def create_visualizations(self, batch, outputs):
        del batch
        return outputs

    def save_visualizations(self, outputs, save_path):
        stage_maps = (
            ('clean', outputs['clean_map']),
            ('pre_render', outputs['pre_render_map']),
            ('corrupted', outputs['corrupted_map']),
            ('reconstructed', outputs['reconstructed_map']),
        )
        save_packed_map_comparison(stage_maps, save_path)
