import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
from src.hair_synthesis_encoder import HairSynthesisEncoder
from src.smirk_encoder import SmirkEncoder
from src.smirk_generator import SmirkGenerator
from src.base_trainer import BaseHairTrainer
from src.helpers import (
    FLAMEHairStrandAttachment,
    HairTemplateManager,
    HairStrandRasterizer,
)
import src.utils.utils as utils
import src.utils.masking as masking_utils

class HairSynthesisTrainer(BaseHairTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        resume_path = getattr(self.config, 'resume', None)
        self.initialize_from_resume = bool(resume_path)
        step2_sparse_cfg = getattr(self.config.train, 'step2_sparse_color', None)
        self.step2_use_hair_mask_union = bool(
            getattr(
                step2_sparse_cfg,
                'use_hair_mask_union',
                getattr(self.config.train, 'step2_use_hair_mask_union', False),
            )
        )
        self.step2_fill_inverse_intersection_samples = bool(
            getattr(
                step2_sparse_cfg,
                'fill_inverse_intersection_samples',
                getattr(self.config.train, 'step2_fill_inverse_intersection_samples', False),
            )
        )

        ### If enable_fuse_generator, train both encoder and generator
        if self.config.arch.enable_fuse_generator:
            # input: strand (C=3), depth (C=1), sampled color image (C=3)
            gen_in_channel = 7 if self.config.arch.depth_branch else 6
            self.smirk_generator = SmirkGenerator(in_channels=gen_in_channel, out_channels=3, init_features=32, res_blocks=5)
            if (not self.initialize_from_resume) and getattr(self.config, 'load_generator', False):
                self._load_smirk_generator_weights(getattr(self.config, 'checkpoint_smirk', None))
            
        self.hair_encoder = HairSynthesisEncoder(
            img2strand_ckpt=None if self.initialize_from_resume else config.checkpoint_img2strand,
            img2depth_ckpt=None if self.initialize_from_resume else config.checkpoint_img2depth,
            config=self.config)

        self.smirk_face_encoder = SmirkEncoder(
            n_exp=self.config.arch.num_expression,
            n_shape=self.config.arch.num_shape,
        )
        self.freeze_smirk_encoder = getattr(self.config.train, 'freeze_smirk_encoder', True)
        if not self.initialize_from_resume:
            self._load_smirk_encoder_weights(getattr(self.config, 'checkpoint_smirk', None))
        if self.freeze_smirk_encoder:
            utils.freeze_module(self.smirk_face_encoder, 'smirk face encoder')
            self.smirk_face_encoder.eval()
        else:
            utils.unfreeze_module(self.smirk_face_encoder, 'smirk face encoder')

        self.flame = FLAME(
            n_exp=self.config.arch.num_expression,
            n_shape=self.config.arch.num_shape,
        )
        self.flame_renderer = Renderer(render_full_head=False)
        self.flame_renderer_full = Renderer(render_full_head=False)
        self.flame_renderer_full.image_size = getattr(
            self.config.arch,
            'hair_render_size',
            getattr(self.config.dataset, 'resolution', 512),
        )

        hair20k_cfg = getattr(self.config.dataset, 'Hair20k', None)
        template_dir_override = None
        blendshape_override = None
        aug_override = None
        load_roots_override = True
        if hair20k_cfg is not None:
            template_dir_override = getattr(hair20k_cfg, 'Hair20k_path', None)
            blendshape_override = getattr(hair20k_cfg, 'blenshape_path', None) or getattr(hair20k_cfg, 'blendshape_path', None)
            aug_override = getattr(hair20k_cfg, 'use_augmentation', None)
            load_roots_override = getattr(hair20k_cfg, 'load_roots', True)

        strand_basis_path = blendshape_override or getattr(self.config.arch, 'strand_basis_path', 'assets/blend-shapes/strands-blend-shapes.npz')
        self.load_template_roots = bool(load_roots_override)

        self.hair_attachment = FLAMEHairStrandAttachment(
            flame_model=self.flame,
            scale_to_perm=getattr(self.config.arch, 'perm_scale', 100.0),
            translation_to_perm=tuple(getattr(self.config.arch, 'perm_translation', (0.0, 0.0, 0.0))),
            rotation_to_perm_euler_deg=tuple(getattr(self.config.arch, 'perm_rotation_euler_deg', (0.0, 0.0, 0.0))),
            perm_head_mesh_path=getattr(self.config.arch, 'perm_head_mesh_path', None),
            scalp_bounds=getattr(self.config.arch, 'scalp_bounds', None),
            mask_threshold=getattr(self.config.arch, 'mask_threshold', 0.5),
            max_mask_samples=getattr(self.config.arch, 'max_mask_samples', None),
            strand_basis_path=strand_basis_path,
            enable_pre_render_culling=getattr(self.config.arch, 'enable_pre_render_culling', False),
        )
        self.hair_rasterizer = HairStrandRasterizer(
            image_size=getattr(self.config.arch, 'hair_render_size', getattr(self.config.dataset, 'resolution', 512)),
            background_color=(0.0, 0.0, 0.0),
            head_scale=getattr(self.config.arch, 'pre_render_head_scale', 0.96),
        )
        template_dir = template_dir_override or getattr(self.config.dataset, 'hair_template_dir', None)
        aug_cfg = self._make_aug_cfg(getattr(self.config.train, 'hair_template_augmentation', None), aug_override)

        self.hair_template_manager = HairTemplateManager(
            template_dir=template_dir,
            aug_cfg=aug_cfg,
            load_roots=self.load_template_roots,
        )
        self.latest_hair_templates = None
        self.return_low_freq_hair = getattr(self.config, 'return_low_freq_hair', False)
        debug_cfg = getattr(self.config.train, 'debug', None)
        self.debug_dump_hair = bool(getattr(debug_cfg, 'dump_hair', getattr(self.config.train, 'debug_dump_hair', False)))
        self.debug_dump_hit_faces = bool(getattr(debug_cfg, 'dump_hit_faces', getattr(self.config.train, 'debug_dump_hit_faces', False)))
        self.debug_save_hair_render = bool(getattr(debug_cfg, 'save_hair_render', getattr(self.config.train, 'debug_save_hair_render', False)))
        self.debug_save_hairstep_maps = bool(getattr(debug_cfg, 'save_hairstep_maps', False))
        self.debug_save_inverse_intersection_mask = bool(getattr(debug_cfg, 'save_inverse_intersection_mask', False))
        self.debug_render_flame_mesh = bool(getattr(debug_cfg, 'render_flame_mesh', getattr(self.config.train, 'debug_render_flame_mesh', False)))
        self.debug_break_step2 = bool(getattr(debug_cfg, 'break_step2', getattr(self.config.train, 'debug_break_step2', False)))
        self.force_closed_mouth = bool(getattr(self.config.train, 'force_closed_mouth', False))
        self.zero_flame_expression = bool(getattr(self.config.train, 'zero_flame_expression', False))
        self.step2_sparse_color_tangent_radius = float(
            getattr(
                step2_sparse_cfg,
                'tangent_radius',
                getattr(self.config.train, 'step2_sparse_color_tangent_radius', 18.0),
            )
        )
        self.step2_sparse_color_normal_radius = float(
            getattr(
                step2_sparse_cfg,
                'normal_radius',
                getattr(self.config.train, 'step2_sparse_color_normal_radius', 6.0),
            )
        )
        self.step2_sparse_color_fallback_radius = float(
            getattr(
                step2_sparse_cfg,
                'fallback_radius',
                getattr(self.config.train, 'step2_sparse_color_fallback_radius', 24.0),
            )
        )
        self.step2_sparse_color_candidate_pool = int(
            getattr(
                step2_sparse_cfg,
                'candidate_pool',
                getattr(self.config.train, 'step2_sparse_color_candidate_pool', 8),
            )
        )

        self.setup_losses()
        self.current_loss = None
        self.current_epoch_idx = None
        self.in_warmup_phase = False

    def _load_smirk_encoder_weights(self, checkpoint_path):
        if not checkpoint_path:
            raise ValueError("config.checkpoint_smirk must be provided to initialize the smirk encoder.")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint_smirk not found at {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location='cpu')
        encoder_state = {key.replace('smirk_encoder.', '', 1): value
                         for key, value in state_dict.items()
                         if key.startswith('smirk_encoder')}
        missing, unexpected = self.smirk_face_encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            print(f"[HairSynthesisTrainer] Missing smirk encoder keys: {missing}")
        if unexpected:
            print(f"[HairSynthesisTrainer] Unexpected smirk encoder keys: {unexpected}")

    def _load_smirk_generator_weights(self, checkpoint_path):
        if not checkpoint_path:
            raise ValueError("config.checkpoint_smirk must be provided to load the smirk generator.")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint_smirk not found at {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location='cpu')
        generator_state = {key.replace('smirk_generator.', '', 1): value
                           for key, value in state_dict.items()
                           if key.startswith('smirk_generator')}
        if not generator_state:
            print("[HairSynthesisTrainer] No smirk_generator weights found in checkpoint.")
            return
        missing, unexpected = self.smirk_generator.load_state_dict(generator_state, strict=False)
        if missing:
            print(f"[HairSynthesisTrainer] Missing smirk generator keys: {missing}")
        if unexpected:
            print(f"[HairSynthesisTrainer] Unexpected smirk generator keys: {unexpected}")

    def load_model(self, resume, load_fuse_generator=True, load_encoder=True, device='cuda'):
        loaded_state_dict = torch.load(resume, map_location=device)

        print(f'Loading checkpoint from {resume}, load_encoder={load_encoder}, load_fuse_generator={load_fuse_generator}')

        filtered_state_dict = {}
        for key, value in loaded_state_dict.items():
            if load_encoder and (key.startswith('hair_encoder') or key.startswith('smirk_face_encoder')):
                filtered_state_dict[key] = value
            if load_fuse_generator and key.startswith('smirk_generator'):
                filtered_state_dict[key] = value

        self.load_state_dict(filtered_state_dict, strict=False)

    def save_model(self, state_dict, save_path):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('hair_encoder') or key.startswith('smirk_face_encoder') or key.startswith('smirk_generator'):
                new_state_dict[key] = value

        torch.save(new_state_dict, save_path)

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

    def _apply_flame_debug_overrides(self, flame_params):
        if not (self.force_closed_mouth or self.zero_flame_expression):
            return flame_params

        overridden = dict(flame_params)

        if self.zero_flame_expression:
            expression_params = overridden.get('expression_params')
            if expression_params is not None:
                overridden['expression_params'] = torch.zeros_like(expression_params)

        if self.force_closed_mouth:
            jaw_params = overridden.get('jaw_params')
            if jaw_params is not None:
                jaw_params = jaw_params.clone()
                jaw_params[..., 0] = 0.0
                overridden['jaw_params'] = jaw_params

        return overridden

    def _get_encoder_masks(self, batch):
        hairmask = batch.get('encoder_hairmask', batch['hairmask'])
        bodymask = batch.get('encoder_bodymask', batch['bodymask'])
        return hairmask.float(), bodymask.float()

    def step1(self, batch):
        B, C, H, W = batch['img'].shape
        encoder_hairmask, encoder_bodymask = self._get_encoder_masks(batch)

        encoder_output = self.hair_encoder(batch['img'], encoder_hairmask, encoder_bodymask)
        
        with torch.no_grad():       # base_encoder = hair_encoder -> later used to regularize the weights
            base_output = self.base_encoder(batch['img'], encoder_hairmask, encoder_bodymask)

        strand_output = encoder_output['strand_params']     # (B, 3, H, W)

        if self.config.arch.depth_branch:
            depth_output = encoder_output['depth_params']       # (B, 1, H, W)
            strand_depth_output = torch.cat([strand_output, depth_output], dim=1)   # (B, 4, H, W)
        else:
            strand_depth_output = strand_output
    
        # ---------------- losses ---------------- #
        losses = {}

        img = batch['img']  # (B, C, H, W)

        #  ---------------- regularization losses ---------------- # 
        # (avoid the weights to change too much)
        if self.config.train.use_base_model_for_regularization:
            with torch.no_grad():
                base_output = self.base_encoder(batch['img'], encoder_hairmask, encoder_bodymask)
        else:       # TODO
            raise NotImplementedError("Initialize base output with zero. Check dimensions to complete this.")
            base_output = {key[0]: torch.zeros(B, key[1]).to(self.config.device) for key in zip(['expression_params', 'shape_params', 'jaw_params'], [self.config.arch.num_expression, self.config.arch.num_shape, 3])}

        # L2 loss on weights for regularization
        losses['strand_regularization'] = torch.mean((encoder_output['strand_params'] - base_output['strand_params'])**2)
        if self.config.arch.depth_branch:
            losses['depth_regularization'] = torch.mean((encoder_output['depth_params'] - base_output['depth_params'])**2)

        if self.config.arch.enable_fuse_generator:
            masks = batch['hairmask']   # (B, 1, H, W)

            # mask out hair and add random points inside the hair
            tmask_ratio = self.config.train.mask_ratio # ratio of number of points to sample

            # select pixel points from hair mask
            hair_sampled_points = masking_utils.mask_uniform_hair(masks, tmask_ratio)   # (B, N, 2), N = number of points
            extra_points = masking_utils.transfer_pixels(img, hair_sampled_points, hair_sampled_points)         # (B, 3, H, W) - mask in the original image which point will be used
            # extra_points = torch.zeros_like(img)  #debug

            # completed masked img - mask out the hair and add the extra points
            masks_rest = 1 - masks      # Take the rest
            masked_img = masking_utils.masking(img, masks_rest, extra_points, self.config.train.mask_dilation_radius)    # (B, 3, H, W)

            # import imageio.v2 as imageio
            # img_tmp = img[0].detach().cpu().numpy().transpose(1, 2, 0)
            # img_tmp = (img_tmp * 255).astype(np.uint8)
            # mask_tmp = masked_img[0].detach().cpu().numpy().transpose(1, 2, 0)
            # mask_tmp = (mask_tmp * 255).astype(np.uint8)
            # print(np.min(mask_tmp), np.max(mask_tmp))
            # imageio.imwrite("results/debug/img.png", img_tmp)
            # imageio.imwrite("results/debug/mask_xpts.png", mask_tmp)

            reconstructed_img = self.smirk_generator(torch.cat([strand_depth_output, masked_img], dim=1))

            # reconstruction loss
            reconstruction_loss = F.l1_loss(reconstructed_img, img, reduction='none')

            # for visualization
            loss_img = reconstruction_loss.mean(dim=1, keepdim=True)
            losses['reconstruction_loss'] = reconstruction_loss.mean()

            # perceptual loss
            losses['perceptual_vgg_loss'] = self.vgg_loss(reconstructed_img, img)

        else:
            losses['reconstruction_loss'] = 0
            losses['perceptual_vgg_loss'] = 0

        # Local smooth loss
        if self.config.train.loss_weights['strand_local_regularization'] > 0:
            eps = 1e-10
            strand_params = encoder_output['strand_params']     # (B, 3, H, W)

            orient = strand_params[:, 1:3, :, :]          # (B, 2, H, W), first channel is mask
            hairmask = batch['hairmask']   # (B, 1, H, W)
            smooth_kernel = self.config.train.loss_weights.strand_local_kernel
            smooth_stride = self.config.train.loss_weights.strand_local_stride
            smooth_pad = self.config.train.loss_weights.strand_local_padding

            if isinstance(smooth_kernel, int):
                kernel_h = kernel_w = int(smooth_kernel)
            else:
                kernel_h, kernel_w = tuple(int(v) for v in smooth_kernel)

            if isinstance(smooth_stride, int):
                stride_h = stride_w = int(smooth_stride)
            else:
                stride_h, stride_w = tuple(int(v) for v in smooth_stride)

            if isinstance(smooth_pad, int):
                pad_h = pad_w = int(smooth_pad)
            else:
                pad_h, pad_w = tuple(int(v) for v in smooth_pad)

            orient_h, orient_w = orient.shape[-2:]
            pooled_h = (orient_h + 2 * pad_h - kernel_h) // stride_h + 1
            pooled_w = (orient_w + 2 * pad_w - kernel_w) // stride_w + 1
            if (pooled_h, pooled_w) != (orient_h, orient_w):
                if stride_h == 1 and stride_w == 1 and kernel_h % 2 == 1 and kernel_w % 2 == 1:
                    pad_h = kernel_h // 2
                    pad_w = kernel_w // 2
                else:
                    raise ValueError(
                        "strand_local_regularization requires pooling params that preserve the "
                        f"orientation map size. Got kernel={smooth_kernel}, stride={smooth_stride}, "
                        f"padding={smooth_pad}, which yields {(pooled_h, pooled_w)} for input "
                        f"{(orient_h, orient_w)}."
                    )

            pool_area = float(kernel_h * kernel_w)

            # Get neighbors: use hairmask to compute only on hair region
            local_sum = F.avg_pool2d(
                orient * hairmask,
                kernel_size=(kernel_h, kernel_w),
                stride=(stride_h, stride_w),
                padding=(pad_h, pad_w),
            ) * pool_area
            local_count = F.avg_pool2d(
                hairmask,
                kernel_size=(kernel_h, kernel_w),
                stride=(stride_h, stride_w),
                padding=(pad_h, pad_w),
            ) * pool_area + eps
            
            local_mean = local_sum / local_count
            local_norm = torch.clamp(torch.linalg.norm(local_mean, dim=1, keepdim=True), min=eps)
            local_dir = local_mean / local_norm     # (B, 2, H, W)
            # print(local_norm.min(), local_norm.max())

            orient_norm = torch.clamp(torch.linalg.norm(orient, dim=1, keepdim=True), min=eps)
            orient_dir = orient / orient_norm       # (B, 2, H, W)
            # print(orient_norm.min(), orient_norm.max())

            # Use cosine similarity, vectors are already normalize to have magnitude 1
            # cos = 1 matches, cos = 0 perpendicular, cos = -1 opposite
            local_similarity = (orient_dir * local_dir).sum(dim=1, keepdim=True)    # (B, 1, H, W)

            # minimize cos/local_similarity
            local_loss = (1 - local_similarity) * hairmask

            # average over hairmask
            losses['strand_local_regularization'] = local_loss.sum() / (hairmask.sum() + 1e-10)
            # print(losses['strand_local_regularization'])
        else:
            losses['strand_local_regularization'] = 0

        strand_along_weight = float(
            getattr(self.config.train.loss_weights, 'strand_along_regularization', 0.0)
        )
        if strand_along_weight > 0:
            losses['strand_along_regularization'] = self._along_strand_consistency_loss(
                orient=encoder_output['strand_params'][:, 1:3, :, :],
                hairmask=batch['hairmask'],
                step_pixels=float(
                    getattr(self.config.train.loss_weights, 'strand_along_step', 1.5)
                ),
            )
        else:
            losses['strand_along_regularization'] = 0

        fuse_generator_losses = (losses['perceptual_vgg_loss'] * self.config.train.loss_weights['perceptual_vgg_loss'] + 
                                losses['reconstruction_loss'] * self.config.train.loss_weights['reconstruction_loss'] + 
                                losses['strand_regularization'] * self.config.train.loss_weights['strand_regularization'] + 
                                losses['strand_local_regularization'] * self.config.train.loss_weights['strand_local_regularization'] +
                                losses['strand_along_regularization'] * strand_along_weight
                                )

        if self.config.arch.depth_branch:
            fuse_generator_losses += losses['depth_regularization'] * self.config.train.loss_weights['depth_regularization']
               
        loss_first_path = (
            (fuse_generator_losses if self.config.arch.enable_fuse_generator else 0)
        )

        losses = {
            key: self._to_scalar(self._get_logged_weighted_loss(key, value))
            for key, value in losses.items()
        }
        losses['loss_first_path'] = self._to_scalar(loss_first_path)

        # ---------------- create a dictionary of outputs to visualize ---------------- #
        outputs = {}
        outputs['img'] = img
        outputs['strand'] = strand_output
        if self.config.arch.depth_branch:
            outputs['depth'] = depth_output
        
        if self.config.arch.enable_fuse_generator:
            outputs['loss_img'] = loss_img
            outputs['reconstructed_img'] = reconstructed_img
            outputs['masked_1st_path'] = masked_img

        for key in outputs.keys():
            outputs[key] = outputs[key].detach().cpu()

        outputs['encoder_output'] = encoder_output

        return outputs, losses, loss_first_path, encoder_output

    # ---------------- second path ---------------- #
    def step2(self, encoder_output, batch, batch_idx, phase='train'):
        with torch.no_grad():
            smirk_img = batch.get('smirk_img', batch['img'])
            flame_params = self.smirk_face_encoder(smirk_img)
            flame_params = self._apply_flame_debug_overrides(flame_params)

        hair_templates = None
        if self.hair_template_manager is not None:
            hair_templates = self.hair_template_manager.sample(batch['img'].shape[0], batch['img'].device)
            if hair_templates is not None:
                self.latest_hair_templates = hair_templates

        attachment_out = None
        if hair_templates is not None:
            try:
                attachment_out = self.hair_attachment(
                    flame_params=flame_params,
                    hair_textures=hair_templates['texture'],
                    roots=hair_templates.get('roots') if self.load_template_roots else None,
                    scalp_masks=hair_templates['mask'],
                    return_low_frequency=self.return_low_freq_hair,
                    return_flame_mesh=self.debug_dump_hair,
                    debug_dump=self.debug_dump_hair,
                    debug_dump_hit_faces=self.debug_dump_hit_faces,
                )
            except Exception as exc:
                print(f"[HairSynthesisTrainer] Hair attachment failed: {exc}")
                attachment_out = None

        outputs = {f'flame_{key}': value.detach().cpu() for key, value in flame_params.items()}
        if hair_templates is not None:
            outputs['hair_template_paths'] = hair_templates['paths']
        crop_cam_params = flame_params.get(
            'cam',
            torch.zeros(
                flame_params['pose_params'].shape[0],
                3,
                device=flame_params['pose_params'].device,
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
        outputs['flame_cam_crop'] = crop_cam_params.detach().cpu()
        outputs['flame_cam_full'] = cam_params.detach().cpu()
        if 'smirk_crop_valid' in batch:
            outputs['smirk_crop_valid'] = batch['smirk_crop_valid'].detach().cpu()
        hair_render = None
        sparse_hair_color_map = None
        if attachment_out is not None:
            outputs['hair_strands_full'] = attachment_out.full_resolution.detach().cpu()
            if attachment_out.low_frequency is not None:
                outputs['hair_strands_low'] = attachment_out.low_frequency.detach().cpu()
            meta = {}
            for key, value in attachment_out.metadata.items():
                if torch.is_tensor(value):
                    meta[key] = value.detach().cpu()
                else:
                    meta[key] = value
            outputs['hair_attachment_meta'] = meta
            try:
                raster_out = self.hair_rasterizer.forward(
                    flame_vertices=attachment_out.metadata['flame_vertices'],
                    flame_faces=attachment_out.metadata['flame_faces'],
                    strands=attachment_out.full_resolution,
                    cam_params=cam_params,
                )
                hair_render = raster_out
                outputs['hair_render_image'] = raster_out.image.detach().cpu()
                outputs['hair_render_mask'] = raster_out.visibility_mask.detach().cpu()
                outputs['hair_render_depth'] = raster_out.depth.detach().cpu()
                inverse_intersection_mask = self._compute_inverse_intersection_mask(
                    batch['hairmask'],
                    raster_out.visibility_mask,
                )
                sparse_hair_color_map = self._build_sparse_hair_color_map(
                    batch['img'],
                    batch['hairmask'],
                    raster_out.image,
                    raster_out.visibility_mask,
                )
                outputs['hair_render_inverse_intersection_mask'] = inverse_intersection_mask.detach().cpu()
                outputs['hair_render_sparse_color_map'] = sparse_hair_color_map.detach().cpu()
            except Exception as exc:
                print(f"[HairSynthesisTrainer] Hair rasterization failed: {exc}")

        flame_render = None
        flame_render_crop = None
        should_visualize = (
            self.config.train.visualize_phase.get(phase, False)
            and (batch_idx % self.config.train.visualize_every == 0)
        )
        if self.debug_render_flame_mesh or should_visualize:
            flame_vertices = None
            if attachment_out is not None:
                flame_vertices = attachment_out.metadata.get('flame_vertices')
            if flame_vertices is None:
                with torch.no_grad():
                    flame_vertices = self.flame.forward(flame_params)['vertices']
            try:
                flame_render_crop = self.flame_renderer.forward(flame_vertices, crop_cam_params)
                flame_render = self.flame_renderer_full.forward(flame_vertices, cam_params)
                outputs['flame_render_crop_image'] = flame_render_crop['rendered_img'].detach().cpu()
                outputs['flame_render_image'] = flame_render['rendered_img'].detach().cpu()
            except Exception as exc:
                print(f"[HairSynthesisTrainer] FLAME mesh rendering failed: {exc}")

        losses = {}
        loss_second_path = torch.zeros(
            1,
            device=batch['img'].device,
            dtype=batch['img'].dtype,
            requires_grad=(phase == 'train'),
        )

        if hair_render is not None and sparse_hair_color_map is not None and self.config.arch.enable_fuse_generator:
            injected_render = self._generator_hair_input(hair_render.image).detach()
            generator_input = torch.cat([injected_render, sparse_hair_color_map.detach()], dim=1)
            reconstructed_img_2nd_path = self.smirk_generator(generator_input)
            if self.config.train.freeze_generator_in_second_path:
                reconstructed_img_2nd_path = reconstructed_img_2nd_path.detach()

            cycle_hair_mask = hair_render.visibility_mask.detach()
            cycle_body_mask = batch.get('bodymask')
            if cycle_body_mask is None:
                cycle_body_mask = torch.zeros_like(cycle_hair_mask)
            else:
                cycle_body_mask = cycle_body_mask.float()
                if cycle_body_mask.shape[-2:] != cycle_hair_mask.shape[-2:]:
                    cycle_body_mask = F.interpolate(
                        cycle_body_mask,
                        size=cycle_hair_mask.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )

            reencoded_hair = self.hair_encoder(
                reconstructed_img_2nd_path,
                cycle_hair_mask,
                cycle_body_mask,
            )

            cycle_orient_loss = self._masked_cosine_angular_loss(
                reencoded_hair['strand_params'][:, 1:3],
                injected_render[:, 1:3],
                cycle_hair_mask,
            )

            cycle_depth_loss = torch.zeros_like(cycle_orient_loss)
            if self.config.arch.depth_branch and 'depth_params' in reencoded_hair:
                cycle_depth_loss = self._masked_l1_loss(
                    reencoded_hair['depth_params'],
                    injected_render[:, 3:4],
                    cycle_hair_mask,
                )

            weighted_cycle_loss = (
                cycle_orient_loss * self.config.train.loss_weights['cycle_orient_loss']
                + cycle_depth_loss * self.config.train.loss_weights['cycle_depth_loss']
            )
            loss_second_path = weighted_cycle_loss * self.config.train.loss_weights['cycle_loss']

            losses['cycle_orient_loss'] = self._to_scalar(
                self._get_logged_weighted_loss('cycle_orient_loss', cycle_orient_loss)
            )
            losses['cycle_depth_loss'] = self._to_scalar(
                self._get_logged_weighted_loss('cycle_depth_loss', cycle_depth_loss)
            )
            losses['cycle_loss'] = self._to_scalar(
                self._get_logged_weighted_loss('cycle_loss', weighted_cycle_loss)
            )
            losses['loss_second_path'] = self._to_scalar(loss_second_path)

            if batch_idx % self.config.train.visualize_every == 0:
                outputs['hair_cycle_reconstruction'] = reconstructed_img_2nd_path.detach().cpu()
                outputs['hair_cycle_strand'] = reencoded_hair['strand_params'].detach().cpu()
                if 'depth_params' in reencoded_hair:
                    outputs['hair_cycle_depth'] = reencoded_hair['depth_params'].detach().cpu()

        if self.debug_save_hair_render and (hair_render is not None or flame_render is not None):
            self._save_debug_hair_render(
                batch['img'][0],
                hair_render.image[0] if hair_render is not None else None,
                batch_idx,
                phase,
                flame_img=flame_render['rendered_img'][0] if flame_render is not None else None,
                flame_crop_img=flame_render_crop['rendered_img'][0] if flame_render_crop is not None else None,
                smirk_img=smirk_img[0],
                sparse_hair_img=outputs.get('hair_render_sparse_color_map', None)[0] if 'hair_render_sparse_color_map' in outputs else None,
            )
        if self.debug_save_hairstep_maps and hair_render is not None:
            self._save_debug_hairstep_maps(
                hair_render.image[0],
                batch_idx,
                phase,
            )
        if self.debug_save_inverse_intersection_mask and 'hair_render_inverse_intersection_mask' in outputs:
            self._save_debug_inverse_intersection_mask(
                outputs['hair_render_inverse_intersection_mask'][0],
                batch_idx,
                phase,
            )

        if self.debug_break_step2:
            import pdb; pdb.set_trace()
        return outputs, losses, loss_second_path

    def _compute_inverse_intersection_mask(self, source_mask, render_mask):
        """Return source hair matte confidence not covered by the rendered hair mask."""
        source_mask = source_mask.float().clamp(0.0, 1.0)
        render_mask = render_mask.float().clamp(0.0, 1.0)
        if source_mask.shape[-2:] != render_mask.shape[-2:]:
            source_mask = F.interpolate(
                source_mask,
                size=render_mask.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).clamp(0.0, 1.0)

        return (source_mask * (1.0 - render_mask)).clamp(0.0, 1.0)

    def _sample_nearest_valid_source_coords(self, valid_source_mask, target_xy):
        """Map each target xy to nearby source pixels, weighted by source confidence."""
        return self._sample_global_source_coords(valid_source_mask.float(), target_xy)

    def _masked_cosine_angular_loss(self, predicted, target, mask, eps=1e-6):
        """Compare 2D orientation fields with a cosine-based angular loss."""
        if mask.shape[-2:] != predicted.shape[-2:]:
            mask = F.interpolate(
                mask.float(),
                size=predicted.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        valid_target = (torch.linalg.norm(target, dim=1, keepdim=True) > eps).float()
        valid_mask = mask.float() * valid_target
        denom = valid_mask.sum().clamp_min(eps)

        predicted_dir = F.normalize(predicted, dim=1, eps=eps)
        target_dir = F.normalize(target, dim=1, eps=eps)
        cosine = (predicted_dir * target_dir).sum(dim=1, keepdim=True)
        loss = (1.0 - cosine) * valid_mask
        return loss.sum() / denom

    def _get_logged_weighted_loss(self, loss_name, loss_value):
        if loss_name == 'cycle_orient_loss':
            weight = (
                float(self.config.train.loss_weights['cycle_orient_loss'])
                * float(self.config.train.loss_weights['cycle_loss'])
            )
            return loss_value * weight
        if loss_name == 'cycle_depth_loss':
            weight = (
                float(self.config.train.loss_weights['cycle_depth_loss'])
                * float(self.config.train.loss_weights['cycle_loss'])
            )
            return loss_value * weight
        if loss_name == 'cycle_loss':
            return loss_value * float(self.config.train.loss_weights['cycle_loss'])

        loss_weights = getattr(self.config.train, 'loss_weights', None)
        if loss_weights is None:
            return loss_value

        if hasattr(loss_weights, loss_name):
            return loss_value * float(getattr(loss_weights, loss_name))

        try:
            if loss_name in loss_weights:
                return loss_value * float(loss_weights[loss_name])
        except Exception:
            pass

        return loss_value

    def _to_scalar(self, value):
        if isinstance(value, torch.Tensor):
            return value.item()
        return float(value)

    def _generator_hair_input(self, hair_features):
        if self.config.arch.depth_branch:
            return hair_features
        return hair_features[:, :3]

    def _warmup_num_epochs(self):
        warmup_cfg = getattr(self.config.train, 'warmup', None)
        if warmup_cfg is not None:
            if not bool(getattr(warmup_cfg, 'enabled', True)):
                return 0
            epochs = getattr(warmup_cfg, 'num_epochs', getattr(warmup_cfg, 'epochs', 0))
        else:
            epochs = getattr(self.config.train, 'warmup_epochs', 0)
        return max(0, int(epochs))

    def is_warmup_epoch(self, epoch_idx=None):
        if epoch_idx is None:
            epoch_idx = self.current_epoch_idx
        if epoch_idx is None:
            return False
        return int(epoch_idx) < self._warmup_num_epochs()

    def set_freeze_status(self, config, batch_idx, epoch_idx):
        self.current_epoch_idx = int(epoch_idx)
        self.in_warmup_phase = self.is_warmup_epoch(epoch_idx)

        self.config.train.freeze_encoder_in_second_path = False
        self.config.train.freeze_generator_in_second_path = False
        if self.in_warmup_phase:
            return

        decision_idx_second_path = batch_idx
        self.config.train.freeze_encoder_in_second_path = decision_idx_second_path % 2 == 0
        self.config.train.freeze_generator_in_second_path = decision_idx_second_path % 2 == 1

    def _along_strand_consistency_loss(self, orient, hairmask, step_pixels=1.5, eps=1e-6):
        """Penalize sign flips by comparing each orientation to samples along its own tangent."""
        if hairmask.shape[-2:] != orient.shape[-2:]:
            hairmask = F.interpolate(
                hairmask.float(),
                size=orient.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        hairmask = hairmask.float()
        orient_dir = F.normalize(orient, dim=1, eps=eps)
        batch_size, _, height, width = orient_dir.shape
        device = orient_dir.device
        dtype = orient_dir.dtype

        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)

        step_x = 2.0 * float(step_pixels) / max(float(width - 1), 1.0)
        step_y = 2.0 * float(step_pixels) / max(float(height - 1), 1.0)
        offset_grid = torch.stack(
            [orient_dir[:, 0] * step_x, orient_dir[:, 1] * step_y],
            dim=-1,
        )

        grid_forward = base_grid + offset_grid
        grid_backward = base_grid - offset_grid

        forward_orient = F.grid_sample(
            orient_dir,
            grid_forward,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        backward_orient = F.grid_sample(
            orient_dir,
            grid_backward,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )

        forward_mask = F.grid_sample(
            hairmask,
            grid_forward,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        ).clamp(0.0, 1.0)
        backward_mask = F.grid_sample(
            hairmask,
            grid_backward,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        ).clamp(0.0, 1.0)

        forward_valid = (torch.linalg.norm(forward_orient, dim=1, keepdim=True) > eps).float()
        backward_valid = (torch.linalg.norm(backward_orient, dim=1, keepdim=True) > eps).float()

        forward_dir = F.normalize(forward_orient, dim=1, eps=eps)
        backward_dir = F.normalize(backward_orient, dim=1, eps=eps)

        center_mask = hairmask.clamp(0.0, 1.0)
        loss_mask_forward = center_mask * forward_mask * forward_valid
        loss_mask_backward = center_mask * backward_mask * backward_valid

        forward_cosine = (orient_dir * forward_dir).sum(dim=1, keepdim=True)
        backward_cosine = (orient_dir * backward_dir).sum(dim=1, keepdim=True)

        forward_loss = (1.0 - forward_cosine) * loss_mask_forward
        backward_loss = (1.0 - backward_cosine) * loss_mask_backward

        denom = (loss_mask_forward.sum() + loss_mask_backward.sum()).clamp_min(eps)
        return (forward_loss.sum() + backward_loss.sum()) / denom

    def _masked_l1_loss(self, predicted, target, mask, eps=1e-6):
        """Average L1 error using a supervision matte as confidence."""
        if mask.shape[-2:] != predicted.shape[-2:]:
            mask = F.interpolate(
                mask.float(),
                size=predicted.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        valid_mask = mask.float()
        denom = valid_mask.sum().clamp_min(eps)
        loss = torch.abs(predicted - target) * valid_mask
        return loss.sum() / denom

    def _sample_patch_candidates(self, candidate_scores, topk):
        """Randomly choose one candidate from the top-k confidence-weighted patch samples."""
        num_targets, num_candidates = candidate_scores.shape
        if num_targets == 0 or num_candidates == 0:
            device = candidate_scores.device
            return (
                torch.zeros((num_targets,), dtype=torch.long, device=device),
                torch.zeros((num_targets,), dtype=torch.bool, device=device),
            )

        k = max(1, min(int(topk), num_candidates))
        scores = candidate_scores.clamp_min(0.0)
        top_vals, top_idx = torch.topk(scores, k=k, dim=1, largest=True)
        found = top_vals.sum(dim=1) > 0

        chosen = torch.zeros((num_targets,), dtype=torch.long, device=candidate_scores.device)
        if found.any():
            probs = top_vals[found] / top_vals[found].sum(dim=1, keepdim=True).clamp_min(1e-8)
            rand_pos = torch.multinomial(probs, 1).squeeze(1)
            chosen[found] = top_idx[found, rand_pos]
        return chosen, found

    def _sample_global_source_coords(self, source_weight, target_xy):
        """Fallback to nearby source pixels, weighted by matte confidence and distance."""
        source_weight = source_weight.float().clamp_min(0.0)
        device = source_weight.device
        num_targets = target_xy.shape[0]
        if num_targets == 0:
            return torch.zeros((0, 2), dtype=torch.long, device=device)

        src_y, src_x = torch.nonzero(source_weight, as_tuple=True)
        if src_x.numel() == 0:
            return torch.zeros((num_targets, 2), dtype=torch.long, device=device)

        src_xy = torch.stack([src_x, src_y], dim=1).float()
        src_weight = source_weight[src_y, src_x].float()
        max_source_candidates = 8192
        if src_xy.shape[0] > max_source_candidates:
            keep = torch.multinomial(src_weight, max_source_candidates, replacement=False)
            src_xy = src_xy[keep]
            src_weight = src_weight[keep]
        delta = src_xy.unsqueeze(0) - target_xy.float().unsqueeze(1)
        distance = (delta ** 2).sum(dim=-1)
        scores = src_weight.unsqueeze(0) / (distance + 1.0)
        topk = max(1, min(int(self.step2_sparse_color_candidate_pool), src_xy.shape[0]))
        top_vals, top_idx = torch.topk(scores, k=topk, dim=1, largest=True)
        probs = top_vals / top_vals.sum(dim=1, keepdim=True).clamp_min(1e-8)
        rand_pos = torch.multinomial(probs, 1).squeeze(1)
        chosen_src = top_idx[torch.arange(num_targets, device=device), rand_pos]
        return src_xy[chosen_src].long()

    def _build_sparse_hair_color_map(self, img, hair_mask, render_image, render_mask):
        """Create a masked image whose sparse hair hints live only inside the rendered hair silhouette."""
        render_size = render_mask.shape[-2:]
        if img.shape[-2:] != render_size:
            img = F.interpolate(img, size=render_size, mode='bilinear', align_corners=False)
        if hair_mask.shape[-2:] != render_size:
            hair_mask = F.interpolate(
                hair_mask.float(),
                size=render_size,
                mode='bilinear',
                align_corners=False,
            )

        hair_mask = hair_mask.float().clamp(0.0, 1.0)
        render_mask = render_mask.float().clamp(0.0, 1.0)
        inverse_intersection_mask = self._compute_inverse_intersection_mask(hair_mask, render_mask)
        render_dxdy = render_image[:, 1:3]

        batch_size, _, height, width = img.shape
        sparse_points = torch.zeros_like(img)
        base_num_targets = int(self.config.train.mask_ratio * height * width)
        if base_num_targets <= 0:
            return masking_utils.masking(
                img,
                1.0 - hair_mask,
                sparse_points,
                self.config.train.mask_dilation_radius,
                rendered_mask=render_mask,
                extra_noise=False,
                random_mask=0.0,
            )

        tangent_radius = max(self.step2_sparse_color_tangent_radius, 1.0)
        normal_radius = max(self.step2_sparse_color_normal_radius, 1.0)
        fallback_radius = max(self.step2_sparse_color_fallback_radius, max(tangent_radius, normal_radius))
        candidate_pool = max(1, self.step2_sparse_color_candidate_pool)

        patch_radius = int(np.ceil(fallback_radius))
        offsets = torch.arange(-patch_radius, patch_radius + 1, device=img.device)
        off_y, off_x = torch.meshgrid(offsets, offsets, indexing='ij')
        off_x = off_x.reshape(1, -1).float()
        off_y = off_y.reshape(1, -1).float()
        radial_distance = off_x.square() + off_y.square()

        for batch_idx in range(batch_size):
            source_weight = hair_mask[batch_idx, 0].clamp(0.0, 1.0)
            render_weight = render_mask[batch_idx, 0].clamp(0.0, 1.0)
            inverse_weight = inverse_intersection_mask[batch_idx, 0].clamp(0.0, 1.0)
            if self.step2_use_hair_mask_union:
                target_weight = torch.maximum(source_weight, render_weight)
            else:
                target_weight = render_weight

            source_mass = source_weight.sum()
            target_mass = target_weight.sum()
            if float(source_mass.item()) <= 1e-8 or float(target_mass.item()) <= 1e-8:
                continue

            scaled_target_count = int(round(base_num_targets * float((target_mass / source_mass).item())))
            if base_num_targets > 0 and scaled_target_count == 0:
                scaled_target_count = 1

            sample_count = max(0, scaled_target_count)
            if sample_count <= 0:
                continue

            target_flat = target_weight.reshape(-1)
            target_idx = torch.multinomial(target_flat, sample_count, replacement=True)
            target_y = target_idx // width
            target_x = target_idx % width

            inverse_probability = (
                inverse_weight[target_y, target_x]
                / target_weight[target_y, target_x].clamp_min(1e-8)
            ).clamp(0.0, 1.0)
            inverse_samples = torch.rand_like(inverse_probability) < inverse_probability

            regular_rows = torch.where(~inverse_samples)[0]
            if regular_rows.numel() > 0:
                regular_target_x = target_x[regular_rows]
                regular_target_y = target_y[regular_rows]

                tangent = render_dxdy[batch_idx, :, regular_target_y, regular_target_x].permute(1, 0)
                tangent_norm = torch.linalg.norm(tangent, dim=-1, keepdim=True)
                tangent_dir = torch.zeros_like(tangent)
                valid_tangent = tangent_norm.squeeze(-1) > 1e-6
                tangent_dir[valid_tangent] = tangent[valid_tangent] / tangent_norm[valid_tangent]
                tangent_dir[~valid_tangent, 0] = 1.0
                normal_dir = torch.stack([-tangent_dir[:, 1], tangent_dir[:, 0]], dim=-1)

                cand_x = regular_target_x.unsqueeze(1).float() + off_x
                cand_y = regular_target_y.unsqueeze(1).float() + off_y
                valid_patch = (cand_x >= 0) & (cand_x < width) & (cand_y >= 0) & (cand_y < height)
                cand_x_clamped = cand_x.clamp(0, width - 1).long()
                cand_y_clamped = cand_y.clamp(0, height - 1).long()

                source_in_patch = source_weight[cand_y_clamped, cand_x_clamped] * valid_patch.float()
                tangential_offset = off_x * tangent_dir[:, :1] + off_y * tangent_dir[:, 1:]
                normal_offset = off_x * normal_dir[:, :1] + off_y * normal_dir[:, 1:]

                oriented_distance = (
                    (tangential_offset / tangent_radius) ** 2
                    + (normal_offset / normal_radius) ** 2
                )
                oriented_scores = source_in_patch * torch.exp(-0.5 * oriented_distance)

                selected_idx, found = self._sample_patch_candidates(
                    oriented_scores,
                    candidate_pool,
                )

                if (~found).any():
                    missing_rows = torch.where(~found)[0]
                    fallback_distance = radial_distance.expand(missing_rows.numel(), -1)
                    fallback_scores = source_in_patch[missing_rows] * torch.exp(
                        -0.5 * fallback_distance / (fallback_radius ** 2)
                    )
                    fallback_idx, fallback_found = self._sample_patch_candidates(
                        fallback_scores,
                        candidate_pool,
                    )
                    selected_idx[missing_rows[fallback_found]] = fallback_idx[fallback_found]
                    found[missing_rows[fallback_found]] = True

                selected_src_x = torch.zeros_like(regular_target_x)
                selected_src_y = torch.zeros_like(regular_target_y)
                if found.any():
                    found_rows = torch.where(found)[0]
                    chosen_cols = selected_idx[found]
                    selected_src_x[found_rows] = cand_x_clamped[found_rows, chosen_cols]
                    selected_src_y[found_rows] = cand_y_clamped[found_rows, chosen_cols]

                if (~found).any():
                    missing_rows = torch.where(~found)[0]
                    fallback_xy = self._sample_global_source_coords(
                        source_weight,
                        torch.stack([regular_target_x[missing_rows], regular_target_y[missing_rows]], dim=1),
                    )
                    selected_src_x[missing_rows] = fallback_xy[:, 0]
                    selected_src_y[missing_rows] = fallback_xy[:, 1]

                sparse_points[batch_idx, :, regular_target_y, regular_target_x] = img[
                    batch_idx,
                    :,
                    selected_src_y,
                    selected_src_x,
                ]

            inverse_rows = torch.where(inverse_samples)[0]
            if inverse_rows.numel() > 0 and self.step2_fill_inverse_intersection_samples:
                inverse_target_x = target_x[inverse_rows]
                inverse_target_y = target_y[inverse_rows]
                valid_surround_mask = (1.0 - inverse_weight) * (1.0 - render_weight)
                nearest_xy = self._sample_nearest_valid_source_coords(
                    valid_surround_mask,
                    torch.stack([inverse_target_x, inverse_target_y], dim=1),
                )
                sparse_points[batch_idx, :, inverse_target_y, inverse_target_x] = img[
                    batch_idx,
                    :,
                    nearest_xy[:, 1],
                    nearest_xy[:, 0],
                ]

        masks_rest = 1.0 - hair_mask
        return masking_utils.masking(
            img,
            masks_rest,
            sparse_points,
            self.config.train.mask_dilation_radius,
            rendered_mask=render_mask,
            extra_noise=False,
            random_mask=0.0,
        )

    def _save_debug_hair_render(self, input_img, hair_img, batch_idx, phase, flame_img=None, flame_crop_img=None, smirk_img=None, sparse_hair_img=None):
        log_root = getattr(self.config.train, 'log_path', None) or '.'
        debug_dir = os.path.join(log_root, 'hair_debug')
        os.makedirs(debug_dir, exist_ok=True)

        img_vis = (input_img.detach().cpu() * 0.5) + 0.5
        img_vis = img_vis.clamp(0.0, 1.0)

        base = f"{phase}_step2_{batch_idx:06d}"
        save_image(img_vis, os.path.join(debug_dir, f"{base}_input.png"))
        save_image(img_vis, "test_step2_input.png")
        if smirk_img is not None:
            smirk_vis = smirk_img.detach().cpu().clamp(0.0, 1.0)
            save_image(smirk_vis, os.path.join(debug_dir, f"{base}_smirk_input.png"))
        if sparse_hair_img is not None:
            sparse_hair_vis = sparse_hair_img.detach().cpu().clamp(0.0, 1.0)
            save_image(sparse_hair_vis, os.path.join(debug_dir, f"{base}_hair_sparse.png"))
        if hair_img is not None:
            hair_img = hair_img.detach().cpu()
            if hair_img.shape[0] == 4:
                hair_vis = self.strand2vis(hair_img[:3].unsqueeze(0))[0]
                if hair_img[:1].max() > 0:
                    hair_depth_vis = self.depth2vis(hair_img[3:].unsqueeze(0), hair_img[:1].unsqueeze(0))[0]
                    save_image(hair_depth_vis, os.path.join(debug_dir, f"{base}_hair_depth.png"))
            else:
                hair_vis = hair_img.clamp(0.0, 1.0)
            save_image(hair_vis, os.path.join(debug_dir, f"{base}_hair.png"))
            save_image(hair_vis, "test_step2_hair.png")
        if flame_img is not None:
            flame_vis = flame_img.detach().cpu().clamp(0.0, 1.0)
            save_image(flame_vis, os.path.join(debug_dir, f"{base}_flame.png"))
            save_image(flame_vis, "test_step2_flame.png")
        if flame_crop_img is not None:
            flame_crop_vis = flame_crop_img.detach().cpu().clamp(0.0, 1.0)
            save_image(flame_crop_vis, os.path.join(debug_dir, f"{base}_flame_crop.png"))

    def _save_debug_hairstep_maps(self, hair_img, batch_idx, phase):
        if hair_img is None:
            return

        hair_img = hair_img.detach().cpu()
        if hair_img.shape[0] != 4:
            return

        log_root = getattr(self.config.train, 'log_path', None) or '.'
        debug_root = os.path.join(log_root, 'hair_debug', 'hairstep')
        strand_dir = os.path.join(debug_root, 'strand_map')
        depth_dir = os.path.join(debug_root, 'depth_map')
        depth_vis_dir = os.path.join(debug_root, 'depth_vis_map')
        orient_vis_dir = os.path.join(debug_root, 'orient_vis_map')
        seg_dir = os.path.join(debug_root, 'seg')
        os.makedirs(strand_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(depth_vis_dir, exist_ok=True)
        os.makedirs(orient_vis_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)

        base = f"{phase}_step2_{batch_idx:06d}"
        mask = hair_img[:1].clamp(0.0, 1.0)
        dxdy = hair_img[1:3].clamp(-1.0, 1.0)
        depth = hair_img[3:].clamp(0.0, 1.0)

        strand_map = torch.cat([mask, ((dxdy + 1.0) * 0.5) * mask], dim=0).clamp(0.0, 1.0)
        save_image(strand_map, os.path.join(strand_dir, f"{base}.png"))
        orient_vis = self.strand2vis(torch.cat([mask, dxdy], dim=0).unsqueeze(0))[0]
        save_image(orient_vis, os.path.join(orient_vis_dir, f"{base}.png"))
        save_image(mask, os.path.join(seg_dir, f"{base}.png"))
        np.save(os.path.join(depth_dir, f"{base}.npy"), depth.squeeze(0).numpy())

        if mask.max() > 0:
            depth_vis = self.depth2vis(depth.unsqueeze(0), mask.unsqueeze(0))[0]
        else:
            depth_vis = torch.zeros(3, depth.shape[-2], depth.shape[-1], dtype=depth.dtype)
        save_image(depth_vis, os.path.join(depth_vis_dir, f"{base}.png"))

    def _save_debug_inverse_intersection_mask(self, mask_img, batch_idx, phase):
        if mask_img is None:
            return

        log_root = getattr(self.config.train, 'log_path', None) or '.'
        debug_dir = os.path.join(log_root, 'hair_debug')
        os.makedirs(debug_dir, exist_ok=True)

        base = f"{phase}_step2_{batch_idx:06d}"
        mask_vis = mask_img.detach().cpu().clamp(0.0, 1.0)
        save_image(mask_vis, os.path.join(debug_dir, f"{base}_hair_inverse_intersection_mask.png"))

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

    def step2_old(self, encoder_output, batch, batch_idx, phase='train'):
        if not self.config.arch.enable_fuse_generator:
            raise RuntimeError("Second path requires arch.enable_fuse_generator=True")

        img = batch['img']
        masks = batch['hairmask']

        strand_output = encoder_output['strand_params']
        if self.config.arch.depth_branch:
            depth_output = encoder_output['depth_params']
            strand_depth_output = torch.cat([strand_output, depth_output], dim=1)
        else:
            strand_depth_output = strand_output

        tmask_ratio = self.config.train.mask_ratio
        hair_sampled_points = masking_utils.mask_uniform_hair(masks, tmask_ratio)
        extra_points = masking_utils.transfer_pixels(img, hair_sampled_points, hair_sampled_points)
        masks_rest = 1 - masks
        masked_img = masking_utils.masking(img, masks_rest, extra_points, self.config.train.mask_dilation_radius)

        reconstructed_img = self.smirk_generator(torch.cat([strand_depth_output, masked_img], dim=1))

        with torch.no_grad():
            target_smirk_feats = self.smirk_face_encoder(img)
        predicted_smirk_feats = self.smirk_face_encoder(reconstructed_img)

        cycle_loss = F.mse_loss(predicted_smirk_feats['expression_params'], target_smirk_feats['expression_params'])
        cycle_loss += F.mse_loss(predicted_smirk_feats['jaw_params'], target_smirk_feats['jaw_params'])
        if 'shape_params' in predicted_smirk_feats and 'shape_params' in target_smirk_feats:
            cycle_loss += F.mse_loss(predicted_smirk_feats['shape_params'], target_smirk_feats['shape_params'])

        raw_losses = {'cycle_loss': cycle_loss}
        loss_second_path = raw_losses['cycle_loss'] * self.config.train.loss_weights['cycle_loss']
        losses = {key: (value.item() if isinstance(value, torch.Tensor) else value)
                  for key, value in raw_losses.items()}

        outputs = {}
        if batch_idx % self.config.train.visualize_every == 0:
            outputs['cycle_reconstruction'] = reconstructed_img.detach().cpu()
        return outputs, losses, loss_second_path

    def _encoder_batchnorm_eval_enabled(self):
        return bool(getattr(self.config.train, 'encoder_batchnorm_eval', False))

    def _set_encoder_batchnorm_eval(self):
        for module in self.hair_encoder.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    def freeze_encoder(self):
        utils.freeze_module(self.hair_encoder.strand_encoder, 'strand encoder')
        if self.config.arch.depth_branch and hasattr(self.hair_encoder, 'depth_encoder'):
            utils.freeze_module(self.hair_encoder.depth_encoder, 'depth encoder')
        
    def unfreeze_encoder(self):
        optimize_strand = getattr(self.config.train, 'optimize_strand', None)
        if optimize_strand is None:
            optimize_strand = getattr(self.config.train, 'optimize_hairstrand', True)
        if optimize_strand:
            utils.unfreeze_module(self.hair_encoder.strand_encoder, 'strand encoder')
        
        optimize_depth = getattr(self.config.train, 'optimize_depth', None)
        if optimize_depth is None:
            optimize_depth = getattr(self.config.train, 'optimize_hairdepth', False)
        if (self.config.arch.depth_branch and optimize_depth and hasattr(self.hair_encoder, 'depth_encoder')):
            utils.unfreeze_module(self.hair_encoder.depth_encoder, 'depth encoder')

        if self._encoder_batchnorm_eval_enabled():
            self._set_encoder_batchnorm_eval()
            
    def step(self, batch, batch_idx, phase='train', epoch_idx=None):
        if epoch_idx is not None:
            self.current_epoch_idx = int(epoch_idx)
            self.in_warmup_phase = self.is_warmup_epoch(epoch_idx)
        is_warmup = self.in_warmup_phase
        did_optimizer_step = False

        # ------- set the model to train or eval mode ------- #
        if phase == 'train':
            self.train()
            torch.set_grad_enabled(True)
            if is_warmup:
                self.freeze_encoder()
            else:
                self.unfreeze_encoder()
        else:
            self.eval()
            torch.set_grad_enabled(False)
        self.base_encoder.eval()
        
        # losses1 is for logging only, loss_first_path is used to update model
        outputs1, losses1, loss_first_path, encoder_output = self.step1(batch)

        if phase == 'train':
            self.optimizers_zero_grad()
            if self._loss_is_finite(loss_first_path, 'loss_first_path', batch_idx, phase):
                loss_first_path.backward()
                self._clip_gradients(clip_encoder=not is_warmup, clip_generator=True)
                self.optimizers_step(step_encoder=not is_warmup, step_fuse_generator=True)
                did_optimizer_step = True
            else:
                losses1['skipped_first_path_update'] = 1.0
             
        should_visualize = (
            self.config.train.visualize_phase.get(phase, False)
            and (batch_idx % self.config.train.visualize_every == 0)
        )
        use_cycle_loss = (not is_warmup) and (self.config.train.loss_weights['cycle_loss'] > 0)
        run_second_path = (not is_warmup) and (use_cycle_loss or should_visualize)

        if run_second_path:
            if phase == 'train' and self.config.train.freeze_encoder_in_second_path:
                self.freeze_encoder()
            if phase == 'train' and self.config.train.freeze_generator_in_second_path:
                utils.freeze_module(self.smirk_generator, 'fuse generator')

            outputs2, losses2, loss_second_path = self.step2(encoder_output, batch, batch_idx, phase)

            losses1.update(losses2)
            outputs1.update(outputs2)

            if use_cycle_loss and (phase == 'train'):
                self.optimizers_zero_grad()
                if self._loss_is_finite(loss_second_path, 'loss_second_path', batch_idx, phase):
                    loss_second_path.backward()

                    clipped = self._clip_gradients(
                        clip_encoder=not self.config.train.freeze_encoder_in_second_path,
                        clip_generator=not self.config.train.freeze_generator_in_second_path,
                    )

                    # Preserve the previous generator-only cycle clipping when no clipping config is present.
                    if (
                        not clipped
                        and not self._has_gradient_clipping_config()
                        and not self.config.train.freeze_generator_in_second_path
                    ):
                        torch.nn.utils.clip_grad_norm_(self.smirk_generator.parameters(), 0.1)

                    self.optimizers_step(step_encoder=not self.config.train.freeze_encoder_in_second_path, 
                                         step_fuse_generator=not self.config.train.freeze_generator_in_second_path)
                    did_optimizer_step = True
                else:
                    losses1['skipped_second_path_update'] = 1.0

            if phase == 'train' and self.config.train.freeze_encoder_in_second_path:
                self.unfreeze_encoder()

            if phase == 'train' and self.config.train.freeze_generator_in_second_path:
                utils.unfreeze_module(self.smirk_generator, 'fuse generator')
        
        losses = losses1
        if 'loss_second_path' not in losses:
            losses['loss_second_path'] = 0.0
        losses['total_loss'] = losses['loss_first_path'] + losses['loss_second_path']
        self.logging(batch_idx, losses, phase)

        self.current_loss = losses

        if phase == 'train' and did_optimizer_step:
            self.scheduler_step()

        return outputs1

    def _loss_is_finite(self, loss, loss_name, batch_idx, phase):
        if isinstance(loss, torch.Tensor):
            detached_loss = loss.detach()
            is_finite = bool(torch.isfinite(detached_loss).all().item())
            loss_value = detached_loss.item() if detached_loss.numel() == 1 else detached_loss
        else:
            loss_value = float(loss)
            is_finite = bool(np.isfinite(loss_value))

        if is_finite:
            return True

        epoch_idx = getattr(self, 'current_epoch_idx', None)
        epoch_msg = f" epoch={epoch_idx + 1}" if epoch_idx is not None else ""
        msg = (
            f"[HairSynthesisTrainer] Skipping {loss_name} update:"
            f"{epoch_msg} phase={phase} batch={batch_idx} loss={loss_value}"
        )
        print(msg)
        logger = getattr(self, 'logger', None)
        if logger is not None:
            logger.warning(msg)
        return False
