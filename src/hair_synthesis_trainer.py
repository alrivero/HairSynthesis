import os
import torch
import torch.nn.functional as F
from src.smirk_encoder import SmirkEncoder, HairStepEncoder
from src.smirk_generator import SmirkGenerator
from src.base_trainer import BaseHairTrainer
import src.utils.utils as utils
import src.utils.masking as masking_utils
from src.models.flame_perm_bridge import FlamePermBridge

class HairSynthesisTrainer(BaseHairTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        ### If enable_fuse_generator, train both encoder and generator
        if self.config.arch.enable_fuse_generator:
            # input: strand (C=3), depth (C=1), sampled color image (C=3)
            gen_in_channel = 7 if self.config.arch.depth_branch else 6
            self.smirk_generator = SmirkGenerator(in_channels=gen_in_channel, out_channels=3, init_features=32, res_blocks=5)
            if getattr(self.config, 'load_generator', False):
                self._load_smirk_generator_weights(getattr(self.config, 'checkpoint_smirk', None))
            
        self.hair_encoder = HairStepEncoder(
            img2strand_ckpt=config.checkpoint_img2strand,
            img2depth_ckpt=config.checkpoint_img2depth,
            config=self.config)

        self.smirk_face_encoder = SmirkEncoder(
            n_exp=self.config.arch.num_expression,
            n_shape=self.config.arch.num_shape,
        )
        self.freeze_smirk_encoder = getattr(self.config.train, 'freeze_smirk_encoder', True)
        self._load_smirk_encoder_weights(getattr(self.config, 'checkpoint_smirk', None))
        if self.freeze_smirk_encoder:
            utils.freeze_module(self.smirk_face_encoder, 'smirk face encoder')
            self.smirk_face_encoder.eval()
        else:
            utils.unfreeze_module(self.smirk_face_encoder, 'smirk face encoder')

        self.flame_perm_bridge = FlamePermBridge(
            n_shape=self.config.arch.num_shape,
            n_expression=self.config.arch.num_expression,
            perm_model_path=getattr(self.config, 'perm_model_path', None),
            perm_head_mesh=getattr(self.config, 'perm_head_mesh', None),
            perm_scalp_bounds=getattr(self.config, 'perm_scalp_bounds', None),
        )

        self.setup_losses()
        self.current_loss = None

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

    def step1(self, batch):
        B, C, H, W = batch['img'].shape
        # batch has img, hairmask, bodymask
        # batch['img']: (B, C, H, W)
        # batch['hairmask']: (B, 1, H, W)
        # batch['bodymask']: (B, 1, H, W)

        encoder_output = self.hair_encoder(batch['img'], batch['hairmask'], batch['bodymask'])
        
        with torch.no_grad():       # base_encoder = hair_encoder -> later used to regularize the weights
            base_output = self.base_encoder(batch['img'], batch['hairmask'], batch['bodymask'])

        strand_output = encoder_output['strand_params']     # (B, 3, H, W)
        # print("step1")
        # tmp = strand_output[0]
        # print("strand", strand_output.shape)
        # print(torch.min(tmp[0, :, :]), torch.max(tmp[0, :, :]))
        # print(torch.min(tmp[1, :, :]), torch.max(tmp[1, :, :]))
        # print(torch.min(tmp[2, :, :]), torch.max(tmp[2, :, :]))

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
                base_output = self.base_encoder(batch['img'], batch['hairmask'], batch['bodymask'])
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

            # Get neighbors: use hairmask to compute only on hair region
            local_sum = F.avg_pool2d(orient * hairmask, kernel_size=smooth_kernel, stride=smooth_stride, padding=smooth_pad) * 9
            local_count = F.avg_pool2d(hairmask, kernel_size=smooth_kernel, stride=smooth_stride, padding=smooth_pad) * 9 + eps
            
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

        fuse_generator_losses = (losses['perceptual_vgg_loss'] * self.config.train.loss_weights['perceptual_vgg_loss'] + 
                                losses['reconstruction_loss'] * self.config.train.loss_weights['reconstruction_loss'] + 
                                losses['strand_regularization'] * self.config.train.loss_weights['strand_regularization'] + 
                                losses['strand_local_regularization'] * self.config.train.loss_weights['strand_local_regularization']
                                )

        if self.config.arch.depth_branch:
            fuse_generator_losses += losses['depth_regularization'] * self.config.train.loss_weights['depth_regularization']
               
        loss_first_path = (
            (fuse_generator_losses if self.config.arch.enable_fuse_generator else 0)
        )

        for key, value in losses.items():
            losses[key] = value.item() if isinstance(value, torch.Tensor) else value

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
        """Temporarily return only FLAME parameters predicted by the frozen smirk encoder."""
        with torch.no_grad():
            flame_params = self.smirk_face_encoder(batch['img'])
        
        import pdb; pdb.set_trace()

        outputs = {f'flame_{key}': value.detach().cpu() for key, value in flame_params.items()}
        losses = {}
        dummy_loss = torch.zeros(1, device=batch['img'].device, dtype=batch['img'].dtype,
                                 requires_grad=(phase == 'train'))

        return outputs, losses, dummy_loss

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

    def freeze_encoder(self):
        utils.freeze_module(self.hair_encoder.strand_encoder, 'strand encoder')
        if self.config.arch.depth_branch and hasattr(self.hair_encoder, 'depth_encoder'):
            utils.freeze_module(self.hair_encoder.depth_encoder, 'depth encoder')
        
    def unfreeze_encoder(self):
        if self.config.train.optimize_strand:
            utils.unfreeze_module(self.hair_encoder.strand_encoder, 'strand encoder')
        
        if (self.config.arch.depth_branch and getattr(self.config.train, 'optimize_depth', False)
                and hasattr(self.hair_encoder, 'depth_encoder')):
            utils.unfreeze_module(self.hair_encoder.depth_encoder, 'depth encoder')
            
    def step(self, batch, batch_idx, phase='train'):
        # ------- set the model to train or eval mode ------- #
        if phase == 'train':
            self.train()
            torch.set_grad_enabled(True)
        else:
            self.eval()
            torch.set_grad_enabled(False)
        self.base_encoder.eval()
        
        # losses1 is for logging only, loss_first_path is used to update model
        outputs1, losses1, loss_first_path, encoder_output = self.step1(batch)

        if phase == 'train':
            self.optimizers_zero_grad()
            loss_first_path.backward()
            self.optimizers_step(step_encoder=True,  step_fuse_generator=True)
             
        if (self.config.train.loss_weights['cycle_loss'] > 0) and (phase == 'train'):
            if self.config.train.freeze_encoder_in_second_path:
                self.freeze_encoder()
            if self.config.train.freeze_generator_in_second_path:
                utils.freeze_module(self.smirk_generator, 'fuse generator')
                    
            outputs2, losses2, loss_second_path = self.step2(encoder_output, batch, batch_idx, phase)
            
            self.optimizers_zero_grad()
            loss_second_path.backward()

            # gradient clip for generator - we want only details to be guided 
            if not self.config.train.freeze_generator_in_second_path:
                torch.nn.utils.clip_grad_norm_(self.smirk_generator.parameters(), 0.1)
            
            self.optimizers_step(step_encoder=not self.config.train.freeze_encoder_in_second_path, 
                                 step_fuse_generator=not self.config.train.freeze_generator_in_second_path)

            losses1.update(losses2)
            outputs1.update(outputs2)

            if self.config.train.freeze_encoder_in_second_path:
                self.unfreeze_encoder()
            
            if self.config.train.freeze_generator_in_second_path:
                utils.unfreeze_module(self.smirk_generator, 'fuse generator')
        
        losses = losses1
        self.logging(batch_idx, losses, phase)

        self.current_loss = losses

        if phase == 'train':
            self.scheduler_step()

        return outputs1
