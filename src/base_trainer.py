import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
import cv2
import math
import os
import random
import copy
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from src.utils.utils import batch_draw_keypoints, make_grid_from_opencv_images


class BaseTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


    def logging(self, batch_idx, losses, phase):

        # ---------------- logging ---------------- #
        if self.config.train.log_losses_every > 0 and batch_idx % self.config.train.log_losses_every == 0:
            # print losses in one line
            loss_str = ''
            for k, v in losses.items():
                loss_str += f'{k}: {v:.6f} '
            print(loss_str)

    def configure_optimizers(self, n_steps):
        
        self.n_steps = n_steps

        # start decaying at max_epochs // 2
        # at the end of training, the lr will be 0.1 * lr
        # lambda_func = lambda epoch: 1.0 - max(0, .1 + epoch - max_epochs//2) / float(max_epochs//2)

        encoder_scale = .25

        # check if self.encoder_optimizer exists
        if hasattr(self, 'encoder_optimizer'):
            for g in self.encoder_optimizer.param_groups:
                g['lr'] = encoder_scale * self.config.train.lr
        else:
            params = []
            if self.config.train.optimize_expression:
                params += list(self.smirk_encoder.expression_encoder.parameters()) 
            if self.config.train.optimize_shape:
                params += list(self.smirk_encoder.shape_encoder.parameters())
            if self.config.train.optimize_pose:
                params += list(self.smirk_encoder.pose_encoder.parameters())

            self.encoder_optimizer = torch.optim.Adam(params, lr= encoder_scale * self.config.train.lr)
                
        # cosine schedulers for both optimizers - per iterations
        self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_optimizer, T_max=n_steps,
                                                                                  eta_min=0.01 * encoder_scale * self.config.train.lr)

        if self.config.arch.enable_fuse_generator:
            if hasattr(self, 'fuse_generator_optimizer'):
                for g in self.smirk_generator_optimizer.param_groups:
                    g['lr'] = self.config.train.lr
            else:
                self.smirk_generator_optimizer = torch.optim.Adam(self.smirk_generator.parameters(), lr= self.config.train.lr, betas=(0.5, 0.999))

            
            self.smirk_generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.smirk_generator_optimizer, T_max=n_steps,
                                                                                    eta_min=0.01 * self.config.train.lr)

        
    def load_random_template(self, num_expressions=50):
        random_key = random.choice(list(self.templates.keys()))
        templates = self.templates[random_key]
        random_index = random.randint(0, templates.shape[0]-1)

        return templates[random_index][:num_expressions]
        

    def setup_losses(self):
        from src.losses.VGGPerceptualLoss import VGGPerceptualLoss
        self.vgg_loss = VGGPerceptualLoss()
        self.vgg_loss.eval()
        for param in self.vgg_loss.parameters():
            param.requires_grad_(False)
        
            
        if self.config.train.loss_weights['emotion_loss'] > 0:
            from src.losses.ExpressionLoss import ExpressionLoss
            self.emotion_loss = ExpressionLoss()
            # freeze the emotion model
            self.emotion_loss.eval()
            for param in self.emotion_loss.parameters():
                param.requires_grad_(False)

        if self.config.train.loss_weights['mica_loss'] > 0:
            from src.models.MICA.mica import MICA

            self.mica = MICA()
            self.mica.eval()

            for param in self.mica.parameters():
                param.requires_grad_(False)

            
    def scheduler_step(self):
        self.encoder_scheduler.step()
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator_scheduler.step()

    def train(self):
        self.smirk_encoder.train()
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator.train()
    
    def eval(self):
        self.smirk_encoder.eval()
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator.eval()

    def optimizers_zero_grad(self):
        self.encoder_optimizer.zero_grad()
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator_optimizer.zero_grad()

    def optimizers_step(self, step_encoder=True, step_fuse_generator=True):
        if step_encoder:
            self.encoder_optimizer.step()
        if step_fuse_generator and self.config.arch.enable_fuse_generator:
            self.smirk_generator_optimizer.step()


    def save_visualizations(self, outputs, save_path, show_landmarks=False):
        nrow = 1
        
        if 'img' in outputs and 'rendered_img' in outputs and 'masked_1st_path' in outputs:
            outputs['overlap_image'] = outputs['img'] * 0.7 + outputs['rendered_img'] * 0.3
            outputs['overlap_image_pixels'] = outputs['img'] * 0.7 +  0.3 * outputs['masked_1st_path']
        
        if show_landmarks:
            original_img_with_landmarks = batch_draw_keypoints(outputs['img'], outputs['landmarks_mp'], color=(0,255,0))
            original_img_with_landmarks = batch_draw_keypoints(original_img_with_landmarks, outputs['landmarks_mp_gt'], color=(0,0,255))
            original_img_with_landmarks = batch_draw_keypoints(original_img_with_landmarks, outputs['landmarks_fan'][:,:17], color=(255,0,255))
            original_img_with_landmarks = batch_draw_keypoints(original_img_with_landmarks, outputs['landmarks_fan_gt'][:,:17], color=(255,255,255))
            original_grid = make_grid_from_opencv_images(original_img_with_landmarks, nrow=nrow)
        else:
            original_img_with_landmarks = outputs['img']
            original_grid = make_grid(original_img_with_landmarks, nrow=nrow)

        image_keys = ['img_mica', 'rendered_img_base', 'rendered_img', 
                      'overlap_image', 'overlap_image_pixels',
                    'rendered_img_mica_zero', 'rendered_img_zero', 
                      'masked_1st_path', 'reconstructed_img', 'loss_img', 
                      '2nd_path']
        
        nrows = [1 if '2nd_path' not in key else 4 * self.config.train.Ke for key in image_keys]
        
        grid = torch.cat([original_grid] + [make_grid(outputs[key].detach().cpu(), nrow=nr) for key, nr in zip(image_keys, nrows) if key in outputs.keys()], dim=2)
            
        grid = grid.permute(1,2,0).cpu().numpy()*255.0
        grid = np.clip(grid, 0, 255)
        grid = grid.astype(np.uint8)
        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

        cv2.imwrite(save_path, grid)


    def create_visualizations(self, batch, outputs):
        zero_pose_cam = torch.tensor([7,0,0]).unsqueeze(0).repeat(batch['img'].shape[0], 1).float().to(self.config.device)

        # batch keys are already in device, so no need to move them
        # outputs are in cpu, so we need to move them to device if we want to use them in the renderer

        visualizations = {}

        visualizations['img'] = batch['img']
        visualizations['rendered_img'] = outputs['rendered_img']
        

        # 2. Base model
        base_output = self.base_encoder(batch['img'])
        flame_output_base = self.flame.forward(base_output)
        rendered_img_base = self.renderer.forward(flame_output_base['vertices'], base_output['cam'])['rendered_img']
        visualizations['rendered_img_base'] = rendered_img_base
    
        flame_output_zero = self.flame.forward(outputs['encoder_output'], zero_expression=True, zero_pose=True)
        rendered_img_zero = self.renderer.forward(flame_output_zero['vertices'].to(self.config.device), zero_pose_cam)['rendered_img']
        visualizations['rendered_img_zero'] = rendered_img_zero
    
    
        if self.config.arch.enable_fuse_generator:
            visualizations['reconstructed_img'] = outputs['reconstructed_img']
            visualizations['masked_1st_path'] = outputs['masked_1st_path']
            visualizations['loss_img'] = outputs['loss_img']

        for key in visualizations.keys():
            visualizations[key] = visualizations[key].detach().cpu()

        # 3. MICA
        if self.config.train.loss_weights['mica_loss'] > 0:
            mica_output_shape = self.mica(batch['img_mica'])
            mica_output = copy.deepcopy(base_output) # just to get the keys and structure
            mica_output['shape_params'] = mica_output_shape['shape_params']

            if self.config.arch.num_shape < 300:
                # WARNING: we are visualizing using only the first num_shape parameters
                mica_output['shape_params'] = mica_output['shape_params'][:, :self.config.arch.num_shape]

            flame_output_mica = self.flame.forward(mica_output, zero_expression=True, zero_pose=True)
            rendered_img_mica_zero = self.renderer.forward(flame_output_mica['vertices'], zero_pose_cam)['rendered_img']
            visualizations['rendered_img_mica_zero'] = rendered_img_mica_zero

            visualizations['img_mica'] = batch['img_mica'].reshape(-1, 3, 112, 112)
            visualizations['img_mica'] = F.interpolate(visualizations['img_mica'], self.config.image_size).detach().cpu()


        if self.config.train.loss_weights['cycle_loss'] > 0:
            if '2nd_path' in outputs:
                visualizations['2nd_path'] = outputs['2nd_path']

        # landmarks
        visualizations['landmarks_mp'] = outputs['landmarks_mp']
        visualizations['landmarks_mp_gt'] = outputs['landmarks_mp_gt']
        visualizations['landmarks_fan'] = outputs['landmarks_fan']
        visualizations['landmarks_fan_gt'] = outputs['landmarks_fan_gt']

        return visualizations

    def save_model(self, state_dict, save_path):
        # remove everything that is not smirk_encoder or smirk_generator
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if key.startswith('smirk_encoder') or key.startswith('smirk_generator'):
                new_state_dict[key] = state_dict[key]

        torch.save(new_state_dict, save_path)


    def create_base_encoder(self):
        self.base_encoder = copy.deepcopy(self.smirk_encoder)
        self.base_encoder.eval()



    def load_model(self, resume, load_fuse_generator=True, load_encoder=True, device='cuda'):
        loaded_state_dict = torch.load(resume, map_location=device)

        print(f'Loading checkpoint from {resume}, load_encoder={load_encoder}, load_fuse_generator={load_fuse_generator}')

        filtered_state_dict = {}
        for key in loaded_state_dict.keys():
            # new
            if (load_encoder and key.startswith('smirk_encoder')) or (load_fuse_generator and key.startswith('smirk_generator')):
                filtered_state_dict[key] = loaded_state_dict[key]
            

        self.load_state_dict(filtered_state_dict, strict=False) # set it false because it asks for mica and other models apart from smirk_encoder and smirk_generator



    def set_freeze_status(self, config, batch_idx, epoch_idx):
        #self.config.train.freeze_encoder_in_first_path = False
        #self.config.train.freeze_generator_in_first_path = False
        self.config.train.freeze_encoder_in_second_path = False
        self.config.train.freeze_generator_in_second_path = False

        #decision_idx = batch_idx if config.train.freeze_schedule.per_iteration else epoch_idx
        decision_idx_second_path = batch_idx #epoch_idx 

        self.config.train.freeze_encoder_in_second_path = decision_idx_second_path % 2 == 0
        self.config.train.freeze_generator_in_second_path = decision_idx_second_path % 2 == 1

class BaseHairTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.logger = None

        # Setup logger to save training loss to a file
        if config.train.log_path is not None and self._is_main_process():
            log_file = os.path.join(config.train.log_path, 'logs.log')
            assert not os.path.exists(log_file), f"{os.path.join(log_file)} existed!"

            self.logger = logging.getLogger()
            logging.getLogger('matplotlib.font_manager').disabled = True
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler(os.path.join(log_file))
            fh.setLevel(logging.INFO)
            self.logger.addHandler(fh)

    def _rank(self):
        try:
            return int(os.environ.get('RANK', '0'))
        except ValueError:
            return 0

    def _is_main_process(self):
        return self._rank() == 0

    @staticmethod
    def _unwrap_ddp_module(module):
        return module.module if hasattr(module, 'module') else module

    def _hair_encoder_module(self):
        return self._unwrap_ddp_module(self.hair_encoder)

    def _smirk_generator_module(self):
        generator = getattr(self, 'smirk_generator', None)
        if generator is None:
            return None
        return self._unwrap_ddp_module(generator)

    def logging(self, batch_idx, losses, phase):
        # ---------------- logging ---------------- #
        if (
            self._is_main_process()
            and self.config.train.log_losses_every > 0
            and batch_idx % self.config.train.log_losses_every == 0
        ):
            # print losses in one line
            loss_str = ''
            for k, v in losses.items():
                loss_str += f'{k}: {v:.6f} '
            print(loss_str)

    def logging_epoch(self, epoch_idx, losses, phase):
        if self.logger is None:
            return
        # Save to log file
        loss_str = f'Epoch {epoch_idx+1}/{self.config.train.num_epochs}. {phase.upper()}. '
        for k, v in losses.items():
            loss_str += f'{k}: {v:.6f} '
        self.logger.info(loss_str)

    def plot_curve(self, train, val, ylabel, title, plot_fig):
        n_epoch = len(train)

        plt.plot(np.arange(n_epoch).tolist(), train, color='blue', label='train')
        plt.plot(np.arange(n_epoch).tolist(), val, color='orange', label='val')
        plt.xlabel('epoch')
        plt.ylabel(ylabel)
        plt.legend(loc='best')
        plt.title(f'Train-val {title}')
        plt.savefig(plot_fig)
        plt.clf()
        plt.close()

    def plot_curve_single(self, data, data_label, ylabel, title, plot_fig):
        n_data = len(data)

        plt.plot(np.arange(n_data).tolist(), data, color='blue', label=data_label)
        plt.xlabel('epoch')
        plt.ylabel(ylabel)
        plt.legend(loc='best')
        plt.title(f'{data_label} {title}')
        plt.savefig(plot_fig)
        plt.clf()
        plt.close()

    def plot_losses(self, losses_hist, plot_batch=True):
        if not self._is_main_process():
            return
        train_losses = losses_hist['train']
        val_losses = losses_hist['val']
        # print("plot_losses", train_losses)

        losses_name = sorted(set(train_losses.keys()) | set(val_losses.keys()))
        for ln in losses_name:
            train_curve = train_losses.get(ln)
            val_curve = val_losses.get(ln)
            plot_path = os.path.join(self.config.train.log_path, "loss_plots", f"loss_{ln}.png")

            if train_curve is not None and val_curve is not None:
                self.plot_curve(train_curve, val_curve, 'losses', ln, plot_path)
            elif train_curve is not None:
                self.plot_curve_single(train_curve, 'train', 'losses', ln, plot_path)
            elif val_curve is not None:
                self.plot_curve_single(val_curve, 'val', 'losses', ln, plot_path)
            
            if plot_batch:
                train_batch_curve = losses_hist['train_b'].get(ln)
                val_batch_curve = losses_hist['val_b'].get(ln)
                if train_batch_curve is not None:
                    self.plot_curve_single(
                        train_batch_curve,
                        'train',
                        'losses',
                        ln,
                        os.path.join(self.config.train.log_path, "loss_plots", f"loss_{ln}_train_b.png"),
                    )
                if val_batch_curve is not None:
                    self.plot_curve_single(
                        val_batch_curve,
                        'val',
                        'losses',
                        ln,
                        os.path.join(self.config.train.log_path, "loss_plots", f"loss_{ln}_val_b.png"),
                    )

    def _cfg_get(self, cfg, key, default=None):
        if cfg is None:
            return default
        try:
            if key in cfg:
                return cfg[key]
        except Exception:
            pass
        return getattr(cfg, key, default)

    def _target_lr(self, name):
        if name == 'encoder':
            lr = self._cfg_get(self.config.train, 'lr_encoder', None)
            if lr is None:
                lr = self._cfg_get(self.config.train, 'encoder_lr', None)
            if lr is None:
                lr = 0.25 * float(self._cfg_get(self.config.train, 'lr'))
            return float(lr)
        if name == 'generator':
            lr = self._cfg_get(self.config.train, 'lr_generator', None)
            if lr is None:
                lr = self._cfg_get(self.config.train, 'generator_lr', None)
            if lr is None:
                lr = float(self._cfg_get(self.config.train, 'lr'))
            return float(lr)
        raise ValueError(f"Unknown lr target: {name}")

    def _ramp_cfg(self, name):
        lr_ramp_cfg = self._cfg_get(self.config.train, 'lr_ramp', None)
        ramp_cfg = self._cfg_get(lr_ramp_cfg, name, None)
        if ramp_cfg is None:
            ramp_cfg = self._cfg_get(self.config.train, f'{name}_lr_ramp', None)
        return ramp_cfg

    def _ramp_duration_steps(self, ramp_cfg):
        duration_steps = self._cfg_get(ramp_cfg, 'duration_steps', None)
        if duration_steps is None:
            duration_steps = self._cfg_get(ramp_cfg, 'duration_batches', None)
        if duration_steps is not None:
            return max(0, int(duration_steps))

        duration_fraction = self._cfg_get(ramp_cfg, 'duration_fraction', None)
        if duration_fraction is not None:
            return max(0, int(round(float(duration_fraction) * max(1, int(self.n_steps)))))

        # Legacy alias from the first implementation. Treat it as a step count
        # rather than an epoch count so ramping always happens within each epoch.
        duration_steps = self._cfg_get(ramp_cfg, 'duration_epochs', self._cfg_get(ramp_cfg, 'epochs', 0))
        return max(0, int(duration_steps))

    def _encoder_decay_lr(self, step_idx=0):
        base_lr = self._target_lr('encoder')
        eta_min = 0.01 * base_lr
        t_max = max(1, int(self.n_steps))
        t_cur = min(max(int(step_idx), 0), t_max)
        cosine = 0.5 * (1.0 + math.cos(math.pi * float(t_cur) / float(t_max)))
        return eta_min + (base_lr - eta_min) * cosine

    def _encoder_ramp_active(self, epoch_idx=None):
        ramp_cfg = self._ramp_cfg('encoder')
        if ramp_cfg is None or not bool(self._cfg_get(ramp_cfg, 'enabled', False)):
            return False

        epoch_idx = int(epoch_idx if epoch_idx is not None else getattr(self, 'current_epoch_idx', 0) or 0)
        start_epoch = int(self._cfg_get(ramp_cfg, 'start_epoch', 0))
        return epoch_idx >= start_epoch

    def _encoder_ramp_factor(self, step_idx=0, epoch_idx=None):
        ramp_cfg = self._ramp_cfg('encoder')
        if not self._encoder_ramp_active(epoch_idx):
            return 1.0

        target_lr = self._target_lr('encoder')
        end_factor = float(self._cfg_get(ramp_cfg, 'end_factor', 1.0))

        start_lr = self._cfg_get(ramp_cfg, 'start_lr', None)
        if start_lr is not None and target_lr > 0:
            start_factor = float(start_lr) / target_lr
        else:
            start_factor = float(self._cfg_get(ramp_cfg, 'start_factor', 0.0))

        if (
            bool(self._cfg_get(ramp_cfg, 'inverse_to_generator_decay', False))
            and self.config.arch.enable_fuse_generator
            and hasattr(self, 'smirk_generator_optimizer')
        ):
            gen_base_lr = self._target_lr('generator')
            gen_min_lr = 0.01 * gen_base_lr
            gen_lr = self.smirk_generator_optimizer.param_groups[0]['lr']
            denom = max(gen_base_lr - gen_min_lr, 1e-12)
            progress = (gen_base_lr - gen_lr) / denom
            progress = min(max(float(progress), 0.0), 1.0)
        else:
            duration_steps = self._ramp_duration_steps(ramp_cfg)
            if duration_steps <= 0:
                return 1.0
            if duration_steps <= 1:
                progress = 1.0
            else:
                progress = min(max(float(step_idx), 0.0) / float(duration_steps - 1), 1.0)

        return start_factor + progress * (end_factor - start_factor)

    def _encoder_scheduled_lr(self, step_idx=0, epoch_idx=None):
        if self._encoder_ramp_active(epoch_idx):
            return self._target_lr('encoder') * self._encoder_ramp_factor(step_idx, epoch_idx)
        return self._encoder_decay_lr(step_idx)

    def _set_optimizer_lr(self, optimizer, lr):
        for group in optimizer.param_groups:
            group['lr'] = lr

    def _apply_encoder_scheduled_lr(self):
        step_idx = getattr(self, '_lr_step_idx', 0)
        epoch_idx = getattr(self, 'current_epoch_idx', None)
        self._set_optimizer_lr(self.encoder_optimizer, self._encoder_scheduled_lr(step_idx, epoch_idx))

    def _has_gradient_clipping_config(self):
        return (
            self._cfg_get(self.config.train, 'gradient_clipping', None) is not None
            or self._cfg_get(self.config.train, 'clip_gradients', None) is not None
        )

    def _clip_gradients(self, *, clip_encoder=True, clip_generator=True):
        clip_cfg = self._cfg_get(self.config.train, 'gradient_clipping', None)
        if clip_cfg is None:
            enabled = bool(self._cfg_get(self.config.train, 'clip_gradients', False))
            max_norm = self._cfg_get(self.config.train, 'clip_grad_norm', None)
            clip_cfg = {
                'enabled': enabled,
                'encoder_max_norm': max_norm,
                'generator_max_norm': max_norm,
                'norm_type': 2.0,
            }

        if not bool(self._cfg_get(clip_cfg, 'enabled', False)):
            return False

        norm_type = float(self._cfg_get(clip_cfg, 'norm_type', 2.0))

        if clip_encoder:
            max_norm = self._cfg_get(
                clip_cfg,
                'encoder_max_norm',
                self._cfg_get(clip_cfg, 'max_norm', None),
            )
            if max_norm is not None and float(max_norm) > 0:
                params = [p for p in self.hair_encoder.parameters() if p.requires_grad and p.grad is not None]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, float(max_norm), norm_type=norm_type)

        if clip_generator and self.config.arch.enable_fuse_generator:
            max_norm = self._cfg_get(
                clip_cfg,
                'generator_max_norm',
                self._cfg_get(clip_cfg, 'max_norm', None),
            )
            if max_norm is not None and float(max_norm) > 0:
                params = [p for p in self.smirk_generator.parameters() if p.requires_grad and p.grad is not None]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, float(max_norm), norm_type=norm_type)

        return True

    def configure_optimizers(self, n_steps, epoch_idx=None):
        self.n_steps = max(1, int(n_steps))
        self.current_epoch_idx = epoch_idx
        self._lr_step_idx = 0
        generator_lr = self._target_lr('generator')
        hair_encoder_module = self._hair_encoder_module()
        smirk_generator_module = self._smirk_generator_module()

        if self.config.arch.enable_fuse_generator:
            if hasattr(self, 'smirk_generator_optimizer'):
                self._set_optimizer_lr(self.smirk_generator_optimizer, generator_lr)
            else:
                self.smirk_generator_optimizer = torch.optim.Adam(smirk_generator_module.parameters(), lr=generator_lr, betas=(0.5, 0.999))
            self.smirk_generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.smirk_generator_optimizer,
                T_max=self.n_steps,
                eta_min=0.01 * generator_lr,
            )

        encoder_lr = self._encoder_scheduled_lr(0, epoch_idx)

        # check if self.encoder_optimizer exists
        if hasattr(self, 'encoder_optimizer'):
            self._set_optimizer_lr(self.encoder_optimizer, encoder_lr)
        else:
            params = []
            encoder_mode = getattr(self.config.arch, 'encoder_mode', 'hairstep_maps')
            optimize_strand = getattr(self.config.train, 'optimize_strand', None)
            if optimize_strand is None:
                optimize_strand = getattr(self.config.train, 'optimize_hairstrand', True)
            if optimize_strand:
                if encoder_mode == 'perm_latent':
                    params += list(hair_encoder_module.parameters())
                else:
                    params += list(hair_encoder_module.strand_encoder.parameters())
            if encoder_mode != 'perm_latent' and self.config.arch.depth_branch:
                if self.config.train.optimize_hairdepth:
                    params += list(hair_encoder_module.depth_encoder.parameters())

            self.encoder_optimizer = torch.optim.Adam(params, lr=encoder_lr)
        
    def load_random_template(self, num_expressions=50):
        random_key = random.choice(list(self.templates.keys()))
        templates = self.templates[random_key]
        random_index = random.randint(0, templates.shape[0]-1)

        return templates[random_index][:num_expressions]

    def setup_losses(self):
        from src.losses.VGGPerceptualLoss import VGGPerceptualLoss
        self.vgg_loss = VGGPerceptualLoss()
        self.vgg_loss.eval()
        for param in self.vgg_loss.parameters():
            param.requires_grad_(False)
        
    def scheduler_step(self):
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator_scheduler.step()
        self._lr_step_idx = min(getattr(self, '_lr_step_idx', 0) + 1, int(self.n_steps))
        self._apply_encoder_scheduled_lr()

    def train(self):
        self.hair_encoder.train()
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator.train()
    
    def eval(self):
        self.hair_encoder.eval()
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator.eval()

    def optimizers_zero_grad(self):
        self.encoder_optimizer.zero_grad()
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator_optimizer.zero_grad()

    def optimizers_step(self, step_encoder=True, step_fuse_generator=True):
        if step_encoder:
            self.encoder_optimizer.step()
        if step_fuse_generator and self.config.arch.enable_fuse_generator:
            self.smirk_generator_optimizer.step()

    def depth2vis(self, depth, mask):
        """Similar as in img2depth.py but do not save temp.png        
        """
        depth_np = depth.detach().cpu().numpy()     # (B, 1, H, W)
        mask_np = mask.detach().cpu().numpy()       # (B, 1, H, W)

        B, _, H, W = depth_np.shape
        depth_vis = []
        for b in range(B):
            depth_b = depth_np[b, 0]    # (H, W)
            mask_b = mask_np[b, 0]      # (H, W)
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

            depth_rgb = depth_rgb * mask_b[..., None]               # (H, W, C)

            vis_b = torch.from_numpy(depth_rgb).permute(2,0,1)      # (C, H, W)
            depth_vis.append(vis_b)

        return torch.stack(depth_vis, dim=0)

    def strand2vis(self, strand):
        """Visualize strand map using HSV color wheel
        strand_map is in [-1, 1], already normalize per pixel by the magnitude
        """
        ### Hairstep convention:
        # Red = binary (0: background, 0.5: face/body, 1: hair)
        # O(x) = (M(x), O_{2D}/2 + 0.5)

        strand_map_batch = strand.detach().cpu().numpy()
        B, _, H, W = strand_map_batch.shape

        strand_vis = []
        for b in range(B):
            strand_map = strand_map_batch[b]        # already normalized to be in [-1, 1]
            # print("strand_map 1", np.min(strand_map[1]), np.max(strand_map[1]))
            # print("strand_map 2", np.min(strand_map[2]), np.max(strand_map[2]))

            green_c = strand_map[1]     # x (to right)
            blue_c = strand_map[2]      # y (down)

            # compute angle 
            theta = np.arctan2(-blue_c, green_c)     # in [-pi, pi]

            # convert to 0-2pi
            theta = (theta + 2*np.pi) % (2*np.pi)
            theta = (2*np.pi) - theta       # counter clockwise, 90 on the left

            # hue mapping
            h = theta / (2*np.pi)       # to [0, 1]
            s = np.ones_like(h)
            v = np.ones_like(h)

            # OpenCV HSV: H in [0,179], S,V in [0,255]
            hsv = np.zeros((H, W, 3), dtype=np.uint8)
            hsv[:, :, 0] = (h * 179)
            hsv[:, :, 1] = (s * 255)
            hsv[:, :, 2] = (v * 255)

            # print("h range:", h.min(), h.max())
            # print("s range:", s.min(), s.max())
            # print("v range:", v.min(), v.max())

            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255.0       # convert to [0, 1] bc save_visualization assume this as input

            # Recover hair support from HairStep's convention:
            # background=0, body/face=0.5, hair=1.
            mask = np.clip((strand_map[0] - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
            rgb = (rgb * mask[...,None])       # (H, W, C)
            vis_b = torch.from_numpy(rgb).permute(2,0,1)      # (C, H, W)
            
            strand_vis.append(vis_b)
        
        return torch.stack(strand_vis, dim=0)

    def save_visualizations(self, outputs, save_path):
        image_keys = [
            'img',
            'strand',
            'depth',
            'flame_mesh_render',
            'flame_mesh_overlay',
            'face_cleanup_render',
            'face_cleanup_before',
            'face_cleanup_after',
            'face_cleanup_input',
            'face_cleanup_base',
            'face_cleanup_hole_mask',
            'sparse_pixel_map',
            'translator_image_render',
            'injected_orient',
            'injected_depth',
            'injected_mask',
            'injected_inverse_intersection_mask',
            'injected_sparse_pixel_map',
            'injected_composite',
            'cycle_translator_render',
            'cycle_encoded_orient',
            'cycle_encoded_depth',
        ]

        depth_mask_map = {
            'depth': 'hairmask',
            'injected_depth': 'injected_mask',
            'cycle_encoded_depth': 'injected_mask',
        }
        orient_keys = {'strand', 'injected_orient', 'cycle_encoded_orient'}
        depth_keys = set(depth_mask_map.keys())
        mask_keys = {'injected_mask', 'injected_inverse_intersection_mask', 'face_cleanup_hole_mask'}

        prepared = []
        for key in image_keys:
            if key not in outputs:
                continue

            tensor = outputs[key].detach().cpu()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)

            if key in orient_keys:
                tensor = self.strand2vis(tensor)
            elif key in depth_keys:
                mask_key = depth_mask_map[key]
                mask = outputs.get(mask_key)
                if mask is None:
                    mask = torch.ones_like(tensor[:, :1])
                else:
                    mask = mask.detach().cpu()
                    if mask.ndim == 3:
                        mask = mask.unsqueeze(0)
                tensor = self.depth2vis(tensor, mask)
            elif key in mask_keys:
                if tensor.shape[1] == 1:
                    tensor = tensor.repeat(1, 3, 1, 1)
                else:
                    tensor = tensor[:, :3]
            else:
                tensor = tensor.clamp(0.0, 1.0)

            prepared.append(make_grid(tensor, nrow=1))

        if not prepared:
            return

        grid = torch.cat(prepared, dim=2)

        grid = grid.permute(1,2,0).cpu().numpy()*255.0
        grid = np.clip(grid, 0, 255)
        grid = grid.astype(np.uint8)
        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

        cv2.imwrite(save_path, grid)

    def save_visualizations_2(self, outputs, save_path):
        """Save reconstructed image too"""
        if 'img' in outputs and 'reconstructed_img' in outputs and 'masked_1st_path' in outputs:
            outputs['overlap_image'] = outputs['img'] * 0.7 + outputs['reconstructed_img'] * 0.3
            outputs['overlap_image_pixels'] = outputs['img'] * 0.7 +  0.3 * outputs['masked_1st_path']
            diff_signed = outputs['reconstructed_img'] - outputs['img']
            diff_signed = (diff_signed + 1) / 2  # [-1,1] to [0,1] for visualization
            outputs['diff_image_signed'] = diff_signed
        # outputs: depth are in [0, 1], strand are in [-1, 1]
        
        # image_keys = ['img', 'strand', 'depth', 'reconstructed_img', 'overlap_image']
        image_keys = ['img', 'strand', 'depth', 'diff_image_signed']
        if 'depth' in outputs:
            depth_vis = self.depth2vis(outputs['depth'], outputs['hairmask'])
            outputs['depth'] = depth_vis.to(outputs['depth'].device)
        if 'strand' in outputs:     # use hsv instead
            strand_vis = self.strand2vis(outputs['strand'])     # in [0, 1]
            outputs['strand'] = strand_vis.to(outputs['strand'].device)
        # print(outputs['reconstructed_img'].shape)   # B, 3, H, W
        # print(torch.equal(outputs['reconstructed_img'], outputs['img']))        # false
        # mae = (outputs['reconstructed_img'] - outputs['img']).abs().mean()
        # print("MAE:", mae.item())

        nrows = [1 if '2nd_path' not in key else 4 * self.config.train.Ke for key in image_keys]

        grid = torch.cat([make_grid(outputs[key].detach().cpu(), nrow=nr) for key, nr in zip(image_keys, nrows) if key in outputs.keys()], dim=2)

        grid = grid.permute(1,2,0).cpu().numpy()*255.0
        grid = np.clip(grid, 0, 255)
        grid = grid.astype(np.uint8)
        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

        cv2.imwrite(save_path, grid)

    def create_visualizations(self, batch, outputs):
        # batch keys are already in device, so no need to move them
        # outputs are in cpu, so we need to move them to device if we want to use them in the renderer

        visualizations = {}
        visualizations['img'] = batch['img']
        visualizations['hairmask'] = batch.get('encoder_hairmask', batch['hairmask'])
        visualizations['strand'] = outputs['strand']
        if self.config.arch.depth_branch:
            visualizations['depth'] = outputs['depth']
        if 'flame_render_image' in outputs:
            visualizations['flame_mesh_render'] = outputs['flame_render_image']
            visualizations['flame_mesh_overlay'] = (
                batch['img'].detach().cpu() * 0.65 + outputs['flame_render_image'].detach().cpu() * 0.35
            ).clamp(0.0, 1.0)
        if 'face_cleanup_render' in outputs:
            visualizations['face_cleanup_render'] = outputs['face_cleanup_render']
        if 'face_cleanup_before' in outputs:
            visualizations['face_cleanup_before'] = outputs['face_cleanup_before']
        if 'face_cleanup_after' in outputs:
            visualizations['face_cleanup_after'] = outputs['face_cleanup_after']
        if 'face_cleanup_input' in outputs:
            visualizations['face_cleanup_input'] = outputs['face_cleanup_input']
        if 'face_cleanup_base' in outputs:
            visualizations['face_cleanup_base'] = outputs['face_cleanup_base']
        if 'face_cleanup_hole_mask' in outputs:
            visualizations['face_cleanup_hole_mask'] = outputs['face_cleanup_hole_mask']

        if self.config.arch.enable_fuse_generator:
            if 'reconstructed_img' in outputs:
                visualizations['translator_image_render'] = outputs['reconstructed_img']
            if 'masked_1st_path' in outputs:
                visualizations['sparse_pixel_map'] = outputs['masked_1st_path']

        if 'hair_render_image' in outputs:
            visualizations['injected_mask'] = outputs['hair_render_image'][:, :1]
            visualizations['injected_orient'] = outputs['hair_render_image'][:, :3]
            if outputs['hair_render_image'].shape[1] >= 4:
                visualizations['injected_depth'] = outputs['hair_render_image'][:, 3:4]
        if 'hair_render_inverse_intersection_mask' in outputs:
            visualizations['injected_inverse_intersection_mask'] = outputs['hair_render_inverse_intersection_mask']
        if 'hair_render_sparse_color_map' in outputs:
            visualizations['injected_sparse_pixel_map'] = outputs['hair_render_sparse_color_map']
        if 'injected_composite' in outputs:
            visualizations['injected_composite'] = outputs['injected_composite']
        if 'hair_cycle_reconstruction' in outputs:
            visualizations['cycle_translator_render'] = outputs['hair_cycle_reconstruction']
        if 'hair_cycle_strand' in outputs:
            cycle_mask = outputs['hair_cycle_strand'][:, :1]
            if 'injected_mask' in visualizations:
                cycle_mask = visualizations['injected_mask']
            visualizations['cycle_encoded_orient'] = torch.cat(
                [cycle_mask, outputs['hair_cycle_strand'][:, 1:3]],
                dim=1,
            )
        if 'hair_cycle_depth' in outputs:
            visualizations['cycle_encoded_depth'] = outputs['hair_cycle_depth']

        for key in visualizations.keys():
            visualizations[key] = visualizations[key].detach().cpu()

        return visualizations

    def save_model(self, state_dict, save_path):
        # remove everything that is not hair_encoder or smirk_generator
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if key.startswith('hair_encoder') or key.startswith('smirk_generator'):
                new_state_dict[key] = state_dict[key]

        torch.save(new_state_dict, save_path)

    def create_base_encoder(self):
        self.base_encoder = copy.deepcopy(self._hair_encoder_module())
        self.base_encoder.eval()
        for p in self.base_encoder.parameters():
            p.requires_grad = False

    def load_model(self, resume, load_fuse_generator=True, load_encoder=True, device='cuda'):
        loaded_state_dict = torch.load(resume, map_location=device)

        print(f'Loading checkpoint from {resume}, load_encoder={load_encoder}, load_fuse_generator={load_fuse_generator}')

        filtered_state_dict = {}
        for key, value in loaded_state_dict.items():
            if load_encoder and (key.startswith('hair_encoder') or key.startswith('smirk_encoder')):
                if key.startswith('smirk_encoder'):
                    new_key = key.replace('smirk_encoder', 'hair_encoder', 1)
                    if new_key in filtered_state_dict:
                        continue
                else:
                    new_key = key
                filtered_state_dict[new_key] = value
            if load_fuse_generator and key.startswith('smirk_generator'):
                filtered_state_dict[key] = value
            
        self.load_state_dict(filtered_state_dict, strict=False) # set it false because it asks for mica and other models apart from hair_encoder and smirk_generator

    def set_freeze_status(self, config, batch_idx, epoch_idx):
        #self.config.train.freeze_encoder_in_first_path = False
        #self.config.train.freeze_generator_in_first_path = False
        self.config.train.freeze_encoder_in_second_path = False
        self.config.train.freeze_generator_in_second_path = False

        #decision_idx = batch_idx if config.train.freeze_schedule.per_iteration else epoch_idx
        decision_idx_second_path = batch_idx #epoch_idx 

        self.config.train.freeze_encoder_in_second_path = decision_idx_second_path % 2 == 0
        self.config.train.freeze_generator_in_second_path = decision_idx_second_path % 2 == 1

    


if __name__ == "__main__":
    """ 
    python -m src.base_trainer
    """
    import types
    import math
    import torch
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    # dummy config for trainer
    dummy_cfg = types.SimpleNamespace()
    dummy_cfg.train = types.SimpleNamespace(
        log_path=None,
        optimize_hairstrand=False,
        optimize_hairdepth=False,
    )

    dummy_cfg.device = "cpu"
    dummy_cfg.image_size = (64, 64)

    trainer = BaseHairTrainer(config=dummy_cfg)

    # Some test angles
    H, W = 64, 64
    angles_deg = [0, 60, 120, 180, 240, 300]
    B = len(angles_deg)

    # strand: (B, 3, H, W)
    strand = torch.zeros(B, 3, H, W, dtype=torch.float32)

    # Channel 0 = hair mask
    strand[:, 0, :, :] = 1.0

    for i, theta_deg in enumerate(angles_deg):
        theta_rad = math.radians(theta_deg)

        # green_c = x, blue_c = y (down positive)
        green_c = math.cos(theta_rad)     # [-1, 1]
        blue_c = -math.sin(theta_rad)     # [-1, 1]     # makes y down, strand2vis expects y-down as input

        strand[i, 1, :, :] = green_c
        strand[i, 2, :, :] = blue_c

    # strand2vis returns RGB values for display, in [0, 255]
    vis = trainer.strand2vis(strand)  # (B, 3, H, W)

    grid = make_grid(vis, nrow=B)
    grid = grid.permute(1,2,0).cpu().numpy()*255.0
    grid = np.clip(grid, 0, 255).astype(np.uint8)
    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

    # Add text
    H, W, _ = grid.shape
    patch_w = W // len(angles_deg)

    for i, deg in enumerate(angles_deg):
        x = int(i * patch_w + 5)
        y = int(H - 10)
        
        cv2.putText(grid, f"{deg}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite("/gpfs/projects/CascanteBonillaGroup/thinguyen/storage/smirk/strand_hsv_test.png", grid)

    # plt.imshow(img)
    # plt.title(f"degrees {angles_deg}")
    # plt.axis("off")
    # plt.savefig("/gpfs/projects/CascanteBonillaGroup/thinguyen/storage/smirk/strand_hsv_test.png", bbox_inches="tight")

    
