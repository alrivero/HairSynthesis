import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
import cv2
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

        # Setup logger to save training loss to a file
        if config.train.log_path is not None:
            log_file = os.path.join(config.train.log_path, 'logs.log')
            assert not os.path.exists(log_file), f"{os.path.join(log_file)} existed!"

            self.logger = logging.getLogger()
            logging.getLogger('matplotlib.font_manager').disabled = True
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler(os.path.join(log_file))
            fh.setLevel(logging.INFO)
            self.logger.addHandler(fh)

    def logging(self, batch_idx, losses, phase):
        # ---------------- logging ---------------- #
        if self.config.train.log_losses_every > 0 and batch_idx % self.config.train.log_losses_every == 0:
            # print losses in one line
            loss_str = ''
            for k, v in losses.items():
                loss_str += f'{k}: {v:.6f} '
            print(loss_str)

    def logging_epoch(self, epoch_idx, losses, phase):
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
        train_losses = losses_hist['train']
        val_losses = losses_hist['val']
        # print("plot_losses", train_losses)

        losses_name = list(train_losses.keys())
        for ln in losses_name:
            self.plot_curve(train_losses[ln], val_losses[ln], 'losses', ln, os.path.join(self.config.train.log_path, "loss_plots", f"loss_{ln}.png"))
            
            if plot_batch:
                self.plot_curve_single(losses_hist['train_b'][ln], 'train', 'losses', ln, os.path.join(self.config.train.log_path, "loss_plots", f"loss_{ln}_train_b.png"))
                self.plot_curve_single(losses_hist['val_b'][ln], 'val', 'losses', ln, os.path.join(self.config.train.log_path, "loss_plots", f"loss_{ln}_val_b.png"))

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
            if self.config.train.optimize_hairstrand:
                params += list(self.smirk_encoder.strand_encoder.parameters()) 
            if self.config.arch.depth_branch:
                if self.config.train.optimize_hairdepth:
                    params += list(self.smirk_encoder.depth_encoder.parameters())

            self.encoder_optimizer = torch.optim.Adam(params, lr= encoder_scale * self.config.train.lr)
                
        # cosine schedulers for both optimizers - per iterations
        self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_optimizer, T_max=n_steps, eta_min=0.01 * encoder_scale * self.config.train.lr)

        if self.config.arch.enable_fuse_generator:
            if hasattr(self, 'fuse_generator_optimizer'):
                for g in self.smirk_generator_optimizer.param_groups:
                    g['lr'] = self.config.train.lr
            else:
                self.smirk_generator_optimizer = torch.optim.Adam(self.smirk_generator.parameters(), lr= self.config.train.lr, betas=(0.5, 0.999))

            
            self.smirk_generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.smirk_generator_optimizer, T_max=n_steps, eta_min=0.01 * self.config.train.lr)
        
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

            masked_img = depth_b * mask_b + (1 - mask_b) * ((depth_b * mask_b - (1 - mask_b) * 100000).max())  # set the value of un-mask to the min-val in mask    # (H, W)
            norm_masked_depth = masked_img / (np.nanmax(masked_img) - np.nanmin(masked_img))  # norm    # (H, W)

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

            # apply hair mask
            mask = (strand_map[0] > 0.75).astype(np.float32)
            rgb = (rgb * mask[...,None])       # (H, W, C)
            vis_b = torch.from_numpy(rgb).permute(2,0,1)      # (C, H, W)
            
            strand_vis.append(vis_b)
        
        return torch.stack(strand_vis, dim=0)

    def save_visualizations(self, outputs, save_path):
        if 'img' in outputs and 'rendered_img' in outputs and 'masked_1st_path' in outputs:
            # TODO: overlap generated img with original
            outputs['overlap_image'] = outputs['img'] * 0.7 + outputs['rendered_img'] * 0.3
            outputs['overlap_image_pixels'] = outputs['img'] * 0.7 +  0.3 * outputs['masked_1st_path']
        # outputs: depth are in [0, 1], strand are in [-1, 1]
        
        image_keys = ['img', 'strand', 'depth']
        if 'depth' in outputs:
            depth_vis = self.depth2vis(outputs['depth'], outputs['hairmask'])
            outputs['depth'] = depth_vis.to(outputs['depth'].device)
        if 'strand' in outputs:     # use hsv instead
            strand_vis = self.strand2vis(outputs['strand'])     # in [0, 1]
            outputs['strand'] = strand_vis.to(outputs['strand'].device)

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
        visualizations['hairmask'] = batch['hairmask']
        visualizations['strand'] = outputs['strand']
        if self.config.arch.depth_branch:
            visualizations['depth'] = outputs['depth']
        # visualizations['rendered_img'] = outputs['rendered_img']

        # 2. Base model
        base_output = self.base_encoder(batch['img'], batch['hairmask'], batch['bodymask'])
        visualizations['strand_base'] = base_output['strand_params']
        if self.config.arch.depth_branch:
            visualizations['depth_base'] = base_output['depth_params']
    
        if self.config.arch.enable_fuse_generator:
            visualizations['reconstructed_img'] = outputs['reconstructed_img']
            visualizations['masked_1st_path'] = outputs['masked_1st_path']
            visualizations['loss_img'] = outputs['loss_img']

        for key in visualizations.keys():
            visualizations[key] = visualizations[key].detach().cpu()

        if self.config.train.loss_weights['cycle_loss'] > 0:
            if '2nd_path' in outputs:
                visualizations['2nd_path'] = outputs['2nd_path']

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
        for p in self.base_encoder.parameters():
            p.requires_grad = False

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

    


