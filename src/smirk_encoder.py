import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import timm
import torchvision.transforms as transforms

# import sys
# sys.path.append("/gpfs/projects/CascanteBonillaGroup/thinguyen/storage/HairStep")
from external.HairStep.lib.model.img2hairstep.UNet import Model as StrandModel
from external.HairStep.lib.model.img2hairstep.hourglass import Model as DepthModel
from utils.torchutils import torch_nanminmax


def create_backbone(backbone_name, pretrained=True):
    backbone = timm.create_model(backbone_name, 
                        pretrained=pretrained,
                        features_only=True)
    feature_dim = backbone.feature_info[-1]['num_chs']
    return backbone, feature_dim

class PoseEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
              
        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_small_minimal_100')
        
        self.pose_cam_layers = nn.Sequential(
            nn.Linear(feature_dim, 6)
        )

        self.init_weights()

    def init_weights(self):
        self.pose_cam_layers[-1].weight.data *= 0.001
        self.pose_cam_layers[-1].bias.data *= 0.001

        self.pose_cam_layers[-1].weight.data[3] = 0
        self.pose_cam_layers[-1].bias.data[3] = 7


    def forward(self, img):
        features = self.encoder(img)[-1]
            
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)

        outputs = {}

        pose_cam = self.pose_cam_layers(features).reshape(img.size(0), -1)
        outputs['pose_params'] = pose_cam[...,:3]
        outputs['cam'] = pose_cam[...,3:]

        return outputs


class ShapeEncoder(nn.Module):
    def __init__(self, n_shape=300) -> None:
        super().__init__()

        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_large_minimal_100')

        self.shape_layers = nn.Sequential(
            nn.Linear(feature_dim, n_shape)
        )

        self.init_weights()


    def init_weights(self):
        self.shape_layers[-1].weight.data *= 0
        self.shape_layers[-1].bias.data *= 0


    def forward(self, img):
        features = self.encoder(img)[-1]
            
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)

        parameters = self.shape_layers(features).reshape(img.size(0), -1)

        return {'shape_params': parameters}


class ExpressionEncoder(nn.Module):
    def __init__(self, n_exp=50) -> None:
        super().__init__()

        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_large_minimal_100')
        
        self.expression_layers = nn.Sequential( 
            nn.Linear(feature_dim, n_exp+2+3) # num expressions + jaw + eyelid
        )

        self.n_exp = n_exp
        self.init_weights()


    def init_weights(self):
        self.expression_layers[-1].weight.data *= 0.1
        self.expression_layers[-1].bias.data *= 0.1


    def forward(self, img):
        features = self.encoder(img)[-1]
            
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)


        parameters = self.expression_layers(features).reshape(img.size(0), -1)

        outputs = {}

        outputs['expression_params'] = parameters[...,:self.n_exp]
        outputs['eyelid_params'] = torch.clamp(parameters[...,self.n_exp:self.n_exp+2], 0, 1)
        outputs['jaw_params'] = torch.cat([F.relu(parameters[...,self.n_exp+2].unsqueeze(-1)), 
                                           torch.clamp(parameters[...,self.n_exp+3:self.n_exp+5], -.2, .2)], dim=-1)

        return outputs


class SmirkEncoder(nn.Module):
    def __init__(self, n_exp=50, n_shape=300) -> None:
        super().__init__()

        self.pose_encoder = PoseEncoder()

        self.shape_encoder = ShapeEncoder(n_shape=n_shape)

        self.expression_encoder = ExpressionEncoder(n_exp=n_exp) 

    def forward(self, img):
        pose_outputs = self.pose_encoder(img)
        shape_outputs = self.shape_encoder(img)
        expression_outputs = self.expression_encoder(img)

        outputs = {}
        outputs.update(pose_outputs)
        outputs.update(shape_outputs)
        outputs.update(expression_outputs)

        return outputs


class HairStepEncoder(nn.Module):
    """Using Hairstep encoder"""
    
    def __init__(self,
                 img2strand_ckpt,
                 img2depth_ckpt,
                 ) -> None:
        super().__init__()
        
        # HairStep SAM mask is precomputed

        # HairStep UNet strand model
        self.strand_encoder = StrandModel()
        self.strand_encoder.load_state_dict(torch.load(img2strand_ckpt))
        # self.strand_encoder.eval()

        self.depth_encoder = DepthModel()
        depth_state = torch.load(img2depth_ckpt)
        # In Hairstep, DepthModel was loaded using torch.nn.DataParallel so need to remove module. prefix
        if "module." in list(depth_state.keys())[0]:
            depth_state = {k.replace("module.", ""): v for k, v in depth_state.items()}
        self.depth_encoder.load_state_dict(depth_state)
        # self.depth_encoder.eval()

    def forward(self, img, hair_mask, body_mask):
        """Extract HairStep features

        Args:
            imgs (torch.Tensor): from resize folder.                                (B, C, H, W)
            hair_masks (torch.Tensor): from seg folder, values are 0 or 1.          (B, 1, H, W)
            body_masks (torch.Tensor): from body_mask folder, values are 0 or 1.    (B, 1, H, W)
        """
        # masked image for both img2strand and img2depth
        img = img * hair_mask                       # (B, C, H, W)

        #################### img2strand.py
        body = body_mask * (1 - hair_mask)          # (B, 1, H, W)

        strand_pred = self.strand_encoder(img)      # (B, C, H, W)
        strand_pred = strand_pred.clamp(0., 1.)     # (B, C, H, W)
        strand_pred = torch.cat([hair_mask+body*0.5, strand_pred*hair_mask], dim=1) # (B, 1, H, W) + (B, 2, H, W) ->(B, 3, H, W)

        #################### img2depth.py
        depth_pred = self.depth_encoder(img)        # (B, 1, H, W)

        # Normalization: min and max should be from mask region only the mask region can have 0 so we cannot simply taking min where depth_pred not 0 in mask. Do this by assigning a large constant to the background to avoid including that when finding the max, min
        abs_max_nonnan = torch.abs(torch_nanminmax(depth_pred, 'max', dim=(1, 2, 3), keepdim=True))     # (B, 1, 1, 1)
        abs_min_nonnan = torch.abs(torch_nanminmax(depth_pred, 'min', dim=(1, 2, 3), keepdim=True))     # (B, 1, 1, 1)
        # print("abs_max_nonnan", abs_max_nonnan.shape, "abs_min_nonnan", abs_min_nonnan.shape)

        depth_pred_masked = depth_pred * hair_mask - (1 - hair_mask) * (abs_max_nonnan + abs_min_nonnan)    # (B, 1, H, W)
        # print("depth_pred_masked", depth_pred_masked.shape)
        max_val = torch_nanminmax(depth_pred_masked, 'max', dim=(1, 2, 3), keepdim=True)        # (B, 1, 1, 1)
        min_val = torch_nanminmax(depth_pred_masked + 2 * (1 - hair_mask) * (abs_max_nonnan + abs_min_nonnan), 'min', dim=(1, 2, 3), keepdim=True)        # (B, 1, 1, 1)
        # print("max_val", max_val.shape, "min_val", min_val.shape)

        depth_pred_norm = (depth_pred_masked - min_val) / (max_val - min_val) * hair_mask   # (B, 1, H, W)
        depth_pred_norm = depth_pred_norm.clamp(0., 1.)                                     # (B, 1, H, W)
        # print("depth_pred_norm", depth_pred_norm.shape)

        hairstep = {
            'strand_params': strand_pred,
            'depth_params': depth_pred_norm
        }

        return hairstep


if __name__ == "__main__":
    a = HairStepEncoder(
        img2strand_ckpt="/gpfs/projects/CascanteBonillaGroup/thinguyen/storage/smirk/external/HairStep/checkpoints/img2hairstep/img2strand.pth",
        img2depth_ckpt="/gpfs/projects/CascanteBonillaGroup/thinguyen/storage/smirk/external/HairStep/checkpoints/img2hairstep/img2depth.pth"
    )
    # print(a)
