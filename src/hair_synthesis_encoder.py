import torch
from torch import nn

from external.HairStep.lib.model.img2hairstep.UNet import Model as StrandModel
from external.HairStep.lib.model.img2hairstep.hourglass import Model as DepthModel
from utils.torchutils import torch_nanminmax


class HairSynthesisEncoder(nn.Module):
    """Wrapper around HairStep's strand and optional depth encoders."""

    def __init__(
        self,
        img2strand_ckpt=None,
        img2depth_ckpt=None,
        config=None,
    ) -> None:
        super().__init__()

        self.strand_encoder = StrandModel()
        if img2strand_ckpt is not None:
            self.strand_encoder.load_state_dict(torch.load(img2strand_ckpt))

        if config.arch.depth_branch:
            self.depth_encoder = DepthModel()
            if img2depth_ckpt is not None:
                depth_state = torch.load(img2depth_ckpt)
                # HairStep trained DepthModel under DataParallel, so strip the module prefix.
                if "module." in list(depth_state.keys())[0]:
                    depth_state = {k.replace("module.", ""): v for k, v in depth_state.items()}
                self.depth_encoder.load_state_dict(depth_state)

        self.config = config

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

            abs_max_nonnan = torch.abs(
                torch_nanminmax(depth_pred, 'max', dim=(1, 2, 3), keepdim=True)
            )
            abs_min_nonnan = torch.abs(
                torch_nanminmax(depth_pred, 'min', dim=(1, 2, 3), keepdim=True)
            )

            depth_pred_masked = depth_pred * hair_mask - (
                1 - hair_mask
            ) * (abs_max_nonnan + abs_min_nonnan)
            max_val = torch_nanminmax(depth_pred_masked, 'max', dim=(1, 2, 3), keepdim=True)
            min_val = torch_nanminmax(
                depth_pred_masked + 2 * (1 - hair_mask) * (abs_max_nonnan + abs_min_nonnan),
                'min',
                dim=(1, 2, 3),
                keepdim=True,
            )

            depth_pred_norm = (depth_pred_masked - min_val) / (max_val - min_val) * hair_mask
            depth_pred_norm = depth_pred_norm.clamp(0.0, 1.0)

            return {
                'strand_params': strand_pred,
                'depth_params': depth_pred_norm,
            }

        return {
            'strand_params': strand_pred,
        }
