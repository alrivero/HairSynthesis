import json
import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import HairStepEncoder
import argparse
import os
import src.utils.masking as masking_utils
from datasets.base_dataset import create_mask
import torch.nn.functional as F
from pathlib import Path
import imageio.v2 as imageio
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='samples/mead_90.png', help='Path to the input image/video')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='trained_models/SMIRK_em1.pt', help='Path to the checkpoint')
    parser.add_argument('--out_path', type=str, default='output', help='Path to save the output (will be created if not exists)')
    parser.add_argument('--use_smirk_generator', action='store_true', help='Use SMIRK neural image to image translator to reconstruct the image')
    parser.add_argument('--render_orig', action='store_true', help='Present the result w.r.t. the original image/video size')

    parser.add_argument('--hairmask_path', type=str)
    parser.add_argument('--bodymask_path', type=str)

    args = parser.parse_args()

    image_size = 224

    with open("/gpfs/projects/CascanteBonillaGroup/thinguyen/datasets/FFHQ256/metadata/splits/splits_smirk.json", 'r') as f:
        val_list = json.load(f)['validation']
    # val_list = [val_list[0]]
    val_list = val_list[:20]

    # ----------------------- initialize configuration ----------------------- #
    smirk_encoder = HairStepEncoder().to(args.device)
    if args.device == 'cuda':
        checkpoint = torch.load(args.checkpoint)
    else:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k} # checkpoint includes both smirk_encoder and smirk_generator

    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()

    if args.use_smirk_generator:
        from src.smirk_generator import SmirkGenerator
        smirk_generator = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5).to(args.device)

        checkpoint_generator = {k.replace('smirk_generator.', ''): v for k, v in checkpoint.items() if 'smirk_generator' in k} # checkpoint includes both smirk_encoder and smirk_generator
        smirk_generator.load_state_dict(checkpoint_generator)
        smirk_generator.eval()

    out_path = Path(args.checkpoint).parent / 'vis' / 'strand'
    out_path.mkdir(parents=True, exist_ok=True)
    print(val_list)
    for i, img_file in enumerate(val_list):
        # ---- visualize the results ---- #
        image = cv2.imread(str(Path(args.input_path) / img_file))
        orig_image_height, orig_image_width, _ = image.shape

        cropped_image = image
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        # cropped_image = cv2.resize(cropped_image, (224,224))
        cropped_image = torch.tensor(cropped_image).permute(2,0,1).unsqueeze(0).float()/255.0
        cropped_image = cropped_image.to(args.device)

        hairmask = (imageio.imread(Path(args.hairmask_path) / img_file)/255.>0.5)[:,:,None]
        hairmask = torch.tensor(hairmask).permute(2,0,1).unsqueeze(0).float()
        hairmask = hairmask.to(args.device)

        bodymask = (imageio.imread(Path(args.bodymask_path) / img_file)[:,:,0]/255.>0.5)[:,:,None]
        bodymask = torch.tensor(bodymask).permute(2,0,1).unsqueeze(0).float()
        bodymask = bodymask*(1-hairmask)
        bodymask = bodymask.to(args.device)

        print(hairmask.shape, bodymask.shape, cropped_image.shape)
        with torch.no_grad():
            outputs = smirk_encoder(cropped_image, hairmask, bodymask)
        print("smirk_encoder output", outputs['strand_params'].shape, outputs['depth_params'].shape)

        # np.save(Path(args.input_path).parent / f"{Path(args.input_path).stem}_strands.npy", outputs['strand_params'].detach().cpu().numpy())
        # np.save(Path(args.input_path).parent / f"{Path(args.input_path).stem}_depth.npy", outputs['depth_params'].detach().cpu().numpy())

        # Strand maps
        # Red = binary (0: background, 0.5: face/body, 1: hair)
        # O(x) = (M(x), O_{2D}/2 + 0.5)

        strand_map = outputs['strand_params'].squeeze(0).detach().cpu().numpy()
        print(strand_map.shape)
        # normalized direction, should be in [-1, 1]
        green_c = (strand_map[1] - 0.5) * 2       # x (to right)
        blue_c = (strand_map[2] - 0.5) * 2        # y (down)

        # compute angle 
        theta = np.arctan2(-blue_c, green_c)     # in [-pi, pi]

        # convert to 0-360
        theta = (theta + 2*np.pi) % (2*np.pi)
        theta = (2*np.pi - theta) % (2*np.pi)   # counter clockwise, 90 on the left

        # hue mapping
        h = theta / (2*np.pi)
        s = np.ones_like(h)
        v = np.ones_like(h)

        hsv = np.stack([h, s, v], axis=-1)
        rgb = cv2.cvtColor((hsv*255).astype(np.uint8), cv2.COLOR_HSV2RGB)

        # apply hair mask
        mask = (strand_map[0] > 0.75).astype(np.float32)
        rgb = (rgb * mask[...,None]).astype(np.uint8)

        # plt.subplot(len(val_list), 2, i*2+1)
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(img_file.split('.')[0])
        plt.axis('off')

        # plt.subplot(len(val_list), 2, i*2+2)
        plt.subplot(1, 2, 2)
        plt.imshow(rgb)
        plt.axis('off')
    
        # plt.savefig(str(out_path/f"strands_{len(val_list)}.png"))
        plt.savefig(str(out_path/img_file))
        plt.close()


        # if args.render_orig:
        #     # if args.crop:
        #     #     rendered_img_numpy = (rendered_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0).astype(np.uint8)               
        #     #     rendered_img_orig = warp(rendered_img_numpy, tform, output_shape=(orig_image_height, orig_image_width), preserve_range=True).astype(np.uint8)
        #     #     # back to pytorch to concatenate with full_image
        #     #     rendered_img_orig = torch.Tensor(rendered_img_orig).permute(2,0,1).unsqueeze(0).float()/255.0
        #     # else:
        #     #     rendered_img_orig = F.interpolate(rendered_img, (orig_image_height, orig_image_width), mode='bilinear').cpu()

        #     rendered_img_orig = F.interpolate(rendered_img, (orig_image_height, orig_image_width), mode='bilinear').cpu()
        #     full_image = torch.Tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float()/255.0
        #     grid = torch.cat([full_image, rendered_img_orig], dim=3)
        # else:
        # #     grid = torch.cat([cropped_image, rendered_img], dim=3)


        # # ---- create the neural renderer reconstructed img ---- #
        # if args.use_smirk_generator:
        #     if (kpt_mediapipe is None):
        #         print('Could not find landmarks for the image using mediapipe and cannot create the hull mask for the smirk generator. Exiting...')
        #         exit()

        #     mask_ratio_mul = 5
        #     mask_ratio = 0.01
        #     mask_dilation_radius = 10

        #     hull_mask = create_mask(cropped_kpt_mediapipe, (224, 224))

        #     face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()  

        #     rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
        #     tmask_ratio = mask_ratio * mask_ratio_mul # upper bound on the number of points to sample
            
        #     npoints, _ = masking_utils.mesh_based_mask_uniform_faces(renderer_output['transformed_vertices'], # sample uniformly from the mesh
        #                                                             flame_faces=flame.faces_tensor,
        #                                                             face_probabilities=face_probabilities,
        #                                                             mask_ratio=tmask_ratio)
            
        #     pmask = torch.zeros_like(rendered_mask)                
        #     rsing = torch.randint(0, 2, (npoints.size(0),)).to(npoints.device) * 2 - 1
        #     rscale = torch.rand((npoints.size(0),)).to(npoints.device) * (mask_ratio_mul - 1) + 1
        #     rbound =(npoints.size(1) * (1/mask_ratio_mul) * (rscale ** rsing)).long()

        #     for bi in range(npoints.size(0)):
        #         pmask[bi, :, npoints[bi, :rbound[bi], 1], npoints[bi, :rbound[bi], 0]] = 1
            
        #     hull_mask = torch.from_numpy(hull_mask).type(dtype = torch.float32).unsqueeze(0).to(args.device)

        #     extra_points = cropped_image * pmask
        #     masked_img = masking_utils.masking(cropped_image, hull_mask, extra_points, mask_dilation_radius, rendered_mask=rendered_mask)

        #     smirk_generator_input = torch.cat([rendered_img, masked_img], dim=1)

        #     reconstructed_img = smirk_generator(smirk_generator_input)

        #     if args.render_orig:
        #         if args.crop:
        #             reconstructed_img_numpy = (reconstructed_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0).astype(np.uint8)               
        #             reconstructed_img_orig = warp(reconstructed_img_numpy, tform, output_shape=(orig_image_height, orig_image_width), preserve_range=True).astype(np.uint8)
        #             # back to pytorch to concatenate with full_image
        #             reconstructed_img_orig = torch.Tensor(reconstructed_img_orig).permute(2,0,1).unsqueeze(0).float()/255.0
        #         else:
        #             reconstructed_img_orig = F.interpolate(reconstructed_img, (orig_image_height, orig_image_width), mode='bilinear').cpu()

        #         grid = torch.cat([grid, reconstructed_img_orig], dim=3)
        #     else:
        #         grid = torch.cat([grid, reconstructed_img], dim=3)

        # grid_numpy = grid.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0
        # grid_numpy = grid_numpy.astype(np.uint8)
        # grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_BGR2RGB)

        # if not os.path.exists(args.out_path):
        #     os.makedirs(args.out_path)

        # image_name = args.input_path.split('/')[-1]

        # cv2.imwrite(f"{args.out_path}/{image_name}", grid_numpy)

