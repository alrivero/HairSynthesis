"""
Compute L2 loss on HiSA between orientation map of HairStep model and current model.
"""
import os
import sys
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

from src.smirk_trainer import SmirkHairTrainer
from datasets.data_utils import load_dataloaders_HiSA


def compute_orientation_dot(orient_gt, orient_current, hairmask):
    """
    l2 loss
    
    orient_gt, orient_current: (B, 3, H, W)     # 3 channels: mask, x, y
    hairmask: (B, 1, H, W), values 0 or 1
    """
    # print("orient_hs", orient_gt.max(), orient_gt.min())
    # print("orient_current", orient_current.max(), orient_current.min())
    device = orient_gt.device
    orient_current = orient_current.to(device)
    hairmask = hairmask.to(device)

    B, C, H, W = orient_current.shape
    assert C == 3, f"Expected 3 channels (mask,x,y), got {C}"

    # mask only 0 or 1
    unique_vals = torch.unique(hairmask)
    assert torch.all((unique_vals == 0) | (unique_vals == 1)), f"hairmask must contain only 0 and 1, but got values: {unique_vals}"
    hairmask = hairmask.squeeze(1).bool()      # (B, H, W)

    # take (u,v) channels only
    G = orient_gt[:, 1:3, :, :]             # (B, 2, H, W)
    M = orient_current[:, 1:3, :, :]           # (B, 2, H, W)

    # norm
    M = M / (torch.norm(M, dim=1, keepdim=True) + 1e-10)  # (B, 2, H, W)
    G = G / (torch.norm(G, dim=1, keepdim=True) + 1e-10)  # (B, 2, H, W)

    # dot product
    dot = (M * G).sum(dim=1)
    dot_abs = dot.abs()

    # mask to only hair region
    scores = []
    for b in range(B):
        mask_b = hairmask[b]                   # (H, W), bool
        if not mask_b.any():
            scores.append(float("nan"))
        else:
            scores.append(dot_abs[b][mask_b].mean().item())
    return scores

def compute_orientation_l2(orient_hs, orient_current, hairmask):
    """
    l2 loss
    
    orient_hs, orient_current: (B, 3, H, W)     # 3 channels: mask, x, y
    hairmask: (B, 1, H, W), values 0 or 1
    """
    device = orient_hs.device
    orient_current = orient_current.to(device)
    hairmask = hairmask.to(device)

    B, C, H, W = orient_hs.shape
    assert C == 3, f"Expected 3 channels (mask,x,y), got {C}"

    # mask only 0 or 1
    unique_vals = torch.unique(hairmask)
    assert torch.all((unique_vals == 0) | (unique_vals == 1)), f"hairmask must contain only 0 and 1, but got values: {unique_vals}"
    hairmask = hairmask.squeeze(1).bool()      # (B, H, W)

    # take (u,v) channels only
    hs = orient_hs[:, 1:3, :, :]             # (B, 2, H, W)
    curr = orient_current[:, 1:3, :, :]           # (B, 2, H, W)

    diff = (hs - curr) ** 2                        # (B, 2, H, W)
    l2_map = diff.sum(dim=1)                   # (B, H, W) per-pixel squared L2

    l2_batch = []
    for b in range(B):
        mask_b = hairmask[b]                   # (H, W), bool
        if not mask_b.any():
            l2_batch.append(float("nan"))
        else:
            l2_batch.append(l2_map[b][mask_b].mean().item())
    return l2_batch

def compute_depth_l2(depth_hs, depth_current, hairmask):
    """
    l2 loss

    depth_hs, depth_current: (B, 1, H, W)
    hairmask: (B, 1, H, W), values 0 or 1
    """
    device = depth_hs.device
    depth_current = depth_current.to(device)
    hairmask = hairmask.to(device)

    B, C, H, W = depth_hs.shape
    assert C == 1, f"Expected 1 channels, got {C}"

    # mask only 0 or 1
    unique_vals = torch.unique(hairmask)
    assert torch.all((unique_vals == 0) | (unique_vals == 1)), f"hairmask must contain only 0 and 1, but got values: {unique_vals}"
    hairmask = hairmask.squeeze(1).bool()      # (B, H, W)

    # take (u,v) channels only
    # hs = depth_hs[:, 1:3, :, :]             # (B, 2, H, W)
    # curr = depth_current[:, 1:3, :, :]           # (B, 2, H, W)

    diff = (depth_hs - depth_current) ** 2                        # (B, 1, H, W)
    l2_map = diff.sum(dim=1)                   # (B, H, W) per-pixel squared L2

    l2_batch = []
    for b in range(B):
        mask_b = hairmask[b]                   # (H, W), bool
        if not mask_b.any():
            l2_batch.append(float("nan"))
        else:
            l2_batch.append(l2_map[b][mask_b].mean().item())
    return l2_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--subset", choices=['val', 'test'], default='val')
    parser.add_argument("--stress", default='blur', choices=['blur', 'noise'])
    parser.add_argument("--noise_sigma", type=float)
    parser.add_argument("--hisa_dot", action='store_true')   # 

    args, unknown = parser.parse_known_args()

    # load config
    conf = OmegaConf.load(args.config)
    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + unknown # Remove the configuration file name from sys.argv. Keep only OmegaConf-style overrides in argv
    conf.merge_with_cli()

    # add the args from argparse
    OmegaConf.set_struct(conf, False)
    conf.subset = args.subset
    conf.stress = args.stress
    conf.noise_sigma = args.noise_sigma
    OmegaConf.set_struct(conf, True)

    return conf

def main():
    # Initialize configuration
    config = parse_args()

    # Initialize log directories
    result_base_dir = Path(config.resume).parent
    print("result_base_dir", result_base_dir)
    result_dir = result_base_dir / "hisa_dot_gt"
    os.makedirs(result_dir)
    OmegaConf.save(config, os.path.join(result_dir, 'config.yaml'))

    # load dataset
    data_loader = load_dataloaders_HiSA(config)   # train, val
    print("data_loader", len(data_loader))
    
    # turn off log during testing
    config.train.log_path = None

    # Hairstep encoder trainer
    trainer_hs = SmirkHairTrainer(config)       # encoder are loaded with pretrained weights from HairStep in init
    trainer_hs = trainer_hs.to(config.device)
    trainer_hs.create_base_encoder()

    trainer_curr = SmirkHairTrainer(config)
    trainer_curr = trainer_curr.to(config.device)

    # load trained model
    trainer_curr.load_model(config.resume, load_fuse_generator=config.load_fuse_generator, load_encoder=config.load_encoder, device=config.device)

    # setup trainer
    trainer_curr.create_base_encoder()

    phase = 'val'
    loss_all_strand_gt_current = []
    loss_all_strand_gt_hs = []

    for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        for k in batch:
            batch[k] = batch[k].to(config.device)

        with torch.no_grad():
            # current model
            outputs = trainer_curr.step(batch, batch_idx, phase=phase)
            strand_current = outputs["strand"]      # (B, 3, H, W)

            # HairStep: encoder, check smirk_encoder for outputs
            outputs_hs = trainer_hs.step(batch, batch_idx, phase=phase)
            strand_hs = outputs_hs["strand"]

            strand_gt = batch["strand_gt"]

            # compute loss
            hairmask = batch["hairmask"]

            # compute against gt
            # strand_flipped = strand_gt.clone()
            # strand_flipped[:, 1:3] = -strand_flipped[:, 1:3]

            # rotated = torch.zeros_like(strand_gt)
            # rotated[:, 0] = strand_gt[:, 0]     # keep mask
            # u = strand_gt[:, 1]
            # v = strand_gt[:, 2]

            # rotated[:, 1] = -v                  # u_rot = -v
            # rotated[:, 2] =  u                  # v_rot = u            
            # tmp = compute_orientation_dot(strand_gt, rotated, hairmask)
            # print("tmp", tmp)
            # exit(0)
            
            loss_batch_strand_gt_current = compute_orientation_dot(strand_gt, strand_current, hairmask)
            loss_all_strand_gt_current.extend(loss_batch_strand_gt_current)

            loss_batch_strand_gt_hairstep = compute_orientation_dot(strand_gt, strand_hs, hairmask)
            loss_all_strand_gt_hs.extend(loss_batch_strand_gt_hairstep)

    loss_all_strand_gt_current = np.array(loss_all_strand_gt_current)
    loss_all_strand_gt_hs = np.array(loss_all_strand_gt_hs)

    stats = {
        "strand_gt_current_mean": float(loss_all_strand_gt_current.mean()),
        "strand_gt_current_std": float(loss_all_strand_gt_current.std()),
        "strand_gt_hs_mean": float(loss_all_strand_gt_hs.mean()),
        "strand_gt_hs_std": float(loss_all_strand_gt_hs.std()),
    }
    
    print("HiSA HairStep–current_model loss:")
    for k, v in stats.items():
        print(f"{k}: {v}")

    with open(os.path.join(result_dir, "hisa_dot_gt.json"), "w") as f:
        json.dump(stats, f, indent=4)



if __name__ == "__main__":
    main()