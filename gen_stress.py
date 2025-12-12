"""
For generative stress tests: 
Use the test set. For example, I plan to do this: 
Given 1 image:
- Blur the input image, and compare the orientation map from the original image and the blurred input image (using cosine similarity or L1) -> this gives sim_1
- Compute the similarity between the orientation map of the original image and of a random person -> sim_2
Then sim_1 should be less than sim_2 for all images.
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
from collections import defaultdict
from src.smirk_trainer import SmirkHairTrainer
from datasets.data_utils import load_dataloaders

from torchvision import transforms


def gaussian_blur(images, kernel_size=13, sigma=2.0):
    """Gaussian blur blurring a batch
    
    images: (B, 3, H, W)
    """
    output = []
    for i in range(images.size(0)):
        output.append(transforms.functional.gaussian_blur(images[i], kernel_size=kernel_size, sigma=sigma))
    return torch.stack(output, dim=0)        

def random_noise(images, sigma=0.1):
    """Random noise

    Args:
        images: (B, 3, H, W)
    """
    # print("random_noise", images.max(), images.min())

    noise = sigma * torch.randn_like(images)
    images_noise = images + noise
    images_noise = torch.clamp(images_noise, 0.0, 1.0)
    return images_noise

def compute_orientation_cosine(orient1, orient2, hairmask):
    """Cosine simiarily
    
    orient1: (B, 3, H, W)   # first dim in 3 is mask
    orient2: (B, 3, H, W)   # first dim in 3 is mask
    hairmask: (B, 1, H, W), 0 or 1
    """
    B, C, H, W = orient1.shape
    unique_vals = torch.unique(hairmask)
    assert torch.all((unique_vals == 0) | (unique_vals == 1)), f"hairmask must contain only 0 and 1, but got values: {unique_vals}"
    hairmask = hairmask.squeeze(1).bool()      # (B, H, W) - need bool for indexing

    # print("compute_orientation_cosine", orient1.shape, orient2.shape, hairmask.shape)
    orient1 = orient1[:, 1:3, :, :]     # (B, H, W)
    orient2 = orient2[:, 1:3, :, :]     # (B, H, W)

    cosine_sim = (orient1 * orient2).sum(dim=1)     # (B, H, W)

    cosine_batch = []
    for b in range(B):
        if hairmask[b].sum() == 0:
            cosine_batch.append(float("nan"))
        else:
            # cosine_sim[b][hairmask[b]]: (N,)
            cosine_batch.append(cosine_sim[b][hairmask[b]].mean().item())

    return cosine_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--subset", choices=['val', 'test'], default='val')
    parser.add_argument("--stress", default='blur', choices=['blur', 'noise'])
    parser.add_argument("--noise_sigma", type=float)

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
    result_dir = Path(config.resume).parent
    split_file = os.path.join(result_dir, 'splits.json')

    if config.subset == 'val':
        _, data_loader = load_dataloaders(config, split_file)   # train, val
    elif config.subset == 'test':
        data_loader = load_dataloaders(config, split_file, test_only=True)

    # turn off log during testing
    config.train.log_path = None

    trainer = SmirkHairTrainer(config)
    trainer = trainer.to(config.device)

    # setup output directory
    if config.stress == 'blur':
        stress_test_dir = os.path.join(result_dir, f'stress_test_blur_{config.subset}')
    elif config.stress == 'noise':
        stress_test_dir = os.path.join(result_dir, f'stress_test_noise_{str(config.noise_sigma)}_{config.subset}')
    else:
        raise ValueError(f"stress {config.stress} not allowed")
    os.makedirs(stress_test_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(stress_test_dir, 'config.yaml'))
    
    # val/test images
    images_save_path = f"{stress_test_dir}/{config.subset}_images"
    os.makedirs(images_save_path)

    # load trained model
    trainer.load_model(config.resume, load_fuse_generator=config.load_fuse_generator, load_encoder=config.load_encoder, device=config.device)

    # setup trainer
    trainer.create_base_encoder()

    phase = 'val'
    epoch = Path(config.resume).stem.split('_')[-1]
    losses_hist = {'train': {}, config.subset: {}, 'train_b': {}, f'{config.subset}_b': {}}

    sim_blur_all = []
    sim_random_all = []
    epoch_loss_sum = defaultdict(float)
    for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        if batch is None:
            continue
        trainer.set_freeze_status(config, batch_idx, None)
        for key in batch:
            batch[key] = batch[key].to(config.device)
        outputs = trainer.step(batch, batch_idx, phase=phase)        

        # save reconstructed image

        ############### Stress test
        with torch.no_grad():
            # print(outputs.keys()) ['img', 'strand', 'depth', 'loss_img', 'reconstructed_img', 'masked_1st_path', 'encoder_output'])
            strand = outputs['strand']          # (B, 3, H, W)
            hairmask = batch['hairmask'].to(strand.device)        # (B, 1, H, W)

            # create blurred images
            batch_stress = {}
            for k, v in batch.items():
                batch_stress[k] = v.clone()
            if config.stress == 'blur':
                batch_stress['img'] = gaussian_blur(batch['img'])
            elif config.stress == 'noise':
                batch_stress['img'] = random_noise(batch['img'], sigma=config.noise_sigma)
            else:
                raise ValueError(f"Got stress {config.stress}")

            # pass blurred images to network again
            outputs_blur = trainer.step(batch_stress, batch_idx, phase=phase)
            strand_blur = outputs_blur['strand']

            # print(torch.equal(batch['img'].to(strand.device), outputs['reconstructed_img']))

            # save current loss and visualization
            current_loss = trainer.current_loss
            for k, v in current_loss.items():
                epoch_loss_sum[k] += v
                losses_hist[f"{config.subset}_b"].setdefault(k, []).append(v)

            if batch_idx % config.train.visualize_every == 0:
                if config.train.visualize_phase.get(phase, False):
                    with torch.no_grad():
                        visualizations = trainer.create_visualizations(batch, outputs)
                        trainer.save_visualizations_2(visualizations, f"{stress_test_dir}/{config.subset}_images/{epoch}_{batch_idx}.jpg")

            # cosine difference between original and blur orienation map
            sim_blur_b = compute_orientation_cosine(strand, strand_blur, hairmask)

            # cosine difference between original and random image
            B, C, H, W = strand.shape
            if B > 1:
                permutation = torch.randperm(B, device=strand.device)

                # avoid comparing with same data (identity)
                if (permutation == torch.arange(B, device=strand.device)).all():
                    permutation = permutation.roll(1)

                strand_rand = strand[permutation]
                sim_random_b = compute_orientation_cosine(strand, strand_rand, hairmask)
            else:
                sim_random_b = [float("nan")] * B

            sim_blur_all.extend(sim_blur_b)
            sim_random_all.extend(sim_random_b)         

    # save 
    sim_blur_all = np.array(sim_blur_all, dtype=np.float32)
    sim_random_all = np.array(sim_random_all, dtype=np.float32)

    stats = {
        "sim_blur_mean": float(sim_blur_all.mean()),
        "sim_blur_std": float(sim_blur_all.std()),
        "sim_random_mean": float(sim_random_all.mean()),
        "sim_random_std": float(sim_random_all.std()),
    }

    print("Stress-test stats:")
    for k, v in stats.items():
        print(f"{k}: {v}")

    with open(os.path.join(stress_test_dir, "cosine_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)    



if __name__ == "__main__":
    main()
