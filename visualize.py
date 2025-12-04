"""Similar to train.py, only run evaluation on validation set to do visualization"""

import sys
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from src.smirk_trainer import SmirkHairTrainer
import os
from datetime import datetime
from collections import defaultdict
from datasets.data_utils import load_dataloaders
from pathlib import Path


def parse_args():
    """For visualization, input the path to trained weights instead of config
    Use the config file in the same folder of the result folder"""

    conf_file = Path(sys.argv[1]).parent / 'config.yaml'

    conf = OmegaConf.load(conf_file)
    conf.resume = sys.argv[1]

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:] # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf


if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()

    # ----------------------- initialize log directories ----------------------- #
    # Load trained model for visualization
    result_dir = Path(config.resume).parent
    # print("result_dir", result_dir)
    split_file = os.path.join(result_dir, 'splits.json')
    # print("split_file", split_file)

    train_loader, val_loader = load_dataloaders(config, split_file)
    print("train_loader", len(train_loader), "val_loader", len(val_loader))

    # no log for visualization
    config.train.log_path = None

    trainer = SmirkHairTrainer(config)
    trainer = trainer.to(config.device)

    # load trained model
    visualize_dir = os.path.join(result_dir, 'val_images_hsv')
    os.makedirs(visualize_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(visualize_dir, 'config.yaml'))

    if config.resume:
        trainer.load_model(config.resume, load_fuse_generator=config.load_fuse_generator, load_encoder=config.load_encoder, device=config.device)

    # after loading, copy the base encoder 
    # this is used for regularization w.r.t. the base model as well as to compare the results    
    trainer.create_base_encoder()
    
    loader = val_loader
    phase = 'val'
    epoch = Path(config.resume).stem.split('_')[-1]
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
        if batch is None:
            continue

        trainer.set_freeze_status(config, batch_idx, None)

        for key in batch:
            batch[key] = batch[key].to(config.device)
        outputs = trainer.step(batch, batch_idx, phase=phase)
        # tmp = outputs['strand'][0]
        # print('----- visualize.py')
        # print("strand", outputs['strand'].shape)
        # print(torch.min(tmp[0, :, :]), torch.max(tmp[0, :, :]))
        # print(torch.min(tmp[1, :, :]), torch.max(tmp[1, :, :]))
        # print(torch.min(tmp[2, :, :]), torch.max(tmp[2, :, :]))

        if batch_idx % config.train.visualize_every == 0:
            if not config.train.visualize_phase.get(phase, False):
                continue
            
            with torch.no_grad():
                visualizations = trainer.create_visualizations(batch, outputs)
                trainer.save_visualizations(visualizations, f"{visualize_dir}/{epoch}_{batch_idx}.jpg")
    
