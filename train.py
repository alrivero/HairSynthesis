import logging
import sys
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from src.smirk_trainer import SmirkHairTrainer
import os
from datetime import datetime
from collections import defaultdict
from datasets.data_utils import load_dataloaders
import time


def parse_args():
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:] # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf


if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()

    # ----------------------- initialize log directories ----------------------- #
    if config.train.log_append_timestamp:
        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.train.log_path = os.path.join(config.train.log_path, start_time)
    print("Output", config.train.log_path)
    os.makedirs(config.train.log_path, exist_ok=True)
    train_images_save_path = os.path.join(config.train.log_path, 'train_images')
    os.makedirs(train_images_save_path, exist_ok=True)
    val_images_save_path = os.path.join(config.train.log_path, 'val_images')
    os.makedirs(val_images_save_path, exist_ok=True)
    loss_plots_save_path = os.path.join(config.train.log_path, 'loss_plots')
    os.makedirs(loss_plots_save_path, exist_ok=True)    
    OmegaConf.save(config, os.path.join(config.train.log_path, 'config.yaml'))
    start_time = time.perf_counter()

    train_loader, val_loader = load_dataloaders(config)
    print("train_loader", len(train_loader), "val_loader", len(val_loader))

    trainer = SmirkHairTrainer(config)
    trainer = trainer.to(config.device)

    if config.resume:
        trainer.load_model(config.resume, load_fuse_generator=config.load_fuse_generator, load_encoder=config.load_encoder, device=config.device)

    # after loading, copy the base encoder 
    # this is used for regularization w.r.t. the base model as well as to compare the results    
    trainer.create_base_encoder()

    losses_hist = {'train': {}, 'val': {}, 'train_b': {}, 'val_b': {}}
    for epoch in tqdm(range(config.train.resume_epoch, config.train.num_epochs)):
        # restart everything at each epoch!
        trainer.configure_optimizers(len(train_loader))

        for phase in ['train', 'val']:
            loader = train_loader if phase == 'train' else val_loader
            epoch_loss_sum = defaultdict(float)
            
            # for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
            for batch_idx, batch in enumerate(loader):
                if batch is None:
                    continue

                trainer.set_freeze_status(config, batch_idx, epoch)

                for key in batch:
                    batch[key] = batch[key].to(config.device)

                outputs = trainer.step(batch, batch_idx, phase=phase)
                current_loss = trainer.current_loss
                for k, v in current_loss.items():
                    epoch_loss_sum[k] += v
                    losses_hist[f"{phase}_b"].setdefault(k, []).append(v)

                if batch_idx % config.train.visualize_every == 0:
                    if not config.train.visualize_phase.get(phase, False):
                        continue
                    
                    with torch.no_grad():
                        visualizations = trainer.create_visualizations(batch, outputs)
                        trainer.save_visualizations(visualizations, f"{config.train.log_path}/{phase}_images/{epoch}_{batch_idx}.jpg")
            
            epoch_loss_mean = {k: v/len(loader) for k, v in epoch_loss_sum.items()}

            trainer.logging_epoch(epoch, epoch_loss_mean, phase)

            for k, v in epoch_loss_mean.items():
                losses_hist[phase].setdefault(k, []).append(v)
            # print("losses_hist", losses_hist)
        trainer.plot_losses(losses_hist)

        if epoch % config.train.save_every == 0 or epoch == (config.train.num_epochs-1):
            trainer.save_model(trainer.state_dict(), os.path.join(config.train.log_path, 'model_{}.pt'.format(epoch)))

    elapsed = time.perf_counter() - start_time
    trainer.logger.info(f"Elapsed: {int(elapsed // 60)} min {(elapsed % 60):.2f} sec")

    logging.shutdown()
