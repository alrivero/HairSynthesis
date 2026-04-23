import logging
import sys
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from src.hair_synthesis_trainer import HairSynthesisTrainer
import os
from datetime import datetime
from collections import defaultdict
from datasets.data_utils import load_dataloaders
import time

try:
    import wandb
except ImportError:
    wandb = None


def ensure_config_defaults(conf):
    if 'train' in conf and 'run_name_suffix' not in conf.train:
        conf.train.run_name_suffix = ''


def format_run_name_suffix(suffix):
    if suffix is None or suffix is False:
        return ''

    suffix = str(suffix).strip()
    if not suffix:
        return ''

    suffix = suffix.replace('/', '_').replace('\\', '_')
    suffix = '_'.join(suffix.split())
    if not suffix:
        return ''

    if suffix.startswith(('_', '-')):
        return suffix
    return f'_{suffix}'


def append_suffix_to_log_path(log_path, suffix):
    if not suffix:
        return log_path

    stripped_path = log_path.rstrip(os.path.sep)
    parent = os.path.dirname(stripped_path)
    run_name = os.path.basename(stripped_path)
    return os.path.join(parent, f'{run_name}{suffix}')


def parse_args():
    conf = OmegaConf.load(sys.argv[1])
    ensure_config_defaults(conf)

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:] # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf


if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()
    wandb_cfg = getattr(config.train, 'wandb', None)

    # ----------------------- initialize log directories ----------------------- #
    run_name_suffix = format_run_name_suffix(getattr(config.train, 'run_name_suffix', None))
    if config.train.log_append_timestamp:
        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.train.log_path = os.path.join(config.train.log_path, f'{start_time}{run_name_suffix}')
    else:
        config.train.log_path = append_suffix_to_log_path(config.train.log_path, run_name_suffix)
    run_name = os.path.basename(config.train.log_path.rstrip(os.path.sep))
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

    wandb_run = None
    use_wandb = bool(
        getattr(wandb_cfg, 'enabled', getattr(config.train, 'use_wandb', False))
    )
    if use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it with `pip install wandb` or disable train.use_wandb.")

        wandb_run_name = getattr(wandb_cfg, 'run_name', getattr(config.train, 'wandb_run_name', None))
        if wandb_run_name:
            wandb_run_name = f'{wandb_run_name}{run_name_suffix}'
        else:
            wandb_run_name = run_name

        wandb_kwargs = {
            'project': getattr(wandb_cfg, 'project', getattr(config.train, 'wandb_project', 'hair-synthesis')),
            'name': wandb_run_name,
            'dir': config.train.log_path,
            'config': OmegaConf.to_container(config, resolve=True),
        }
        wandb_entity = getattr(wandb_cfg, 'entity', getattr(config.train, 'wandb_entity', None))
        if wandb_entity:
            wandb_kwargs['entity'] = wandb_entity
        wandb_mode = getattr(wandb_cfg, 'mode', getattr(config.train, 'wandb_mode', None))
        if wandb_mode:
            wandb_kwargs['mode'] = wandb_mode
        wandb_run = wandb.init(**wandb_kwargs)

    train_loader, val_loader = load_dataloaders(config)
    print("train_loader", len(train_loader), "val_loader", len(val_loader))

    trainer = HairSynthesisTrainer(config)
    trainer = trainer.to(config.device)

    if config.resume:
        trainer.load_model(config.resume, load_fuse_generator=config.load_fuse_generator, load_encoder=config.load_encoder, device=config.device)

    # after loading, copy the base encoder 
    # this is used for regularization w.r.t. the base model as well as to compare the results    
    trainer.create_base_encoder()

    losses_hist = {'train': {}, 'val': {}, 'train_b': {}, 'val_b': {}}
    global_train_step = 0
    for epoch in range(config.train.resume_epoch, config.train.num_epochs):
        # Configure epoch-dependent learning rates.
        trainer.configure_optimizers(len(train_loader), epoch_idx=epoch)

        for phase in ['train', 'val']:
            loader = train_loader if phase == 'train' else val_loader
            epoch_loss_sum = defaultdict(float)

            progress = tqdm(
                enumerate(loader),
                total=len(loader),
                desc=f"Epoch {epoch + 1}/{config.train.num_epochs} - {phase}",
                leave=(phase == 'val'),
            )
            for batch_idx, batch in progress:
                if batch is None:
                    continue

                trainer.set_freeze_status(config, batch_idx, epoch)

                for key in batch:
                    batch[key] = batch[key].to(config.device)

                outputs = trainer.step(batch, batch_idx, phase=phase, epoch_idx=epoch)
                current_loss = trainer.current_loss
                for k, v in current_loss.items():
                    epoch_loss_sum[k] += v
                    losses_hist[f"{phase}_b"].setdefault(k, []).append(v)

                if wandb_run is not None and phase == 'train':
                    batch_metrics = {
                        f'{phase}/batch/{k}': float(v)
                        for k, v in current_loss.items()
                    }
                    batch_metrics['epoch'] = epoch
                    if hasattr(trainer, 'encoder_optimizer'):
                        batch_metrics['train/lr_encoder'] = float(trainer.encoder_optimizer.param_groups[0]['lr'])
                    if getattr(config.arch, 'enable_fuse_generator', False) and hasattr(trainer, 'smirk_generator_optimizer'):
                        batch_metrics['train/lr_generator'] = float(trainer.smirk_generator_optimizer.param_groups[0]['lr'])
                    wandb.log(batch_metrics, step=global_train_step)
                    global_train_step += 1

                if batch_idx % config.train.visualize_every == 0:
                    if not config.train.visualize_phase.get(phase, False):
                        continue
                    
                    with torch.no_grad():
                        visualizations = trainer.create_visualizations(batch, outputs)
                        trainer.save_visualizations(visualizations, f"{config.train.log_path}/{phase}_images/{epoch}_{batch_idx}.jpg")
            
            epoch_loss_mean = {k: v/len(loader) for k, v in epoch_loss_sum.items()}

            trainer.logging_epoch(epoch, epoch_loss_mean, phase)
            if wandb_run is not None:
                epoch_metrics = {
                    f'{phase}/epoch/{k}': float(v)
                    for k, v in epoch_loss_mean.items()
                }
                epoch_metrics['epoch'] = epoch
                wandb.log(epoch_metrics, step=global_train_step)

            for k, v in epoch_loss_mean.items():
                losses_hist[phase].setdefault(k, []).append(v)
            # print("losses_hist", losses_hist)
        trainer.plot_losses(losses_hist)

        if epoch % config.train.save_every == 0 or epoch == (config.train.num_epochs-1):
            trainer.save_model(trainer.state_dict(), os.path.join(config.train.log_path, 'model_{}.pt'.format(epoch)))

    elapsed = time.perf_counter() - start_time
    trainer.logger.info(f"Elapsed: {int(elapsed // 60)} min {(elapsed % 60):.2f} sec")
    if wandb_run is not None:
        wandb.log({'runtime/elapsed_sec': elapsed}, step=global_train_step)
        wandb.finish()

    logging.shutdown()
