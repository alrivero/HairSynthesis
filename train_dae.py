import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from datasets.data_utils import load_dataloaders
from src.dae_trainer import HairMapDAETrainer
from src.utils.runtime_diagnostics import RuntimeDiagnostics

try:
    import wandb
except ImportError:
    wandb = None


def ensure_config_defaults(conf):
    if 'train' in conf and 'run_name_suffix' not in conf.train:
        conf.train.run_name_suffix = ''
    if 'train' in conf and 'resume_epoch' not in conf.train:
        conf.train.resume_epoch = 0
    if 'resume' not in conf:
        conf.resume = False


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
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    conf.merge_with_cli()
    return conf


if __name__ == '__main__':
    config = parse_args()
    wandb_cfg = getattr(config.train, 'wandb', None)

    run_name_suffix = format_run_name_suffix(getattr(config.train, 'run_name_suffix', None))
    if config.train.log_append_timestamp:
        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.train.log_path = os.path.join(config.train.log_path, f'{start_time}{run_name_suffix}')
    else:
        config.train.log_path = append_suffix_to_log_path(config.train.log_path, run_name_suffix)
    os.makedirs(config.train.log_path, exist_ok=True)
    os.makedirs(os.path.join(config.train.log_path, 'train_images'), exist_ok=True)
    os.makedirs(os.path.join(config.train.log_path, 'val_images'), exist_ok=True)
    os.makedirs(os.path.join(config.train.log_path, 'loss_plots'), exist_ok=True)
    OmegaConf.save(config, os.path.join(config.train.log_path, 'config.yaml'))

    diagnostics = RuntimeDiagnostics.from_config(config)
    run_name = os.path.basename(config.train.log_path.rstrip(os.path.sep))
    diagnostics.record_event(
        'run_start',
        run_name=run_name,
        resume=bool(config.resume),
    )

    wandb_run = None
    use_wandb = bool(getattr(wandb_cfg, 'enabled', getattr(config.train, 'use_wandb', False)))
    if use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it with `pip install wandb` or disable train.wandb.enabled.")
        wandb_kwargs = {
            'project': getattr(wandb_cfg, 'project', 'hair-map-dae'),
            'name': getattr(wandb_cfg, 'run_name', run_name),
            'dir': config.train.log_path,
            'config': OmegaConf.to_container(config, resolve=True),
        }
        if getattr(wandb_cfg, 'entity', None):
            wandb_kwargs['entity'] = wandb_cfg.entity
        if getattr(wandb_cfg, 'mode', None):
            wandb_kwargs['mode'] = wandb_cfg.mode
        wandb_run = wandb.init(**wandb_kwargs)

    train_loader, val_loader = load_dataloaders(config)
    trainer = HairMapDAETrainer(config).to(config.device)
    trainer.runtime_diagnostics = diagnostics
    train_batches_seen = config.train.resume_epoch * len(train_loader)
    global_train_step = 0
    losses_hist = {'train': {}, 'val': {}, 'train_b': {}, 'val_b': {}}

    resume_epoch = config.train.resume_epoch
    trainer.configure_optimizers(len(train_loader), epoch_idx=resume_epoch)
    if config.resume:
        loaded_epoch = trainer.load_model(config.resume, device=config.device)
        if loaded_epoch is not None:
            resume_epoch = int(loaded_epoch) + 1

    start_time_perf = time.perf_counter()
    try:
        for epoch in range(resume_epoch, config.train.num_epochs):
            trainer.configure_optimizers(len(train_loader), epoch_idx=epoch)
            diagnostics.record_event('epoch_start', epoch_idx=epoch)
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
                    for key in batch:
                        batch[key] = batch[key].to(config.device)

                    outputs = trainer.step(batch, batch_idx, phase=phase, epoch_idx=epoch)
                    if phase == 'train':
                        train_batches_seen += 1

                    current_loss = trainer.current_loss
                    for key, value in current_loss.items():
                        epoch_loss_sum[key] += value
                        losses_hist[f'{phase}_b'].setdefault(key, []).append(value)

                    if wandb_run is not None and phase == 'train':
                        batch_metrics = {f'{phase}/batch/{k}': float(v) for k, v in current_loss.items()}
                        batch_metrics['epoch'] = epoch
                        batch_metrics['train/lr_dae'] = float(trainer.encoder_optimizer.param_groups[0]['lr'])
                        wandb.log(batch_metrics, step=global_train_step)
                        global_train_step += 1

                    should_visualize = bool(
                        config.train.visualize_phase.get(phase, False)
                        and (batch_idx % config.train.visualize_every == 0)
                    )
                    if should_visualize and outputs:
                        vis_path = os.path.join(config.train.log_path, f'{phase}_images', f'{epoch}_{batch_idx}.jpg')
                        trainer.save_visualizations(trainer.create_visualizations(batch, outputs), vis_path)

                epoch_loss_mean = {k: v / max(1, len(loader)) for k, v in epoch_loss_sum.items()}
                trainer.logging_epoch(epoch, epoch_loss_mean, phase)
                for key, value in epoch_loss_mean.items():
                    losses_hist[phase].setdefault(key, []).append(value)
                if wandb_run is not None:
                    epoch_metrics = {f'{phase}/epoch/{k}': float(v) for k, v in epoch_loss_mean.items()}
                    epoch_metrics['epoch'] = epoch
                    wandb.log(epoch_metrics, step=global_train_step)

            trainer.plot_losses(losses_hist)
            if epoch % config.train.save_every == 0 or epoch == (config.train.num_epochs - 1):
                checkpoint_path = os.path.join(config.train.log_path, f'model_{epoch}.pt')
                trainer.save_model(trainer.state_dict(), checkpoint_path, epoch_idx=epoch)
            diagnostics.record_event('epoch_end', epoch_idx=epoch)
    except Exception as exc:
        diagnostics.record_event(
            'run_exception',
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
        raise
    finally:
        elapsed = time.perf_counter() - start_time_perf
        diagnostics.record_event('run_end', elapsed_sec=elapsed)
        diagnostics.close()
        if wandb_run is not None:
            try:
                wandb.log({'runtime/elapsed_sec': elapsed}, step=global_train_step)
            except Exception:
                pass
            try:
                wandb.finish()
            except Exception:
                pass
        logging.shutdown()
