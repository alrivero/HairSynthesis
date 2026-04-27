import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from datasets.data_utils import load_dataloaders
from datasets.ffhq_dataset import ensure_ffhq_split_manifest
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
    if 'train' in conf and 'ddp' not in conf.train:
        conf.train.ddp = OmegaConf.create()
    if 'train' in conf and 'ddp' in conf.train:
        if 'find_unused_parameters' not in conf.train.ddp:
            conf.train.ddp.find_unused_parameters = False
        if 'sync_batchnorm' not in conf.train.ddp:
            conf.train.ddp.sync_batchnorm = False
        if 'broadcast_buffers' not in conf.train.ddp:
            conf.train.ddp.broadcast_buffers = True


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


def cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    try:
        if key in cfg:
            return cfg[key]
    except Exception:
        pass
    return getattr(cfg, key, default)


def init_distributed():
    required_env = ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')
    missing = [name for name in required_env if name not in os.environ]
    if missing:
        raise RuntimeError(
            "train_dae_ddp.py must be launched with torchrun; missing env vars: "
            + ', '.join(missing)
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA.")

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return rank, world_size, local_rank


def broadcast_object(value, src=0):
    object_list = [value]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def build_diagnostics(config, log_dir, device):
    diagnostics_cfg = getattr(config.train, 'diagnostics', None)
    return RuntimeDiagnostics(
        log_dir=log_dir,
        device=device,
        enabled=bool(cfg_get(diagnostics_cfg, 'enabled', True)),
        heartbeat_interval_sec=float(cfg_get(diagnostics_cfg, 'heartbeat_interval_sec', 10.0)),
        batch_log_every=int(cfg_get(diagnostics_cfg, 'batch_log_every', 25)),
        trace_batches_every=int(cfg_get(diagnostics_cfg, 'trace_batches_every', 0)),
        enable_faulthandler=bool(cfg_get(diagnostics_cfg, 'enable_faulthandler', True)),
    )


def append_progress_line(progress_log_path, message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(progress_log_path, 'a', encoding='utf-8') as handle:
        handle.write(f'{timestamp} {message}\n')


def reduce_loss_dict(loss_dict):
    gathered = [None for _ in range(dist.get_world_size())]
    serializable = {key: float(value) for key, value in loss_dict.items()}
    dist.all_gather_object(gathered, serializable)

    reduced = defaultdict(float)
    counts = defaultdict(int)
    for item in gathered:
        if item is None:
            continue
        for key, value in item.items():
            reduced[key] += float(value)
            counts[key] += 1

    return {
        key: reduced[key] / max(1, counts[key])
        for key in sorted(reduced.keys())
    }


def all_ranks_have_batch(batch, device, world_size):
    local_has_batch = torch.tensor(
        0 if batch is None else 1,
        device=device,
        dtype=torch.int64,
    )
    dist.all_reduce(local_has_batch, op=dist.ReduceOp.SUM)
    return int(local_has_batch.item()) == int(world_size)


def maybe_convert_sync_batchnorm(trainer, config):
    ddp_cfg = getattr(config.train, 'ddp', None)
    if not bool(cfg_get(ddp_cfg, 'sync_batchnorm', False)):
        return trainer

    trainer.hair_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(trainer.hair_encoder)
    return trainer


def wrap_trainable_modules(trainer, local_rank, config):
    ddp_cfg = getattr(config.train, 'ddp', None)
    find_unused_parameters = bool(cfg_get(ddp_cfg, 'find_unused_parameters', False))
    broadcast_buffers = bool(cfg_get(ddp_cfg, 'broadcast_buffers', True))

    trainer.hair_encoder = DDP(
        trainer.hair_encoder,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
    )
    return trainer


if __name__ == '__main__':
    rank, world_size, local_rank = init_distributed()
    is_main_process = rank == 0
    config = parse_args()
    config.device = f'cuda:{local_rank}'
    wandb_cfg = getattr(config.train, 'wandb', None)

    run_name_suffix = format_run_name_suffix(getattr(config.train, 'run_name_suffix', None))
    if is_main_process:
        if config.train.log_append_timestamp:
            start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.train.log_path = os.path.join(config.train.log_path, f'{start_timestamp}{run_name_suffix}')
        else:
            config.train.log_path = append_suffix_to_log_path(config.train.log_path, run_name_suffix)
    config.train.log_path = broadcast_object(config.train.log_path, src=0)
    run_name = os.path.basename(config.train.log_path.rstrip(os.path.sep))

    if not is_main_process:
        config.train.visualize_phase.train = False
        config.train.visualize_phase.val = False

    if is_main_process:
        print("Output", config.train.log_path, flush=True)
        os.makedirs(config.train.log_path, exist_ok=True)
        os.makedirs(os.path.join(config.train.log_path, 'train_images'), exist_ok=True)
        os.makedirs(os.path.join(config.train.log_path, 'val_images'), exist_ok=True)
        os.makedirs(os.path.join(config.train.log_path, 'loss_plots'), exist_ok=True)
        OmegaConf.save(config, os.path.join(config.train.log_path, 'config.yaml'))
    dist.barrier()

    split_manifest_path = os.path.join(config.train.log_path, 'splits.json')
    if is_main_process:
        ensure_ffhq_split_manifest(config, split_manifest_path)
    dist.barrier()

    runtime_log_dir = os.path.join(config.train.log_path, 'runtime', f'rank_{rank:02d}')
    diagnostics = build_diagnostics(config, runtime_log_dir, config.device)
    progress_log_dir = os.path.join(config.train.log_path, 'progress')
    os.makedirs(progress_log_dir, exist_ok=True)
    rank_progress_log_path = os.path.join(progress_log_dir, f'rank_{rank:02d}.log')
    main_progress_log_path = os.path.join(progress_log_dir, 'progress.log')
    diagnostics.record_event(
        'run_start',
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        run_name=run_name,
        resume=bool(config.resume),
        split_manifest_path=split_manifest_path,
    )
    append_progress_line(
        rank_progress_log_path,
        f'[rank {rank}] run_start world_size={world_size} local_rank={local_rank} run_name={run_name}',
    )
    if is_main_process:
        append_progress_line(
            main_progress_log_path,
            f'[main] run_start world_size={world_size} run_name={run_name}',
        )

    wandb_run = None
    use_wandb = is_main_process and bool(
        getattr(wandb_cfg, 'enabled', getattr(config.train, 'use_wandb', False))
    )
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

    train_loader, val_loader = load_dataloaders(
        config,
        split_file=split_manifest_path,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )
    diagnostics.record_event(
        'dataloaders_ready',
        rank=rank,
        train_batches=len(train_loader),
        val_batches=len(val_loader),
    )
    if is_main_process:
        print("train_loader", len(train_loader), "val_loader", len(val_loader), flush=True)
        append_progress_line(
            main_progress_log_path,
            f'[main] dataloaders_ready train_batches={len(train_loader)} val_batches={len(val_loader)}',
        )

    trainer_created = False
    trainer = None
    start_time = time.perf_counter()
    global_train_step = 0

    try:
        diagnostics.record_event('trainer_init_start', rank=rank)
        with diagnostics.stage('trainer.init', rank=rank):
            trainer = HairMapDAETrainer(config)
        trainer.runtime_diagnostics = diagnostics
        trainer = maybe_convert_sync_batchnorm(trainer, config)
        trainer = trainer.to(config.device)
        diagnostics.record_event('trainer_init_end', rank=rank)
        append_progress_line(
            rank_progress_log_path,
            f'[rank {rank}] trainer_init_complete device={config.device}',
        )
        if is_main_process:
            append_progress_line(main_progress_log_path, '[main] trainer_init_complete')

        resume_epoch = int(getattr(config.train, 'resume_epoch', 0))
        trainer.configure_optimizers(len(train_loader), epoch_idx=resume_epoch)
        if config.resume:
            loaded_epoch = trainer.load_model(config.resume, device=config.device)
            if loaded_epoch is not None:
                resume_epoch = int(loaded_epoch) + 1

        with diagnostics.stage('trainer.wrap_ddp', rank=rank):
            trainer = wrap_trainable_modules(trainer, local_rank, config)
        trainer_created = True

        losses_hist = {'train': {}, 'val': {}, 'train_b': {}, 'val_b': {}} if is_main_process else None
        global_train_step = 0
        train_batches_seen = resume_epoch * len(train_loader)

        for epoch in range(resume_epoch, config.train.num_epochs):
            diagnostics.record_event('epoch_start', epoch_idx=epoch, rank=rank)

            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            if isinstance(val_loader.sampler, DistributedSampler):
                val_loader.sampler.set_epoch(epoch)

            trainer.configure_optimizers(len(train_loader), epoch_idx=epoch)
            append_progress_line(
                rank_progress_log_path,
                f'[rank {rank}] epoch_start epoch={epoch + 1}/{config.train.num_epochs}',
            )
            if is_main_process:
                append_progress_line(
                    main_progress_log_path,
                    f'[main] epoch_start epoch={epoch + 1}/{config.train.num_epochs}',
                )

            for phase in ['train', 'val']:
                loader = train_loader if phase == 'train' else val_loader
                epoch_loss_sum = defaultdict(float)
                processed_batches = 0

                diagnostics.record_event(
                    'phase_start',
                    epoch_idx=epoch,
                    phase=phase,
                    rank=rank,
                    num_batches=len(loader),
                )
                append_progress_line(
                    rank_progress_log_path,
                    f'[rank {rank}] phase_start epoch={epoch + 1} phase={phase} num_batches={len(loader)}',
                )
                if is_main_process:
                    append_progress_line(
                        main_progress_log_path,
                        f'[main] phase_start epoch={epoch + 1} phase={phase} num_batches={len(loader)}',
                    )

                progress = tqdm(
                    enumerate(loader),
                    total=len(loader),
                    desc=f"[rank {rank}] Epoch {epoch + 1}/{config.train.num_epochs} - {phase}",
                    position=rank,
                    leave=False,
                    disable=False,
                    dynamic_ncols=True,
                    mininterval=1.0,
                    file=sys.stdout,
                )

                for batch_idx, batch in progress:
                    batch_start_time = time.perf_counter()
                    if not all_ranks_have_batch(batch, config.device, world_size):
                        diagnostics.record_event(
                            'batch_skipped_distributed_empty',
                            epoch_idx=epoch,
                            phase=phase,
                            batch_idx=batch_idx,
                            rank=rank,
                        )
                        append_progress_line(
                            rank_progress_log_path,
                            f'[rank {rank}] batch_skipped_distributed_empty epoch={epoch + 1} phase={phase} batch={batch_idx + 1}/{len(loader)}',
                        )
                        continue

                    should_visualize = bool(
                        is_main_process
                        and config.train.visualize_phase.get(phase, False)
                        and (batch_idx % config.train.visualize_every == 0)
                    )
                    log_this_batch = diagnostics.should_log_batch(batch_idx, force=should_visualize)
                    diagnostics.set_batch_context(
                        epoch_idx=epoch,
                        phase=phase,
                        batch_idx=batch_idx,
                        train_batch_step=train_batches_seen,
                        should_visualize=should_visualize,
                        stage='batch_start',
                        status='running',
                    )
                    if log_this_batch:
                        diagnostics.record_event(
                            'batch_start',
                            epoch_idx=epoch,
                            phase=phase,
                            batch_idx=batch_idx,
                            train_batch_step=train_batches_seen,
                            should_visualize=should_visualize,
                            rank=rank,
                        )

                    with (
                        diagnostics.stage(
                            'batch.move_to_device',
                            epoch_idx=epoch,
                            phase=phase,
                            batch_idx=batch_idx,
                            rank=rank,
                        ) if log_this_batch else nullcontext()
                    ):
                        for key in batch:
                            batch[key] = batch[key].to(config.device)

                    with (
                        diagnostics.stage(
                            'trainer.step',
                            epoch_idx=epoch,
                            phase=phase,
                            batch_idx=batch_idx,
                            train_batch_step=train_batches_seen,
                            should_visualize=should_visualize,
                            rank=rank,
                        ) if log_this_batch else nullcontext()
                    ):
                        outputs = trainer.step(batch, batch_idx, phase=phase, epoch_idx=epoch)

                    reduced_loss = reduce_loss_dict(trainer.current_loss)

                    if phase == 'train':
                        train_batches_seen += 1

                    processed_batches += 1
                    if is_main_process:
                        for key, value in reduced_loss.items():
                            epoch_loss_sum[key] += value
                            losses_hist[f'{phase}_b'].setdefault(key, []).append(value)

                        if wandb_run is not None and phase == 'train':
                            batch_metrics = {
                                f'{phase}/batch/{key}': float(value)
                                for key, value in reduced_loss.items()
                            }
                            batch_metrics['epoch'] = epoch
                            if hasattr(trainer, 'encoder_optimizer'):
                                batch_metrics['train/lr_dae'] = float(trainer.encoder_optimizer.param_groups[0]['lr'])
                            wandb.log(batch_metrics, step=global_train_step)
                            global_train_step += 1

                        if should_visualize and outputs:
                            vis_path = os.path.join(config.train.log_path, f'{phase}_images', f'{epoch}_{batch_idx}.jpg')
                            diagnostics.record_event(
                                'visualization_requested',
                                epoch_idx=epoch,
                                phase=phase,
                                batch_idx=batch_idx,
                                save_path=vis_path,
                                rank=rank,
                            )
                            with torch.no_grad():
                                with diagnostics.stage(
                                    'visualization.create',
                                    epoch_idx=epoch,
                                    phase=phase,
                                    batch_idx=batch_idx,
                                    save_path=vis_path,
                                    rank=rank,
                                ):
                                    visualizations = trainer.create_visualizations(batch, outputs)
                                with diagnostics.stage(
                                    'visualization.save',
                                    epoch_idx=epoch,
                                    phase=phase,
                                    batch_idx=batch_idx,
                                    save_path=vis_path,
                                    rank=rank,
                                ):
                                    trainer.save_visualizations(visualizations, vis_path)
                            diagnostics.record_event(
                                'visualization_saved',
                                epoch_idx=epoch,
                                phase=phase,
                                batch_idx=batch_idx,
                                save_path=vis_path,
                                rank=rank,
                            )
                            del visualizations

                    batch_duration_sec = time.perf_counter() - batch_start_time
                    total_loss_value = reduced_loss.get('total_loss')
                    total_loss_text = (
                        f'{float(total_loss_value):.4f}'
                        if total_loss_value is not None
                        else 'n/a'
                    )
                    progress.set_postfix_str(
                        f'loss={total_loss_text} batch={batch_duration_sec:.1f}s',
                        refresh=False,
                    )
                    append_progress_line(
                        rank_progress_log_path,
                        (
                            f'[rank {rank}] batch_end epoch={epoch + 1} phase={phase} '
                            f'batch={batch_idx + 1}/{len(loader)} total_loss={total_loss_text} '
                            f'batch_sec={batch_duration_sec:.2f}'
                        ),
                    )
                    if is_main_process:
                        append_progress_line(
                            main_progress_log_path,
                            (
                                f'[main] batch_end epoch={epoch + 1} phase={phase} '
                                f'batch={batch_idx + 1}/{len(loader)} total_loss={total_loss_text} '
                                f'batch_sec={batch_duration_sec:.2f}'
                            ),
                        )

                    if log_this_batch:
                        diagnostics.record_event(
                            'batch_end',
                            epoch_idx=epoch,
                            phase=phase,
                            batch_idx=batch_idx,
                            train_batch_step=train_batches_seen,
                            total_loss=reduced_loss.get('total_loss'),
                            rank=rank,
                        )
                    diagnostics.set_batch_context(stage='batch_complete', status='idle')

                progress.close()
                if is_main_process:
                    epoch_loss_mean = {
                        key: value / max(1, processed_batches)
                        for key, value in epoch_loss_sum.items()
                    }

                    trainer.logging_epoch(epoch, epoch_loss_mean, phase)
                    diagnostics.record_event(
                        'phase_end',
                        epoch_idx=epoch,
                        phase=phase,
                        epoch_loss_mean=epoch_loss_mean,
                        processed_batches=processed_batches,
                        rank=rank,
                    )

                    if wandb_run is not None:
                        epoch_metrics = {
                            f'{phase}/epoch/{key}': float(value)
                            for key, value in epoch_loss_mean.items()
                        }
                        epoch_metrics['epoch'] = epoch
                        wandb.log(epoch_metrics, step=global_train_step)

                    for key, value in epoch_loss_mean.items():
                        losses_hist[phase].setdefault(key, []).append(value)
                    append_progress_line(
                        main_progress_log_path,
                        f'[main] phase_end epoch={epoch + 1} phase={phase} processed_batches={processed_batches}',
                    )
                else:
                    diagnostics.record_event(
                        'phase_end',
                        epoch_idx=epoch,
                        phase=phase,
                        processed_batches=processed_batches,
                        rank=rank,
                    )
                append_progress_line(
                    rank_progress_log_path,
                    f'[rank {rank}] phase_end epoch={epoch + 1} phase={phase} processed_batches={processed_batches}',
                )

            if is_main_process:
                trainer.plot_losses(losses_hist)

                if epoch % config.train.save_every == 0 or epoch == (config.train.num_epochs - 1):
                    checkpoint_path = os.path.join(config.train.log_path, f'model_{epoch}.pt')
                    diagnostics.record_event(
                        'checkpoint_save_start',
                        epoch_idx=epoch,
                        save_path=checkpoint_path,
                        rank=rank,
                    )
                    trainer.save_model(None, checkpoint_path, epoch_idx=epoch)
                    diagnostics.record_event(
                        'checkpoint_save_end',
                        epoch_idx=epoch,
                        save_path=checkpoint_path,
                        rank=rank,
                    )

            dist.barrier()
            diagnostics.record_event('epoch_end', epoch_idx=epoch, rank=rank)
            append_progress_line(
                rank_progress_log_path,
                f'[rank {rank}] epoch_end epoch={epoch + 1}/{config.train.num_epochs}',
            )
            if is_main_process:
                append_progress_line(
                    main_progress_log_path,
                    f'[main] epoch_end epoch={epoch + 1}/{config.train.num_epochs}',
                )
    except Exception as exc:
        diagnostics.record_event(
            'run_exception',
            rank=rank,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
        diagnostics.dump_cuda_memory(label=f'run_exception_rank_{rank}')
        append_progress_line(
            rank_progress_log_path,
            f'[rank {rank}] run_exception type={type(exc).__name__} message={str(exc)}',
        )
        if is_main_process:
            append_progress_line(
                main_progress_log_path,
                f'[main] run_exception type={type(exc).__name__} message={str(exc)}',
            )
        raise
    finally:
        elapsed = time.perf_counter() - start_time
        if trainer_created and getattr(trainer, 'logger', None) is not None:
            trainer.logger.info(f"Elapsed: {int(elapsed // 60)} min {(elapsed % 60):.2f} sec")
        diagnostics.record_event('run_end', elapsed_sec=elapsed, rank=rank)
        append_progress_line(
            rank_progress_log_path,
            f'[rank {rank}] run_end elapsed_sec={elapsed:.2f}',
        )
        if is_main_process:
            append_progress_line(
                main_progress_log_path,
                f'[main] run_end elapsed_sec={elapsed:.2f}',
            )
        if wandb_run is not None:
            try:
                wandb.log({'runtime/elapsed_sec': elapsed}, step=global_train_step)
            except Exception:
                pass
            try:
                wandb.finish()
            except Exception:
                pass
        diagnostics.close()
        logging.shutdown()
        if dist.is_initialized():
            dist.destroy_process_group()
