import os
import random
import sys
from dataclasses import fields
from datetime import datetime

import numpy as np
from omegaconf import OmegaConf
import torch

from datasets.data_utils import load_dataloaders
from src.hair_map_corruption import HairMapCorruptor
from src.hair_map_visualization import save_packed_map_comparison
from src.synthetic_hair_map_generator import SyntheticHairMapBatch, SyntheticHairMapGenerator


FAMILY_DESCRIPTIONS = {
    'support': (
        "Post-render support corruption. It perturbs the hair silhouette itself, then zeroes orientation "
        "and depth outside the new support."
    ),
    'strand_dropout': (
        "Pre-render sparsification. It drops a subset of strands before rerasterization so support, "
        "orientation, and depth stay physically coupled."
    ),
    'blur_resample': (
        "Post-render low-detail corruption. It downsamples and upsamples the packed map to soften edges "
        "and remove fine strand detail."
    ),
    'orientation_jitter': (
        "Post-render orientation-field rotation. It adds a smooth spatially varying angular perturbation "
        "to the dx/dy channels."
    ),
    'orientation_sign_flip': (
        "Post-render local orientation discontinuity. It flips the sign of dx/dy together inside small "
        "support-overlapping windows to create sharp local flow inconsistencies."
    ),
    'orientation_confidence': (
        "Post-render orientation weakening. It attenuates dx/dy magnitude with a smooth field to mimic "
        "low-confidence direction estimates."
    ),
    'depth_holes': (
        "Post-render depth dropout. It removes depth values in local regions while keeping the rest of the "
        "packed map intact."
    ),
    'depth_drift': (
        "Post-render depth bias. It adds a smooth local offset field to the depth channel."
    ),
    'misregistration': (
        "Post-render channel alignment error. It shifts orientation and depth relative to the support mask."
    ),
    'partial_occlusion': (
        "Post-render large support removal. It zeros larger regions of the support mask to mimic strong "
        "foreground occlusion. It is implemented but disabled in the default mix."
    ),
    'channel_inconsistency': (
        "Post-render channel-specific corruption. It intentionally corrupts only orientation or only depth "
        "so the channels do not fully agree."
    ),
}


PARAM_DESCRIPTIONS = {
    'support': {
        'weight': "Sampling weight when this family competes with the others in the mixed corruption draw.",
        'kernel_min': "Minimum morphology kernel size when the support corruption chooses erosion or dilation.",
        'kernel_max': "Maximum morphology kernel size when the support corruption chooses erosion or dilation.",
        'max_holes': "Upper bound on how many internal rectangular holes can be cut from the support mask.",
        'hole_size_min': "Minimum rectangular hole size as a fraction of the image height and width.",
        'hole_size_max': "Maximum rectangular hole size as a fraction of the image height and width.",
        'trim_fraction_min': "Minimum fraction of the image width or height removed when boundary trimming is chosen.",
        'trim_fraction_max': "Maximum fraction of the image width or height removed when boundary trimming is chosen.",
    },
    'strand_dropout': {
        'weight': "Sampling weight in the mixed corruption draw.",
        'keep_ratio_min': "Lower bound on the fraction of strands kept before rerendering.",
        'keep_ratio_max': "Upper bound on the fraction of strands kept before rerendering.",
    },
    'blur_resample': {
        'weight': "Sampling weight in the mixed corruption draw.",
        'scale_min': "Lower bound on the temporary downsample scale before upsampling back to full resolution.",
        'scale_max': "Upper bound on the temporary downsample scale before upsampling back to full resolution.",
    },
    'orientation_jitter': {
        'weight': "Sampling weight in the mixed corruption draw.",
        'max_angle_deg': "Maximum absolute orientation rotation in degrees.",
        'low_res': "Controls how smooth the angular noise field is. Larger values produce broader, smoother regions.",
    },
    'orientation_sign_flip': {
        'weight': "Sampling weight in the mixed corruption draw.",
        'max_regions': "Upper bound on how many small support-overlapping windows can flip orientation sign.",
        'size_min': "Minimum sign-flip window size as a fraction of the image height and width.",
        'size_max': "Maximum sign-flip window size as a fraction of the image height and width.",
    },
    'orientation_confidence': {
        'weight': "Sampling weight in the mixed corruption draw.",
        'low_res': "Controls the smoothness of the attenuation field applied to dx/dy magnitude.",
        'attenuation_min': "Minimum multiplicative attenuation applied to orientation magnitude.",
        'attenuation_max': "Maximum multiplicative attenuation applied to orientation magnitude.",
    },
    'depth_holes': {
        'weight': "Sampling weight in the mixed corruption draw.",
        'max_holes': "Upper bound on how many local depth holes can be cut.",
        'hole_size_min': "Minimum hole size as a fraction of the image height and width.",
        'hole_size_max': "Maximum hole size as a fraction of the image height and width.",
    },
    'depth_drift': {
        'weight': "Sampling weight in the mixed corruption draw.",
        'max_abs_bias': "Maximum absolute additive depth bias applied by the smooth drift field.",
        'low_res': "Controls how smooth the depth drift field is. Larger values produce broader, slower-varying regions.",
    },
    'misregistration': {
        'weight': "Sampling weight in the mixed corruption draw.",
        'max_shift_px': "Maximum integer pixel translation applied to orientation and depth relative to the support mask.",
    },
    'partial_occlusion': {
        'weight': "Sampling weight in the mixed corruption draw. Zero means disabled unless you force it in the demo.",
        'max_regions': "Upper bound on how many larger occlusion regions can be carved from the support mask.",
        'size_min': "Minimum occlusion size as a fraction of the image height and width.",
        'size_max': "Maximum occlusion size as a fraction of the image height and width.",
    },
    'channel_inconsistency': {
        'weight': "Sampling weight in the mixed corruption draw.",
    },
}


def ensure_config_defaults(conf):
    if 'train' in conf and 'run_name_suffix' not in conf.train:
        conf.train.run_name_suffix = ''
    if 'train' in conf and 'resume_epoch' not in conf.train:
        conf.train.resume_epoch = 0
    if 'resume' not in conf:
        conf.resume = False


def parse_args():
    config_path = sys.argv[1]
    conf = OmegaConf.load(config_path)
    ensure_config_defaults(conf)
    OmegaConf.set_struct(conf, True)
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    conf.merge_with_cli()
    return conf, config_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_output_dir(config) -> str:
    demo_cfg = config.dae.demo
    if getattr(demo_cfg, 'output_dir', None):
        output_dir = str(demo_cfg.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(config.train.log_path, f"corruption_demo_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def trim_batch(batch, max_samples: int):
    if max_samples <= 0:
        return batch
    trimmed = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            trimmed[key] = value[:max_samples]
        elif isinstance(value, list):
            trimmed[key] = value[:max_samples]
        elif isinstance(value, tuple):
            trimmed[key] = value[:max_samples]
        else:
            trimmed[key] = value
    return trimmed


def trim_bundle(bundle: SyntheticHairMapBatch, max_samples: int) -> SyntheticHairMapBatch:
    if max_samples <= 0 or bundle.clean_map.shape[0] <= max_samples:
        return bundle

    data = {}
    for field in fields(SyntheticHairMapBatch):
        value = getattr(bundle, field.name)
        if torch.is_tensor(value):
            data[field.name] = value[:max_samples]
        elif isinstance(value, list):
            data[field.name] = value[:max_samples]
        else:
            data[field.name] = value
    return SyntheticHairMapBatch(**data)


def first_valid_batch(loader, device: str, max_samples: int):
    for batch in loader:
        if batch is None:
            continue
        moved = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return trim_batch(moved, max_samples)
    raise RuntimeError("No valid batch was available from the selected dataloader.")


def write_summary(config, config_path: str, output_dir: str, families: list[str], weighted_families) -> None:
    demo_cfg = config.dae.demo
    corruption_cfg = config.dae.corruption
    lines = [
        "# DAE Corruption Demo",
        "",
        f"- Config: `{os.path.abspath(config_path)}`",
        f"- Output directory: `{output_dir}`",
        f"- Phase: `{demo_cfg.phase}`",
        f"- Max samples shown per image: `{demo_cfg.max_samples}`",
        f"- Mixed draw ops per sample: `{corruption_cfg.min_ops_per_sample}` to `{corruption_cfg.max_ops_per_sample}`",
        "",
        "## Stage Layout",
        "",
        "- `clean`: pristine synthetic render from the current FLAME + Hair20k synthesis path.",
        "- `pre_render`: result after pre-render corruption only, currently the geometry-aware rerender step such as `strand_dropout`.",
        "- `corrupted`: final DAE input after all selected corruptions have been applied.",
        "",
    ]

    if weighted_families is not None:
        lines.extend([
            "## Mixed Sample",
            "",
            "- `00_weighted_mix.jpg` shows the normal config-weighted corruption sampling.",
            f"- Applied families per sample in that image: `{weighted_families}`",
            "",
        ])

    lines.extend([
        "## Family Files",
        "",
        "Each `NN_<family>.jpg` image forces exactly one corruption family across every sample in the row-major grid.",
        "",
    ])

    for family in families:
        family_cfg = getattr(config.dae.corruption, family)
        family_dict = OmegaConf.to_container(family_cfg, resolve=True)
        lines.append(f"### `{family}`")
        lines.append("")
        lines.append(FAMILY_DESCRIPTIONS.get(family, "No description available."))
        lines.append("")
        for key, value in family_dict.items():
            description = PARAM_DESCRIPTIONS.get(family, {}).get(key, "No additional description recorded.")
            lines.append(f"- `{family}.{key} = {value}`: {description}")
        lines.append("")

    with open(os.path.join(output_dir, 'README.md'), 'w', encoding='utf-8') as handle:
        handle.write("\n".join(lines))


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python demo_dae_corruptions.py <config.yaml> [key=value ...]")

    config, config_path = parse_args()
    demo_cfg = config.dae.demo
    set_seed(int(demo_cfg.seed))
    output_dir = build_output_dir(config)
    OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    train_loader, val_loader = load_dataloaders(config)
    loader = train_loader if str(demo_cfg.phase).lower() == 'train' else val_loader

    generator = SyntheticHairMapGenerator(config).to(config.device)
    corruptor = HairMapCorruptor(config).to(config.device)
    generator.eval()
    corruptor.eval()

    batch = first_valid_batch(loader, config.device, int(demo_cfg.max_samples))
    with torch.no_grad():
        bundle = generator(batch)
    if bundle is None:
        raise RuntimeError("SyntheticHairMapGenerator returned None for the selected batch.")
    bundle = trim_bundle(bundle, int(demo_cfg.max_samples))

    weighted_families = None
    if bool(getattr(demo_cfg, 'include_weighted_mix', True)):
        with torch.no_grad():
            weighted = corruptor(bundle, generator, phase=str(demo_cfg.phase))
        weighted_families = weighted.applied_families
        save_packed_map_comparison(
            (
                ('clean', bundle.clean_map.detach().cpu()),
                ('pre_render', weighted.pre_render_map.detach().cpu()),
                ('corrupted', weighted.corrupted_map.detach().cpu()),
            ),
            os.path.join(output_dir, '00_weighted_mix.jpg'),
        )

    if getattr(demo_cfg, 'families', None):
        families = [str(name) for name in demo_cfg.families]
    else:
        families = corruptor.available_families(
            include_disabled=bool(getattr(demo_cfg, 'include_disabled_families', True))
        )
    available = set(corruptor.available_families(include_disabled=True))
    invalid = [name for name in families if name not in available]
    if invalid:
        raise ValueError(f"Unknown demo corruption families: {invalid}. Available: {sorted(available)}")

    for family_idx, family in enumerate(families, start=1):
        families_per_sample = [[family] for _ in range(bundle.clean_map.shape[0])]
        with torch.no_grad():
            corruption = corruptor(
                bundle,
                generator,
                phase=str(demo_cfg.phase),
                families_per_sample=families_per_sample,
            )

        filename = f"{family_idx:02d}_{family}.jpg"
        save_packed_map_comparison(
            (
                ('clean', bundle.clean_map.detach().cpu()),
                ('pre_render', corruption.pre_render_map.detach().cpu()),
                ('corrupted', corruption.corrupted_map.detach().cpu()),
            ),
            os.path.join(output_dir, filename),
        )

    write_summary(config, config_path, output_dir, families, weighted_families)
    print(f"Saved corruption demo to {output_dir}")


if __name__ == '__main__':
    main()
