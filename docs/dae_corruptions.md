# Hair Map DAE Corruptions

This document lists the corruption families available for `train_dae.py`, including the v1 defaults and the families kept disabled for later experiments.

## Packed Map Contract

The DAE input and target use the packed hair-map format:

- `mask`: hair support mask
- `dx, dy`: 2D orientation field
- `depth`: normalized visible-hair depth

## V1 Enabled Default Mix

The starting mix used by `configs/config_train_dae.yaml` is:

- `support` at weight `0.25`
- `strand_dropout` at weight `0.35`
- `blur_resample` at weight `0.30`
- `orientation_jitter` at weight `0.12`
- `orientation_sign_flip` at weight `0.10`
- `orientation_confidence` at weight `0.12`
- `depth_holes` at weight `0.12`
- `depth_drift` at weight `0.12`
- `misregistration` at weight `0.08`
- `partial_occlusion` at weight `0.20`
- `channel_inconsistency` at weight `0.08`

These weights are sampling weights, not direct percentages. Each sample currently draws `8-10` corruption families.

In addition to the sampled families, the config can also apply an optional **final pose jitter**
to the already-corrupted map. That stage is separate from the weighted family draw.

## Demo Script

Use the standalone demo script to visualize one corruption family at a time from the current config:

```bash
python demo_dae_corruptions.py configs/config_train_dae.yaml
```

Useful overrides:

```bash
python demo_dae_corruptions.py configs/config_train_dae.yaml dae.demo.phase=train dae.demo.max_samples=2
python demo_dae_corruptions.py configs/config_train_dae.yaml 'dae.demo.families=[support,depth_drift,misregistration]'
python demo_dae_corruptions.py configs/config_train_dae.yaml dae.demo.include_disabled_families=false
```

The script saves:

- `00_weighted_mix.jpg`: a normal config-weighted mixed corruption sample
- `NN_<family>.jpg`: one image per forced corruption family
- `README.md`: descriptions plus the resolved parameter values used for the run
- `config.yaml`: the resolved config snapshot used for that demo

Each image is laid out with one sample per row and stage triplets left-to-right:

- `clean orientation`, `clean depth`, `clean mask`
- `pre_render orientation`, `pre_render depth`, `pre_render mask`
- `corrupted orientation`, `corrupted depth`, `corrupted mask`

`pre_render` only differs from `clean` when a geometry-aware pre-render corruption such as `strand_dropout` is active.

## Available Corruption Families

### Pre-render

- `strand_dropout`
  Randomly drops a subset of strands before rerasterization. This keeps support, orientation, and depth physically coupled and is the main pre-render sparsification corruption in v1.

### Post-render

- `support`
  Applies support corruption directly to the rendered hair silhouette and zeros orientation/depth outside the corrupted support. The current implementation mixes:
  - morphology-based erosion or dilation
  - internal rectangular hole carving
  - boundary trimming

- `blur_resample`
  Downsamples and upsamples the packed map to simulate softened or low-detail predictions.

- `orientation_jitter`
  Rotates the orientation field by a smooth spatially varying angle field.

- `orientation_sign_flip`
  Flips the sign of the orientation field inside small local windows. This creates short-range direction discontinuities and encourages the DAE to restore locally coherent strand flow.

- `orientation_confidence`
  Reduces orientation magnitude with a smooth attenuation field to mimic weak or low-confidence strand direction estimates.

- `depth_holes`
  Removes depth in local regions while keeping the rest of the map intact.

- `depth_drift`
  Adds a smooth local bias field to the depth channel.

- `misregistration`
  Shifts orientation and depth relative to the support mask to simulate small channel alignment errors.

- `channel_inconsistency`
  Applies channel-specific corruption without enforcing full cross-channel consistency. In the current implementation this chooses either orientation-only attenuation or depth-only drift.

### Final-stage Transform

- `final_pose_jitter`
  Applies a small rigid 2D transform to the final corrupted map after the sampled corruption families have already run. This is intended to introduce slight pose variation in the DAE input without changing the clean reconstruction target. The orientation vectors are rotated consistently with the image transform.

## Config Knobs

### Global Sampler

- `dae.corruption.min_ops_per_sample`
  Minimum number of corruption families randomly sampled per training example.

- `dae.corruption.max_ops_per_sample`
  Maximum number of corruption families randomly sampled per training example.

These apply to normal DAE training, not the forced one-family-per-image demo mode.

### Final-stage Pose Jitter

- `dae.corruption.final_pose_jitter.enabled`
  Enables the last-step rigid jitter on the corrupted map.

- `dae.corruption.final_pose_jitter.apply_probability`
  Probability of applying the final rigid jitter to a sample.

- `dae.corruption.final_pose_jitter.max_rotation_deg`
  Maximum absolute in-plane rotation in degrees applied to the corrupted map.

- `dae.corruption.final_pose_jitter.max_translation_px`
  Maximum absolute x/y translation in pixels applied to the corrupted map.

### Family Parameters

#### `support`

- `weight`
  Relative sampling weight in the mixed corruption draw.

- `kernel_min`, `kernel_max`
  Range of odd morphology kernel sizes used when the support corruption chooses erosion or dilation.

- `max_holes`
  Maximum number of rectangular holes carved into the support mask when the hole branch is chosen.

- `hole_size_min`, `hole_size_max`
  Hole size as a fraction of image height and width.

- `trim_fraction_min`, `trim_fraction_max`
  Fraction of the image width or height removed from one randomly chosen boundary when the boundary-trim branch is chosen.

#### `strand_dropout`

- `weight`
  Relative sampling weight in the mixed corruption draw.

- `keep_ratio_min`, `keep_ratio_max`
  Range for the fraction of strands kept before rerendering. Lower values create more severe sparsification.

#### `blur_resample`

- `weight`
  Relative sampling weight in the mixed corruption draw.

- `scale_min`, `scale_max`
  Temporary downsample scale range before the packed map is upsampled back to full resolution. Smaller values remove more high-frequency detail.

#### `orientation_jitter`

- `weight`
  Relative sampling weight in the mixed corruption draw.

- `max_angle_deg`
  Maximum absolute local orientation rotation.

- `low_res`
  Smoothness control for the noise field. Larger values produce broader, slower-varying angular regions.

#### `orientation_sign_flip`

- `weight`
  Relative sampling weight in the mixed corruption draw.

- `max_regions`
  Maximum number of small local windows that can invert the orientation sign.

- `size_min`, `size_max`
  Window size as a fraction of image height and width.

#### `orientation_confidence`

- `weight`
  Relative sampling weight in the mixed corruption draw.

- `low_res`
  Smoothness control for the attenuation field applied to orientation magnitude.

- `attenuation_min`, `attenuation_max`
  Lower and upper bounds for the multiplicative attenuation applied to `dx/dy`.

#### `depth_holes`

- `weight`
  Relative sampling weight in the mixed corruption draw.

- `max_holes`
  Maximum number of local depth-dropout regions.

- `hole_size_min`, `hole_size_max`
  Region size as a fraction of image height and width.

#### `depth_drift`

- `weight`
  Relative sampling weight in the mixed corruption draw.

- `max_abs_bias`
  Maximum absolute additive bias applied to the normalized depth channel.

- `low_res`
  Smoothness control for the drift field. Larger values produce broader, slower-varying depth bias.

#### `misregistration`

- `weight`
  Relative sampling weight in the mixed corruption draw.

- `max_shift_px`
  Maximum integer pixel translation applied to orientation and depth relative to the support mask.

#### `partial_occlusion`

- `weight`
  Relative sampling weight in the mixed corruption draw. `0.0` disables it in normal training while still allowing the demo script to force it.

- `max_regions`
  Maximum number of larger occlusion regions carved from the support mask.

- `size_min`, `size_max`
  Occlusion size as a fraction of image height and width.

#### `channel_inconsistency`

- `weight`
  Relative sampling weight in the mixed corruption draw.

This family currently has no extra parameters because it chooses between an orientation-only corruption and a depth-only corruption internally.

## Demo Config

- `dae.demo.phase`
  Which dataloader split the demo batch is drawn from.

- `dae.demo.max_samples`
  Number of samples shown in each saved image.

- `dae.demo.include_weighted_mix`
  Whether to save one mixed-sampling example in addition to the per-family images.

- `dae.demo.include_disabled_families`
  Whether families with `weight: 0.0` should still appear in the forced demo list.

- `dae.demo.families`
  Optional explicit subset of family names to render. Leave empty to use every configured family.

- `dae.demo.output_dir`
  Optional fixed output directory. If omitted, the script writes under `train.log_path/corruption_demo_<timestamp>`.

- `dae.demo.seed`
  Random seed used for the demo run.

## Available But Disabled In V1

- `partial_occlusion`
  Removes larger contiguous support regions to mimic partial foreground occlusion. This is implemented but disabled by default in v1.

## Not Yet Implemented, But Tracked

These were discussed and should stay documented even if not active in the first implementation:

- explicit face or shoulder-shaped occluders instead of simple rectangles
- stronger crop/scale mismatch between channels
- renderer-domain artifacts such as structured raster gaps
- corruption families tied to specific thin-wisp failure cases
- deterministic validation corruption presets for strict metric comparability

## Notes

- The DAE should not use long input-output skip connections.
- The corruption config is intentionally separate from the main hair-synthesis config.
- The enabled mix is meant to mimic plausible map failures without destroying hairstyle identity.
