## Installation
Using smirk env, install as needed:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Folder
```bash
cd /gpfs/projects/CascanteBonillaGroup/thinguyen/storage/smirk/external/HairStep
```

## Commands
### Precomputed hairmask
```bash
CUDA_VISIBLE_DEVICES=3 python -m scripts.img2masks --root_real_imgs "/gpfs/projects/CascanteBonillaGroup/thinguyen/datasets/FFHQ256"

CUDA_VISIBLE_DEVICES=3 python -m scripts.img2masks --root_real_imgs "/gpfs/projects/CascanteBonillaGroup/thinguyen/datasets/CelebA"
```

### Training
```bash
python train.py configs/config_train_hs.yaml
```

### Evaluation
```bash
python demo.py --input_path samples/test_image2.png --out_path results/ --checkpoint logs/smirkhair/20251115_202905/model_0.pt
python demo_hair.py --input_path samples/00000.png --out_path results/ --checkpoint logs/smirkhair/20251116_035918/model_32.pt --device cpu --hairmask_path /gpfs/projects/CascanteBonillaGroup/thinguyen/datasets/FFHQ256/processed/hairstep/seg --bodymask_path /gpfs/projects/CascanteBonillaGroup/thinguyen/datasets/FFHQ256/processed/hairstep/body_img


# python eval.py configs/config_eval_hs.yaml

python demo_hair.py --input_path /gpfs/projects/CascanteBonillaGroup/thinguyen/datasets/FFHQ256/processed/hairstep/resized_img --out_path results/ --checkpoint logs/smirkhair/20251116_035918/model_32.pt --device cpu --hairmask_path /gpfs/projects/CascanteBonillaGroup/thinguyen/datasets/FFHQ256/processed/hairstep/seg --bodymask_path /gpfs/projects/CascanteBonillaGroup/thinguyen/datasets/FFHQ256/processed/hairstep/body_img
```

