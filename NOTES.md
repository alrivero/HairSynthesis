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
```
