import os
import cv2
import json
import numpy as np
from pathlib import Path
import imageio.v2 as imageio
from collections import defaultdict
from PIL import Image
from torchvision import transforms
from datasets.base_dataset import BaseHairDataset
from datasets.config_utils import get_dataset_section


def load_hisa_strand(strand_path, load_size=512):
    """
    Load HiSA strand ground truth
        (3, H, W): 3 channels: hair mask, x, y

    Follows load_hairstep() in HairStep/scripts/recon3D.py
    """
    img_to_tensor = transforms.Compose([
        transforms.Resize(load_size),
        transforms.ToTensor(),                     # [0,1]
        transforms.Normalize((0.5, 0.5, 0.5),      # to [-1,1]
                             (0.5, 0.5, 0.5)),
    ])

    raw = Image.open(strand_path).convert("RGB")
    strand = img_to_tensor(raw).float()           # (3, H, W)
    return strand


def load_soft_mask(mask_path):
    mask = imageio.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(np.float32)
    max_value = float(mask.max()) if mask.size else 0.0
    if max_value > 1.0:
        mask = mask / max_value
    return np.clip(mask, 0.0, 1.0)[..., None]


class HiSADataset(BaseHairDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.name = 'HiSA'
        self.dataset_cfg = get_dataset_section(config, 'HiSA')
        self.mediapipe_dir = getattr(self.dataset_cfg, 'HiSA_mediapipe_landmarks_path', None)

    def __getitem_aux__(self, index):
        img_path = self.data_list[index][0]     # input image
        hair_path = self.data_list[index][1]    # hair segmentation image
        body_path = self.data_list[index][2]
        strand_path = self.data_list[index][3]

        # check if paths exist
        if not os.path.exists(img_path):
            print('Image not found for %s'%(img_path))
            return None
        if not os.path.exists(hair_path):
            print('Hairmask not found for %s'%(hair_path))
            return None
        if not os.path.exists(body_path):
            print('Bodymask not found for %s'%(body_path))
            return None
        
        image = cv2.imread(img_path)        # (H, W, 3)
        hairmask = load_soft_mask(hair_path)
        bodymask = load_soft_mask(body_path) * (1.0 - hairmask)
        landmarks_mediapipe = self._load_mediapipe_landmarks(img_path)

        data_dict = self.prepare_data(
            image=image,
            hairmask=hairmask,
            bodymask=bodymask,
            landmarks_mediapipe=landmarks_mediapipe,
        )
        
        # load strand gt from HiSA
        strand_gt = load_hisa_strand(strand_path)
        data_dict['strand_gt'] = strand_gt

        return data_dict

    def _load_mediapipe_landmarks(self, img_path):
        if not self.mediapipe_dir:
            return None

        filename = os.path.basename(img_path)
        landmark_name = os.path.splitext(filename)[0] + '.npy'
        landmark_path = os.path.join(self.mediapipe_dir, landmark_name)
        if not os.path.exists(landmark_path):
            return None
        return np.load(landmark_path, allow_pickle=True).astype(np.float32)


def get_datasets_HiSA(config):
    # No train, validation, test; load everything
    hisa_cfg = get_dataset_section(config, 'HiSA')
    image_list = sorted(os.listdir(hisa_cfg.HiSA_path))

    data_list = [
        [os.path.join(hisa_cfg.HiSA_path, i),
        os.path.join(hisa_cfg.HiSA_hairmask, i),
        os.path.join(hisa_cfg.HiSA_bodymask, i),
        os.path.join(hisa_cfg.HiSA_strand, i),
        ]
        for i in sorted(image_list)
    ]
        
    dataset = HiSADataset(data_list, config, test=True)
    return dataset
