import os
import cv2
import json
from pathlib import Path
import imageio.v2 as imageio
from collections import defaultdict
from PIL import Image
from torchvision import transforms
from datasets.base_dataset import BaseHairDataset


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


class HiSADataset(BaseHairDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.name = 'HiSA'

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
        hairmask = (imageio.imread(hair_path)[:,:,0]/255.>0.5)[:,:,None]        # (H, W, 1)
        bodymask = ((imageio.imread(body_path)[:,:,0]/255.>0.5)[:,:,None])*(1-hairmask)     # (H, W, 1)

        data_dict = self.prepare_data(image=image, hairmask=hairmask, bodymask=bodymask)
        
        # load strand gt from HiSA
        strand_gt = load_hisa_strand(strand_path)
        data_dict['strand_gt'] = strand_gt

        return data_dict


def get_datasets_HiSA(config):
    # No train, validation, test; load everything
    image_list = sorted(os.listdir(config.dataset.HiSA_path))

    data_list = [
        [os.path.join(config.dataset.HiSA_path, i),
        os.path.join(config.dataset.HiSA_hairmask, i),
        os.path.join(config.dataset.HiSA_bodymask, i),
        os.path.join(config.dataset.HiSA_strand, i),
        ]
        for i in sorted(image_list)
    ]
        
    dataset = HiSADataset(data_list, config, test=True)
    return dataset
