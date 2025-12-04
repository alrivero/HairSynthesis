import os
import numpy as np
from datasets.base_dataset import BaseHairDataset
import cv2
import numpy as np
import json
from collections import defaultdict
import imageio
from pathlib import Path


class FFHQDataset(BaseHairDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.name = 'FFHQ'

    def __getitem_aux__(self, index):
        img_path = self.data_list[index][0]
        hair_path = self.data_list[index][1]
        body_path = self.data_list[index][2]

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
        
        image = cv2.imread(img_path)
        hairmask = (imageio.imread(hair_path)/255.>0.5)[:,:,None]
        bodymask = ((imageio.imread(body_path)[:,:,0]/255.>0.5)[:,:,None])*(1-hairmask)

        data_dict = self.prepare_data(image=image, hairmask=hairmask, bodymask=bodymask)
        
        return data_dict


def get_datasets_FFHQ(config, split_file=None):
    from sklearn.model_selection import train_test_split
    import os.path as osp
    
    if split_file is None:
        # Read train and validation list
        # {'training': <list of train images>, 'validation': <list of val images>}
        with open(config.dataset.FFHQ_meta_path, 'r') as f:
            meta = json.load(f)
        categories = defaultdict(list)
        for v in meta.values():
            category = v['category']
            file_path = v['image']['file_path'].split('/')[-1]

            categories[category].append(file_path)
        categories = dict(categories)

        train_idx, val_idx = train_test_split(categories['training'], test_size=0.2, random_state=42)
        test_idx = categories['validation']

        if config.dataset.FFHQ_percentage_subset != 1:
            subset_perc = config.dataset.FFHQ_percentage_subset
            train_idx = train_idx[:int(len(train_idx)*subset_perc)]
            val_idx = val_idx[:int(len(val_idx)*subset_perc)]
            test_idx = test_idx[:int(len(test_idx)*subset_perc)]

        split_dir = Path(config.train.log_path)
        final_splits = {'training': train_idx, 'validation': val_idx, 'test': test_idx}
        split_json = split_dir/'splits.json'
        assert not split_json.exists(), f"{split_json} existed"
        with open(split_json, 'w') as f:
            json.dump(final_splits, f, indent=4)
    else:
        with open(split_file, 'r') as f:
            final_splits = json.load(f)
        train_idx = final_splits['training']
        val_idx = final_splits['validation']
        test_idx = final_splits['test']

    train_list = [
        [osp.join(config.dataset.FFHQ_path, i),
        osp.join(config.dataset.FFHQ_hairmask, i),
        osp.join(config.dataset.FFHQ_bodymask, i)]
        for i in sorted(train_idx)
    ]
    val_list = [
        [osp.join(config.dataset.FFHQ_path, i),
        osp.join(config.dataset.FFHQ_hairmask, i),
        osp.join(config.dataset.FFHQ_bodymask, i)]
        for i in sorted(val_idx)
    ]
    test_list = [
        [osp.join(config.dataset.FFHQ_path, i),
        osp.join(config.dataset.FFHQ_hairmask, i),
        osp.join(config.dataset.FFHQ_bodymask, i)]
        for i in sorted(test_idx)
    ]
        
    dataset = FFHQDataset(train_list, config), FFHQDataset(val_list, config, test=True), FFHQDataset(test_list, config, test=True)
    return dataset





