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
        image = cv2.imread(self.data_list[index][0])

        # check if paths exist
        if not os.path.exists(self.data_list[index][1]):
            print('Hairmask not found for %s'%(self.data_list[index]))
            return None
        if not os.path.exists(self.data_list[index][2]):
            print('Bodymask not found for %s'%(self.data_list[index]))
            return None

        # hairmask = cv2.imread(self.data_list[index][1], cv2.IMREAD_GRAYSCALE)
        # bodymask = cv2.imread(self.data_list[index][2], cv2.IMREAD_GRAYSCALE)
        hairmask = (imageio.imread(self.data_list[index][1])/255.>0.5)[:,:,None]
        bodymask = ((imageio.imread(self.data_list[index][2])[:,:,0]/255.>0.5)[:,:,None])*(1-hairmask)

        data_dict = self.prepare_data(image=image, hairmask=hairmask, bodymask=bodymask)
        
        return data_dict


def get_datasets_FFHQ(config):
    from sklearn.model_selection import train_test_split
    import os.path as osp

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

    train_idx, val_idx = train_test_split(categories['training'], test_size=0.2, random_state=1234)
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





