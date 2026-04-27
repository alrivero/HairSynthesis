import os
import os.path as osp
import numpy as np
from datasets.base_dataset import BaseHairDataset
import cv2
import json
from collections import defaultdict
import imageio.v2 as imageio
from pathlib import Path
from datasets.config_utils import get_dataset_section


class FFHQDataset(BaseHairDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.name = 'FFHQ'
        self.dataset_cfg = get_dataset_section(config, 'FFHQ')
        self.mediapipe_dir = getattr(self.dataset_cfg, 'FFHQ_mediapipe_landmarks_path', None)
        self.face_mask_dir = getattr(
            self.dataset_cfg,
            'FFHQ_facemask',
            getattr(self.dataset_cfg, 'FFHQ_Facemask', None),
        )
        apply_person_mask = _get_bool(self.dataset_cfg, 'FFHQ_apply_person_mask_to_image', False)
        apply_body_mask = _get_bool(self.dataset_cfg, 'FFHQ_apply_body_mask_to_image', False)
        apply_face_hair_mask = _get_bool(self.dataset_cfg, 'FFHQ_apply_face_hair_mask_to_image', False)
        image_mask_mode = getattr(self.dataset_cfg, 'FFHQ_apply_mask_to_image', 'none')
        if apply_person_mask or apply_body_mask:
            image_mask_mode = 'body'
        if apply_face_hair_mask:
            image_mask_mode = 'face_hair'
        self.image_mask_mode = self._resolve_image_mask_mode(
            image_mask_mode
        )

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
        soft_hairmask = self._load_soft_mask(hair_path)
        soft_bodymask = self._load_soft_mask(body_path)
        soft_personmask = np.maximum(soft_bodymask, soft_hairmask)
        hairmask = soft_hairmask
        personmask = soft_personmask
        encoder_hairmask = soft_hairmask
        landmarks_mediapipe = self._load_mediapipe_landmarks(img_path)
        face_mask = self._load_face_mask(img_path) if self.face_mask_dir else None
        image_mask = soft_personmask if self.image_mask_mode == 'body' else None
        if self.image_mask_mode == 'face_hair':
            if face_mask is None:
                return None
            image_mask = np.maximum(face_mask, soft_hairmask)

        data_dict = self.prepare_data(
            image=image,
            hairmask=hairmask,
            bodymask=personmask,
            landmarks_mediapipe=landmarks_mediapipe,
            image_mask_mode=self.image_mask_mode,
            image_mask=image_mask,
            face_mask=face_mask,
            encoder_hairmask=encoder_hairmask,
            encoder_bodymask=soft_personmask,
        )
        
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

    def _load_face_mask(self, img_path):
        if not self.face_mask_dir:
            print('Face mask directory not configured for FFHQ face_hair image masking')
            return None

        filename = os.path.basename(img_path)
        face_mask_path = os.path.join(self.face_mask_dir, filename)
        if not os.path.exists(face_mask_path):
            print('Face mask not found for %s' % (face_mask_path))
            return None
        return self._load_soft_mask(face_mask_path)

    @staticmethod
    def _load_soft_mask(mask_path):
        mask = imageio.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]

        mask = mask.astype(np.float32)
        max_value = float(mask.max()) if mask.size else 0.0
        if max_value > 1.0:
            mask = mask / max_value
        return np.clip(mask, 0.0, 1.0)[..., None]


def get_datasets_FFHQ(config, split_file=None):
    from sklearn.model_selection import train_test_split
    ffhq_cfg = get_dataset_section(config, 'FFHQ')
    include_names = _load_include_names(_get_optional_path(ffhq_cfg, 'FFHQ_include_list'))
    split_data = _load_or_create_split_manifest(
        config=config,
        split_file=split_file,
        include_names=include_names,
        train_test_split=train_test_split,
    )
    train_idx = split_data['training']
    val_idx = split_data['validation']
    test_idx = split_data['test']

    train_list = [
        [osp.join(ffhq_cfg.FFHQ_path, i),
        osp.join(ffhq_cfg.FFHQ_hairmask, i),
        osp.join(ffhq_cfg.FFHQ_bodymask, i)]
        for i in sorted(train_idx)
    ]
    val_list = [
        [osp.join(ffhq_cfg.FFHQ_path, i),
        osp.join(ffhq_cfg.FFHQ_hairmask, i),
        osp.join(ffhq_cfg.FFHQ_bodymask, i)]
        for i in sorted(val_idx)
    ]
    test_list = [
        [osp.join(ffhq_cfg.FFHQ_path, i),
        osp.join(ffhq_cfg.FFHQ_hairmask, i),
        osp.join(ffhq_cfg.FFHQ_bodymask, i)]
        for i in sorted(test_idx)
    ]
        
    dataset = FFHQDataset(train_list, config), FFHQDataset(val_list, config, test=True), FFHQDataset(test_list, config, test=True)
    return dataset


def ensure_ffhq_split_manifest(config, split_file=None):
    ffhq_cfg = get_dataset_section(config, 'FFHQ')
    include_names = _load_include_names(_get_optional_path(ffhq_cfg, 'FFHQ_include_list'))
    manifest_path = _resolve_split_manifest_path(config, split_file)

    if manifest_path.exists():
        return str(manifest_path)

    from sklearn.model_selection import train_test_split

    _load_or_create_split_manifest(
        config=config,
        split_file=str(manifest_path),
        include_names=include_names,
        train_test_split=train_test_split,
    )
    return str(manifest_path)


def _resolve_split_manifest_path(config, split_file=None):
    if split_file is not None:
        return Path(split_file)
    return Path(config.train.log_path) / 'splits.json'


def _load_or_create_split_manifest(config, split_file, include_names, train_test_split):
    manifest_path = _resolve_split_manifest_path(config, split_file)
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        return _normalize_split_manifest(manifest)

    manifest = _build_split_manifest(
        config=config,
        include_names=include_names,
        train_test_split=train_test_split,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    return _normalize_split_manifest(manifest)


def _build_split_manifest(config, include_names, train_test_split):
    ffhq_cfg = get_dataset_section(config, 'FFHQ')
    split_seed = 42

    with open(ffhq_cfg.FFHQ_meta_path, 'r') as f:
        meta = json.load(f)

    categories = defaultdict(list)
    for value in meta.values():
        category = value['category']
        file_path = value['image']['file_path'].split('/')[-1]
        categories[category].append(file_path)
    categories = dict(categories)
    categories = {
        category: _filter_include_names(files, include_names)
        for category, files in categories.items()
    }

    train_idx, val_idx = train_test_split(
        categories['training'],
        test_size=0.2,
        random_state=split_seed,
    )
    test_idx = categories['validation']

    subset_fraction = float(ffhq_cfg.FFHQ_percentage_subset)
    if subset_fraction != 1.0:
        train_idx = train_idx[:int(len(train_idx) * subset_fraction)]
        val_idx = val_idx[:int(len(val_idx) * subset_fraction)]
        test_idx = test_idx[:int(len(test_idx) * subset_fraction)]

    split_names = {
        'training': sorted(train_idx),
        'validation': sorted(val_idx),
        'test': sorted(test_idx),
    }

    return {
        'format_version': 2,
        'dataset': 'FFHQ',
        'seed': split_seed,
        'counts': {name: len(entries) for name, entries in split_names.items()},
        'source': {
            'ffhq_meta_path': ffhq_cfg.FFHQ_meta_path,
            'include_list_path': _get_optional_path(ffhq_cfg, 'FFHQ_include_list'),
            'percentage_subset': subset_fraction,
            'ffhq_path': ffhq_cfg.FFHQ_path,
            'hairmask_path': ffhq_cfg.FFHQ_hairmask,
            'bodymask_path': ffhq_cfg.FFHQ_bodymask,
        },
        'training': split_names['training'],
        'validation': split_names['validation'],
        'test': split_names['test'],
        'records': {
            split_name: _build_split_records(entries, ffhq_cfg)
            for split_name, entries in split_names.items()
        },
    }


def _build_split_records(entries, ffhq_cfg):
    return [
        {
            'file_name': file_name,
            'image_path': osp.join(ffhq_cfg.FFHQ_path, file_name),
            'hairmask_path': osp.join(ffhq_cfg.FFHQ_hairmask, file_name),
            'bodymask_path': osp.join(ffhq_cfg.FFHQ_bodymask, file_name),
        }
        for file_name in entries
    ]


def _normalize_split_manifest(manifest):
    if isinstance(manifest, dict):
        training = _extract_split_entries(manifest, 'training')
        validation = _extract_split_entries(manifest, 'validation')
        test = _extract_split_entries(manifest, 'test')
        if training is not None and validation is not None and test is not None:
            return {
                'training': training,
                'validation': validation,
                'test': test,
            }
    raise ValueError("Split manifest must contain training/validation/test entries.")


def _extract_split_entries(manifest, split_name):
    entries = manifest.get(split_name)
    if isinstance(entries, list):
        if not entries:
            return []
        if isinstance(entries[0], dict):
            return [entry['file_name'] for entry in entries]
        return list(entries)

    records = manifest.get('records', {}).get(split_name)
    if isinstance(records, list):
        return [entry['file_name'] for entry in records]

    return None


def _get_optional_path(cfg, key):
    value = getattr(cfg, key, None)
    if value is None:
        return None

    path = str(value).strip()
    if path.lower() in ('', 'none', 'null'):
        return None
    return path


def _get_bool(cfg, key, default=False):
    value = getattr(cfg, key, default)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    return bool(value)


def _load_include_names(include_list_path):
    if include_list_path is None:
        return None

    path = Path(include_list_path)
    if path.suffix.lower() == '.json':
        with open(path, 'r') as f:
            names = _extract_include_names(json.load(f))
    else:
        with open(path, 'r') as f:
            names = [line.strip() for line in f]

    include_names = {
        os.path.basename(str(name).strip())
        for name in names
        if str(name).strip()
    }
    if not include_names:
        raise ValueError(f"FFHQ include list is empty: {include_list_path}")
    return include_names


def _extract_include_names(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ('kept', 'included', 'include', 'images', 'names'):
            if key in data:
                return data[key]
    raise ValueError("JSON include list must be a list or contain one of: kept, included, include, images, names.")


def _filter_include_names(names, include_names):
    if include_names is None:
        return names
    return [name for name in names if os.path.basename(str(name)) in include_names]
