import torch.utils.data
from skimage.transform import estimate_transform, warp
import albumentations as A
import numpy as np
from skimage import transform as trans
import cv2
from collections import abc


def create_mask(landmarks, shape):
    landmarks = landmarks.astype(np.int32)[...,:2]
    hull = cv2.convexHull(landmarks)
    mask = np.ones(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 0)

    return mask

# these are the indices of the mediapipe landmarks that correspond to the mediapipe landmark barycentric coordinates provided by FLAME2020
mediapipe_indices = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336,  46,  52,  53,
        55,  63,  65,  66,  70, 105, 107, 249, 263, 362, 373, 374, 380,
       381, 382, 384, 385, 386, 387, 388, 390, 398, 466,   7,  33, 133,
       144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
       168,   6, 197, 195,   5,   4, 129,  98,  97,   2, 326, 327, 358,
         0,  13,  14,  17,  37,  39,  40,  61,  78,  80,  81,  82,  84,
        87,  88,  91,  95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
       308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
       415]


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    try:
        if key in cfg:
            return cfg[key]
    except Exception:
        pass
    return getattr(cfg, key, default)


def _cfg_enabled(cfg, default=True):
    return bool(_cfg_get(cfg, 'enabled', default))


def _as_tuple(value):
    if isinstance(value, abc.Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return value


def _append_if_enabled(transforms, cfg, default_p, factory):
    p = float(_cfg_get(cfg, 'p', default_p))
    if not _cfg_enabled(cfg, p > 0.0) or p <= 0.0:
        return
    transforms.append(factory(cfg, p))


def build_color_augmentation(config):
    return [
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
        A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.25),
        A.CLAHE(p=0.255),
        A.RGBShift(p=0.25),
        A.Blur(p=0.1),
        A.GaussNoise(p=0.5),
    ]


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, config, test=False):
        self.data_list = data_list
        self.config = config
        self.image_size = config.image_size
        self.test = test

        if not self.test:
            self.scale = [config.train.train_scale_min, config.train.train_scale_max] 
        else:
            self.scale = config.train.test_scale
        
        self.transform = A.Compose([
                # color ones
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.25),
                A.CLAHE(p=0.255),
                A.RGBShift(p=0.25),
                A.Blur(p=0.1),
                A.GaussNoise(p=0.5),
                # affine ones
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.9),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),  additional_targets={'mediapipe_keypoints': 'keypoints'})


        self.arcface_dst = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32)


    def estimate_norm(self, lmk, image_size=112,mode='arcface'):
        assert lmk.shape == (5, 2)
        assert image_size%112==0 or image_size%128==0
        if image_size%112==0:
            ratio = float(image_size)/112.0
            diff_x = 0
        else:
            ratio = float(image_size)/128.0
            diff_x = 8.0*ratio
        dst = self.arcface_dst * ratio
        dst[:,0] += diff_x
        tform = trans.SimilarityTransform()
        tform.estimate(lmk, dst)
        M = tform.params[0:2, :]
        return M

    @staticmethod
    def crop_face(frame, landmarks, scale=1.0, image_size=224):
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])

        h, w, _ = frame.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        return tform

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        landmarks_not_checked = True
        while landmarks_not_checked:
            try:
                data_dict = self.__getitem_aux__(index)
                # check if landmarks are not None
                if data_dict is not None:
                    landmarks = data_dict['landmarks_fan']
                    if landmarks is not None and (landmarks.shape[-2] == 68):
                        landmarks_not_checked = False
                        break
                #else:
                print("Error in loading data. Trying again...")
                index = np.random.randint(0, len(self.data_list))
            except Exception as e:
                # raise e
                print('Error in loading data. Trying again...', e)
                index = np.random.randint(0, len(self.data_list))


        return data_dict

    def prepare_data(self, image, landmarks_fan, landmarks_mediapipe):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


        if landmarks_fan is None:
            flag_landmarks_fan = False
            landmarks_fan = np.zeros((68,2))
        else:
            flag_landmarks_fan = True
            
        # crop the image using the landmarks
        if isinstance(self.scale, list):
            # select randomly a zoom scale during training for cropping
            scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        else:
            scale = self.scale

        tform = self.crop_face(image,landmarks_mediapipe,scale,image_size=self.image_size)
        
        landmarks_mediapipe = landmarks_mediapipe[...,:2]
        
        cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size), preserve_range=True).astype(np.uint8)
        cropped_landmarks_fan = np.dot(tform.params, np.hstack([landmarks_fan, np.ones([landmarks_fan.shape[0],1])]).T).T
        cropped_landmarks_fan = cropped_landmarks_fan[:,:2]


        cropped_landmarks_mediapipe = np.dot(tform.params, np.hstack([landmarks_mediapipe, np.ones([landmarks_mediapipe.shape[0],1])]).T).T
        cropped_landmarks_mediapipe = cropped_landmarks_mediapipe[:,:2]

        # find convex hull for masking the face 
        hull_mask = create_mask(cropped_landmarks_mediapipe, (self.image_size, self.image_size))

        cropped_landmarks_mediapipe = cropped_landmarks_mediapipe[mediapipe_indices,:2]


        # augment
        if not self.test:
            transformed = self.transform(image=cropped_image, mask= 1 - hull_mask, keypoints=cropped_landmarks_fan, mediapipe_keypoints=cropped_landmarks_mediapipe)

            cropped_image = (transformed['image']/255.0).astype(np.float32)
            cropped_landmarks_fan = np.array(transformed['keypoints']).astype(np.float32)
            cropped_landmarks_mediapipe = np.array(transformed['mediapipe_keypoints']).astype(np.float32)
            hull_mask = 1 - transformed['mask']
        else: 
            cropped_image = (cropped_image/255.0).astype(np.float32)
            cropped_landmarks_fan = cropped_landmarks_fan.astype(np.float32)
            cropped_landmarks_mediapipe = cropped_landmarks_mediapipe.astype(np.float32)
            

        cropped_landmarks_fan[:,:2] = cropped_landmarks_fan[:,:2]/self.image_size * 2  - 1
        cropped_landmarks_mediapipe[:,:2] = cropped_landmarks_mediapipe[:,:2]/self.image_size * 2  - 1
        masked_cropped_image = cropped_image * hull_mask[...,None]
        

        cropped_image = cropped_image.transpose(2,0,1)
        masked_cropped_image = masked_cropped_image.transpose(2,0,1)
        hull_mask = hull_mask[...,None]
        hull_mask = hull_mask.transpose(2,0,1)


        # ----------- mica images ---------------- #
        landmarks_arcface_crop = landmarks_fan[[36,45,32,48,54]].copy()
        landmarks_arcface_crop[0] = (landmarks_fan[36] + landmarks_fan[39])/2
        landmarks_arcface_crop[1] = (landmarks_fan[42] + landmarks_fan[45])/2

        tform = self.estimate_norm(landmarks_arcface_crop, 112)

        image = image/255.
        mica_image = cv2.warpAffine(image, tform, (112, 112), borderValue=0.0)
        mica_image = mica_image.transpose(2,0,1)



        image = torch.from_numpy(cropped_image).type(dtype = torch.float32) 
        masked_image = torch.from_numpy(masked_cropped_image).type(dtype = torch.float32)
        landmarks_fan = torch.from_numpy(cropped_landmarks_fan).type(dtype = torch.float32) 
        landmarks_mediapipe = torch.from_numpy(cropped_landmarks_mediapipe).type(dtype = torch.float32)
        hull_mask = torch.from_numpy(hull_mask).type(dtype = torch.float32)
        mica_image = torch.from_numpy(mica_image).type(dtype = torch.float32) 


        data_dict = {
            'img': image,
            'landmarks_fan': landmarks_fan[...,:2],
            'flag_landmarks_fan': flag_landmarks_fan, # if landmarks are not available
            'landmarks_mp': landmarks_mediapipe[...,:2],
            'mask': hull_mask,
            'img_mica': mica_image
        }


        return data_dict


class BaseHairDataset(BaseDataset):
    def __init__(self, data_list, config, test=False):
        self.data_list = data_list
        self.config = config
        self.test = test
        self.target_resolution = self._resolve_target_resolution(config)
        self.smirk_image_size = int(getattr(config, 'image_size', 224))
        self.smirk_crop_scale = self._resolve_smirk_crop_scale(config)

        self.transform = A.Compose(build_color_augmentation(config), additional_targets={
                'hairmask': 'mask',
                'bodymask': 'mask',
                'face_mask': 'mask',
                'image_mask': 'mask',
                'encoder_hairmask': 'mask',
                'encoder_bodymask': 'mask',
            })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        max_attempts = max(1, min(len(self.data_list), 20))
        for attempt in range(max_attempts):
            try:
                data_dict = self.__getitem_aux__(index)
                if data_dict is not None:
                    return data_dict
                print(f"Invalid hair sample at index {index}. Trying again...")
            except Exception as exc:
                print(f"Error in loading hair data at index {index}. Trying again... {exc}")

            index = np.random.randint(0, len(self.data_list))

        raise RuntimeError("Failed to fetch a valid hair sample after multiple attempts.")

    def prepare_data(
        self,
        image,
        hairmask,
        bodymask,
        landmarks_mediapipe=None,
        image_mask_mode='none',
        image_mask=None,
        face_mask=None,
        encoder_hairmask=None,
        encoder_bodymask=None,
    ):
        orig_h, orig_w = image.shape[:2]
        image = self._resize_image(image)
        hairmask = self._resize_soft_mask(hairmask)
        bodymask = self._resize_soft_mask(bodymask)
        if image_mask is not None:
            image_mask = self._resize_soft_mask(image_mask)
        if face_mask is None:
            face_mask = np.zeros_like(hairmask, dtype=np.float32)
        else:
            face_mask = self._resize_soft_mask(face_mask)
        if encoder_hairmask is None:
            encoder_hairmask = hairmask
        else:
            encoder_hairmask = self._resize_soft_mask(encoder_hairmask)
        if encoder_bodymask is None:
            encoder_bodymask = bodymask
        else:
            encoder_bodymask = self._resize_soft_mask(encoder_bodymask)
        hairmask = hairmask.astype(np.float32)
        bodymask = bodymask.astype(np.float32)
        if image_mask is not None:
            image_mask = image_mask.astype(np.float32)
        face_mask = face_mask.astype(np.float32)
        encoder_hairmask = encoder_hairmask.astype(np.float32)
        encoder_bodymask = encoder_bodymask.astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target_h, target_w = image.shape[:2]

        resized_landmarks = None
        if landmarks_mediapipe is not None:
            resized_landmarks = self._resize_landmarks(landmarks_mediapipe, orig_h, orig_w, target_h, target_w)
        # augment
        if not self.test:
            transform_inputs = {
                'image': image,
                'hairmask': hairmask,
                'bodymask': bodymask,
                'face_mask': face_mask,
                'encoder_hairmask': encoder_hairmask,
                'encoder_bodymask': encoder_bodymask,
            }
            if image_mask is not None:
                transform_inputs['image_mask'] = image_mask

            transformed = self.transform(
                **transform_inputs,
            )

            image = transformed['image']
            hairmask = transformed['hairmask'].astype(np.float32)
            bodymask = transformed['bodymask'].astype(np.float32)
            face_mask = transformed['face_mask'].astype(np.float32)
            encoder_hairmask = transformed['encoder_hairmask'].astype(np.float32)
            encoder_bodymask = transformed['encoder_bodymask'].astype(np.float32)
            if image_mask is not None:
                image_mask = transformed['image_mask'].astype(np.float32)
        else: 
            hairmask = hairmask.astype(np.float32)
            bodymask = bodymask.astype(np.float32)
            face_mask = face_mask.astype(np.float32)
            encoder_hairmask = encoder_hairmask.astype(np.float32)
            encoder_bodymask = encoder_bodymask.astype(np.float32)
            if image_mask is not None:
                image_mask = image_mask.astype(np.float32)

        image = self._apply_image_mask(
            image,
            hairmask,
            bodymask,
            image_mask_mode,
            image_mask=image_mask,
        )
        image = (image / 255.0).astype(np.float32)

        smirk_data = self._prepare_smirk_data((image * 255.0).astype(np.uint8), resized_landmarks)

        if float(np.asarray(hairmask, dtype=np.float32).sum()) <= 1e-6:
            return None
        
        image = image.transpose(2,0,1)
        hairmask = hairmask.transpose(2,0,1)
        bodymask = bodymask.transpose(2,0,1)
        face_mask = face_mask.transpose(2,0,1)
        encoder_hairmask = encoder_hairmask.transpose(2,0,1)
        encoder_bodymask = encoder_bodymask.transpose(2,0,1)

        data_dict = {
            'img': image,
            'hairmask': hairmask,
            'bodymask': bodymask,
            'face_mask': face_mask,
            'encoder_hairmask': encoder_hairmask,
            'encoder_bodymask': encoder_bodymask,
        }
        data_dict.update(smirk_data)
        return data_dict

    def _resolve_image_mask_mode(self, mask_mode):
        if mask_mode is None:
            return 'none'

        normalized = str(mask_mode).strip().lower()
        aliases = {
            '': 'none',
            'none': 'none',
            'off': 'none',
            'false': 'none',
            'no': 'none',
            'body': 'body',
            'bodymask': 'body',
            'body_mask': 'body',
            'hair': 'hair',
            'hairmask': 'hair',
            'hair_mask': 'hair',
            'face_hair': 'face_hair',
            'facehair': 'face_hair',
            'face-hair': 'face_hair',
            'face+hair': 'face_hair',
            'face_and_hair': 'face_hair',
        }
        if normalized not in aliases:
            raise ValueError(
                f"Unsupported image mask mode '{mask_mode}'. Expected one of: none, body, hair, face_hair."
            )
        return aliases[normalized]

    def _apply_image_mask(self, image, hairmask, bodymask, mask_mode, image_mask=None):
        resolved_mode = self._resolve_image_mask_mode(mask_mode)
        if resolved_mode == 'none':
            return image

        if resolved_mode in {'body', 'face_hair'} and image_mask is not None:
            mask = image_mask
        else:
            mask = bodymask if resolved_mode == 'body' else hairmask
        if mask.ndim == 2:
            mask = mask[..., None]
        if mask.shape[:2] != image.shape[:2]:
            mask = self._resize_soft_mask(mask)
            if mask.shape[:2] != image.shape[:2]:
                target_h, target_w = image.shape[:2]
                resized_channels = []
                for channel_idx in range(mask.shape[2]):
                    resized_channels.append(
                        cv2.resize(
                            mask[..., channel_idx].astype(np.float32),
                            (target_w, target_h),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    )
                mask = np.stack(resized_channels, axis=-1)
        original_dtype = image.dtype
        masked = image.astype(np.float32) * mask.astype(np.float32, copy=False)
        if np.issubdtype(original_dtype, np.integer):
            max_value = np.iinfo(original_dtype).max
            return np.clip(masked, 0, max_value).astype(original_dtype)
        return masked.astype(original_dtype, copy=False)

    def _resolve_smirk_crop_scale(self, config):
        train_cfg = getattr(config, 'train', None)
        if train_cfg is None:
            return 1.6

        if hasattr(train_cfg, 'smirk_crop_scale'):
            return float(train_cfg.smirk_crop_scale)
        if hasattr(train_cfg, 'test_scale'):
            return float(train_cfg.test_scale)
        return 1.6

    def _resize_landmarks(self, landmarks, orig_h, orig_w, target_h, target_w):
        resized = np.array(landmarks, dtype=np.float32, copy=True)
        if resized.ndim != 2 or resized.shape[-1] < 2:
            return None
        resized[:, 0] *= float(target_w) / max(float(orig_w), 1.0)
        resized[:, 1] *= float(target_h) / max(float(orig_h), 1.0)
        return resized

    def _prepare_smirk_data(self, image_rgb, landmarks_mediapipe):
        crop_size = self.smirk_image_size
        crop_valid = False
        if landmarks_mediapipe is not None and landmarks_mediapipe.shape[0] > max(mediapipe_indices):
            crop_tform = self.crop_face(image_rgb, landmarks_mediapipe[..., :2], self.smirk_crop_scale, image_size=crop_size)
            crop_valid = True
        else:
            crop_tform = self._build_resize_transform(image_rgb.shape[0], image_rgb.shape[1], crop_size)

        smirk_img = warp(
            image_rgb,
            crop_tform.inverse,
            output_shape=(crop_size, crop_size),
            preserve_range=True,
        ).astype(np.uint8)
        smirk_img = (smirk_img / 255.0).astype(np.float32).transpose(2, 0, 1)

        return {
            'smirk_img': smirk_img,
            'smirk_crop_transform': crop_tform.params.astype(np.float32),
            'smirk_crop_valid': np.array(float(crop_valid), dtype=np.float32),
        }

    @staticmethod
    def _build_resize_transform(full_h, full_w, crop_size):
        sx = float(crop_size) / max(float(full_w), 1.0)
        sy = float(crop_size) / max(float(full_h), 1.0)
        matrix = np.array(
            [
                [sx, 0.0, 0.0],
                [0.0, sy, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return trans.ProjectiveTransform(matrix=matrix)

    def _resolve_target_resolution(self, config):
        dataset_cfg = getattr(config, 'dataset', None)
        resolution = None
        if dataset_cfg is not None and hasattr(dataset_cfg, 'resolution'):
            resolution = dataset_cfg.resolution
        if resolution is None and hasattr(config, 'image_size'):
            resolution = config.image_size
        return self._normalize_resolution(resolution)

    def _normalize_resolution(self, resolution):
        if resolution is None:
            return None

        # Handle mappings like {"height": 512, "width": 512}
        if isinstance(resolution, abc.Mapping):
            height = resolution.get('height') or resolution.get('h')
            width = resolution.get('width') or resolution.get('w')
        elif isinstance(resolution, abc.Sequence) and not isinstance(resolution, (str, bytes)):
            if len(resolution) == 0:
                return None
            if len(resolution) == 1:
                height = width = resolution[0]
            else:
                height, width = resolution[:2]
        else:
            height = width = resolution

        if height is None or width is None:
            return None

        return int(height), int(width)

    def _resize_image(self, image):
        if self.target_resolution is None:
            return image
        target_h, target_w = self.target_resolution
        h, w = image.shape[:2]
        if (h, w) == (target_h, target_w):
            return image

        interpolation = cv2.INTER_AREA if target_h < h or target_w < w else cv2.INTER_LINEAR
        return cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    def _resize_mask(self, mask):
        if self.target_resolution is None:
            return mask
        target_h, target_w = self.target_resolution
        h, w = mask.shape[:2]
        if (h, w) == (target_h, target_w):
            return mask

        mask_3d = mask if mask.ndim == 3 else mask[..., None]
        channels = mask_3d.shape[2]
        resized_channels = []
        for c in range(channels):
            channel = mask_3d[..., c].astype(np.float32)
            resized_channel = cv2.resize(channel, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            resized_channels.append(resized_channel)
        resized = np.stack(resized_channels, axis=-1)
        if mask.ndim == 2:
            return resized[..., 0]
        return resized

    def _resize_soft_mask(self, mask):
        if self.target_resolution is None:
            return mask
        target_h, target_w = self.target_resolution
        h, w = mask.shape[:2]
        if (h, w) == (target_h, target_w):
            return mask

        mask_3d = mask if mask.ndim == 3 else mask[..., None]
        interpolation = cv2.INTER_AREA if target_h < h or target_w < w else cv2.INTER_LINEAR
        resized_channels = []
        for c in range(mask_3d.shape[2]):
            channel = mask_3d[..., c].astype(np.float32)
            resized_channel = cv2.resize(channel, (target_w, target_h), interpolation=interpolation)
            resized_channels.append(resized_channel)
        resized = np.clip(np.stack(resized_channels, axis=-1), 0.0, 1.0)
        if mask.ndim == 2:
            return resized[..., 0]
        return resized
    
