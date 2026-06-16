import logging
import os
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        mask_dir: str,
        scale: float = 1.0,
        mask_suffix: str = '',
        mask_values=None,
        weight_dir: str | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.weight_dir = Path(weight_dir) if weight_dir else None
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        missing_masks = [
            idx for idx in self.ids
            if len(list(self.mask_dir.glob(idx + self.mask_suffix + '.*'))) != 1
        ]
        if missing_masks:
            preview = ', '.join(missing_masks[:5])
            raise RuntimeError(
                f'Missing or ambiguous masks for {len(missing_masks)} examples '
                f'with suffix "{self.mask_suffix}". First examples: {preview}'
            )
        if self.weight_dir is not None:
            missing_weights = [
                idx for idx in self.ids
                if len(list(self.weight_dir.glob(idx + '_weight.npy'))) != 1
            ]
            if missing_weights:
                preview = ', '.join(missing_weights[:5])
                raise RuntimeError(
                    f'Missing weight maps for {len(missing_weights)} examples. '
                    f'First examples: {preview}'
                )
        if mask_values is None:
            logging.info('Scanning mask files to determine unique values')
            if os.name == 'nt':
                unique = [
                    unique_mask_values(idx, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix)
                    for idx in tqdm(self.ids, total=len(self.ids))
                ]
            else:
                with Pool() as p:
                    unique = list(tqdm(
                        p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                        total=len(self.ids)
                    ))

            self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        else:
            self.mask_values = list(mask_values)
            logging.info('Using provided mask values; skipping mask scan')
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    @staticmethod
    def preprocess_weight(weight, scale):
        h, w = weight.shape
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized weight maps would have no pixel'
        if (newW, newH) != (w, h):
            weight_img = Image.fromarray(weight.astype(np.float32), mode='F')
            weight = np.asarray(weight_img.resize((newW, newH), resample=Image.BICUBIC), dtype=np.float32)
        return np.maximum(weight, 0.0)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        sample = {
            'name': name,
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
        if self.weight_dir is not None:
            weight = np.load(self.weight_dir / f'{name}_weight.npy').astype(np.float32)
            sample['weight'] = torch.as_tensor(
                self.preprocess_weight(weight, self.scale).copy()
            ).float().contiguous()
        return sample


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1, mask_values=None, weight_dir=None):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask', mask_values=mask_values, weight_dir=weight_dir)
