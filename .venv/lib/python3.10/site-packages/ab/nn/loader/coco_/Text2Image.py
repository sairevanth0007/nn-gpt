
import os
import random
from os.path import join
from PIL import Image

import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_and_extract_archive
import torchvision.transforms as T

from ab.nn.util.Const import data_dir

# --- Configuration ---
COCO_ANN_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
COCO_IMG_URL_TEMPLATE = 'http://images.cocodataset.org/zips/{}2017.zip'
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_DEV = (0.5, 0.5, 0.5)
IMAGE_SIZE = 256


class Text2Image(Dataset):
    def __init__(self, root, split='train', transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.split = split

        ann_dir = join(root, 'annotations')
        if not os.path.exists(ann_dir):
            os.makedirs(root, exist_ok=True)
            download_and_extract_archive(COCO_ANN_URL, root, filename='annotations_trainval2017.zip')

        ann_file = join(ann_dir, f'captions_{split}2017.json')
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.img_dir = join(root, f'{split}2017')
        if self.ids and not os.path.exists(join(self.img_dir, self.coco.loadImgs(self.ids[0])[0]['file_name'])):
            download_and_extract_archive(COCO_IMG_URL_TEMPLATE.format(split), root, filename=f'{split}2017.zip')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = join(self.img_dir, img_info['file_name'])

        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, FileNotFoundError):
            return self.__getitem__((index + 1) % len(self))

        if self.transform:
            image = self.transform(image)


        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in anns if 'caption' in ann]
        text_prompt = random.choice(captions) if captions else "an image"
        return image, text_prompt


def loader(transform_fn, task, **kwargs):
    if 'txt-image' not in task.strip().lower():
        raise ValueError(f"The task '{task}' is not a text-to-image task for this dataloader.")


    transform = transform_fn((NORM_MEAN, NORM_DEV))

    path = join(data_dir, 'coco')

    train_dataset = Text2Image(root=path, split='train', transform=transform)
    val_dataset = Text2Image(root=path, split='val', transform=transform)

    metadata = (None,)
    performance_goal = 0.0
    return metadata, performance_goal, train_dataset, val_dataset