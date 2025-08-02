from os import makedirs
from os.path import join, exists, dirname

import requests
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm

from ab.nn.util.Const import data_dir

# Standard module-level constants
__norm_mean = (104.01362025, 114.03422265, 119.9165958)
__norm_dev = (73.6027665, 69.89082075, 70.9150767)
MINIMUM_ACCURACY = 0.005  # Minimum accuracy for object detection

# COCO URLs
coco_ann_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
coco_img_url = 'http://images.cocodataset.org/zips/{}2017.zip'

# Class definitions
#MIN_CLASS_LIST = list(range(91))
MIN_CLASS_LIST = [0, 1, 2, 3, 4, 5]
MIN_CLASS_N = len(MIN_CLASS_LIST)

def class_n():
    return MIN_CLASS_N


class COCODetectionDataset(Dataset):
    num_workers = 0

    def __init__(self, transform, root, split='train', class_list=None, preprocess=True):
        """
        Initialize COCO detection dataset
        
        Parameters:
        -----------
        root : str
            Path to COCO dataset root directory
        split : str
            'train' or 'val'
        transform : callable, optional
            Transform to apply to images and targets
        class_list : list, optional
            List of class IDs to use (for subset of classes)
        """
        valid_splits = ['train', 'val']
        if split not in valid_splits:
            raise ValueError(f'Invalid split: {split}')
        self.root = root
        self.transform = transform
        self.split = split  
        self.class_list = class_list or MIN_CLASS_LIST
       
        
        ann_file = join(root, 'annotations', f'instances_{split}2017.json')
        if not exists(join(root, 'annotations')):
            print('Annotation file doesn\'t exist! Downloading')
            makedirs(root, exist_ok=True)
            download_and_extract_archive(coco_ann_url, root, filename='annotations_trainval2017.zip')
            print('Annotation file preparation complete')
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.img_dir = join(root, f'{split}2017')
        
        self.preprocess = preprocess
        if self.preprocess:
            self.__preprocess__() 
        first_image_info = self.coco.loadImgs(self.ids[0])[0]
        first_file_path = join(self.img_dir, first_image_info['file_name'])
        if not exists(first_file_path):
            print(f'Image dataset doesn\'t exist! Downloading {split} split...')
            download_and_extract_archive(coco_img_url.format(split), root, filename=f'{split}2017.zip')
            print('Image dataset preparation complete')

    def __preprocess__(self):
        list_file = join(self.root, 'preprocessed', f"{self.split}2017_filtered_class{'-'.join(map(str, self.class_list))}.list")
        makedirs(join(self.root, 'preprocessed'), exist_ok=True)
        if exists(list_file):
            with open(list_file, 'r') as f:
                lines = f.readlines()
                filtered_ids = [int(line.strip()) for line in lines]
            self.ids = filtered_ids
            print(f"Loaded {len(self.ids)} filtered image IDs from {list_file}")
        else:
            print(f"Preprocessing to filter image IDs for class list {self.class_list}")
            filtered_ids = []
            for img_id in tqdm(self.ids, desc="Preprocessing"):
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                has_class = any(ann['category_id'] in self.class_list for ann in anns)
                if has_class:
                    filtered_ids.append(img_id)
            self.ids = filtered_ids
            # Save the list to file
            with open(list_file, 'w') as f:
                for img_id in self.ids:
                    f.write(f"{img_id}\n")
            print(f"Saved {len(self.ids)} filtered image IDs to {list_file}")
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        file_path = join(self.img_dir, img_info['file_name'])
        try:
            with Image.open(file_path) as img_file:
                image = img_file.convert('RGB')
                # Create a copy in memory before closing the file
                image = image.copy()

        except:
            if not hasattr(self, 'no_missing_img'):
                print('Failed to read image(s). Download will be performed as needed.')
                self.no_missing_img = True
            response = requests.get(img_info['coco_url'])
            if response.status_code == 200:
                makedirs(dirname(file_path), exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                with Image.open(file_path) as img_file:
                    image = img_file.convert('RGB')
                    image = image.copy()

            else:
                raise RuntimeError(f"Failed to download image: {img_info['file_name']}")
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            cat_id = ann['category_id']
            if cat_id not in self.class_list:
                continue
            cat_id = self.class_list.index(cat_id)
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(cat_id)
            areas.append(ann['area'])
            iscrowd.append(0)
        target = {}
        if boxes:
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
            target['area'] = torch.as_tensor(areas, dtype=torch.float32)
            target['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.int64)
            if self.transform is not None:
                image = self.transform(image)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['area'] = torch.zeros((0,), dtype=torch.float32)
            target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)
            if self.transform is not None:
                image = self.transform(image)
        target['image_id'] = torch.tensor([img_id])
        target['orig_size'] = torch.as_tensor([img_info['height'], img_info['width']])
        return image, target


    @staticmethod
    def collate_fn(batch):
        """
        Default collate function for the dataset.
        """
        images = []
        targets = []

        for image, target in batch:
            images.append(image)
            # Keep original dictionary structure for each target
            targets.append({
                'boxes': target['boxes'],
                'labels': target['labels'],
                'image_id': target['image_id'],
                'orig_size': target['orig_size'],
                'area': target['area'],
                'iscrowd': target['iscrowd']
            })

        images = torch.stack(images, dim=0)

        class TargetsWrapper:
            def __init__(self, targets_list):
                self.targets = targets_list
                
            def to(self, device):
                # Move each tensor to device while preserving list of dicts structure
                self.targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                               for k, v in t.items()} for t in self.targets]
                return self
            
            def __getitem__(self, idx):
                return self.targets[idx]
            
            def __len__(self):
                return len(self.targets)

        return images, TargetsWrapper(targets)

    # Override the default collate function
    def __collate__(self, batch):
    
        return self.collate_fn(batch)
    

    def __len__(self):
        return len(self.ids)
      

def loader(transform_fn, task):
    """
    Main entry point following repository pattern.
    Returns train and validation datasets for COCO object detection.
    
    Parameters:
    -----------
    path : str
        Path to COCO dataset root directory
    transform : callable, optional
        Transform to apply to images and targets
    class_list : list, optional
        List of class IDs to use (for subset of classes)
    **kwargs : dict
        Additional arguments passed to dataset
    
    Returns:
    --------
    tuple: (train_dataset, val_dataset)
    """

    path = join(data_dir, 'coco')
    transform = transform_fn((__norm_mean, __norm_dev))
    resize = None
    train_dataset = COCODetectionDataset(transform=transform, root=path, split="train", class_list=MIN_CLASS_LIST, preprocess=True)
    val_dataset = COCODetectionDataset(transform=transform, root=path, split="val", class_list=MIN_CLASS_LIST, preprocess=True)
    
    return (class_n(),), MINIMUM_ACCURACY, train_dataset, val_dataset
