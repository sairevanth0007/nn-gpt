import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    transforms.RandomApply(transforms=[ColorJitter(brightness=(0.8, 1.2), contrast=None, saturation=None, hue=None)], p=0.7),
    transforms.RandomVerticalFlip(p=0.41),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
