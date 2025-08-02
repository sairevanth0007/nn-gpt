import torchvision

from ab.nn.util.Const import data_dir

__norm_mean = (0.485, 0.456, 0.406)
__norm_dev = (0.229, 0.224, 0.225)

__class_quantity = 365
MINIMUM_ACCURACY = 1.0 / __class_quantity

def loader(transform_fn, task):
    transform = transform_fn((__norm_mean, __norm_dev))
    download =  not (data_dir / 'data_256_standard').exists()
    train_set = torchvision.datasets.Places365(root=data_dir, small=True, split='train-standard', transform=transform, download=download)
    test_set = torchvision.datasets.Places365(root=data_dir, small=True, split='val', transform=transform, download=download)
    return (__class_quantity,), MINIMUM_ACCURACY, train_set, test_set