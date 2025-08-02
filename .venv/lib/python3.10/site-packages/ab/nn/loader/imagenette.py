import torchvision

from ab.nn.util.Const import data_dir

__norm_mean = (0.485, 0.456, 0.406)
__norm_dev = (0.229, 0.224, 0.225)

__class_quantity = 10
MINIMUM_ACCURACY = 1.0 / __class_quantity

def loader(transform_fn, task):
    transform = transform_fn((__norm_mean, __norm_dev))
    download =  not (data_dir / 'imagenette2').exists()
    train_set = torchvision.datasets.Imagenette(root=data_dir, split='train', transform=transform, download=download)
    test_set = torchvision.datasets.Imagenette(root=data_dir, split='val', transform=transform, download=download)
    return (__class_quantity,), MINIMUM_ACCURACY, train_set, test_set