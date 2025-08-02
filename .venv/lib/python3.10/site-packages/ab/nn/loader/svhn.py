import torchvision

from ab.nn.util.Const import data_dir

__norm_mean = (0.4377, 0.4438, 0.4728)
__norm_dev = (0.1980, 0.2010, 0.1970)

__class_quantity = 10
MINIMUM_ACCURACY = 1.0 / __class_quantity

def loader(transform_fn, task):
    transform = transform_fn((__norm_mean, __norm_dev))
    train_set = torchvision.datasets.SVHN(root=data_dir, split='train', transform=transform, download=True)
    test_set = torchvision.datasets.SVHN(root=data_dir, split='test', transform=transform, download=True)
    return (__class_quantity,), MINIMUM_ACCURACY, train_set, test_set