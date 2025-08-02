import torchvision

from ab.nn.util.Const import data_dir

__norm_mean = (0.4914, 0.4822, 0.4465)
__norm_dev = (0.2470, 0.2435, 0.2616)

__class_quantity = 10
MINIMUM_ACCURACY = 1.0 / __class_quantity

def loader(transform_fn, task):
    transform = transform_fn((__norm_mean, __norm_dev))
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)
    return (__class_quantity,), MINIMUM_ACCURACY, train_set, test_set