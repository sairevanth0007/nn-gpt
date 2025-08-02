import torchvision

from ab.nn.util.Const import data_dir

__norm_mean = (0.485, 0.456, 0.406)
__norm_dev = (0.229, 0.224, 0.225)

# for all (__first_class_quantity,) + ((__class_quantity,) * 40)
# __first_class_quantity = 5
# __output_quantity = 40
# MINIMUM_ACCURACY = 1.0 / (__class_quantity ** __output_quantity)

__class_quantity = 2
MINIMUM_ACCURACY = 1.0 / __class_quantity

def get_gender(attr):
    return attr[20]


def loader(transform_fn, task):
    transform = transform_fn((__norm_mean, __norm_dev))
    train_set = torchvision.datasets.CelebA(root=data_dir, split='train', transform=transform,
                                            target_type='attr', target_transform=get_gender, download=True)
    test_set = torchvision.datasets.CelebA(root=data_dir, split='test', transform=transform,
                                           target_type='attr', target_transform=get_gender, download=True)
    return (__class_quantity,), MINIMUM_ACCURACY, train_set, test_set


# 00 - 5_o_Clock_Shadow
#     01 - Arched_Eyebrows
#     02 - Attractive
#     03 - Bags_Under_Eyes
#     04 - Bald
#     05 - Bangs
#     06 - Big_Lips
#     07 - Big_Nose
#     08 - Black_Hair
#     09 - Blond_Hair
#     10 - Blurry
#     11 - Brown_Hair
#     12 - Bushy_Eyebrows
#     13 - Chubby
#     14 - Double_Chin
#     15 - Eyeglasses
#     16 - Goatee
#     17 - Gray_Hair
#     18 - Heavy_Makeup
#     19 - High_Cheekbones
#     20 - Male
#     21 - Mouth_Slightly_Open
#     22 - Mustache
#     23 - Narrow_Eyes
#     24 - No_Beard
#     25 - Oval_Face
#     26 - Pale_Skin
#     27 - Pointy_Nose
#     28 - Receding_Hairline
#     29 - Rosy_Cheeks
#     30 - Sideburns
#     31 - Smiling
#     32 - Straight_Hair
#     33 - Wavy_Hair
#     34 - Wearing_Earrings
#     35 - Wearing_Hat
#     36 - Wearing_Lipstick
#     37 - Wearing_Necklace
#     38 - Wearing_Necktie
#     39 - Young