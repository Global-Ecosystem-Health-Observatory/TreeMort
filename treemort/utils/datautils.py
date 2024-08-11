import h5py

from collections import defaultdict
from sklearn.model_selection import train_test_split


['S4344A_1.tif', 'P4113C_1.tif', 'N5442C_2014_1.tiff', 'L4411F_tile_3_band_1multiband_tile_3.tif', 'L4424B_tile_1_band_1multiband_tile_1.tif', 'P4341G_1.tif', 'Q3334C_2019_2.tif', 'T4323E_1.tif', 'N5141A_1.tif', 'M4123D_2015_1.tiff', 'M4313B_2015_1.tiff', 'N-33-127-A-c-3-3_1.tiff', 'M5221F_2016_1.tiff', 'M5224A_1.tif', 'N-34-104-B-c-2-2_1.tiff', 'L3341B_1.tif', 'P5322F_2_1.tiff', 'Q4323B_1.tif', 'L3312A_2017_1.tiff', 'M-34-105-B-b-2-4_1.tiff', 'L3322E_2015_1.tiff', 'N5132F_1.tif', 'L3434C_2019_1.tif', 'T4123G_1.tif', 'L4412B_2015_1.tiff', 'Q4211E_2019_1.tif', 'S5112B_1.tif', 'U5214H_2018_1.tif', 'N-34-104-B-d-2-3_1.tiff', 'L3343A_2019_1.tif', 'L3343D_2019_1.tif', 'L5242G_2017_1.tif', 'L4412D_2015_1.tiff', 'U5224F_2018_1.tif', 'N-34-104-B-c-4-4_1.tiff', 'P5322A_2017_1.tif', 'V4331F_2018_2.tif', 'N-34-90-A-a-1-3_1.tiff', 'U5224D_1.tif', 'L4423C_2_1.tiff', 'L3312D_2017_1.tiff', 'M-34-105-B-b-3-1_1.tiff', 'M-34-105-B-b-2-2_1.tiff', 'N-34-90-A-c-1-1_1.tiff', 'M-34-56-B-d-2-1_1.tiff', '63611_3_1.tiff', 'L3344B_2019_1.tif', 'Q3334C_2019_1.tif', 'L4431B_4_1.tiff', 'L3433D_2019_1.tif', 'L4431D_5_1.tiff', 'U5222C_2018_1.tif', 'N4212G_2013_1.tiff', 'N4212C_2013_1.tiff', 'S5321C_2017_1.tif', 'M4311G_2.tif', 'M4424E_4_1.tiff', 'M3322F_1.tif', 'L3211A_2.tif', 'M-34-56-B-d-3-2_1.tiff', 'L4414G_2.tif', 'N-33-127-A-c-1-1_1.tiff', 'M-34-105-B-b-1-3_1.tiff', 'L3211A_1.tif', 'N4331F_1.tif', 'L4414G_1.tif', 'N5143C_2017_1.tif', '63224_1_1.tiff', 'M3322F_2.tif', 'U4442B_2018_2.tif', 'U5242A_1.tif', 'V4311C_1.tif', 'V4323C_1.tif', 'M3421F_2_1.tiff', 'P4343H_1.tif', '63432_4_1.tiff', 'M4424E_5_1.tiff', 'M4311G_1.tif', '63443_4_1.tiff', 'V4331A_2018_1.tif', '63224_2_1.tiff', 'L4431D_2_1.tiff', 'L3333F_2019_1.tif', 'V4331B_2018_1.tif', '63223_3_1.tiff', 'N-34-90-A-a-3-1_1.tiff', 'M-34-56-B-d-2-4_1.tiff', 'M4414F_4_1.tiff', 'M4441C_1_1.tiff', 'N-34-90-A-a-1-4_1.tiff', 'M5133H_2017_1.tif', 'U4442B_2018_1.tif', '63471_4_1.tiff', '63472_2_1.tiff', 'L4134E_2013_1.tiff', 'M-34-57-A-c-1-1_1.tiff', 'M3422E_2016_1.tiff', 'M4122B_2014_1.tiff', 'M-34-56-B-d-1-2_1.tiff', 'M4124C_2017_1.tiff', 'M4213A_2017_1.tiff', 'V4331A_2018_2.tif', 'M4213A_2014_1.tiff', 'U4324B_1.tif', 'V4314H_1.tif', '63224_4.tif', 'L4134A_2013_1.tiff', '63223_1.tif', '63224_3.tif', '63223_2.tif', 'L3324A_3_1.tif', 'M4331C_2016_2.tiff', 'M4211G_2023_1.tif', 'M4211G_2023_2.tif', 'M3442B_2011_1.tiff', '63224_2.tif', 'L2313D_4.tif', 'M4213A_2023_2.tif', '63223_3.tif', 'M4331C_2016_1.tiff', '63224_1.tif', 'L3324B_3_1.tif', '63223_5.tif', '63223_4.tif', 'P4131H_2019_1.tif', 'N-34-90-A-a-2-3_1.tiff', 'P4131H_2019_2.tif', 'R4234D_2019_1.tif']
['L3243F_2022.jp2', 'L3333C_2022.jp2']

def load_and_organize_data(hdf5_file_path):
    image_patch_map = defaultdict(list)

    with h5py.File(hdf5_file_path, "r") as hf:
        for key in hf.keys():
            contains_dead_tree = hf[key].attrs.get("contains_dead_tree", 0)
            filename = hf[key].attrs.get("source_image", "")
            image_patch_map[filename].append((key, contains_dead_tree))

    return image_patch_map


def bin_images_by_patch_count(image_patch_map, val_ratio, test_ratio):
    """
    Bin images such that validation and test bins fulfill the given ratios in terms of patch count.
    """
    keys = list(image_patch_map.keys())
    random.seed(42)
    random.shuffle(keys)
    shuffled_images = [(key, image_patch_map[key]) for key in keys]

    total_patches = sum(len(patches) for patches in image_patch_map.values())

    target_val_patches = int(val_ratio * total_patches)
    target_test_patches = int(test_ratio * total_patches)

    val_patches_count = 0
    test_patches_count = 0

    train_images = []
    val_images = []
    test_images = []

    for img, patches in shuffled_images:
        if val_patches_count < target_val_patches:
            val_images.append(img)
            val_patches_count += len(patches)
        elif test_patches_count < target_test_patches:
            test_images.append(img)
            test_patches_count += len(patches)
        else:
            train_images.append(img)

    return train_images, val_images, test_images

def extract_keys_from_images(image_patch_map, images):
    """
    Extract keys corresponding to images for a specific bin (train/val/test).
    """
    keys = []
    for img in images:
        keys.extend([key for key, _ in image_patch_map[img]])
    return keys

def stratify_images_by_patch_count(image_patch_map, val_ratio, test_ratio):
    """
    Stratify images into training, validation, and test bins based on patch count.
    """
    train_images, val_images, test_images = bin_images_by_patch_count(image_patch_map, val_ratio, test_ratio)

    print(len(train_images) + len(val_images) + len(test_images))

    train_keys = extract_keys_from_images(image_patch_map, train_images)
    val_keys = extract_keys_from_images(image_patch_map, val_images)
    test_keys = extract_keys_from_images(image_patch_map, test_images)

    return train_keys, val_keys, test_keys