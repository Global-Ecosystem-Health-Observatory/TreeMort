import h5py
import random

from collections import defaultdict
from sklearn.model_selection import train_test_split


def load_and_organize_data(hdf5_file_path):
    image_patch_map = defaultdict(list)

    with h5py.File(hdf5_file_path, "r") as hf:
        for key in hf.keys():
            contains_dead_tree = hf[key].attrs.get("contains_dead_tree", 0)
            filename = hf[key].attrs.get("source_image", "")
            image_patch_map[filename].append((key, contains_dead_tree))

    return image_patch_map


def bin_images_by_patch_count(image_patch_map, val_ratio, test_ratio, seed=42):
    keys = list(image_patch_map.keys())

    random.seed(seed) # for replication
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
    keys = []
    for img in images:
        keys.extend([key for key, _ in image_patch_map[img]])
    return keys

def stratify_images_by_patch_count(image_patch_map, val_ratio, test_ratio):
    train_images, val_images, test_images = bin_images_by_patch_count(image_patch_map, val_ratio, test_ratio)

    train_keys = extract_keys_from_images(image_patch_map, train_images)
    val_keys = extract_keys_from_images(image_patch_map, val_images)
    test_keys = extract_keys_from_images(image_patch_map, test_images)

    return train_keys, val_keys, test_keys