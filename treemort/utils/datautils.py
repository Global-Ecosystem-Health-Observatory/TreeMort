import h5py

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


def stratified_split(image_patch_map, val_ratio, test_ratio):
    patches_with_dead_tree = []
    patches_without_dead_tree = []

    for img, patches in image_patch_map.items():
        for patch in patches:
            if patch[1] == 1:
                patches_with_dead_tree.append((img, patch))
            else:
                patches_without_dead_tree.append((img, patch))

    train_patches_with_dead_tree, val_test_patches_with_dead_tree = train_test_split(
        patches_with_dead_tree,
        test_size=(val_ratio + test_ratio),
        random_state=42,
        stratify=[img for img, patch in patches_with_dead_tree],  # stratify by image
    )
    val_patches_with_dead_tree, test_patches_with_dead_tree = train_test_split(
        val_test_patches_with_dead_tree,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=42,
        stratify=[
            img for img, patch in val_test_patches_with_dead_tree
        ],  # stratify by image
    )

    if patches_without_dead_tree:
        train_patches_without_dead_tree, val_test_patches_without_dead_tree = (
            train_test_split(
                patches_without_dead_tree,
                test_size=(val_ratio + test_ratio),
                random_state=42,
                stratify=[
                    img for img, patch in patches_without_dead_tree
                ],  # stratify by image
            )
        )
        val_patches_without_dead_tree, test_patches_without_dead_tree = (
            train_test_split(
                val_test_patches_without_dead_tree,
                test_size=(test_ratio / (val_ratio + test_ratio)),
                random_state=42,
                stratify=[
                    img for img, patch in val_test_patches_without_dead_tree
                ],  # stratify by image
            )
        )
    else:
        train_patches_without_dead_tree = []
        val_patches_without_dead_tree = []
        test_patches_without_dead_tree = []

    train_patches = train_patches_with_dead_tree + train_patches_without_dead_tree
    val_patches = val_patches_with_dead_tree + val_patches_without_dead_tree
    test_patches = test_patches_with_dead_tree + test_patches_without_dead_tree

    return train_patches, val_patches, test_patches
