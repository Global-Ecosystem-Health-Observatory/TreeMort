import h5py
import random

import pandas as pd

from collections import defaultdict


def load_and_organize_data(hdf5_file_path):
    image_patch_map = defaultdict(list)

    with h5py.File(hdf5_file_path, "r") as hf:
        for key in hf.keys():
            contains_dead_tree = hf[key].attrs.get("contains_dead_tree", 0)
            latitude = hf[key].attrs.get("latitude", None)
            longitude = hf[key].attrs.get("longitude", None)
            filename = hf[key].attrs.get("source_image", "")
            image_patch_map[filename].append((key, contains_dead_tree, latitude, longitude))

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


def stratify_images_by_region(image_patch_map, train_ratio=0.7, val_ratio=0.15, lat_bin_size=2.0, lon_bin_size=2.0):
    rows = []

    for filename, patches in image_patch_map.items():
        for key, contains_dead_tree, latitude, longitude in patches:
            rows.append({
                "Key": key,
                "Filename": filename,
                "ContainsDeadTree": contains_dead_tree,
                "Latitude": latitude,
                "Longitude": longitude
            })

    df = pd.DataFrame(rows)

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    df = df.dropna(subset=["Latitude", "Longitude"])

    df["LatBin"] = (df["Latitude"] // lat_bin_size).astype(int)
    df["LonBin"] = (df["Longitude"] // lon_bin_size).astype(int)
    df["Region"] = df["LatBin"].astype(str) + "_" + df["LonBin"].astype(str)

    grouped = df.groupby("Region")
    train_keys, val_keys, test_keys = [], [], []
    for _, group in grouped:
        group = group.sample(frac=1, random_state=42)  # Shuffle within the region
        n_train = int(len(group) * train_ratio)
        n_val = int(len(group) * val_ratio)

        train_keys.extend(group.iloc[:n_train]["Key"].tolist())
        val_keys.extend(group.iloc[n_train:n_train + n_val]["Key"].tolist())
        test_keys.extend(group.iloc[n_train + n_val:]["Key"].tolist())

    return train_keys, val_keys, test_keys