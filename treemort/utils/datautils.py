import h5py
import random

import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN


def load_and_organize_data(hdf5_file_path):
    image_patch_map = defaultdict(list)

    with h5py.File(hdf5_file_path, "r") as hf:
        for key in hf.keys():
            num_trees = hf[key].attrs.get("num_trees", 0)
            latitude = hf[key].attrs.get("latitude", None)
            longitude = hf[key].attrs.get("longitude", None)
            pixel_x = hf[key].attrs.get("pixel_x", None)
            pixel_y = hf[key].attrs.get("pixel_y", None)
            source_image = hf[key].attrs.get("source_image", "")
            image_patch_map[source_image].append((key, num_trees, latitude, longitude, pixel_x, pixel_y))

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


def stratify_images_by_region(
    image_patch_map,
    val_ratio=0.2,
    test_ratio=0.1,
    lat_bin_size=2.0,
    lon_bin_size=2.0,
    eps=0.5,
):
    """
    Memory-efficient spatial stratification that mimics the previous GeoPandas-based logic.

    We aggregate patches into coarse lat/lon bins first, then run DBSCAN on the bin centroids.
    This keeps the clustering workload bounded by the number of bins instead of the number of
    individual patches (which can be in the hundreds of thousands).
    """
    bin_map = {}
    missing_coords_keys = []

    def _safe_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    for patches in image_patch_map.values():
        for key, dead_tree_count, latitude, longitude, _, _ in patches:
            lat = _safe_float(latitude)
            lon = _safe_float(longitude)
            if lat is None or lon is None:
                missing_coords_keys.append((key, dead_tree_count))
                continue

            lat_bin = int(lat // lat_bin_size)
            lon_bin = int(lon // lon_bin_size)
            bin_key = (lat_bin, lon_bin)

            if bin_key not in bin_map:
                bin_map[bin_key] = {
                    "keys": [],
                    "dead_trees": 0.0,
                    "lat_sum": 0.0,
                    "lon_sum": 0.0,
                    "count": 0,
                }

            entry = bin_map[bin_key]
            entry["keys"].append(key)
            entry["dead_trees"] += float(dead_tree_count)
            entry["lat_sum"] += lat
            entry["lon_sum"] += lon
            entry["count"] += 1

    if not bin_map:
        # No usable coordinates; fall back to simple stratification.
        return stratify_images_by_patch_count(image_patch_map, val_ratio, test_ratio)

    coords = []
    bin_entries = []

    for entry in bin_map.values():
        mean_lat = entry["lat_sum"] / max(entry["count"], 1)
        mean_lon = entry["lon_sum"] / max(entry["count"], 1)
        coords.append([mean_lon, mean_lat])
        bin_entries.append(entry)

    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0:
        return stratify_images_by_patch_count(image_patch_map, val_ratio, test_ratio)

    dbscan = DBSCAN(eps=eps, min_samples=1).fit(coords)
    labels = dbscan.labels_

    cluster_bins = defaultdict(list)
    cluster_dead_counts = defaultdict(float)

    for label, entry in zip(labels, bin_entries):
        cluster_bins[label].append(entry)
        cluster_dead_counts[label] += entry["dead_trees"]

    cluster_order = sorted(
        cluster_bins.keys(),
        key=lambda c: cluster_dead_counts[c],
        reverse=True,
    )

    total_dead_trees = sum(cluster_dead_counts.values())
    if total_dead_trees <= 0:
        # Degenerate case: no dead-tree counts; fall back to uniform patch split.
        return stratify_images_by_patch_count(image_patch_map, val_ratio, test_ratio)

    desired_ratios = np.array([1 - val_ratio - test_ratio, val_ratio, test_ratio])
    target_counts = desired_ratios * total_dead_trees
    cumulative = np.zeros(3, dtype=float)

    train_keys, val_keys, test_keys = [], [], []
    splits = [train_keys, val_keys, test_keys]

    for cluster in cluster_order:
        cluster_keys = []
        for entry in cluster_bins[cluster]:
            cluster_keys.extend(entry["keys"])

        cluster_dead = cluster_dead_counts[cluster]

        ratios = np.divide(
            cumulative,
            target_counts,
            out=np.full_like(cumulative, np.inf),
            where=target_counts > 0,
        )
        target_idx = int(np.argmin(ratios))

        splits[target_idx].extend(cluster_keys)
        cumulative[target_idx] += cluster_dead

    # Append missing-coordinate keys to whichever split is shortest
    if missing_coords_keys:
        splits = [train_keys, val_keys, test_keys]
        target_idx = int(np.argmin([len(k) for k in splits]))
        splits[target_idx].extend(key for key, _ in missing_coords_keys)

    return train_keys, val_keys, test_keys
