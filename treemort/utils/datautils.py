import h5py
import random

import numpy as np
import pandas as pd
import geopandas as gpd

from collections import defaultdict
from sklearn.cluster import DBSCAN
from shapely.geometry import Point


def load_and_organize_data(hdf5_file_path):
    image_patch_map = defaultdict(list)

    with h5py.File(hdf5_file_path, "r") as hf:
        for key in hf.keys():
            dead_tree_count = hf[key].attrs.get("dead_tree_count", 0)
            latitude = hf[key].attrs.get("latitude", None)
            longitude = hf[key].attrs.get("longitude", None)
            filename = hf[key].attrs.get("source_image", "")
            image_patch_map[filename].append((key, dead_tree_count, latitude, longitude))

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


def stratify_images_by_region(image_patch_map, val_ratio=0.2, test_ratio=0.1, lat_bin_size=2.0, lon_bin_size=2.0, eps = 0.5):
    rows = []

    for filename, patches in image_patch_map.items():
        for key, dead_tree_count, latitude, longitude in patches:
            rows.append({
                "Key": key,
                "Filename": filename,
                "DeadTreeCount": dead_tree_count,
                "Latitude": latitude,
                "Longitude": longitude
            })

    df = pd.DataFrame(rows)

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    df = df.dropna(subset=["Latitude", "Longitude"])

    df["geometry"] = [Point(lon, lat) for lon, lat in zip(df["Longitude"], df["Latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    gdf["LatBin"] = (gdf.geometry.y // lat_bin_size).astype(int)
    gdf["LonBin"] = (gdf.geometry.x // lon_bin_size).astype(int)
    gdf["Region"] = gdf["LatBin"].astype(str) + "_" + gdf["LonBin"].astype(str)

    bin_aggregates = (
        gdf.groupby("Region")["DeadTreeCount"]
        .sum()
        .reset_index()
        .rename(columns={"DeadTreeCount": "TotalDeadTrees"})
    )
    gdf = gdf.merge(bin_aggregates, on="Region", how="left")

    coords = np.array([(geom.x, geom.y) for geom in gdf.geometry])
    dbscan = DBSCAN(eps=eps, min_samples=1).fit(coords)
    gdf["Cluster"] = dbscan.labels_

    cluster_aggregates = (
        gdf.groupby("Cluster")["TotalDeadTrees"]
        .sum()
        .reset_index()
        .rename(columns={"TotalDeadTrees": "ClusterDeadTrees"})
    )
    cluster_aggregates = cluster_aggregates.sort_values("ClusterDeadTrees", ascending=False)

    total_dead_trees = cluster_aggregates["ClusterDeadTrees"].sum()
    desired_ratios = np.array([1 - val_ratio - test_ratio, val_ratio, test_ratio])
    target_counts = (desired_ratios * total_dead_trees).round()

    train_keys, val_keys, test_keys = [], [], []
    train_count, val_count, test_count = 0, 0, 0

    for _, row in cluster_aggregates.iterrows():
        cluster = row["Cluster"]
        cluster_dead_trees = row["ClusterDeadTrees"]
        cluster_keys = gdf[gdf["Cluster"] == cluster]["Key"].tolist()

        if train_count < target_counts[0]:
            train_keys.extend(cluster_keys)
            train_count += cluster_dead_trees
        elif val_count < target_counts[1]:
            val_keys.extend(cluster_keys)
            val_count += cluster_dead_trees
        elif test_count < target_counts[2]:
            test_keys.extend(cluster_keys)
            test_count += cluster_dead_trees

    return train_keys, val_keys, test_keys