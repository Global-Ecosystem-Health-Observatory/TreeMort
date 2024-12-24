import os
import torch
import argparse
import rasterio

import numpy as np
import geopandas as gpd

from shapely.geometry import shape
from rasterio.features import rasterize, shapes
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

from misc.refine import UNetWithDeepSupervision
from misc.utils import find_file_pairs


def geojson_to_mask(geojson_path, image_shape, transform):
    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        return np.zeros(image_shape, dtype=np.uint8)
    
    shapes = [(geom, 1) for geom in gdf.geometry]
    mask = rasterize(shapes, out_shape=image_shape, transform=transform, fill=0, all_touched=True)
    return mask

def mask_to_geojson(mask, transform, crs, output_path):
    results = (
        {"properties": {}, "geometry": shape(geom)}
        for geom, value in shapes(mask, transform=transform)
        if value != 0  # Exclude background values
    )
    gdf = gpd.GeoDataFrame.from_features(results)
    gdf.set_crs(crs, inplace=True)
    gdf.to_file(output_path, driver="GeoJSON")

def combine_patches(patches, image_shape, patch_size, stride):
    combined_mask = np.zeros(image_shape, dtype=np.uint8)
    counts = np.zeros(image_shape, dtype=np.uint8)
    step = patch_size - stride

    for (i, j), patch in patches:
        combined_mask[i:i+patch_size, j:j+patch_size] += patch
        counts[i:i+patch_size, j:j+patch_size] += 1

    combined_mask = (combined_mask / np.maximum(counts, 1)).astype(np.uint8)
    return combined_mask

def prepare_patches(image, patch_size, stride):
    patches = []
    positions = []
    h, w = image.shape

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
            positions.append((i, j))

    return patches, positions

def get_shape_transform_crs(img_path):
    with rasterio.open(img_path) as src:
        image_shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs
    return image_shape, transform, crs

def process_file(image_path, geojson_path, output_path, model, device, patch_size=64, stride=64):
    if os.path.exists(output_path):
        print(f"Skipping {output_path} as it already exists.")
        return
    
    image_shape, transform, crs = get_shape_transform_crs(image_path)

    mask = geojson_to_mask(geojson_path, image_shape, transform)

    patches, patch_positions = prepare_patches(mask, patch_size, stride)

    processed_patches = []
    for (i, j), patch in zip(patch_positions, patches):
        input_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_tensor = torch.sigmoid(model(input_tensor)[0])
        pred_mask = (pred_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        processed_patches.append(((i, j), pred_mask))

    combined_mask = combine_patches(processed_patches, image_shape, patch_size, stride)

    mask_to_geojson(combined_mask, transform, crs, output_path)
    print(f"Processed {geojson_path} -> {output_path}")

def process_single_file(file_pair):
    global model  # Use the global model
    image_path, _, pred_path, output_path, patch_size, stride, model, device = file_pair

    try:
        process_file(image_path, pred_path, output_path, model, device, patch_size=patch_size, stride=stride)
        return f"Processed {image_path} -> {output_path}"
    except Exception as e:
        return f"Error processing {image_path}: {e}"
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine predictions and save as GeoJSON.")
    parser.add_argument("--best-model", type=str, required=True, help="Path to the best model weights.")
    parser.add_argument("--input-folder", type=str, required=True, help="Folder containing input Images and/or GeoJSON files.")
    parser.add_argument("--output-folder", type=str, required=True, help="Folder to save processed GeoJSON files.")
    parser.add_argument("--prediction-folder", type=str, default=None, help="Folder containing only GeoJSON files.")
    parser.add_argument("--patch-size", type=int, default=64, help="Size of the patches for processing.")
    parser.add_argument("--stride", type=int, default=64, help="Stride for sliding window.")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    prediction_folder = args.prediction_folder
    best_model_path = args.best_model
    patch_size = args.patch_size
    stride = args.stride

    input_folder = "/Users/anisr/Documents/dead_trees/Finland"
    prediciton_folder = "/Users/anisr/Documents/dead_trees/Finland/Predictions"

    file_pairs = find_file_pairs(input_folder, prediction_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNetWithDeepSupervision()

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        print(f"Loaded model weights from {best_model_path}")
    else:
        print("No previous model found.")

    model.eval()
    model.to(device)
    
    os.makedirs(output_folder, exist_ok=True)

    file_pairs_with_args = [
        (
            image_path,
            _,
            pred_path,
            os.path.join(output_folder, os.path.basename(pred_path)),
            patch_size,
            stride,
            model,
            device
        )
        for image_path, _, pred_path in file_pairs
    ]

    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_single_file, file_pairs_with_args)

    for result in results:
        print(result)

    print("Processing completed.")

'''

Usage:

PYTHONPATH="/Users/anisr/Documents/TreeSeg" python misc/refine_fin.py \
    --best-model ./output/refine/best.weights.pth \
    --input-folder /Users/anisr/Documents/dead_trees/Finland/Predictions \
    --output-folder /Users/anisr/Documents/dead_trees/Finland/Predictions_r

'''