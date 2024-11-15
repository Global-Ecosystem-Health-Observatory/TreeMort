import os

import numpy as np
import geopandas as gpd

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from shapely.validation import make_valid


def calculate_pixel_iou(prediction_gdf, ground_truth_gdf):
    intersection_gdf = gpd.overlay(prediction_gdf, ground_truth_gdf, how='intersection')

    intersection_gdf['iou'] = intersection_gdf['geometry'].apply(
        lambda geom: geom.area / (
            prediction_gdf.loc[prediction_gdf.intersects(geom), 'geometry'].area.values[0] +
            ground_truth_gdf.loc[ground_truth_gdf.intersects(geom), 'geometry'].area.values[0] -
            geom.area
        )
    )

    if intersection_gdf.empty:
        return 0
    
    iou_pixels = intersection_gdf['iou'].mean()
    return iou_pixels


def calculate_tree_iou(prediction_gdf, ground_truth_gdf):
    prediction_gdf = prediction_gdf.copy()
    ground_truth_gdf = ground_truth_gdf.copy()

    prediction_gdf.loc[:, 'geometry'] = prediction_gdf['geometry'].apply(lambda x: x.buffer(0) if not x.is_valid else x)
    ground_truth_gdf.loc[:, 'geometry'] = ground_truth_gdf['geometry'].apply(lambda x: x.buffer(0) if not x.is_valid else x)
    
    prediction_union = prediction_gdf.geometry.union_all()

    filtered_ground_truth_gdf = ground_truth_gdf[ground_truth_gdf.intersects(prediction_union)]
    
    tp_trees, fp_trees, fn_trees = 0, 0, 0
    
    for pred_geom in prediction_gdf['geometry']:
        if filtered_ground_truth_gdf.intersects(pred_geom).any():
            tp_trees += 1
        else:
            fp_trees += 1

    for gt_geom in filtered_ground_truth_gdf['geometry']:
        if not prediction_gdf.intersects(gt_geom).any():
            fn_trees += 1

    if (tp_trees + fp_trees + fn_trees) == 0:
        iou_trees = 1.0 if len(filtered_ground_truth_gdf) == 0 else 0.0
    else:
        iou_trees = tp_trees / (tp_trees + fp_trees + fn_trees)
    
    return iou_trees


def validate_and_fix_geometry(geom):
    try:
        valid_geom = make_valid(geom)

        if valid_geom.is_valid and not valid_geom.is_empty:
            return valid_geom
        else:
            return None
        
    except Exception:
        return None

def process_prediction_file(prediction_path, ground_truth_gdf):
    try:
        prediction_gdf = gpd.read_file(prediction_path)
        
        if prediction_gdf.empty:
            print(f"File {prediction_path} is empty. Skipping...")
            return 0, 0
        
        prediction_gdf['geometry'] = prediction_gdf['geometry'].apply(validate_and_fix_geometry)
        prediction_gdf = prediction_gdf[prediction_gdf['geometry'].notnull()]

        if prediction_gdf.empty:
            print(f"File {prediction_path} has no valid geometries after cleaning. Skipping...")
            return 0, 0
        
        if ground_truth_gdf.crs != prediction_gdf.crs:
            ground_truth_gdf = ground_truth_gdf.to_crs(prediction_gdf.crs)

        filtered_ground_truth_gdf = ground_truth_gdf[ground_truth_gdf.intersects(prediction_gdf.geometry.union_all())]

        mean_pixel_iou = calculate_pixel_iou(prediction_gdf, filtered_ground_truth_gdf)
        mean_tree_iou = calculate_tree_iou(prediction_gdf, filtered_ground_truth_gdf)

        return mean_pixel_iou, mean_tree_iou

    except Exception as e:
        print(f"Error processing {prediction_path}: {e}")
        return 0, 0  # Return default IoUs on error


def calculate_mean_ious(predictions_dir, ground_truth_path):
    ground_truth_gdf = gpd.read_file(ground_truth_path)

    if ground_truth_gdf.empty:
        print("The ground truth GeoDataFrame is empty. Exiting...")
        return 0, 0  # mean pixel-level IoU, mean tree-level IoU

    ground_truth_gdf['geometry'] = ground_truth_gdf['geometry'].apply(validate_and_fix_geometry)
    ground_truth_gdf = ground_truth_gdf[ground_truth_gdf['geometry'].notnull()]

    pixel_iou_values = []
    tree_iou_values = []

    prediction_files = [os.path.join(predictions_dir, filename) for filename in os.listdir(predictions_dir) if filename.endswith('.geojson')]

    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_prediction_file, prediction_path, ground_truth_gdf): prediction_path for prediction_path in prediction_files}
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing Files"):
            try:
                pixel_iou, tree_iou = future.result()
                pixel_iou_values.append(pixel_iou)
                tree_iou_values.append(tree_iou)
            except Exception as e:
                print(f"Error processing file {future_to_file[future]}: {e}")

    mean_pixel_iou = np.mean(pixel_iou_values) if pixel_iou_values else 0
    mean_tree_iou = np.mean(tree_iou_values) if tree_iou_values else 0

    return mean_pixel_iou, mean_tree_iou

if __name__ == "__main__":
    predictions_dir = "/Users/anisr/Documents/copenhagen_data/Predictions"
    ground_truth_path = "/Users/anisr/Documents/copenhagen_data/labels/target_features_20241031.gpkg"

    mean_pixel_iou, mean_tree_iou = calculate_mean_ious(predictions_dir, ground_truth_path)
    print(f"Mean Pixel IoU for all predictions: {mean_pixel_iou:.4f}")
    print(f"Mean Tree IoU for all predictions: {mean_tree_iou:.4f}")