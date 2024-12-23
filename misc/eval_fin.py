import os
import fiona
import numpy as np
import geopandas as gpd

from tqdm import tqdm
from typing import Tuple, List, Dict
from shapely.geometry import shape
from concurrent.futures import ThreadPoolExecutor, as_completed

from misc.utils import find_file_pairs


def validate_geometry(geom):
    try:
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.geom_type not in ["Polygon", "MultiPolygon"]:
            return None
        return geom
    except Exception:
        return None


def load_geodata_with_unique_ids(file_path: str) -> gpd.GeoDataFrame:
    with fiona.open(file_path) as src:
        features = [
            {
                **feature,
                "id": str(idx),  # Ensure unique IDs
                "geometry": shape(feature["geometry"]) if feature["geometry"] else None,
            }
            for idx, feature in enumerate(src)
        ]
        geometries = [feature["geometry"] for feature in features]
        gdf = gpd.GeoDataFrame(features, geometry=geometries, crs=src.crs)
        gdf["geometry"] = gdf["geometry"].apply(validate_geometry)
        return gdf[gdf["geometry"].notnull()]


def calculate_iou_metrics(prediction_gdf: gpd.GeoDataFrame, ground_truth_gdf: gpd.GeoDataFrame) -> Tuple[float, float]:

    def calculate_pixel_iou():
        intersection = gpd.overlay(prediction_gdf, ground_truth_gdf, how="intersection")
        if intersection.empty:
            return 0.0
        intersection["iou"] = intersection["geometry"].apply(
            lambda geom: geom.area
            / (
                prediction_gdf.loc[prediction_gdf.intersects(geom), "geometry"].area.values[0]
                + ground_truth_gdf.loc[ground_truth_gdf.intersects(geom), "geometry"].area.values[0]
                - geom.area
            )
        )
        return intersection["iou"].mean()

    def calculate_tree_iou():
        prediction_union = prediction_gdf.geometry.unary_union
        filtered_gt = ground_truth_gdf[ground_truth_gdf.intersects(prediction_union)]
        tp, fp, fn = 0, 0, 0

        for pred_geom in prediction_gdf["geometry"]:
            if filtered_gt.intersects(pred_geom).any():
                tp += 1
            else:
                fp += 1
        for gt_geom in filtered_gt["geometry"]:
            if not prediction_gdf.intersects(gt_geom).any():
                fn += 1

        return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0

    return calculate_pixel_iou(), calculate_tree_iou()


def calculate_centroid_association_metrics(
    prediction_gdf: gpd.GeoDataFrame, ground_truth_gdf: gpd.GeoDataFrame
) -> Tuple[int, int, int]:
    if prediction_gdf.empty or ground_truth_gdf.empty:
        tp = 0
        fp = len(prediction_gdf) if not prediction_gdf.empty else 0
        fn = len(ground_truth_gdf) if not ground_truth_gdf.empty else 0
        return tp, fp, fn

    prediction_gdf["centroid"] = prediction_gdf["geometry"].centroid
    ground_truth_gdf["centroid"] = ground_truth_gdf["geometry"].centroid

    matched_preds, matched_gts = set(), set()

    for gt_idx, gt_row in ground_truth_gdf.iterrows():
        distances = prediction_gdf["centroid"].distance(gt_row["centroid"])
        if distances.empty:
            continue  # Skip if there are no predictions to match
        nearest_idx = distances.idxmin()
        if nearest_idx not in matched_preds:
            matched_preds.add(nearest_idx)
            matched_gts.add(gt_idx)

    tp = len(matched_gts)
    fp = len(prediction_gdf) - len(matched_preds)
    fn = len(ground_truth_gdf) - len(matched_gts)
    return tp, fp, fn


def process_prediction_file(prediction_path: str, ground_truth_path: str) -> Tuple[float, float, int, int, int]:
    try:
        prediction_gdf = load_geodata_with_unique_ids(prediction_path)
        ground_truth_gdf = load_geodata_with_unique_ids(ground_truth_path)

        if prediction_gdf.crs != ground_truth_gdf.crs:
            ground_truth_gdf = ground_truth_gdf.to_crs(prediction_gdf.crs)

        pixel_iou, tree_iou = calculate_iou_metrics(prediction_gdf, ground_truth_gdf)
        tp_centroid, fp_centroid, fn_centroid = calculate_centroid_association_metrics(prediction_gdf, ground_truth_gdf)

        '''
        print(f"\nEvaluation Results  : {os.path.basename(prediction_path)}")
        print("=" * 30)
        print(f"Mean Pixel IoU        : {pixel_iou:.4f}")
        print(f"Mean Tree IoU         : {tree_iou:.4f}")
        print(f"Total True Positives  : {tp_centroid}")
        print(f"Total False Positives : {fp_centroid}")
        print(f"Total False Negatives : {fn_centroid}")
        print("=" * 30)
        '''

        return pixel_iou, tree_iou, tp_centroid, fp_centroid, fn_centroid

    except Exception as e:
        print(f"Error processing files: {prediction_path} or {ground_truth_path}. Error: {e}")
        return 0, 0, 0, 0, 0


def calculate_mean_ious(data_folder: str, predictions_folder: str = None) -> Dict[str, float]:
    file_pairs = find_file_pairs(data_folder, predictions_folder)
    
    metrics = {"pixel_iou": [], "tree_iou": [], "tp": 0, "fp": 0, "fn": 0}

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_prediction_file, pred_path, gt_path): (
                pred_path,
                gt_path,
            )
            for _, pred_path, gt_path in file_pairs
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing File Pairs"):
            try:
                pixel_iou, tree_iou, tp, fp, fn = future.result()
                metrics["pixel_iou"].append(pixel_iou)
                metrics["tree_iou"].append(tree_iou)
                metrics["tp"] += tp
                metrics["fp"] += fp
                metrics["fn"] += fn
            except Exception as e:
                print(f"Error in processing: {e}")

    return {
        "mean_pixel_iou": (np.mean(metrics["pixel_iou"]) if metrics["pixel_iou"] else 0.0),
        "mean_tree_iou": np.mean(metrics["tree_iou"]) if metrics["tree_iou"] else 0.0,
        "total_tp": metrics["tp"],
        "total_fp": metrics["fp"],
        "total_fn": metrics["fn"],
    }


if __name__ == "__main__":
    data_folder = "/Users/anisr/Documents/dead_trees/Finland"
    predictions_folder = "/Users/anisr/Documents/dead_trees/Finland/Predictions_r"

    results = calculate_mean_ious(data_folder, predictions_folder)

    print("\nEvaluation Results:")
    print("=" * 30)
    print(f"Mean Pixel IoU        : {results['mean_pixel_iou']:.4f}")
    print(f"Mean Tree IoU         : {results['mean_tree_iou']:.4f}")
    print(f"Total True Positives  : {results['total_tp']}")
    print(f"Total False Positives : {results['total_fp']}")
    print(f"Total False Negatives : {results['total_fn']}")
    print("=" * 30)


'''

Usage:

PYTHONPATH="/Users/anisr/Documents/TreeSeg" python misc/eval_fin.py

'''