import os
import fiona

import numpy as np
import geopandas as gpd

from tqdm import tqdm
from typing import Tuple, List, Dict
from shapely.geometry import shape
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        # Reproject to a projected CRS for area calculation
        crs_proj = "EPSG:3857"  # Use a suitable projected CRS (e.g., Web Mercator)
        pred_projected = prediction_gdf.to_crs(crs_proj)
        gt_projected = ground_truth_gdf.to_crs(crs_proj)

        intersection = gpd.overlay(pred_projected, gt_projected, how="intersection")
        if intersection.empty:
            return 0.0
        intersection["iou"] = intersection["geometry"].apply(
            lambda geom: geom.area
            / (
                pred_projected.loc[pred_projected.intersects(geom), "geometry"].area.values[0]
                + gt_projected.loc[gt_projected.intersects(geom), "geometry"].area.values[0]
                - geom.area
            )
        )
        return intersection["iou"].mean()

    def calculate_tree_iou():
        prediction_union = prediction_gdf.geometry.union_all()
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

    # Reproject to a suitable projected CRS (e.g., EPSG:3857 or UTM)
    crs_proj = "EPSG:3857"  # Replace with the appropriate CRS for your area if necessary
    pred_projected = prediction_gdf.to_crs(crs_proj)
    gt_projected = ground_truth_gdf.to_crs(crs_proj)

    pred_projected["centroid"] = pred_projected["geometry"].centroid
    gt_projected["centroid"] = gt_projected["geometry"].centroid

    matched_preds, matched_gts = set(), set()

    for gt_idx, gt_row in gt_projected.iterrows():
        distances = pred_projected["centroid"].distance(gt_row["centroid"])
        if distances.empty:
            continue  # Skip if there are no predictions to match
        nearest_idx = distances.idxmin()
        if nearest_idx not in matched_preds:
            matched_preds.add(nearest_idx)
            matched_gts.add(gt_idx)

    tp = len(matched_gts)
    fp = len(pred_projected) - len(matched_preds)
    fn = len(gt_projected) - len(matched_gts)
    return tp, fp, fn

def calculate_mean_ious(pred_dir: str, gt_path: str) -> Dict[str, float]:
    pred_paths = [os.path.join(pred_dir, filename) for filename in os.listdir(pred_dir) if filename.endswith('.geojson')]

    ground_truth_gdf = load_geodata_with_unique_ids(gt_path)

    with fiona.open(pred_paths[0]) as src:
        prediction_crs = src.crs

    if ground_truth_gdf.crs != prediction_crs:
        ground_truth_gdf = ground_truth_gdf.to_crs(prediction_crs)

    metrics = {"pixel_iou": [], "tree_iou": [], "tp": 0, "fp": 0, "fn": 0}

    def process_prediction_file(prediction_path: str) -> Tuple[float, float, int, int, int]:
        try:
            prediction_gdf = load_geodata_with_unique_ids(prediction_path)

            filtered_ground_truth_gdf = ground_truth_gdf[
                ground_truth_gdf.intersects(prediction_gdf.geometry.union_all())
            ]

            pixel_iou, tree_iou = calculate_iou_metrics(prediction_gdf, filtered_ground_truth_gdf)
            tp_centroid, fp_centroid, fn_centroid = calculate_centroid_association_metrics(
                prediction_gdf, filtered_ground_truth_gdf
            )

            return pixel_iou, tree_iou, tp_centroid, fp_centroid, fn_centroid

        except Exception as e:
            print(f"Error processing files: {prediction_path}. Error: {e}")
            return 0, 0, 0, 0, 0

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_prediction_file, pred_path): pred_path for pred_path in pred_paths}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Files"):
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
    predictions_dir = "/Users/anisr/Documents/copenhagen_data/Predictions_filtered"
    ground_truth_path = "/Users/anisr/Documents/copenhagen_data/labels/target_features_20241031.gpkg"

    results = calculate_mean_ious(predictions_dir, ground_truth_path)

    print("\nEvaluation Results:")
    print("=" * 30)
    print(f"Mean Pixel IoU        : {results['mean_pixel_iou']:.4f}")
    print(f"Mean Tree IoU         : {results['mean_tree_iou']:.4f}")
    print(f"Total True Positives  : {results['total_tp']}")
    print(f"Total False Positives : {results['total_fp']}")
    print(f"Total False Negatives : {results['total_fn']}")
    print("=" * 30)
