import os
import fiona
import rasterio
import numpy as np
import geopandas as gpd

from math import sqrt
from typing import Tuple, List, Dict
from shapely.geometry import shape
from rasterio.warp import transform


def find_file_pairs(
    data_folder: str,
    predictions_folder: str = None,
    image_dir_name: str = "Images",
    geojsons_dir_name: str = "Geojsons",
    predictions_dir_name: str = "Predictions",
    file_ext: str = ".geojson",
) -> List[Tuple[str, str, str]]:
    pairs = []
    pred_folder = predictions_folder if predictions_folder else os.path.join(data_folder, predictions_dir_name)

    for root, dirs, _ in os.walk(data_folder):
        if {image_dir_name, geojsons_dir_name}.issubset(dirs):
            image_files = {
                os.path.splitext(f)[0]: os.path.join(root, image_dir_name, f)
                for f in os.listdir(os.path.join(root, image_dir_name))
                if f.endswith(".tiff") or f.endswith(".tif")  # Assuming images are in TIFF format
            }
            gt_files = {
                os.path.splitext(f)[0]: os.path.join(root, geojsons_dir_name, f)
                for f in os.listdir(os.path.join(root, geojsons_dir_name))
                if f.endswith(file_ext)
            }
            pred_files = {
                os.path.splitext(f)[0]: os.path.join(pred_folder, f)
                for f in os.listdir(pred_folder)
                if f.endswith(file_ext)
            }
            # Match files by name
            common_files = image_files.keys() & gt_files.keys() & pred_files.keys()
            for fname in common_files:
                pairs.append((image_files[fname], gt_files[fname], pred_files[fname]))
    return pairs


def filter_file_pairs(file_pairs, keys):
    return [
        pair for pair in file_pairs 
        if any(key in pair[1] for key in keys)
    ]


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
    

def calculate_iou_metrics(prediction_gdf: gpd.GeoDataFrame, ground_truth_gdf: gpd.GeoDataFrame, overlap_threshold=0.5) -> Tuple[float, float]:

    def calculate_pixel_iou():
        try:
            if prediction_gdf.empty or ground_truth_gdf.empty:
                return 0.0

            ious = []
            for _, pred_row in prediction_gdf.iterrows():
                pred_geom = pred_row["geometry"]

                intersecting_gts = ground_truth_gdf[ground_truth_gdf.intersects(pred_geom)]

                if intersecting_gts.empty:
                    continue

                best_match = None
                best_iou = 0.0
                for _, gt_row in intersecting_gts.iterrows():
                    gt_geom = gt_row["geometry"]
                    intersect_area = pred_geom.intersection(gt_geom).area
                    union_area = pred_geom.area + gt_geom.area - intersect_area
                    iou = intersect_area / union_area if union_area > 0 else 0.0

                    if iou > best_iou:
                        best_iou = iou
                        best_match = gt_geom

                if best_iou > 0.0:
                    ious.append(best_iou)

            return np.mean(ious) if ious else 0.0

        except Exception as e:
            print(f"Error in calculate_pixel_iou: {e}")
            return 0.0

    def calculate_tree_iou():
        if prediction_gdf.empty or ground_truth_gdf.empty:
            return 0.0  # Return 0 IoU with all counts as 0

        matched_preds = set()
        matched_gts = set()

        # Check each predicted tree segment against all ground truth segments
        for pred_idx, pred_row in prediction_gdf.iterrows():
            pred_geom = pred_row["geometry"]
            best_overlap = 0.0
            best_match = None

            for gt_idx, gt_row in ground_truth_gdf.iterrows():
                gt_geom = gt_row["geometry"]
                intersect_area = pred_geom.intersection(gt_geom).area
                pred_area = pred_geom.area

                if pred_area > 0:
                    overlap_ratio = intersect_area / pred_area
                else:
                    overlap_ratio = 0.0

                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_match = gt_idx

            if best_overlap >= overlap_threshold and best_match not in matched_gts:
                matched_preds.add(pred_idx)
                matched_gts.add(best_match)

        # Calculate TP, FP, FN counts
        tp = len(matched_gts)  # True Positives: ground truth segments matched correctly
        fp = len(prediction_gdf) - len(matched_preds)  # False Positives: extra predicted segments
        fn = len(ground_truth_gdf) - len(matched_gts)  # False Negatives: missed ground truth segments

        tree_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        return tree_iou
    
    return calculate_pixel_iou(), calculate_tree_iou()


def calculate_centroid_errors(prediction_gdf: gpd.GeoDataFrame, ground_truth_gdf: gpd.GeoDataFrame) -> Tuple[int, int, int, float]:
    if prediction_gdf.empty or ground_truth_gdf.empty:
        tp = 0
        fp = len(prediction_gdf) if not prediction_gdf.empty else 0
        fn = len(ground_truth_gdf) if not ground_truth_gdf.empty else 0
        centroid_error = float('nan')
        return tp, fp, fn, centroid_error

    prediction_gdf["centroid"] = prediction_gdf["geometry"].centroid
    ground_truth_gdf["centroid"] = ground_truth_gdf["geometry"].centroid

    matched_preds, matched_gts = set(), set()
    total_error = 0.0

    for gt_idx, gt_row in ground_truth_gdf.iterrows():
        distances = prediction_gdf["centroid"].distance(gt_row["centroid"])
        if distances.empty:
            continue
        nearest_idx = distances.idxmin()
        if nearest_idx not in matched_preds:
            matched_preds.add(nearest_idx)
            matched_gts.add(gt_idx)
            total_error += sqrt((prediction_gdf.loc[nearest_idx, "centroid"].x - gt_row["centroid"].x)**2 +
                                (prediction_gdf.loc[nearest_idx, "centroid"].y - gt_row["centroid"].y)**2)

    tp = len(matched_gts)
    fp = len(prediction_gdf) - len(matched_preds)
    fn = len(ground_truth_gdf) - len(matched_gts)
    centroid_error = total_error / tp if tp > 0 else float('nan')
    return tp, fp, fn, centroid_error


def calculate_precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1_score


def extract_centroid_from_metadata(image_path: str) -> Tuple[float, float]:
    try:
        with rasterio.open(image_path) as dataset:
            bounds = dataset.bounds  # Bounds are in the CRS of the dataset
            centroid_x = (bounds.left + bounds.right) / 2
            centroid_y = (bounds.top + bounds.bottom) / 2

            if dataset.crs.is_geographic:
                return centroid_y, centroid_x  # Already in lat/lon
            else:
                centroid_lon, centroid_lat = transform(
                    dataset.crs,
                    "EPSG:4326",
                    [centroid_x],
                    [centroid_y]
                )
                return centroid_lat[0], centroid_lon[0]
    except Exception as e:
        print(f"Error extracting centroid from metadata for {image_path}: {e}")
        return None, None