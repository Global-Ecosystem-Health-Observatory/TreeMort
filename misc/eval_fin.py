import os
import csv
import argparse
import numpy as np
import geopandas as gpd

from collections import OrderedDict

from tqdm import tqdm
from typing import Tuple, List, Dict
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor, as_completed

from misc.utils import (
    find_file_pairs,
    filter_file_pairs,
    load_geodata_with_unique_ids,
    calculate_iou_metrics,
    calculate_centroid_errors,
    calculate_precision_recall_f1,
    extract_centroid_from_metadata,
)


test_keys = [
    'L2344D_2022_1_ITD.geojson',
    'L3211A_2022_1_ITD.geojson',
    'L3211A_2022_2_ITD.geojson',
    'L4134A_2013_1.geojson',
    'L4134E_2013_1.geojson',
    'L5242G_2017_1_ITD.geojson',
    'L5244D_2017_1_ITD.geojson',
    'M3442B_2011_1.geojson',
    'N4242H_2019_1_ITD.geojson',
    'N5132F_2022_1_ITD.geojson',
    'N5412A_tile_0_2023_ITD.geojson',
    'N5412A_tile_1_2023_ITD.geojson',
    'N5412A_tile_3_2023_ITD.geojson',
    'N5412B_tile_0_2023_ITD.geojson',
    'N5412B_tile_2_2023_ITD.geojson',
    'N5412C_tile_2_2023_ITD.geojson',
    'N5412D_tile_1_2023_ITD.geojson',
    'N5412E_tile_1_2023_ITD.geojson',
    'N5412F_tile_1_2023_ITD.geojson',
    'N5442C_2014_1.geojson',
    'P4131H_2019_1_ITD.geojson',
    'P4131H_2019_2_ITD.geojson',
    'P4341G_2022_1_ITD.geojson',
    'P4343H_2022_1_ITD.geojson',
    'P5322F_2_1.geojson',
    'Q3334C_2019_1_ITD.geojson',
    'Q3334C_2019_2_ITD.geojson',
    'Q4211E_2019_1_ITD.geojson',
    'Q4323B_2022_1_ITD.geojson',
    'Q5422F_2022_1_ITD.geojson',
    'Q5422H_2022_1_ITD.geojson',
    'R4234D_2019_1_ITD.geojson',
    'R4414E_tile_0_2023_ITD.geojson',
    'R4414E_tile_3_2023_ITD.geojson',
    'R4414F_tile_0_2023_ITD.geojson',
    'R4414F_tile_1_2023_ITD.geojson',
    'R4414F_tile_3_2023_ITD.geojson',
    'R4423E_tile_1_2023_ITD.geojson',
    'R4423E_tile_2_2023_ITD.geojson',
    'S5112B_2022_1_ITD.geojson',
    'T4123G_2022_1_ITD.geojson',
    'U4324B_2022_1_ITD.geojson',
    'U5224D_2022_1_ITD.geojson',
    'U5242A_2022_1_ITD.geojson',
    'V4311C_2022_1_ITD.geojson',
    'V4314G_2022_1_ITD.geojson',
    'V4314G_2022_2_ITD.geojson',
    'V4314H_2022_1_ITD.geojson',
    'V4323C_2022_1_ITD.geojson',
    'V4331A_2022_1_ITD.geojson',
    'V4331A_2022_2_ITD.geojson',
    'V4341C_2022_1_ITD.geojson',
]

# test_keys = ['L2344D_2022_1_ITD.geojson']

def process_prediction_file(
    image_path: str, ground_truth_path: str, prediction_path: str
) -> Tuple[float, float, int, int, int, float, float, float, float, float, float, float, float, float, float]:
    try:
        prediction_gdf = load_geodata_with_unique_ids(prediction_path)
        ground_truth_gdf = load_geodata_with_unique_ids(ground_truth_path)

        if prediction_gdf.crs != ground_truth_gdf.crs:
            ground_truth_gdf = ground_truth_gdf.to_crs(prediction_gdf.crs)

        pixel_iou, tree_iou = calculate_iou_metrics(prediction_gdf, ground_truth_gdf)

        tp_centroid, fp_centroid, fn_centroid, centroid_error = calculate_centroid_errors(
            prediction_gdf, ground_truth_gdf
        )

        instance_precision, instance_recall, instance_f1_score = calculate_precision_recall_f1(tp_centroid, fp_centroid, fn_centroid)

        # New: Compute pixel-level Precision, Recall, F1
        # Assuming calculate_iou_metrics returns total_intersection_area, total_union_area, pred_area, gt_area
        # Update calculate_iou_metrics in utils to return these, or compute here
        total_intersection_area, total_union_area, total_pred_area, total_gt_area = calculate_area_metrics(prediction_gdf, ground_truth_gdf)  # Assume new function in utils

        pixel_precision = total_intersection_area / total_pred_area if total_pred_area > 0 else 0
        pixel_recall = total_intersection_area / total_gt_area if total_gt_area > 0 else 0
        pixel_f1_score = 2 * (pixel_precision * pixel_recall) / (pixel_precision + pixel_recall) if (pixel_precision + pixel_recall) > 0 else 0

        latitude, longitude = extract_centroid_from_metadata(image_path)

        prediction_count = len(prediction_gdf)
        ground_truth_count = len(ground_truth_gdf)

        return (
            pixel_iou,
            tree_iou,
            tp_centroid,
            fp_centroid,
            fn_centroid,
            instance_precision,
            instance_recall,
            instance_f1_score,
            pixel_precision,
            pixel_recall,
            pixel_f1_score,
            centroid_error,
            latitude,
            longitude,
            prediction_count,
            ground_truth_count,
        )

    except Exception as e:
        print(f"Error processing files: {prediction_path} or {ground_truth_path}. Error: {e}")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, 0, 0


# New helper function (add to misc.utils.py)
def calculate_area_metrics(pred_gdf, gt_gdf):
    # Compute total intersection, union, pred area, gt area from vector overlaps
    # Example logic (simplified; implement fully based on your calculate_iou_metrics)
    pred_gdf['area'] = pred_gdf.geometry.area
    gt_gdf['area'] = gt_gdf.geometry.area
    total_pred_area = pred_gdf['area'].sum()
    total_gt_area = gt_gdf['area'].sum()
    # Total intersection from pairwise overlaps (use sjoin or similar for efficiency)
    intersections = gpd.overlay(pred_gdf, gt_gdf, how='intersection')
    intersections['inter_area'] = intersections.geometry.area
    total_intersection_area = intersections['inter_area'].sum()
    total_union_area = total_pred_area + total_gt_area - total_intersection_area
    return total_intersection_area, total_union_area, total_pred_area, total_gt_area


def save_results_to_csv(results, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "File",
                "Pixel IoU",
                "Tree IoU",
                "True Positives",
                "False Positives",
                "False Negatives",
                "Instance Precision",
                "Instance Recall",
                "Instance F1-Score",
                "Pixel Precision",
                "Pixel Recall",
                "Pixel F1-Score",
                "Centroid Error",
                "Latitude",
                "Longitude",
                "Prediction Count",
                "Ground Truth Count",
            ]
        )
        for result in results:
            writer.writerow(result)
    print(f"Results saved to {output_file}")


def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    clean_data = [x for x in data if not np.isnan(x)]

    if len(clean_data) == 0:
        return float('nan'), float('nan')

    mean = np.nanmean(clean_data)
    std_dev = np.nanstd(clean_data, ddof=1)
    n = len(clean_data)
    z_score = norm.ppf((1 + confidence) / 2)
    margin_of_error = z_score * (std_dev / np.sqrt(n))

    return mean - margin_of_error, mean + margin_of_error


def calculate_mean_ious(data_folder: str, predictions_folder: str = None, output_csv: str = None, eval_test_only: bool = True) -> Dict[str, float]:
    file_pairs = find_file_pairs(data_folder, predictions_folder)

    if eval_test_only:
        filtered_file_pairs = filter_file_pairs(file_pairs, test_keys)
    else:
        filtered_file_pairs = file_pairs

    metrics = {
        "pixel_iou": [],
        "tree_iou": [],
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "instance_precision": [],
        "instance_recall": [],
        "instance_f1_score": [],
        "pixel_precision": [],
        "pixel_recall": [],
        "pixel_f1_score": [],
        "centroid_err": [],
        "pred_count": [],
        "gt_count": [],
    }
    detailed_results = []

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_prediction_file, image_path, gt_path, pred_path): (
                gt_path,
                pred_path,
            )
            for image_path, gt_path, pred_path in filtered_file_pairs
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing File Pairs"):
            try:
                pixel_iou, tree_iou, tp, fp, fn, i_p, i_r, i_f1, p_p, p_r, p_f1, cerr, lat, lon, pred_count, gt_count = future.result()

                metrics["pixel_iou"].append(pixel_iou * pred_count)
                metrics["tree_iou"].append(tree_iou * gt_count)
                metrics["pred_count"].append(pred_count)
                metrics["gt_count"].append(gt_count)
                metrics["tp"] += tp
                metrics["fp"] += fp
                metrics["fn"] += fn
                metrics["instance_precision"].append(i_p)
                metrics["instance_recall"].append(i_r)
                metrics["instance_f1_score"].append(i_f1)
                metrics["pixel_precision"].append(p_p)
                metrics["pixel_recall"].append(p_r)
                metrics["pixel_f1_score"].append(p_f1)
                metrics["centroid_err"].append(cerr)

                detailed_results.append(
                    [
                        os.path.basename(futures[future][0]),
                        pixel_iou,
                        tree_iou,
                        tp,
                        fp,
                        fn,
                        i_p,
                        i_r,
                        i_f1,
                        p_p,
                        p_r,
                        p_f1,
                        cerr,
                        lat,
                        lon,
                        pred_count,
                        gt_count,
                    ]
                )
            except Exception as e:
                print(f"Error in processing: {e}")

    # Save detailed results to CSV if output_csv is provided
    if output_csv:
        save_results_to_csv(detailed_results, output_csv)

    # Use CSV-based approach to compute summary statistics
    import pandas as pd
    # If output_csv is provided, load the CSV; otherwise, create a DataFrame from detailed_results
    if output_csv and os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
    else:
        df = pd.DataFrame(detailed_results, columns=[
            "File",
            "Pixel IoU",
            "Tree IoU",
            "True Positives",
            "False Positives",
            "False Negatives",
            "Instance Precision",
            "Instance Recall",
            "Instance F1-Score",
            "Pixel Precision",
            "Pixel Recall",
            "Pixel F1-Score",
            "Centroid Error",
            "Latitude",
            "Longitude",
            "Prediction Count",
            "Ground Truth Count",
        ])

    summary = df.describe()

    if df.empty or "mean" not in df.describe().index:
        print("Evaluation aborted: DataFrame is empty or missing summary statistics.")
        return {}

    mean_pixel_iou = summary.loc["mean", "Pixel IoU"]
    std_pixel_iou = summary.loc["std", "Pixel IoU"]
    ci_pixel_iou = compute_confidence_interval(df["Pixel IoU"].tolist())

    mean_tree_iou = summary.loc["mean", "Tree IoU"]
    std_tree_iou = summary.loc["std", "Tree IoU"]
    ci_tree_iou = compute_confidence_interval(df["Tree IoU"].tolist())

    mean_instance_precision = summary.loc["mean", "Instance Precision"]
    std_instance_precision = summary.loc["std", "Instance Precision"]
    ci_instance_precision = compute_confidence_interval(df["Instance Precision"].tolist())

    mean_instance_recall = summary.loc["mean", "Instance Recall"]
    std_instance_recall = summary.loc["std", "Instance Recall"]
    ci_instance_recall = compute_confidence_interval(df["Instance Recall"].tolist())

    mean_instance_f1_score = summary.loc["mean", "Instance F1-Score"]
    std_instance_f1_score = summary.loc["std", "Instance F1-Score"]
    ci_instance_f1_score = compute_confidence_interval(df["Instance F1-Score"].tolist())

    mean_pixel_precision = summary.loc["mean", "Pixel Precision"]
    std_pixel_precision = summary.loc["std", "Pixel Precision"]
    ci_pixel_precision = compute_confidence_interval(df["Pixel Precision"].tolist())

    mean_pixel_recall = summary.loc["mean", "Pixel Recall"]
    std_pixel_recall = summary.loc["std", "Pixel Recall"]
    ci_pixel_recall = compute_confidence_interval(df["Pixel Recall"].tolist())

    mean_pixel_f1_score = summary.loc["mean", "Pixel F1-Score"]
    std_pixel_f1_score = summary.loc["std", "Pixel F1-Score"]
    ci_pixel_f1_score = compute_confidence_interval(df["Pixel F1-Score"].tolist())

    mean_centroid_err = summary.loc["mean", "Centroid Error"]
    std_centroid_err = summary.loc["std", "Centroid Error"]
    ci_centroid_err = compute_confidence_interval(df["Centroid Error"].tolist())

    total_tp = df["True Positives"].sum()
    total_fp = df["False Positives"].sum()
    total_fn = df["False Negatives"].sum()

    results = {
        "mean_pixel_iou": mean_pixel_iou,
        "std_pixel_iou": std_pixel_iou,
        "ci_pixel_iou": ci_pixel_iou,
        "mean_tree_iou": mean_tree_iou,
        "std_tree_iou": std_tree_iou,
        "ci_tree_iou": ci_tree_iou,
        "mean_instance_precision": mean_instance_precision,
        "std_instance_precision": std_instance_precision,
        "ci_instance_precision": ci_instance_precision,
        "mean_instance_recall": mean_instance_recall,
        "std_instance_recall": std_instance_recall,
        "ci_instance_recall": ci_instance_recall,
        "mean_instance_f1_score": mean_instance_f1_score,
        "std_instance_f1_score": std_instance_f1_score,
        "ci_instance_f1_score": ci_instance_f1_score,
        "mean_pixel_precision": mean_pixel_precision,
        "std_pixel_precision": std_pixel_precision,
        "ci_pixel_precision": ci_pixel_precision,
        "mean_pixel_recall": mean_pixel_recall,
        "std_pixel_recall": std_pixel_recall,
        "ci_pixel_recall": ci_pixel_recall,
        "mean_pixel_f1_score": mean_pixel_f1_score,
        "std_pixel_f1_score": std_pixel_f1_score,
        "ci_pixel_f1_score": ci_pixel_f1_score,
        "mean_centroid_err": mean_centroid_err,
        "std_centroid_err": std_centroid_err,
        "ci_centroid_err": ci_centroid_err,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IoU metrics for predictions")
    parser.add_argument("data_folder", help="Path to the data folder containing ground truth subfolder.")
    parser.add_argument("--pred-folder", default="", help="Path to the data folder containing predictions subfolders.")
    parser.add_argument("--output-folder", default="./output/eval", help="Folder to store output CSV files.")
    # By default, eval_test_only is True; use --all to evaluate all files (not only test keys)
    parser.add_argument("--all", dest="eval_test_only", action="store_false", help="Evaluate all files (not only test keys).")
    args = parser.parse_args()

    data_folder = args.data_folder
    pred_folder = args.pred_folder
    output_folder = args.output_folder
    eval_test_only = args.eval_test_only

    if not pred_folder:
        pred_folder = data_folder
    
    # Find all subdirectories in data_folder that start with 'Predictions'
    predictions_folders = []
    for entry in os.listdir(pred_folder):
        full_path = os.path.join(pred_folder, entry)
        if os.path.isdir(full_path) and entry.startswith("Predictions"):
            predictions_folders.append(full_path)

    if not predictions_folders:
        print("No predictions folders found in the data folder.")
        exit(1)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    summary_all = OrderedDict()

    for predictions_folder in predictions_folders:
        folder_name = os.path.basename(predictions_folder)
        output_csv = os.path.join(output_folder, f"eval_{folder_name}.csv")
        print(f"\nProcessing Predictions Folder: {predictions_folder}")

        results = calculate_mean_ious(data_folder, predictions_folder, output_csv, eval_test_only=eval_test_only)

        # Add results to summary_all for later summary
        summary_all[folder_name] = {
            "Mean Pixel IoU": results["mean_pixel_iou"],
            "Mean Tree IoU": results["mean_tree_iou"],
            "Instance Precision": results["mean_instance_precision"],
            "Instance Recall": results["mean_instance_recall"],
            "Instance F1-Score": results["mean_instance_f1_score"],
            "Pixel Precision": results["mean_pixel_precision"],
            "Pixel Recall": results["mean_pixel_recall"],
            "Pixel F1-Score": results["mean_pixel_f1_score"],
            "Mean Centroid Error": results["mean_centroid_err"],
            "Total True Positives": results["total_tp"],
            "Total False Positives": results["total_fp"],
            "Total False Negatives": results["total_fn"]
        }

        print(f"\nEvaluation Results for folder: {folder_name}")
        print("=" * 50)
        print(
            f"Mean Pixel IoU        : {results['mean_pixel_iou']:.4f} "
            f"(CI: {results['ci_pixel_iou'][0]:.4f} - {results['ci_pixel_iou'][1]:.4f}, "
            f"Std: {results['std_pixel_iou']:.4f})"
        )
        print(
            f"Mean Tree IoU         : {results['mean_tree_iou']:.4f} "
            f"(CI: {results['ci_tree_iou'][0]:.4f} - {results['ci_tree_iou'][1]:.4f}, "
            f"Std: {results['std_tree_iou']:.4f})"
        )
        print(
            f"Instance Precision    : {results['mean_instance_precision']:.4f} "
            f"(CI: {results['ci_instance_precision'][0]:.4f} - {results['ci_instance_precision'][1]:.4f}, "
            f"Std: {results['std_instance_precision']:.4f})"
        )
        print(
            f"Instance Recall       : {results['mean_instance_recall']:.4f} "
            f"(CI: {results['ci_instance_recall'][0]:.4f} - {results['ci_instance_recall'][1]:.4f}, "
            f"Std: {results['std_instance_recall']:.4f})"
        )
        print(
            f"Instance F1-Score     : {results['mean_instance_f1_score']:.4f} "
            f"(CI: {results['ci_instance_f1_score'][0]:.4f} - {results['ci_instance_f1_score'][1]:.4f}, "
            f"Std: {results['std_instance_f1_score']:.4f})"
        )
        print(
            f"Pixel Precision       : {results['mean_pixel_precision']:.4f} "
            f"(CI: {results['ci_pixel_precision'][0]:.4f} - {results['ci_pixel_precision'][1]:.4f}, "
            f"Std: {results['std_pixel_precision']:.4f})"
        )
        print(
            f"Pixel Recall          : {results['mean_pixel_recall']:.4f} "
            f"(CI: {results['ci_pixel_recall'][0]:.4f} - {results['ci_pixel_recall'][1]:.4f}, "
            f"Std: {results['std_pixel_recall']:.4f})"
        )
        print(
            f"Pixel F1-Score        : {results['mean_pixel_f1_score']:.4f} "
            f"(CI: {results['ci_pixel_f1_score'][0]:.4f} - {results['ci_pixel_f1_score'][1]:.4f}, "
            f"Std: {results['std_pixel_f1_score']:.4f})"
        )
        print(
            f"Mean Centroid Error   : {results['mean_centroid_err']:.4f} "
            f"(CI: {results['ci_centroid_err'][0]:.4f} - {results['ci_centroid_err'][1]:.4f}, "
            f"Std: {results['std_centroid_err']:.4f})"
        )
        print("=" * 50)
        print(f"Total True Positives  : {results['total_tp']}")
        print(f"Total False Positives : {results['total_fp']}")
        print(f"Total False Negatives : {results['total_fn']}")
        # Save formatted results to CSV
        summary_csv = os.path.join(output_folder, f"summary_{folder_name}.csv")
        with open(summary_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value", "95% CI Lower", "95% CI Upper", "Std"])
            writer.writerow(["Mean Pixel IoU", results['mean_pixel_iou'], results['ci_pixel_iou'][0], results['ci_pixel_iou'][1], results['std_pixel_iou']])
            writer.writerow(["Mean Tree IoU", results['mean_tree_iou'], results['ci_tree_iou'][0], results['ci_tree_iou'][1], results['std_tree_iou']])
            writer.writerow(["Instance Precision", results['mean_instance_precision'], results['ci_instance_precision'][0], results['ci_instance_precision'][1], results['std_instance_precision']])
            writer.writerow(["Instance Recall", results['mean_instance_recall'], results['ci_instance_recall'][0], results['ci_instance_recall'][1], results['std_instance_recall']])
            writer.writerow(["Instance F1-Score", results['mean_instance_f1_score'], results['ci_instance_f1_score'][0], results['ci_instance_f1_score'][1], results['std_instance_f1_score']])
            writer.writerow(["Pixel Precision", results['mean_pixel_precision'], results['ci_pixel_precision'][0], results['ci_pixel_precision'][1], results['std_pixel_precision']])
            writer.writerow(["Pixel Recall", results['mean_pixel_recall'], results['ci_pixel_recall'][0], results['ci_pixel_recall'][1], results['std_pixel_recall']])
            writer.writerow(["Pixel F1-Score", results['mean_pixel_f1_score'], results['ci_pixel_f1_score'][0], results['ci_pixel_f1_score'][1], results['std_pixel_f1_score']])
            writer.writerow(["Mean Centroid Error", results['mean_centroid_err'], results['ci_centroid_err'][0], results['ci_centroid_err'][1], results['std_centroid_err']])
            writer.writerow(["Total True Positives", results['total_tp'], "", "", ""])
            writer.writerow(["Total False Positives", results['total_fp'], "", "", ""])
            writer.writerow(["Total False Negatives", results['total_fn'], "", "", ""])
        print(f"Summary written to {summary_csv}")
        print("=" * 50)

    # Save summary across all folders
    summary_all_csv = os.path.join(output_folder, "summary_all_folders.csv")
    with open(summary_all_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        metrics = list(next(iter(summary_all.values())).keys())
        header = ["Folder"] + metrics
        writer.writerow(header)
        for folder, values in summary_all.items():
            row = [folder] + [values[m] for m in metrics]
            writer.writerow(row)
    print(f"Summary across all folders written to {summary_all_csv}")