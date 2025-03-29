import os
import csv
import argparse
import numpy as np

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
) -> Tuple[float, float, int, int, int, float, float, float, float, float, float, float]:
    try:
        prediction_gdf = load_geodata_with_unique_ids(prediction_path)
        ground_truth_gdf = load_geodata_with_unique_ids(ground_truth_path)

        if prediction_gdf.crs != ground_truth_gdf.crs:
            ground_truth_gdf = ground_truth_gdf.to_crs(prediction_gdf.crs)

        pixel_iou, tree_iou = calculate_iou_metrics(prediction_gdf, ground_truth_gdf)

        tp_centroid, fp_centroid, fn_centroid, centroid_error = calculate_centroid_errors(
            prediction_gdf, ground_truth_gdf
        )

        precision, recall, f1_score = calculate_precision_recall_f1(tp_centroid, fp_centroid, fn_centroid)

        latitude, longitude = extract_centroid_from_metadata(image_path)

        prediction_count = len(prediction_gdf)
        ground_truth_count = len(ground_truth_gdf)

        return (
            pixel_iou,
            tree_iou,
            tp_centroid,
            fp_centroid,
            fn_centroid,
            precision,
            recall,
            f1_score,
            centroid_error,
            latitude,
            longitude,
            prediction_count,
            ground_truth_count,
        )

    except Exception as e:
        print(f"Error processing files: {prediction_path} or {ground_truth_path}. Error: {e}")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, 0, 0


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
                "Precision",
                "Recall",
                "F1-Score",
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
        "precision": [],
        "recall": [],
        "f1_score": [],
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
                pixel_iou, tree_iou, tp, fp, fn, p, r, f1, cerr, lat, lon, pred_count, gt_count = future.result()

                metrics["pixel_iou"].append(pixel_iou * pred_count)
                metrics["tree_iou"].append(tree_iou * gt_count)
                metrics["pred_count"].append(pred_count)
                metrics["gt_count"].append(gt_count)
                metrics["tp"] += tp
                metrics["fp"] += fp
                metrics["fn"] += fn
                metrics["precision"].append(p)
                metrics["recall"].append(r)
                metrics["f1_score"].append(f1)
                metrics["centroid_err"].append(cerr)

                detailed_results.append(
                    [
                        os.path.basename(futures[future][0]),
                        pixel_iou,
                        tree_iou,
                        tp,
                        fp,
                        fn,
                        p,
                        r,
                        f1,
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
            "Precision",
            "Recall",
            "F1-Score",
            "Centroid Error",
            "Latitude",
            "Longitude",
            "Prediction Count",
            "Ground Truth Count",
        ])

    summary = df.describe()

    mean_pixel_iou = summary.loc["mean", "Pixel IoU"]
    std_pixel_iou = summary.loc["std", "Pixel IoU"]
    ci_pixel_iou = compute_confidence_interval(df["Pixel IoU"].tolist())

    mean_tree_iou = summary.loc["mean", "Tree IoU"]
    std_tree_iou = summary.loc["std", "Tree IoU"]
    ci_tree_iou = compute_confidence_interval(df["Tree IoU"].tolist())

    mean_precision = summary.loc["mean", "Precision"]
    std_precision = summary.loc["std", "Precision"]
    ci_precision = compute_confidence_interval(df["Precision"].tolist())

    mean_recall = summary.loc["mean", "Recall"]
    std_recall = summary.loc["std", "Recall"]
    ci_recall = compute_confidence_interval(df["Recall"].tolist())

    mean_f1_score = summary.loc["mean", "F1-Score"]
    std_f1_score = summary.loc["std", "F1-Score"]
    ci_f1_score = compute_confidence_interval(df["F1-Score"].tolist())

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
        "mean_precision": mean_precision,
        "std_precision": std_precision,
        "ci_precision": ci_precision,
        "mean_recall": mean_recall,
        "std_recall": std_recall,
        "ci_recall": ci_recall,
        "mean_f1_score": mean_f1_score,
        "std_f1_score": std_f1_score,
        "ci_f1_score": ci_f1_score,
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
    parser.add_argument("data_folder", help="Path to the data folder containing ground truth and predictions subfolders.")
    parser.add_argument("--output-folder", default="./output/eval", help="Folder to store output CSV files.")
    # By default, eval_test_only is True; use --all to evaluate all files (not only test keys)
    parser.add_argument("--all", dest="eval_test_only", action="store_false", help="Evaluate all files (not only test keys).")
    args = parser.parse_args()

    data_folder = args.data_folder
    output_folder = args.output_folder
    eval_test_only = args.eval_test_only

    # Find all subdirectories in data_folder that start with 'Predictions'
    predictions_folders = []
    for entry in os.listdir(data_folder):
        full_path = os.path.join(data_folder, entry)
        if os.path.isdir(full_path) and entry.startswith("Predictions"):
            predictions_folders.append(full_path)

    if not predictions_folders:
        print("No predictions folders found in the data folder.")
        exit(1)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for predictions_folder in predictions_folders:
        folder_name = os.path.basename(predictions_folder)
        output_csv = os.path.join(output_folder, f"eval_{folder_name}.csv")
        print(f"\nProcessing Predictions Folder: {predictions_folder}")

        results = calculate_mean_ious(data_folder, predictions_folder, output_csv, eval_test_only=eval_test_only)

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
            f"Precision             : {results['mean_precision']:.4f} "
            f"(CI: {results['ci_precision'][0]:.4f} - {results['ci_precision'][1]:.4f}, "
            f"Std: {results['std_precision']:.4f})"
        )
        print(
            f"Recall                : {results['mean_recall']:.4f} "
            f"(CI: {results['ci_recall'][0]:.4f} - {results['ci_recall'][1]:.4f}, "
            f"Std: {results['std_recall']:.4f})"
        )
        print(
            f"F1-Score              : {results['mean_f1_score']:.4f} "
            f"(CI: {results['ci_f1_score'][0]:.4f} - {results['ci_f1_score'][1]:.4f}, "
            f"Std: {results['std_f1_score']:.4f})"
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
        print("=" * 50)