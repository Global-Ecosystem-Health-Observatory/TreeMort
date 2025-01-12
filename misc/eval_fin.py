import os
import csv
import numpy as np

from tqdm import tqdm
from typing import Tuple, List, Dict
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor, as_completed

from misc.utils import (
    find_file_pairs,
    load_geodata_with_unique_ids,
    calculate_iou_metrics,
    calculate_centroid_errors,
    calculate_precision_recall_f1,
    extract_centroid_from_metadata,
)


def process_prediction_file(
    image_path: str, prediction_path: str, ground_truth_path: str
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
    if len(data) == 0:
        return float('nan'), float('nan')
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # Use sample standard deviation
    n = len(data)
    z_score = norm.ppf((1 + confidence) / 2)
    margin_of_error = z_score * (std_dev / np.sqrt(n))
    return mean - margin_of_error, mean + margin_of_error


def calculate_mean_ious(data_folder: str, predictions_folder: str = None, output_csv: str = None) -> Dict[str, float]:
    file_pairs = find_file_pairs(data_folder, predictions_folder)

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
    }
    detailed_results = []

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_prediction_file, image_path, pred_path, gt_path): (
                pred_path,
                gt_path,
            )
            for image_path, pred_path, gt_path in file_pairs
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing File Pairs"):
            try:
                pixel_iou, tree_iou, tp, fp, fn, p, r, f1, cerr, lat, lon, pred_count, gt_count = future.result()
                metrics["pixel_iou"].append(pixel_iou)
                metrics["tree_iou"].append(tree_iou)
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

    if output_csv:
        save_results_to_csv(detailed_results, output_csv)

    results = {
        "mean_pixel_iou": np.mean(metrics["pixel_iou"]),
        "std_pixel_iou": np.std(metrics["pixel_iou"]),
        "ci_pixel_iou": compute_confidence_interval(metrics["pixel_iou"]),
        "mean_tree_iou": np.mean(metrics["tree_iou"]),
        "std_tree_iou": np.std(metrics["tree_iou"]),
        "ci_tree_iou": compute_confidence_interval(metrics["tree_iou"]),
        "mean_precision": np.mean(metrics["precision"]),
        "std_precision": np.std(metrics["precision"]),
        "ci_precision": compute_confidence_interval(metrics["precision"]),
        "mean_recall": np.mean(metrics["recall"]),
        "std_recall": np.std(metrics["recall"]),
        "ci_recall": compute_confidence_interval(metrics["recall"]),
        "mean_f1_score": np.mean(metrics["f1_score"]),
        "std_f1_score": np.std(metrics["f1_score"]),
        "ci_f1_score": compute_confidence_interval(metrics["f1_score"]),
        "mean_centroid_err": np.nanmean(metrics["centroid_err"]),
        "std_centroid_err": np.nanstd(metrics["centroid_err"]),
        "ci_centroid_err": compute_confidence_interval(metrics["centroid_err"]),
        "total_tp": metrics["tp"],
        "total_fp": metrics["fp"],
        "total_fn": metrics["fn"],
    }
    return results


if __name__ == "__main__":
    data_folder = "/Users/anisr/Documents/dead_trees/Finland"
    predictions_folders = [
        "/Users/anisr/Documents/dead_trees/Finland/Predictions",
        "/Users/anisr/Documents/dead_trees/Finland/Predictions_r",
    ]
    output_csvs = [
        "./output/eval/eval_fin.csv",
        "./output/eval/eval_fin_r.csv",
    ]

    for predictions_folder, output_csv in zip(predictions_folders, output_csvs):
        print(f"\nProcessing Predictions Folder: {predictions_folder}")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        results = calculate_mean_ious(data_folder, predictions_folder, output_csv)

        print("\nEvaluation Results:")
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