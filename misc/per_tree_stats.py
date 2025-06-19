import os
import argparse
import pandas as pd
from tqdm import tqdm
from misc.utils import (
    find_file_pairs,
    filter_file_pairs,
    load_geodata_with_unique_ids,
    calculate_centroid_errors,
)


test_keys = [
    'M-33-21-B-d-2-3_0.geojson',
    'M-33-21-C-d-2-4.geojson',
    'M-33-32-A-c-2-1.geojson',
    'M-33-44-B-b-2-4_0.geojson',
    'M-33-57-B-d-4-2_0.geojson',
    'M-34-21-D-d-4-2.geojson',
    'M-34-5-B-a-1-3.geojson',
    'M-34-51-A-c-2-3_0.geojson',
    'M-34-51-D-d-2-2.geojson',
    'M-34-52-C-c-4-2_0.geojson',
    'M-34-55-A-c-1-3_0.geojson',
    'M-34-63-B-b-3-1.geojson',
    'M-34-64-A-a-1-2_0.geojson',
    'N-33-126-B-a-3-3.geojson',
    'N-33-127-A-c-1-1_1.geojson',
    'N-33-127-A-c-3-3_1.geojson',
    'N-33-127-A-d-1-1.geojson',
    'N-34-106-D-b-2-2_0.geojson',
    'N-34-107-B-c-1-4.geojson',
    'N-34-134-A-a-2-1.geojson',
    'N-34-63-C-b-3-4_0.geojson',
    'N-34-64-B-c-2-4.geojson',
    'N-34-66-B-a-4-2.geojson',
    'N-34-68-B-d-2-4.geojson',
    'N-34-69-C-b-1-3.geojson',
    'N-34-70-A-c-2-3_0.geojson',
    'N-34-70-D-b-3-3.geojson',
    'N-34-75-B-b-1-2.geojson',
    'N-34-75-D-b-2-4_0.geojson',
    'N-34-81-B-a-4-2_0.geojson',
    'N-34-82-A-b-1-2_0.geojson',
    'N-34-82-C-a-4-4_0.geojson',
    'N-34-94-A-d-2-3_0.geojson',
    'N-34-96-A-a-3-4.geojson',
    'N-34-99-B-b-3-1_0.geojson',
]



def analyze_per_tree(predictions_folder: str, data_folder: str = "data/Poland") -> None:
    print(data_folder)
    print(predictions_folder)

    file_pairs = find_file_pairs(data_folder, predictions_folder=predictions_folder)
    file_pairs = filter_file_pairs(file_pairs, test_keys)
    
    print(f"Evaluating per-tree stats for {len(file_pairs)} test files...\n")

    # Collect stats into a list
    stats = []

    for image_path, gt_path, pred_path in tqdm(file_pairs):
        try:
            prediction_gdf = load_geodata_with_unique_ids(pred_path)
            ground_truth_gdf = load_geodata_with_unique_ids(gt_path)

            if prediction_gdf.crs != ground_truth_gdf.crs:
                ground_truth_gdf = ground_truth_gdf.to_crs(prediction_gdf.crs)

            # Ensure both GeoDataFrames are projected to a metric CRS
            if prediction_gdf.crs.is_geographic:
                target_crs = "EPSG:3857"
                prediction_gdf = prediction_gdf.to_crs(target_crs)
                ground_truth_gdf = ground_truth_gdf.to_crs(target_crs)

            prediction_gdf = prediction_gdf[
                ~prediction_gdf.geometry.is_empty &
                prediction_gdf.geometry.notna() &
                prediction_gdf.is_valid
            ]
            prediction_areas = prediction_gdf.geometry.area.fillna(0.0)
            prediction_gdf = prediction_gdf[prediction_areas > 0]

            ground_truth_gdf = ground_truth_gdf[
                ~ground_truth_gdf.geometry.is_empty &
                ground_truth_gdf.geometry.notna() &
                ground_truth_gdf.is_valid
            ]
            ground_truth_areas = ground_truth_gdf.geometry.area.fillna(0.0)
            ground_truth_gdf = ground_truth_gdf[ground_truth_areas > 0]

            tp, fp, fn, _ = calculate_centroid_errors(prediction_gdf, ground_truth_gdf)

            print(f"{os.path.basename(gt_path)}:")
            print(f"  True Positives : {tp}")
            print(f"  False Positives: {fp}")
            print(f"  False Negatives: {fn}")
            print("-" * 50)

            stats.append({
                "filename": os.path.basename(gt_path),
                "TP": tp,
                "FP": fp,
                "FN": fn
            })

        except Exception as e:
            print(f"Error processing {pred_path}: {e}")

    # Save to CSV
    stats_df = pd.DataFrame(stats)
    output_csv = os.path.join(predictions_folder, "per_tree_summary.csv")
    stats_df.to_csv(output_csv, index=False)
    print(f"\nSaved summary to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-tree statistics for one predictions folder")
    parser.add_argument("predictions_folder", help="Path to the Predictions folder")
    parser.add_argument("--data-folder", default="data/Poland", help="Path to the data folder with ground truth")

    args = parser.parse_args()
    analyze_per_tree(args.predictions_folder, args.data_folder)