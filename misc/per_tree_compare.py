import pandas as pd

def compare_per_tree_metrics(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    df1.set_index("File", inplace=True)
    df2.set_index("File", inplace=True)

    common_files = df1.index.intersection(df2.index)

    comparison_rows = []
    for fname in common_files:
        row1 = df1.loc[fname]
        row2 = df2.loc[fname]
        diff = row2 - row1

        comparison = {
            "File": fname,
            "Pixel IoU Δ": diff["Pixel IoU"],
            "Tree IoU Δ": diff["Tree IoU"],
            "TP Δ": diff["True Positives"],
            "FP Δ": diff["False Positives"],
            "FN Δ": diff["False Negatives"],
            "Precision Δ": diff["Precision"],
            "Recall Δ": diff["Recall"],
            "F1-Score Δ": diff["F1-Score"],
            "Centroid Error Δ": diff["Centroid Error"],
            "Prediction Count Δ": diff["Prediction Count"],
            "Ground Truth Count Δ": diff["Ground Truth Count"]
        }

        comparison_rows.append(comparison)

    comparison_df = pd.DataFrame(comparison_rows)
    return comparison_df

if __name__ == "__main__":
    import sys

    # if len(sys.argv) != 3:
    #     print("Usage: python per_tree_compare.py file1.csv file2.csv")
    #     sys.exit(1)

    # file1, file2 = sys.argv[1], sys.argv[2]
    file1 = 'output/eval/poland/eval_Predictions_flair_unet_feature.csv'
    file2 = 'output/eval/poland/eval_Predictions_flair_unet_ensemble.csv'

    result = compare_per_tree_metrics(file1, file2)

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df1.set_index("File", inplace=True)
    df2.set_index("File", inplace=True)

    # Weighted mean for File 1
    print("\nWeighted Mean Scores for File 1:")
    weights1 = df1["Ground Truth Count"]
    weighted_means1 = (df1.mul(weights1, axis=0).sum(numeric_only=True) / weights1.sum())
    print(weighted_means1.to_string())

    # Weighted mean for File 2
    print("\nWeighted Mean Scores for File 2:")
    weights2 = df2["Ground Truth Count"]
    weighted_means2 = (df2.mul(weights2, axis=0).sum(numeric_only=True) / weights2.sum())
    print(weighted_means2.to_string())

    # print(result.to_string(index=False))
    output_file = "output/eval/poland/comparison_output.csv"
    result.to_csv(output_file, index=False)
    print(f"\nComparison saved to {output_file}")

    # mean_scores = result.drop(columns=["File"]).mean()
    # print("\nMean Differences Across All Files:")
    # print(mean_scores.to_string())