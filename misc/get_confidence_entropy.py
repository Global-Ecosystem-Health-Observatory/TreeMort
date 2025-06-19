import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
import os

def compute_confidence_and_entropy(logits):
    probs = torch.sigmoid(logits)  # shape: (B, 1, H, W)
    confidence = probs.squeeze(1)  # shape: (B, H, W)
    entropy = - (probs * torch.log(probs + 1e-6) +
                 (1 - probs) * torch.log(1 - probs + 1e-6)).squeeze(1)
    return confidence, entropy

def load_image_as_tensor(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0)  # (1, C, H, W)

def process_test_loader(model, test_loader, device="cpu", relevant_ids=None):
    model.eval()
    model.to(device)

    confidence_means = []
    entropy_means = []

    for batch_idx, batch in enumerate(test_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(images)  # (B, 2, H, W)
            confidence_map, entropy_map = compute_confidence_and_entropy(logits)

        for i in range(images.size(0)):
            if relevant_ids:
                filename = Path(labels[i]['filename']).stem if isinstance(labels[i], dict) and 'filename' in labels[i] else None
                if filename and filename not in relevant_ids:
                    continue

            conf_np = confidence_map[i].cpu().numpy()
            entr_np = entropy_map[i].cpu().numpy()
            if not np.isnan(conf_np).all() and not np.all(conf_np == 0):
                confidence_means.append(conf_np.mean())
            if not np.isnan(entr_np).all() and not np.all(entr_np == 0):
                entropy_means.append(entr_np.mean())

    return confidence_means, entropy_means

def analyze_confidence_results(confidence_means, entropy_means, output_dir):
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Confidence: mean={np.mean(confidence_means):.4f}, std={np.std(confidence_means):.4f}")
    print(f"Entropy: mean={np.mean(entropy_means):.4f}, std={np.std(entropy_means):.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(confidence_means, bins=20, color="steelblue", edgecolor="black")
    plt.title("Average Confidence per Image")
    plt.xlabel("Confidence")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(entropy_means, bins=20, color="orangered", edgecolor="black")
    plt.title("Average Entropy per Image")
    plt.xlabel("Entropy")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_dir / "confidence_entropy_summary.png", dpi=300)

# --- Correlate confidence and false positives ---
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns

def correlate_confidence_fp(summary_csv, confidence_dir):
    summary_csv = Path(summary_csv)
    confidence_dir = Path(confidence_dir)

    df = pd.read_csv(summary_csv)

    confidence_means = {}
    for conf_file in confidence_dir.glob("*_confidence.npy"):
        base_name = conf_file.name.replace("_confidence.npy", "")
        conf_map = np.load(conf_file)
        confidence_means[base_name] = conf_map.mean()

    # Assumes 'filename' column exists without extension
    df["mean_confidence"] = df["filename"].apply(lambda f: confidence_means.get(Path(f).stem, np.nan))

    # Drop rows with missing confidence
    df = df.dropna(subset=["mean_confidence"])

    # Plot and correlate
    sns.set(style="whitegrid")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="mean_confidence", y="FP", s=60)
    r, p = pearsonr(df["mean_confidence"], df["FP"])
    plt.title(f"Confidence vs FP\nPearson r = {r:.2f}, p = {p:.3f}")
    plt.xlabel("Mean Confidence per Image")
    plt.ylabel("False Positives")
    plt.tight_layout()
    plt.savefig(confidence_dir / "confidence_vs_fp.png", dpi=300)
    print(f"Saved correlation plot to {confidence_dir/'confidence_vs_fp.png'}")

def load_relevant_ids(summary_csv):
    df = pd.read_csv(summary_csv)
    relevant = df[(df["TP"] > 0) | (df["FN"] > 0)]
    return set(relevant["filename"].str.replace(".geojson", ""))

def main():
    from treemort.utils.config import setup, expand_path
    from treemort.data.loader import prepare_datasets
    from treemort.modeling.builder import resume_or_load

    id2label = {0: "alive", 1: "dead"}

    device = torch.device("cpu")

    config_file_path = expand_path("${TREEMORT_REPO_PATH}/configs/model/flair_unet.txt")
    data_config_file_path = expand_path("${TREEMORT_REPO_PATH}/configs/data/finland.txt")

    conf = setup(config_file_path, data_config=data_config_file_path)

    train_loader, val_loader, test_loader = prepare_datasets(conf)

    model, optimizer, schedular, criterion, metrics, callbacks = resume_or_load(conf, id2label, len(train_loader), device)

    summary_csv_path = "/Users/anisr/Documents/dead_trees/Poland/Predictions_flair_unet_transfer/per_tree_summary_finetune.csv"
    relevant_ids = load_relevant_ids(summary_csv_path)
    confidence_means, entropy_means = process_test_loader(model, val_loader, device=device, relevant_ids=relevant_ids)

    analyze_confidence_results(confidence_means, entropy_means, "output/confidence_entropy")

if __name__ == "__main__":
    main()


'''

Remember to add repo path

export PYTHONPATH=/path/to/TreeSeg:$PYTHONPATH

'''