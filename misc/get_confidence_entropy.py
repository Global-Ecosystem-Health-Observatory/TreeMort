import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
import os

def compute_confidence_and_entropy(logits):
    probs = F.softmax(logits, dim=1)
    confidence = probs[:, 0]  # single-class foreground (model output has only one class channel)
    entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
    return confidence, entropy

def load_image_as_tensor(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0)  # (1, C, H, W)

def process_test_loader(model, test_loader, output_dir, device="cpu"):
    model.eval()
    model.to(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch in test_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        filenames = None  # update this if filenames are passed differently

        with torch.no_grad():
            logits = model(images)  # (B, 2, H, W)
            confidence_map, entropy_map = compute_confidence_and_entropy(logits)

        for i in range(images.size(0)):
            base_name = Path(filenames[i]).stem if filenames else f"sample_{i}"
            conf_np = confidence_map[i].cpu().numpy()
            entr_np = entropy_map[i].cpu().numpy()
            np.save(output_dir / f"{base_name}_confidence.npy", conf_np)
            np.save(output_dir / f"{base_name}_entropy.npy", entr_np)
            print(f"Saved: {base_name}")

import matplotlib.pyplot as plt

def summarize_confidence_entropy(output_dir):
    output_dir = Path(output_dir)
    confidence_means = []
    entropy_means = []

    for conf_file in output_dir.glob("*_confidence.npy"):
        entropy_file = output_dir / conf_file.name.replace("_confidence.npy", "_entropy.npy")

        conf_map = np.load(conf_file)
        entr_map = np.load(entropy_file)

        confidence_means.append(conf_map.mean())
        entropy_means.append(entr_map.mean())

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

# Example usage
# from model_loader import load_model
# model = load_model("path/to/checkpoint.pth")
# process_directory(model, "input/images", "output/confidence_entropy")

from treemort.utils.config import setup, expand_path
from treemort.data.loader import prepare_datasets
from treemort.modeling.builder import resume_or_load

id2label = {0: "alive", 1: "dead"}

device = torch.device("cpu")

config_file_path = expand_path("${TREEMORT_REPO_PATH}/configs/model/flair_unet.txt")
data_config_file_path = expand_path("${TREEMORT_REPO_PATH}/configs/data/poland.txt")

conf = setup(config_file_path, data_config=data_config_file_path)


train_loader, val_loader, test_loader = prepare_datasets(conf)

model, optimizer, schedular, criterion, metrics, callbacks = resume_or_load(conf, id2label, len(train_loader), device)

process_test_loader(model, val_loader, "output/confidence_entropy", device="cpu")


summarize_confidence_entropy("output/confidence_entropy")


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
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="mean_confidence", y="FP", s=60)
    r, p = pearsonr(df["mean_confidence"], df["FP"])
    plt.title(f"Confidence vs FP\nPearson r = {r:.2f}, p = {p:.3f}")
    plt.xlabel("Mean Confidence per Image")
    plt.ylabel("False Positives")
    plt.tight_layout()
    plt.savefig(confidence_dir / "confidence_vs_fp.png", dpi=300)
    print(f"Saved correlation plot to {confidence_dir/'confidence_vs_fp.png'}")

# Example usage:
# correlate_confidence_fp("output/per_tree_summary.csv", "output/confidence_entropy")

'''

Remember to add repo path

export PYTHONPATH=/path/to/TreeSeg:$PYTHONPATH

'''