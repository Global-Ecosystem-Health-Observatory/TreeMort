"""
Regenerate representational analysis figures (Figures 3-6) from saved features.

Expects features saved by scripts/extract_features.py in directories:
  feature_root/
    features_poland_baseline/   features_{layer}_batch*.pt
    features_poland_transfer/
    features_poland_feature/
    features_finland_baseline/
    features_finland_transfer/
    features_finland_feature/

Usage:
  python3 misc/representational_analysis.py \
    --feature-root /scratch/.../output \
    --report-images report/images
"""

import os
import sys
import argparse
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings("ignore")

LAYERS = ["layer1", "layer2", "layer3", "layer4"]
LAYER_LABELS = ["Layer 1", "Layer 2", "Layer 3", "Layer 4"]


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_features(feature_dir, layer, spatial=False, max_batches=None):
    """Load and concatenate per-batch feature files. Returns (N, C) or (N, C, H, W)."""
    key = "spatial_" if spatial else ""
    files = sorted(Path(feature_dir).glob(f"features_{layer}_{key}batch*.pt"))
    if not files:
        raise FileNotFoundError(f"No features found in {feature_dir} for layer={layer}")
    chunks = []
    for i, f in enumerate(files):
        if max_batches and i >= max_batches:
            break
        chunks.append(torch.load(f, weights_only=True))
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    K, L = X @ X.T, Y @ Y.T
    Kc = K - K.mean(0) - K.mean(1, keepdim=True) + K.mean()
    Lc = L - L.mean(0) - L.mean(1, keepdim=True) + L.mean()
    return ((Kc * Lc).sum() / (torch.norm(Kc) * torch.norm(Lc))).item()


def mean_cosine_similarity(A, B):
    """Mean cosine similarity between corresponding rows of A and B."""
    A = torch.nn.functional.normalize(A, dim=1)
    B = torch.nn.functional.normalize(B, dim=1)
    return (A * B).sum(dim=1).mean().item()


def mean_spatial_ssim(feat_a, feat_b):
    """SSIM between mean feature maps (H x W averaged over channels)."""
    a = feat_a.mean(dim=(0, 1)).numpy()   # H x W
    b = feat_b.mean(dim=(0, 1)).numpy()
    data_range = max(a.max() - a.min(), b.max() - b.min(), 1e-6)
    score, _ = ssim(a, b, data_range=data_range, full=True)
    return float(score)


def tsne_cluster_metrics(feat_a, feat_b, n_samples=500):
    """Compute t-SNE cluster metrics for two feature sets."""
    n = min(n_samples, len(feat_a), len(feat_b))
    idx_a = torch.randperm(len(feat_a))[:n]
    idx_b = torch.randperm(len(feat_b))[:n]
    fa = feat_a[idx_a].numpy()
    fb = feat_b[idx_b].numpy()

    combined = np.concatenate([fa, fb], axis=0)
    labels = np.array([0] * n + [1] * n)

    perplexity = min(30, n - 1)
    emb = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(combined)

    ea, eb = emb[:n], emb[n:]
    centroid_a, centroid_b = ea.mean(axis=0), eb.mean(axis=0)

    sil = silhouette_score(emb, labels)
    centroid_dist = np.linalg.norm(centroid_a - centroid_b)
    compactness_a = np.mean(np.linalg.norm(ea - centroid_a, axis=1))
    compactness_b = np.mean(np.linalg.norm(eb - centroid_b, axis=1))

    # Overlap: fraction of points from A closer to centroid_b than centroid_a
    d_a_to_a = np.linalg.norm(ea - centroid_a, axis=1)
    d_a_to_b = np.linalg.norm(ea - centroid_b, axis=1)
    d_b_to_a = np.linalg.norm(eb - centroid_a, axis=1)
    d_b_to_b = np.linalg.norm(eb - centroid_b, axis=1)
    overlap = (((d_a_to_b < d_a_to_a).sum() + (d_b_to_a < d_b_to_b).sum()) / (2 * n))

    return {
        "silhouette": sil,
        "centroid_distance": centroid_dist,
        "compactness_a": compactness_a,
        "compactness_b": compactness_b,
        "overlap": overlap,
        "embedding": emb,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Figure 3: Inter-model representational similarity
# ---------------------------------------------------------------------------

def figure_inter_model(feature_root, out_path):
    configs = [
        ("Poland", "features_poland_baseline",  "features_poland_transfer",
                   "features_poland_feature"),
        ("Finland", "features_finland_baseline", "features_finland_transfer",
                    "features_finland_feature"),
    ]
    pair_labels = ["B vs. F", "B vs. KD", "F vs. KD"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metrics_fns = [("Mean Cosine", mean_cosine_similarity),
                   ("Linear CKA",  lambda a, b: linear_cka(a.float(), b.float()))]

    for row, (dataset, dir_base, dir_trans, dir_feat) in enumerate(configs):
        for col, (metric_name, metric_fn) in enumerate(metrics_fns):
            ax = axes[row, col]
            pairs = [
                (dir_base, dir_trans, "B vs. F"),
                (dir_base, dir_feat,  "B vs. KD"),
                (dir_trans, dir_feat, "F vs. KD"),
            ]
            for label, (da, db) in [(p[2], (p[0], p[1])) for p in pairs]:
                vals = []
                for layer in LAYERS:
                    try:
                        fa = load_features(os.path.join(feature_root, da), layer)
                        fb = load_features(os.path.join(feature_root, db), layer)
                        n = min(len(fa), len(fb))
                        vals.append(metric_fn(fa[:n].float(), fb[:n].float()))
                    except FileNotFoundError:
                        vals.append(float("nan"))
                ax.plot(LAYER_LABELS, vals, marker="o", label=label)

            ax.set_title(f"{dataset} — {metric_name}")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(metric_name)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Intra-model domain invariance (Finland vs Poland)
# ---------------------------------------------------------------------------

def figure_intra_model(feature_root, out_path):
    models = [
        ("Baseline",      "features_finland_baseline", "features_poland_baseline"),
        ("Fine-tuned",    "features_finland_transfer", "features_poland_transfer"),
        ("Feature-level KD", "features_finland_feature", "features_poland_feature"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for model_label, fin_dir, pol_dir in models:
        cos_vals, ssim_vals = [], []
        for layer in LAYERS:
            try:
                f_fin = load_features(os.path.join(feature_root, fin_dir), layer)
                f_pol = load_features(os.path.join(feature_root, pol_dir), layer)
                n = min(len(f_fin), len(f_pol))
                # cosine between mean feature vectors across domains
                m_fin = f_fin[:n].float().mean(dim=0, keepdim=True)
                m_pol = f_pol[:n].float().mean(dim=0, keepdim=True)
                cos_vals.append(mean_cosine_similarity(m_fin, m_pol))
            except FileNotFoundError:
                cos_vals.append(float("nan"))

            try:
                fs_fin = load_features(os.path.join(feature_root, fin_dir), layer, spatial=True)
                fs_pol = load_features(os.path.join(feature_root, pol_dir), layer, spatial=True)
                ssim_vals.append(mean_spatial_ssim(fs_fin.float(), fs_pol.float()))
            except FileNotFoundError:
                ssim_vals.append(float("nan"))

        axes[0].plot(LAYER_LABELS, cos_vals, marker="o", label=model_label)
        axes[1].plot(LAYER_LABELS, ssim_vals, marker="o", label=model_label)

    for ax, title, ylabel in zip(
        axes,
        ["Intra-model Cosine Similarity (Finland vs Poland)",
         "Intra-model SSIM (Poland)"],
        ["Mean Cosine Similarity", "SSIM"],
    ):
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: t-SNE scatterplots
# ---------------------------------------------------------------------------

def figure_tsne(feature_root, out_dir):
    panel_configs = [
        # (file_a, label_a, file_b, label_b, filename)
        ("features_poland_baseline",  "Baseline",    "features_poland_transfer", "Fine-tuned",
         "tsne_poland_baseline_vs_poland_transfer.png"),
        ("features_poland_baseline",  "Baseline",    "features_poland_feature",  "Feature-KD",
         "tsne_poland_baseline_vs_poland_feature.png"),
        ("features_poland_transfer",  "Fine-tuned",  "features_poland_feature",  "Feature-KD",
         "tsne_poland_transfer_vs_poland_feature.png"),
        ("features_finland_baseline", "Baseline",    "features_finland_transfer","Fine-tuned",
         "tsne_finland_baseline_vs_finland_transfer.png"),
        ("features_finland_baseline", "Baseline",    "features_finland_feature", "Feature-KD",
         "tsne_finland_baseline_vs_finland_feature.png"),
    ]
    colors = ["#2166ac", "#d6604d"]

    print("\nt-SNE Cluster Metrics:")
    print(f"{'Panel':<50} {'Sil':>6} {'CentDist':>10} {'CompA':>8} {'CompB':>8} {'Overlap':>9}")
    print("-" * 95)

    for dir_a, lbl_a, dir_b, lbl_b, fname in panel_configs:
        fig, ax = plt.subplots(figsize=(5, 5))
        try:
            fa = load_features(os.path.join(feature_root, dir_a), "layer4")
            fb = load_features(os.path.join(feature_root, dir_b), "layer4")
            metrics = tsne_cluster_metrics(fa.float(), fb.float(), n_samples=500)

            emb, labels = metrics["embedding"], metrics["labels"]
            for i, (lbl, col) in enumerate(zip([lbl_a, lbl_b], colors)):
                mask = labels == i
                ax.scatter(emb[mask, 0], emb[mask, 1], c=col, alpha=0.5, s=8, label=lbl)
            ax.legend(fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{lbl_a} vs. {lbl_b}", fontsize=10)

            panel_label = fname.replace(".png", "").replace("tsne_", "")
            print(f"{panel_label:<50} {metrics['silhouette']:>6.3f} "
                  f"{metrics['centroid_distance']:>10.3f} "
                  f"{metrics['compactness_a']:>8.3f} "
                  f"{metrics['compactness_b']:>8.3f} "
                  f"{metrics['overlap']:>9.3f}")
        except FileNotFoundError as e:
            ax.text(0.5, 0.5, "features missing", ha="center", va="center", transform=ax.transAxes)
            print(f"  SKIPPED {fname}: {e}")

        out_path = os.path.join(out_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 6: Linear probing
# ---------------------------------------------------------------------------

def load_labels(feature_dir, max_batches=None):
    """Load binary patch labels (dead tree present) saved alongside features."""
    label_files = sorted(Path(feature_dir).glob("labels_batch*.pt"))
    if not label_files:
        return None
    chunks = []
    for i, f in enumerate(label_files):
        if max_batches and i >= max_batches:
            break
        chunks.append(torch.load(f, weights_only=True))
    return torch.cat(chunks, dim=0).numpy()


def linear_probe_layer(features, labels):
    """Train/eval logistic regression on frozen features. Returns (acc, f1, auc)."""
    n = len(features)
    split = int(0.8 * n)
    X_tr, X_te = features[:split], features[split:]
    y_tr, y_te = labels[:split], labels[split:]

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return float("nan"), float("nan"), float("nan")

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, zero_division=0)
    auc = roc_auc_score(y_te, y_prob)
    return acc, f1, auc


def figure_linear_probing(feature_root, out_path):
    model_configs = [
        ("Baseline (Finland)",  "features_finland_baseline"),
        ("Fine-tuned (Poland)", "features_poland_transfer"),
        ("Feature-KD (Poland)", "features_poland_feature"),
    ]
    metrics_names = ["acc", "f1", "auc"]
    colors = ["#4393c3", "#d6604d", "#74c476"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for (model_label, feat_dir), color in zip(model_configs, colors):
        labels = load_labels(os.path.join(feature_root, feat_dir))
        if labels is None:
            print(f"  No labels found in {feat_dir} — skipping linear probing for {model_label}")
            continue

        accs, f1s, aucs = [], [], []
        for layer in LAYERS:
            try:
                feats = load_features(os.path.join(feature_root, feat_dir), layer).numpy()
                n = min(len(feats), len(labels))
                acc, f1, auc = linear_probe_layer(feats[:n], labels[:n])
            except FileNotFoundError:
                acc, f1, auc = float("nan"), float("nan"), float("nan")
            accs.append(acc); f1s.append(f1); aucs.append(auc)

        for ax, vals, mname in zip(axes, [accs, f1s, aucs], metrics_names):
            ax.plot(LAYER_LABELS, vals, marker="o", label=model_label, color=color)

    for ax, mname in zip(axes, metrics_names):
        ax.set_title(mname.upper())
        ax.set_ylabel(mname)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Linear Probing — Dead Tree Presence", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Regenerate representational analysis figures.")
    parser.add_argument("--feature-root",   required=True,
                        help="Directory containing features_*/ subdirectories")
    parser.add_argument("--report-images",  default="report/images",
                        help="Output directory for figure PNGs")
    args = parser.parse_args()

    os.makedirs(args.report_images, exist_ok=True)

    print("=== Figure 3: Inter-model similarity ===")
    figure_inter_model(
        args.feature_root,
        os.path.join(args.report_images, "inter_model_similarity.png"),
    )

    print("\n=== Figure 4: Intra-model domain invariance ===")
    figure_intra_model(
        args.feature_root,
        os.path.join(args.report_images, "intra_model_similarity.png"),
    )

    print("\n=== Figure 5: t-SNE scatterplots ===")
    figure_tsne(args.feature_root, args.report_images)

    print("\n=== Figure 6: Linear probing ===")
    figure_linear_probing(
        args.feature_root,
        os.path.join(args.report_images, "linear_probe.png"),
    )

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
