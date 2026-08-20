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
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings("ignore")

LAYERS = ["layer1", "layer2", "layer3", "layer4"]
LAYER_LABELS = ["L1", "L2", "L3", "L4"]

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------

MODEL_PALETTE = {
    "Baseline":      "#1565c0",   # deep blue
    "Fine-tuned":    "#c62828",   # deep red
    "Feature-KD":    "#2e7d32",   # forest green
}
PAIR_PALETTE = {
    "B vs. F":  "#1565c0",
    "B vs. KD": "#2e7d32",
    "F vs. KD": "#e65100",
}
MODEL_MARKERS = {
    "Baseline":   "o",
    "Fine-tuned": "s",
    "Feature-KD": "D",
}
PAIR_MARKERS = {
    "B vs. F":  "o",
    "B vs. KD": "D",
    "F vs. KD": "s",
}
PAIR_LS = {
    "B vs. F":  "-",
    "B vs. KD": "--",
    "F vs. KD": ":",
}

DATASET_COLORS = {
    "Poland":  "#6a1b9a",
    "Finland": "#00796b",
}

# Display names for legend / labels
DISPLAY_NAMES = {
    "Baseline":      "Baseline (teacher)",
    "Fine-tuned":    "Fine-tuned",
    "Feature-KD":    "Feature-level KD",
    "Baseline (Finland)":  "Baseline",
    "Fine-tuned (Poland)": "Fine-tuned",
    "Feature-KD (Poland)": "Feature-level KD",
}


def setup_style():
    plt.rcParams.update({
        "font.family":           "DejaVu Sans",
        "font.size":             11,
        "axes.titlesize":        12,
        "axes.titleweight":      "bold",
        "axes.labelsize":        10,
        "xtick.labelsize":       10,
        "ytick.labelsize":       10,
        "legend.fontsize":       9,
        "legend.framealpha":     0.9,
        "legend.edgecolor":      "#cccccc",
        "legend.borderpad":      0.5,
        "figure.facecolor":      "white",
        "axes.facecolor":        "#fafafa",
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.spines.left":      True,
        "axes.spines.bottom":    True,
        "axes.edgecolor":        "#555555",
        "axes.grid":             True,
        "grid.alpha":            0.35,
        "grid.linestyle":        "--",
        "grid.linewidth":        0.6,
        "grid.color":            "#bbbbbb",
        "lines.linewidth":       2.2,
        "lines.markersize":      8,
        "lines.markeredgewidth": 1.2,
        "lines.markeredgecolor": "white",
        "xtick.direction":       "out",
        "ytick.direction":       "out",
    })


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_features(feature_dir, layer, spatial=False, max_batches=None):
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
    A = torch.nn.functional.normalize(A, dim=1)
    B = torch.nn.functional.normalize(B, dim=1)
    return (A * B).sum(dim=1).mean().item()


def mean_spatial_ssim(feat_a, feat_b):
    a = feat_a.mean(dim=(0, 1)).numpy()
    b = feat_b.mean(dim=(0, 1)).numpy()
    data_range = max(a.max() - a.min(), b.max() - b.min(), 1e-6)
    score, _ = ssim(a, b, data_range=data_range, full=True)
    return float(score)


def tsne_cluster_metrics(feat_a, feat_b, n_samples=500):
    n = min(n_samples, len(feat_a), len(feat_b))
    rng = torch.Generator().manual_seed(42)
    idx_a = torch.randperm(len(feat_a), generator=rng)[:n]
    idx_b = torch.randperm(len(feat_b), generator=rng)[:n]
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
        "centroid_a": centroid_a,
        "centroid_b": centroid_b,
        "points_a": ea,
        "points_b": eb,
    }


def confidence_ellipse(points, ax, n_std=1.5, **kwargs):
    """Draw a covariance ellipse around a 2-D point cloud."""
    cov = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width  = 2 * n_std * np.sqrt(max(eigenvalues[0], 1e-9))
    height = 2 * n_std * np.sqrt(max(eigenvalues[1], 1e-9))
    ellipse = Ellipse(xy=points.mean(axis=0), width=width, height=height,
                      angle=angle, **kwargs)
    return ax.add_patch(ellipse)


def annotate_endpoint(ax, x_idx, y_val, color, offset=(6, 0)):
    """Add a right-aligned value label at the last data point."""
    ax.annotate(
        f"{y_val:.3f}",
        xy=(x_idx, y_val),
        xytext=(offset[0], offset[1]),
        textcoords="offset points",
        fontsize=8,
        color=color,
        va="center",
        fontweight="bold",
    )


# ---------------------------------------------------------------------------
# Figure 3: Inter-model representational similarity
# ---------------------------------------------------------------------------

def figure_inter_model(feature_root, out_path):
    configs = [
        ("Poland",  "features_poland_baseline",  "features_poland_transfer",  "features_poland_feature"),
        ("Finland", "features_finland_baseline", "features_finland_transfer", "features_finland_feature"),
    ]
    pairs_def = [
        ("B vs. F",  0, 1),
        ("B vs. KD", 0, 2),
        ("F vs. KD", 1, 2),
    ]
    metrics_fns = [
        ("Mean Cosine Similarity", mean_cosine_similarity),
        ("Linear CKA",             lambda a, b: linear_cka(a.float(), b.float())),
    ]
    y_ranges = [(0.45, 1.02), (0.0, 1.02)]

    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("white")

    print("\nInter-model Similarity Results:")
    print(f"{'Dataset':<10} {'Metric':<26} {'Pair':<10} {'L1':>6} {'L2':>6} {'L3':>6} {'L4':>6}")
    print("-" * 72)

    for row, (dataset, dir_base, dir_trans, dir_feat) in enumerate(configs):
        dirs = [dir_base, dir_trans, dir_feat]
        for col, (metric_name, metric_fn) in enumerate(metrics_fns):
            ax = axes[row, col]

            all_vals = {}
            for pair_label, i, j in pairs_def:
                vals = []
                for layer in LAYERS:
                    try:
                        fa = load_features(os.path.join(feature_root, dirs[i]), layer)
                        fb = load_features(os.path.join(feature_root, dirs[j]), layer)
                        n = min(len(fa), len(fb))
                        vals.append(metric_fn(fa[:n].float(), fb[:n].float()))
                    except FileNotFoundError:
                        vals.append(float("nan"))
                all_vals[pair_label] = vals
                print(f"  {dataset:<8} {metric_name:<26} {pair_label:<10} " +
                      " ".join(f"{v:>6.3f}" for v in vals))

            # Shade region between B vs. F and B vs. KD to show relative gap
            xs = list(range(len(LAYER_LABELS)))
            v_f  = all_vals["B vs. F"]
            v_kd = all_vals["B vs. KD"]
            ax.fill_between(xs, v_f, v_kd,
                            color="#aaaaaa", alpha=0.12, zorder=0)

            for pair_label, i, j in pairs_def:
                vals = all_vals[pair_label]
                col_c = PAIR_PALETTE[pair_label]
                ls    = PAIR_LS[pair_label]
                mk    = PAIR_MARKERS[pair_label]
                ax.plot(xs, vals,
                        color=col_c, linestyle=ls, marker=mk,
                        markersize=9, linewidth=2.2,
                        markeredgecolor="white", markeredgewidth=1.5,
                        label=pair_label, zorder=3)
                # Annotate L4 endpoint
                if not np.isnan(vals[-1]):
                    annotate_endpoint(ax, xs[-1], vals[-1], col_c, offset=(8, 0))

            # Dataset badge top-left
            if col == 0:
                ax.text(-0.42, 0.5, dataset,
                        transform=ax.transAxes,
                        fontsize=13, fontweight="bold",
                        color=DATASET_COLORS[dataset],
                        va="center", ha="center",
                        rotation=90)

            ax.set_xticks(xs)
            ax.set_xticklabels(LAYER_LABELS)
            ax.set_xlim(-0.4, len(xs) - 0.4)
            ax.set_ylim(*y_ranges[col])
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

            if row == 0:
                ax.set_title(metric_name, fontsize=12, fontweight="bold", pad=10)
            if col == 0:
                ax.set_ylabel("Similarity", fontsize=10)

            # Light annotation of "Low-level" vs "Semantic" regions
            if row == 1 and col == 1:
                ax.axvspan(-0.4, 1.4, color="#e3f2fd", alpha=0.25, zorder=0)
                ax.axvspan(1.6, 3.4, color="#fce4ec", alpha=0.18, zorder=0)
                ax.text(0.5, y_ranges[col][0] + 0.02, "low-level", fontsize=7.5,
                        ha="center", color="#1565c0", style="italic")
                ax.text(2.5, y_ranges[col][0] + 0.02, "semantic", fontsize=7.5,
                        ha="center", color="#c62828", style="italic")

    # Shared legend at bottom
    legend_elements = [
        Line2D([0], [0], color=PAIR_PALETTE[p], linestyle=PAIR_LS[p],
               marker=PAIR_MARKERS[p], markersize=8,
               markeredgecolor="white", markeredgewidth=1.2,
               linewidth=2.2, label=p)
        for p in ["B vs. F", "B vs. KD", "F vs. KD"]
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.02),
               title="Model pair  (B = Baseline · F = Fine-tuned · KD = Feature-level KD)",
               title_fontsize=9)

    fig.suptitle("Inter-Model Representational Similarity", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Intra-model domain invariance (Finland vs Poland)
# ---------------------------------------------------------------------------

def figure_intra_model(feature_root, out_path):
    models = [
        ("Baseline",         "features_finland_baseline", "features_poland_baseline"),
        ("Fine-tuned",       "features_finland_transfer", "features_poland_transfer"),
        ("Feature-KD",       "features_finland_feature",  "features_poland_feature"),
    ]
    display = ["Baseline (teacher)", "Fine-tuned", "Feature-level KD"]
    keys    = ["Baseline", "Fine-tuned", "Feature-KD"]
    ls_map  = {"Baseline": "-", "Fine-tuned": "--", "Feature-KD": "-."}

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    titles  = ["Cross-Domain Cosine Similarity\n(Finland → Poland)",
               "Cross-Domain Spatial SSIM\n(Finland → Poland)"]
    ylabels = ["Mean Cosine Similarity", "SSIM"]
    y_ranges = [(0.75, 1.02), (0.0, 1.02)]

    print("\nIntra-model Domain Invariance (Finland vs Poland):")
    print(f"{'Model':<20} {'Metric':<8} {'L1':>6} {'L2':>6} {'L3':>6} {'L4':>6}")
    print("-" * 52)

    xs = list(range(len(LAYER_LABELS)))

    for mi, (key, disp, (_, fin_dir, pol_dir)) in enumerate(zip(keys, display, models)):
        cos_vals, ssim_vals = [], []
        color = MODEL_PALETTE[key]
        ls    = ls_map[key]
        mk    = MODEL_MARKERS[key]

        for layer in LAYERS:
            try:
                f_fin = load_features(os.path.join(feature_root, fin_dir), layer)
                f_pol = load_features(os.path.join(feature_root, pol_dir), layer)
                n = min(len(f_fin), len(f_pol))
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

        print(f"  {key:<18} {'Cosine':<8} " + " ".join(f"{v:>6.3f}" for v in cos_vals))
        print(f"  {key:<18} {'SSIM':<8} "   + " ".join(f"{v:>6.3f}" for v in ssim_vals))

        for ax, vals in zip(axes, [cos_vals, ssim_vals]):
            ax.plot(xs, vals,
                    color=color, linestyle=ls, marker=mk,
                    markersize=9, linewidth=2.2,
                    markeredgecolor="white", markeredgewidth=1.5,
                    label=disp, zorder=3)
            if not np.isnan(vals[-1]):
                annotate_endpoint(ax, xs[-1], vals[-1], color, offset=(8, 0))

    # Shade the "gap" between Feature-KD SSIM and Fine-tuned SSIM in L4
    # (already plotted, just add region annotation for the big SSIM L4 drop)
    axes[1].annotate("sharp drop\n(Feature-KD L4)",
                      xy=(3, 0.414), xytext=(1.7, 0.25),
                      arrowprops=dict(arrowstyle="->", color="#2e7d32",
                                      connectionstyle="arc3,rad=-0.25", lw=1.3),
                      fontsize=8.5, color="#2e7d32", ha="center")

    for ax, title, ylabel, yr in zip(axes, titles, ylabels, y_ranges):
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(LAYER_LABELS)
        ax.set_xlim(-0.4, len(xs) - 0.4)
        ax.set_ylim(*yr)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    legend_elements = [
        Line2D([0], [0], color=MODEL_PALETTE[k], linestyle=ls_map[k],
               marker=MODEL_MARKERS[k], markersize=8,
               markeredgecolor="white", markeredgewidth=1.2,
               linewidth=2.2, label=d)
        for k, d in zip(keys, display)
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Intra-Model Domain Invariance  (Finland → Poland)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0.0, 0.06, 1.0, 1.0])
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: t-SNE scatterplots
# ---------------------------------------------------------------------------

MODEL_COLORS_TSNE = {
    "Baseline":   "#1565c0",
    "Fine-tuned": "#c62828",
    "Feature-KD": "#2e7d32",
}

def figure_tsne(feature_root, out_dir):
    panel_configs = [
        ("features_poland_baseline",  "Baseline",    "features_poland_transfer", "Fine-tuned",
         "tsne_poland_baseline_vs_poland_transfer.png", "Poland"),
        ("features_poland_baseline",  "Baseline",    "features_poland_feature",  "Feature-KD",
         "tsne_poland_baseline_vs_poland_feature.png",  "Poland"),
        ("features_poland_transfer",  "Fine-tuned",  "features_poland_feature",  "Feature-KD",
         "tsne_poland_transfer_vs_poland_feature.png",  "Poland"),
        ("features_finland_baseline", "Baseline",    "features_finland_transfer","Fine-tuned",
         "tsne_finland_baseline_vs_finland_transfer.png", "Finland"),
        ("features_finland_baseline", "Baseline",    "features_finland_feature", "Feature-KD",
         "tsne_finland_baseline_vs_finland_feature.png",  "Finland"),
    ]

    print("\nt-SNE Cluster Metrics:")
    print(f"{'Panel':<50} {'Sil':>6} {'CentDist':>10} {'CompA':>8} {'CompB':>8} {'Overlap':>9}")
    print("-" * 95)

    setup_style()

    for dir_a, lbl_a, dir_b, lbl_b, fname, domain in panel_configs:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f8f9fa")

        col_a = MODEL_COLORS_TSNE[lbl_a]
        col_b = MODEL_COLORS_TSNE[lbl_b]

        try:
            fa = load_features(os.path.join(feature_root, dir_a), "layer4")
            fb = load_features(os.path.join(feature_root, dir_b), "layer4")
            metrics = tsne_cluster_metrics(fa.float(), fb.float(), n_samples=500)

            ea = metrics["points_a"]
            eb = metrics["points_b"]
            ca = metrics["centroid_a"]
            cb = metrics["centroid_b"]

            # Scatter points
            ax.scatter(ea[:, 0], ea[:, 1], c=col_a, alpha=0.30, s=12,
                       linewidths=0, zorder=2)
            ax.scatter(eb[:, 0], eb[:, 1], c=col_b, alpha=0.30, s=12,
                       linewidths=0, zorder=2)

            # Confidence ellipses (1.5-sigma)
            confidence_ellipse(ea, ax, n_std=1.5,
                               facecolor=col_a, alpha=0.13,
                               edgecolor=col_a, linewidth=1.8,
                               linestyle="--", zorder=3)
            confidence_ellipse(eb, ax, n_std=1.5,
                               facecolor=col_b, alpha=0.13,
                               edgecolor=col_b, linewidth=1.8,
                               linestyle="--", zorder=3)

            # Centroid markers
            ax.scatter(*ca, c=col_a, s=160, marker="*",
                       edgecolors="white", linewidths=1.2, zorder=5)
            ax.scatter(*cb, c=col_b, s=160, marker="*",
                       edgecolors="white", linewidths=1.2, zorder=5)

            # Arrow between centroids
            dx, dy = cb - ca
            ax.annotate("", xy=cb, xytext=ca,
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color="#555555", lw=1.5,
                            mutation_scale=14,
                            connectionstyle="arc3,rad=0.15",
                        ), zorder=4)

            # Centroid distance label along the arrow midpoint
            mid = (ca + cb) / 2
            ax.text(mid[0], mid[1], f"d={metrics['centroid_distance']:.1f}",
                    fontsize=8, ha="center", va="bottom",
                    color="#333333",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#cccccc", alpha=0.85))

            # Stats box
            stats_text = (
                f"Silhouette = {metrics['silhouette']:.3f}\n"
                f"Overlap     = {metrics['overlap']:.3f}"
            )
            ax.text(0.03, 0.97, stats_text,
                    transform=ax.transAxes,
                    fontsize=8.5, va="top", ha="left",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="#cccccc", alpha=0.90))

            panel_label = fname.replace(".png", "").replace("tsne_", "")
            print(f"{panel_label:<50} {metrics['silhouette']:>6.3f} "
                  f"{metrics['centroid_distance']:>10.3f} "
                  f"{metrics['compactness_a']:>8.3f} "
                  f"{metrics['compactness_b']:>8.3f} "
                  f"{metrics['overlap']:>9.3f}")

        except FileNotFoundError as e:
            ax.text(0.5, 0.5, "features missing", ha="center", va="center",
                    transform=ax.transAxes, color="#aaaaaa")
            print(f"  SKIPPED {fname}: {e}")

        # Legend
        legend_handles = [
            mpatches.Patch(facecolor=col_a, alpha=0.7, label=lbl_a),
            mpatches.Patch(facecolor=col_b, alpha=0.7, label=lbl_b),
        ]
        ax.legend(handles=legend_handles, fontsize=9.5, loc="lower right",
                  framealpha=0.9, edgecolor="#cccccc")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{lbl_a} vs. {lbl_b}   [{domain} · Layer 4]",
                     fontsize=11, fontweight="bold", pad=10)
        ax.text(0.5, -0.03, "t-SNE projection of encoder representations",
                transform=ax.transAxes, fontsize=8, ha="center",
                color="#888888", style="italic")

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#cccccc")

        plt.tight_layout()
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 6: Linear probing
# ---------------------------------------------------------------------------

def load_labels(feature_dir, max_batches=None):
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
        ("Baseline (Finland)",  "features_finland_baseline", "Baseline"),
        ("Fine-tuned (Poland)", "features_poland_transfer",  "Fine-tuned"),
        ("Feature-KD (Poland)", "features_poland_feature",   "Feature-KD"),
    ]
    metric_names   = ["Accuracy", "F1 Score", "AUC-ROC"]
    metric_keys    = ["acc", "f1", "auc"]
    ls_map = {"Baseline": "-", "Fine-tuned": "--", "Feature-KD": "-."}

    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.patch.set_facecolor("white")

    xs = list(range(len(LAYER_LABELS)))

    print(f"\nLinear Probing Results:")
    print(f"{'Model':<25} {'Layer':<8} {'Acc':>6} {'F1':>6} {'AUC':>6}")
    print("-" * 55)

    all_results = {}
    for (model_label, feat_dir, key) in model_configs:
        color  = MODEL_PALETTE[key]
        ls     = ls_map[key]
        mk     = MODEL_MARKERS[key]
        labels = load_labels(os.path.join(feature_root, feat_dir))
        if labels is None:
            print(f"  No labels found in {feat_dir} — skipping {model_label}")
            continue

        accs, f1s, aucs = [], [], []
        for layer, ll in zip(LAYERS, LAYER_LABELS):
            try:
                feats = load_features(os.path.join(feature_root, feat_dir), layer).numpy()
                n = min(len(feats), len(labels))
                acc, f1, auc = linear_probe_layer(feats[:n], labels[:n])
            except FileNotFoundError:
                acc, f1, auc = float("nan"), float("nan"), float("nan")
            accs.append(acc); f1s.append(f1); aucs.append(auc)
            print(f"  {model_label:<23} {ll:<8} {acc:>6.3f} {f1:>6.3f} {auc:>6.3f}")

        all_results[key] = {"acc": accs, "f1": f1s, "auc": aucs,
                            "color": color, "ls": ls, "mk": mk, "label": model_label}

    # Shade region between Fine-tuned and Feature-KD to highlight the gap
    if "Fine-tuned" in all_results and "Feature-KD" in all_results:
        for ax, mk_key in zip(axes, metric_keys):
            v_ft = all_results["Fine-tuned"][mk_key]
            v_kd = all_results["Feature-KD"][mk_key]
            if not any(np.isnan(v_ft)) and not any(np.isnan(v_kd)):
                ax.fill_between(xs, v_ft, v_kd,
                                color="#2e7d32", alpha=0.10, zorder=0,
                                label="_nolegend_")

    # Plot all lines
    for key, res in all_results.items():
        for ax, mk_key in zip(axes, metric_keys):
            vals = res[mk_key]
            ax.plot(xs, vals,
                    color=res["color"], linestyle=res["ls"], marker=res["mk"],
                    markersize=9, linewidth=2.2,
                    markeredgecolor="white", markeredgewidth=1.5,
                    label=DISPLAY_NAMES.get(res["label"], res["label"]),
                    zorder=3)
            if not np.isnan(vals[-1]):
                annotate_endpoint(ax, xs[-1], vals[-1], res["color"], offset=(8, 0))

    for ax, title in zip(axes, metric_names):
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(LAYER_LABELS)
        ax.set_xlim(-0.4, len(xs) - 0.4)
        ax.set_ylim(0.65, 1.02)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Single legend on the AUC panel
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Linear Probing — Dead Tree Presence Separability by Layer",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0.0, 0.06, 1.0, 1.0])
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Regenerate representational analysis figures.")
    parser.add_argument("--feature-root",  required=True,
                        help="Directory containing features_*/ subdirectories")
    parser.add_argument("--report-images", default="report/images",
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
