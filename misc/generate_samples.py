"""
Generate Figure 2 sample prediction comparison figures.

Loads 3 checkpoints, runs inference on Polish test patches, selects
patches with high tree density and inter-model disagreement, and saves
4 five-column comparison figures.

Columns: NIR-R-G composite | GT | Baseline | Fine-tuned | Feature-level KD

Usage on LUMI (via submit_generate_samples.sh), or locally:
  python3 misc/generate_samples.py \\
    --config        configs/model/flair_unet_highrecall.txt \\
    --data-config   configs/data/poland.txt \\
    --baseline-ckpt output/flair_unet/high_recall/best.weights.pth \\
    --ft-ckpt       output/flair_unet/Poland_RGBNIR_25cm/best.weights.pth \\
    --feature-ckpt  output/flair_unet/Poland_RGBNIR_25cm/best.weights.feature.pth \\
    --output-dir    report/images \\
    --n-samples     4
"""

import os
import sys
import argparse
import random

import h5py
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from treemort.modeling.model_config import configure_model
from treemort.utils.config import setup
from treemort.utils.datautils import load_and_organize_data, stratify_images_by_region

# Semi-transparent fill RGBA + contour hex per column
OVERLAY = {
    "gt":      ((1.00, 0.10, 0.10, 0.40), "#ff2222"),
    "base":    ((0.10, 0.35, 0.90, 0.40), "#1a5ae8"),
    "ft":      ((0.10, 0.72, 0.20, 0.40), "#19b833"),
    "feature": ((0.62, 0.08, 0.85, 0.40), "#9e14d9"),
}

COL_TITLES = [
    "NIR-R-G Composite",
    "Ground Truth",
    "Baseline",
    "Fine-tuned",
    "Feature-level KD",
]
COL_COLORS = ["#222222", "#ff2222", "#1a5ae8", "#19b833", "#9e14d9"]


def load_model(conf, checkpoint_path, device):
    id2label = {0: "alive", 1: "dead"}
    model = configure_model(conf, id2label)
    model.to(device)
    model.eval()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] Missing keys in {os.path.basename(checkpoint_path)}: {missing[:3]}...")
    print(f"  Loaded: {checkpoint_path}")
    return model


def predict_mask(model, image_numpy_hwc, device, threshold=0.5):
    """Return binary [H, W] uint8 mask.  image_numpy_hwc is float32 0-1."""
    img_t = torch.from_numpy(image_numpy_hwc).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t)
        logits = out[0] if isinstance(out, tuple) else out
        prob = torch.sigmoid(logits[0, 0]).cpu().numpy()
    return (prob > threshold).astype(np.uint8)


def cir_composite(image_hwc):
    """CIR false-colour from HDF5 channel order [NIR, R, G, B] → display [NIR, R, G]."""
    nir, r, g = image_hwc[:, :, 0], image_hwc[:, :, 1], image_hwc[:, :, 2]
    cir = np.stack([nir, r, g], axis=-1)
    lo, hi = np.percentile(cir, 1), np.percentile(cir, 99)
    return np.clip((cir - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def draw_mask_overlay(ax, background, mask, fill_rgba, contour_hex, lw=0.8):
    ax.imshow(background)
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[mask > 0] = fill_rgba
    ax.imshow(overlay, interpolation="nearest")
    for c in measure.find_contours(mask.astype(float), 0.5):
        ax.plot(c[:, 1], c[:, 0], color=contour_hex, linewidth=lw, antialiased=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_figure(cir, gt, pred_base, pred_ft, pred_feat, out_path):
    masks = [gt, pred_base, pred_ft, pred_feat]
    keys = ["gt", "base", "ft", "feature"]

    fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.0))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.0, right=1.0, top=0.84, bottom=0.0, wspace=0.025)

    axes[0].imshow(cir)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for spine in axes[0].spines.values():
        spine.set_visible(False)

    for ax, mask, key in zip(axes[1:], masks, keys):
        fill, cont = OVERLAY[key]
        draw_mask_overlay(ax, cir, mask, fill, cont)

    for ax, title, color in zip(axes, COL_TITLES, COL_COLORS):
        ax.set_title(title, fontsize=7.5, fontweight="bold", color=color, pad=3)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        required=True, help="Model config (flair_unet_highrecall.txt)")
    parser.add_argument("--data-config",   required=True, help="Data config (poland.txt)")
    parser.add_argument("--baseline-ckpt", required=True, help="Finnish teacher checkpoint")
    parser.add_argument("--ft-ckpt",       required=True, help="Fine-tuned student checkpoint")
    parser.add_argument("--feature-ckpt",  required=True, help="Feature-KD student checkpoint")
    parser.add_argument("--output-dir",    default="report/images")
    parser.add_argument("--n-samples",     type=int,   default=4)
    parser.add_argument("--n-candidates",  type=int,   default=100)
    parser.add_argument("--min-trees",     type=int,   default=8,
                        help="Minimum num_trees attribute for a patch to be considered")
    parser.add_argument("--threshold",     type=float, default=0.5)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Parsing config...")
    conf = setup(args.config, data_config=args.data_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    print("[INFO] Loading models...")
    model_base = load_model(conf, args.baseline_ckpt, device)
    model_ft   = load_model(conf, args.ft_ckpt,       device)
    model_feat = load_model(conf, args.feature_ckpt,  device)

    hdf5_path = Path(conf.data_folder).parent / conf.hdf5_file
    print(f"[INFO] HDF5: {hdf5_path}")

    image_patch_map = load_and_organize_data(hdf5_path)
    _, _, test_keys = stratify_images_by_region(
        image_patch_map,
        val_ratio=conf.val_size,
        test_ratio=conf.test_size,
    )
    print(f"[INFO] Test patches total: {len(test_keys)}")

    with h5py.File(hdf5_path, "r") as hf:
        rich_keys = sorted(
            [k for k in test_keys if hf[k].attrs.get("num_trees", 0) >= args.min_trees],
            key=lambda k: -hf[k].attrs.get("num_trees", 0),
        )
    print(f"[INFO] Patches with >= {args.min_trees} trees: {len(rich_keys)}")

    # Take top-N by tree count, shuffle within top-2× candidates to allow seed control
    pool = rich_keys[:args.n_candidates * 2]
    random.shuffle(pool)
    candidate_keys = pool[:args.n_candidates]
    print(f"[INFO] Running inference on {len(candidate_keys)} candidates...")

    scored = []
    with h5py.File(hdf5_path, "r") as hf:
        for i, key in enumerate(candidate_keys):
            grp   = hf[key]
            image = grp["image"][()].astype(np.float32) / 255.0   # HxWxC, NIRGB
            gt    = grp["labels"]["mask"][()].astype(np.uint8)

            pred_base = predict_mask(model_base, image, device, args.threshold)
            pred_ft   = predict_mask(model_ft,   image, device, args.threshold)
            pred_feat = predict_mask(model_feat, image, device, args.threshold)

            gt_px = int(gt.sum())
            # Inter-model disagreement as selection signal
            disagree = (
                int(np.sum(pred_base != pred_ft))
                + int(np.sum(pred_base != pred_feat))
                + int(np.sum(pred_ft   != pred_feat))
            )
            scored.append((key, image, gt, pred_base, pred_ft, pred_feat, gt_px, disagree))

            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(candidate_keys)}")

    # Primary sort: num GT tree pixels; secondary: inter-model disagreement
    scored.sort(key=lambda x: (-x[6], -x[7]))
    top_pool = scored[:args.n_samples * 4]
    top_pool.sort(key=lambda x: -x[7])
    selected = top_pool[:args.n_samples]

    print(f"[INFO] Saving figures to {args.output_dir} ...")
    for i, (key, image, gt, pb, pf, pk, gt_px, dis) in enumerate(selected):
        cir = cir_composite(image)
        out = os.path.join(args.output_dir, f"s{i + 1}.png")
        save_figure(cir, gt, pb, pf, pk, out)
        print(f"  s{i + 1}.png  patch={key}  gt_px={gt_px}  disagree={dis}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
