"""
Generate Figure 2 sample prediction comparison figures.

Loads 3 checkpoints, runs inference on Polish test patches, selects
patches with high adaptation gain (baseline fails, KD/fine-tuned succeeds),
enforcing one patch per source tile for geographic diversity.

Columns: NIR-R-G + GT overlay | Baseline | Fine-tuned | KD

Usage on LUMI (via submit_generate_samples.sh), or locally:
  python3 misc/generate_samples.py \\
    --config        configs/model/flair_unet_highrecall.txt \\
    --data-config   configs/data/poland.txt \\
    --baseline-ckpt output/flair_unet/high_recall/best.weights.pth \\
    --ft-ckpt       output/flair_unet/Poland_RGBNIR_25cm/best.weights.pth \\
    --kd-ckpt       output/flair_unet/Poland_RGBNIR_25cm/best.weights.feature.pth \\
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

# Semi-transparent fill RGBA + contour hex per role
OVERLAY = {
    "gt":   ((1.00, 0.10, 0.10, 0.40), "#ff2222"),
    "base": ((0.10, 0.35, 0.90, 0.40), "#1a5ae8"),
    "ft":   ((0.10, 0.72, 0.20, 0.40), "#19b833"),
    "kd":   ((0.62, 0.08, 0.85, 0.40), "#9e14d9"),
}


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


def save_figure(cir, gt, pred_base, pred_ft, pred_kd, out_path):
    """Four columns: NIR+GT overlay | Baseline | Fine-tuned | KD."""
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.0))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.025)

    # Col 1: NIR composite with GT overlay
    draw_mask_overlay(axes[0], cir, gt, *OVERLAY["gt"])

    # Cols 2-4: model predictions on NIR composite background
    for ax, mask, key in zip(axes[1:], [pred_base, pred_ft, pred_kd], ["base", "ft", "kd"]):
        draw_mask_overlay(ax, cir, mask, *OVERLAY[key])

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def iou(pred, gt):
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        required=True)
    parser.add_argument("--data-config",   required=True)
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--ft-ckpt",       required=True)
    parser.add_argument("--kd-ckpt",       required=True,
                        help="KD checkpoint (e.g. best.weights.feature.pth or best.weights.self.pth)")
    parser.add_argument("--output-dir",    default="report/images")
    parser.add_argument("--n-samples",     type=int,   default=4)
    parser.add_argument("--n-candidates",  type=int,   default=200,
                        help="Max patches to run inference on before selection")
    parser.add_argument("--min-trees",     type=int,   default=5,
                        help="Minimum num_trees attribute for a patch to be considered")
    parser.add_argument("--min-gt-px",     type=int,   default=200,
                        help="Minimum GT mask pixels (filters out near-empty patches)")
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
    model_kd   = load_model(conf, args.kd_ckpt,       device)

    hdf5_path = Path(conf.data_folder).parent / conf.hdf5_file
    print(f"[INFO] HDF5: {hdf5_path}")

    image_patch_map = load_and_organize_data(hdf5_path)
    _, _, test_keys = stratify_images_by_region(
        image_patch_map,
        val_ratio=conf.val_size,
        test_ratio=conf.test_size,
    )
    print(f"[INFO] Test patches total: {len(test_keys)}")

    # Build candidate pool: patches with enough trees, shuffled for seed reproducibility
    with h5py.File(hdf5_path, "r") as hf:
        rich_keys = [
            (k, hf[k].attrs.get("source_image", ""), hf[k].attrs.get("num_trees", 0))
            for k in test_keys
            if hf[k].attrs.get("num_trees", 0) >= args.min_trees
        ]
    rich_keys.sort(key=lambda x: -x[2])
    random.shuffle(rich_keys[:args.n_candidates])          # shuffle within top pool
    candidate_keys = rich_keys[:args.n_candidates]
    print(f"[INFO] Candidates (>= {args.min_trees} trees): {len(candidate_keys)}")

    # Run inference on all candidates
    print(f"[INFO] Running inference on {len(candidate_keys)} patches...")
    scored = []
    with h5py.File(hdf5_path, "r") as hf:
        for i, (key, source_img, _) in enumerate(candidate_keys):
            grp   = hf[key]
            image = grp["image"][()].astype(np.float32) / 255.0
            gt    = grp["labels"]["mask"][()].astype(np.uint8)

            gt_px = int(gt.sum())
            if gt_px < args.min_gt_px:
                continue

            pred_base = predict_mask(model_base, image, device, args.threshold)
            pred_ft   = predict_mask(model_ft,   image, device, args.threshold)
            pred_kd   = predict_mask(model_kd,   image, device, args.threshold)

            iou_base = iou(pred_base, gt)
            iou_ft   = iou(pred_ft,   gt)
            iou_kd   = iou(pred_kd,   gt)

            adaptation_gain = max(iou_ft, iou_kd) - iou_base
            kd_vs_ft        = iou_kd - iou_ft

            scored.append(dict(
                key=key, source=source_img, image=image, gt=gt,
                pred_base=pred_base, pred_ft=pred_ft, pred_kd=pred_kd,
                gt_px=gt_px,
                iou_base=iou_base, iou_ft=iou_ft, iou_kd=iou_kd,
                adaptation_gain=adaptation_gain, kd_vs_ft=kd_vs_ft,
            ))

            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(candidate_keys)}  (valid so far: {len(scored)})")

    print(f"[INFO] Valid patches after gt_px filter: {len(scored)}")

    # Sort by adaptation_gain descending — patches where baseline struggles most
    # relative to the adapted models make the clearest visual argument
    scored.sort(key=lambda x: -x["adaptation_gain"])

    # Enforce one patch per source image for geographic diversity
    seen_sources = set()
    selected = []
    for rec in scored:
        src = rec["source"]
        if src not in seen_sources:
            seen_sources.add(src)
            selected.append(rec)
        if len(selected) == args.n_samples:
            break

    # If strict diversity left us short, fill from remaining patches
    if len(selected) < args.n_samples:
        for rec in scored:
            if rec not in selected:
                selected.append(rec)
            if len(selected) == args.n_samples:
                break

    print(f"\n[INFO] Selected {len(selected)} patches:")
    for rec in selected:
        print(f"  patch={rec['key']}  source={rec['source']}")
        print(f"    GT px={rec['gt_px']}  IoU: base={rec['iou_base']:.3f}  "
              f"ft={rec['iou_ft']:.3f}  kd={rec['iou_kd']:.3f}  "
              f"gain={rec['adaptation_gain']:+.3f}  kd_vs_ft={rec['kd_vs_ft']:+.3f}")

    print(f"\n[INFO] Saving figures to {args.output_dir} ...")
    for i, rec in enumerate(selected):
        cir = cir_composite(rec["image"])
        out = os.path.join(args.output_dir, f"s{i + 1}.png")
        save_figure(cir, rec["gt"], rec["pred_base"], rec["pred_ft"], rec["pred_kd"], out)
        print(f"  Saved s{i + 1}.png")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
