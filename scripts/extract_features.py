"""
Extract encoder layer features from a flair_unet checkpoint and save to disk.

Each layer's features are saved as features_{layer}_batch{i}.pt (shape: B x C,
after global-average-pooling) plus features_{layer}_spatial_batch{i}.pt
(shape: B x C x H x W, kept for SSIM computation).

Usage:
  python3 scripts/extract_features.py \
    --config      configs/model/flair_unet_transfer.txt \
    --data-config configs/data/poland.txt \
    --checkpoint  output/flair_unet/best.weights.pth \
    --output-dir  /scratch/.../features_poland_transfer \
    --max-patches 1000
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from treemort.data.loader import prepare_datasets
from treemort.modeling.model_config import configure_model
from treemort.utils.config import setup
from treemort.utils.logger import get_logger, configure_logger

LAYER_NAMES = ["layer1", "layer2", "layer3", "layer4"]


def extract(conf, checkpoint_path, output_dir, max_patches):
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        return  # only rank 0 extracts features

    logger = get_logger()
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    id2label = {0: "alive", 1: "dead"}
    model = configure_model(conf, id2label)
    model.to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.info(f"Missing keys: {missing}")
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    _, _, test_loader = prepare_datasets(conf, rank=0, world_size=1)
    logger.info(f"Test batches: {len(test_loader)}")

    patches_seen = 0
    batch_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(device)
            B = images.size(0)

            if patches_seen >= max_patches:
                break

            # CombinedModel.forward returns (logits, encoder_features)
            _, enc_feats = model(images)

            for li, layer_name in enumerate(LAYER_NAMES):
                feat = enc_feats[li].cpu()               # B x C x H x W
                feat_gap = feat.mean(dim=(2, 3))          # B x C  (for CKA/cosine)

                torch.save(feat_gap, os.path.join(output_dir, f"features_{layer_name}_batch{batch_idx}.pt"))
                torch.save(feat,     os.path.join(output_dir, f"features_{layer_name}_spatial_batch{batch_idx}.pt"))

            # Save binary patch labels: 1 if any dead-tree mask pixel is set
            if len(batch) > 1:
                masks = batch[1]              # B x C x H x W or B x H x W
                if masks.ndim == 4:
                    masks = masks[:, 0]       # take first channel (binary mask)
                labels = (masks.view(B, -1).sum(dim=1) > 0).long()
                torch.save(labels, os.path.join(output_dir, f"labels_batch{batch_idx}.pt"))

            patches_seen += B
            batch_idx += 1

            if batch_idx % 20 == 0:
                logger.info(f"  Batch {batch_idx}, patches {patches_seen}/{max_patches}")

    logger.info(f"Done. {patches_seen} patches, {batch_idx} batches saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract flair_unet encoder features.")
    parser.add_argument("--config",       required=True)
    parser.add_argument("--data-config",  required=True)
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--max-patches",  type=int, default=1000)
    parser.add_argument("--verbosity",    default="info")
    parser.add_argument("--run-id",       default=None)
    args = parser.parse_args()

    configure_logger(verbosity=args.verbosity)

    conf = setup(args.config, data_config=args.data_config)
    if args.run_id:
        conf.run_id = args.run_id
        conf.run_dir = os.path.join(conf.output_dir, conf.model, conf.run_id)

    extract(conf, args.checkpoint, args.output_dir, args.max_patches)


if __name__ == "__main__":
    main()
