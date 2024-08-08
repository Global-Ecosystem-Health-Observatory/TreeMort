import os
import torch
import argparse

from treemort.utils.config import setup
from treemort.utils.callbacks import build_callbacks

from treemort.data.dataset import prepare_datasets

from treemort.modeling.builder import resume_or_load
from treemort.modeling.trainer import trainer
from treemort.evaluation.evaluator import evaluator


def run(conf, eval_only):
    assert os.path.exists(
        conf.data_folder
    ), f"[ERROR] Data folder {conf.data_folder} does not exist."

    if not os.path.exists(conf.output_dir):
        os.makedirs(conf.output_dir)
        print(f"[INFO] Created output directory: {conf.output_dir}")
    else:
        print(f"[INFO] Output directory already exists: {conf.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(conf)
    print(f"[INFO] Datasets prepared: Train({len(train_dataset)}), Val({len(val_dataset)}), Test({len(test_dataset)})")

    print("[INFO] Loading or resuming model...")
    model, optimizer, criterion, metrics = resume_or_load(conf, device=device)
    print(f"[INFO] Model, optimizer, criterion, and metrics are set up.")

    n_batches = len(train_dataset)

    print(f"[INFO] Setting up callbacks for {n_batches} batches...")
    callbacks = build_callbacks(n_batches, conf.output_dir, optimizer)
    print(f"[INFO] Callbacks set up: {len(callbacks)} callbacks configured.")

    if eval_only:
        print("[INFO] Evaluation-only mode started.")
        evaluator(
            model,
            dataset=test_dataset,
            num_samples=len(test_dataset),
            batch_size=conf.test_batch_size,
            threshold=conf.threshold,
            device=device,
        )
        print("[INFO] Evaluation completed.")

    else:
        print("[INFO] Training mode started.")
        trainer(
            model,
            optimizer=optimizer,
            criterion=criterion,
            metrics=metrics,
            train_loader=train_dataset,
            val_loader=val_dataset,
            conf=conf,
            callbacks=callbacks,
            device=device,
        )
        print("[INFO] Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration setup for network.")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="If set, only evaluate the model without training",
    )

    args = parser.parse_args()

    conf = setup(args.config)

    run(conf, args.eval_only)
