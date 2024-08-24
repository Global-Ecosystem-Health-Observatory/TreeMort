import os
import torch
import argparse

from treemort.data.loader import prepare_datasets
from treemort.modeling.builder import resume_or_load
from treemort.training.trainer import trainer
from treemort.evaluation.evaluator import evaluator
from treemort.utils.config import setup


def run(conf, eval_only):
    assert os.path.exists(conf.data_folder), f"[ERROR] Data folder {conf.data_folder} does not exist."

    if not os.path.exists(conf.output_dir):
        os.makedirs(conf.output_dir)
        print(f"[INFO] Created output directory: {conf.output_dir}")
    else:
        print(f"[INFO] Output directory already exists: {conf.output_dir}")

    id2label = {0: "alive", 1: "dead"}

    class_weights = torch.tensor([0.1, 0.9], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Preparing datasets...")
    train_dataset, val_dataset, test_dataset, image_processor = prepare_datasets(conf)
    print(f"[INFO] Datasets prepared: Train({len(train_dataset)}), Val({len(val_dataset)}), Test({len(test_dataset)})")

    print("[INFO] Loading or resuming model...")
    model, optimizer, criterion, metrics, callbacks = resume_or_load(conf, id2label, len(train_dataset), device)
    print(f"[INFO] Model, optimizer, criterion, metrics, and callbacks are set up.")

    if eval_only:
        print("[INFO] Evaluation-only mode started.")
        evaluator(
            model,
            dataset=test_dataset,
            num_samples=len(test_dataset),
            batch_size=conf.test_batch_size,
            threshold=conf.threshold,
            model_name=conf.model,
            image_processor=image_processor,
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
            image_processor=image_processor,
            class_weights=class_weights,
        )
        print("[INFO] Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration setup for network.")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("--eval-only", action="store_true", help="If set, only evaluate the model without training",)

    args = parser.parse_args()

    conf = setup(args.config)

    run(conf, args.eval_only)
