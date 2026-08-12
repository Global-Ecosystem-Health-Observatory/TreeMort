import math
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import argparse

from treemort.data.loader import prepare_datasets
from treemort.modeling.builder import resume_or_load
from treemort.evaluation.evaluator import evaluator
from treemort.training.output_processing import process_model_output, prepare_pred_and_target
from treemort.utils.config import setup
from treemort.utils.logger import get_logger, configure_logger

logger = get_logger(__name__)


def _dann_lambda(epoch, total_epochs, step, steps_per_epoch, gamma=10.0):
    """Annealed GRL strength from 0→1 following Ganin et al. (2016)."""
    p = (epoch * steps_per_epoch + step) / max(1, total_epochs * steps_per_epoch)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


def run(conf, source_conf, eval_only):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0

    if eval_only and not is_main:
        return

    is_distributed = world_size > 1 and not eval_only
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if is_distributed:
        dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
        torch.cuda.set_device(0)
    if is_main:
        logger.info(f"Using device: {device}  |  world_size={world_size}")

    assert os.path.exists(conf.data_folder), f"Data folder not found: {conf.data_folder}"

    run_dir = getattr(conf, "run_dir", os.path.join(conf.output_dir, conf.model))
    if is_main:
        os.makedirs(conf.output_dir, exist_ok=True)
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Run directory: {run_dir}")

    if is_distributed:
        dist.barrier()

    id2label = {0: "alive", 1: "dead"}
    lambda_dann = getattr(conf, "lambda_dann", 0.1)

    data_rank = 0 if eval_only else rank
    data_world = 1 if eval_only else world_size

    # Target domain loader (train+val+test)
    train_loader, val_loader, test_loader = prepare_datasets(conf, rank=data_rank, world_size=data_world)
    train_len = len(train_loader) if train_loader is not None else 0
    val_len   = len(val_loader)   if val_loader   is not None else 0
    test_len  = len(test_loader)  if test_loader  is not None else 0
    if is_main:
        logger.info(f"Target datasets: Train({train_len}), Val({val_len}), Test({test_len})")

    num_steps = train_len if train_len > 0 else test_len
    model, optimizer, scheduler, criterion, metrics, callbacks = resume_or_load(
        conf, id2label, num_steps, device, is_main=is_main
    )

    if eval_only:
        conf.resume = True
        conf.best_model = "best.weights.dann.pth"
        if test_loader is None or test_len == 0:
            raise RuntimeError("No test_loader available for evaluation.")
        if is_main:
            logger.info("Evaluation-only mode.")
        evaluator(model, test_loader, test_len, metrics, conf)
        if is_main:
            logger.info("Evaluation completed.")
        return

    # Source domain loader (Finland) — train split only; val/test not needed from source
    assert source_conf is not None, "--source-data-config is required for DANN training."
    src_train, _, _ = prepare_datasets(source_conf, rank=data_rank, world_size=data_world)
    src_len = len(src_train) if src_train is not None else 0
    if is_main:
        logger.info(f"Source (Finland) train: {src_len} batches")

    best_val_iou = float("-inf")
    best_ckpt = os.path.join(run_dir, "best.weights.dann.pth")

    for epoch in range(conf.epochs):
        if is_main:
            logger.info(f"Epoch {epoch + 1}/{conf.epochs}")

        model.train()
        epoch_task_loss = 0.0
        epoch_domain_loss = 0.0
        steps = 0

        src_iter = iter(src_train)

        for step, (tgt_images, tgt_labels) in enumerate(train_loader):
            try:
                src_images, src_labels = next(src_iter)
            except StopIteration:
                src_iter = iter(src_train)
                src_images, src_labels = next(src_iter)

            tgt_images = tgt_images.to(device)
            tgt_labels = tgt_labels.to(device)
            src_images = src_images.to(device)
            src_labels = src_labels.to(device)

            lam = _dann_lambda(epoch, conf.epochs, step, train_len)

            optimizer.zero_grad()

            # --- Task loss: target domain ---
            logits_t, feats_t = process_model_output(model, tgt_images, conf.model)
            _, _, ht, wt = tgt_labels.shape
            preds_t, targets_t, buf_t = prepare_pred_and_target(logits_t, tgt_labels, (ht, wt))
            loss_task_t = criterion(preds_t, targets_t, buffer=buf_t)

            # --- Task loss: source domain ---
            logits_s, feats_s = process_model_output(model, src_images, conf.model)
            _, _, hs, ws = src_labels.shape
            preds_s, targets_s, buf_s = prepare_pred_and_target(logits_s, src_labels, (hs, ws))
            loss_task_s = criterion(preds_s, targets_s, buffer=buf_s)

            loss_task = (loss_task_t + loss_task_s) * 0.5

            # --- Domain adversarial loss via GRL ---
            B_t = tgt_images.shape[0]
            B_s = src_images.shape[0]

            dom_t = model.discriminate(feats_t, lambda_val=lam)
            dom_s = model.discriminate(feats_s, lambda_val=lam)

            labels_t = torch.ones(B_t, 1, device=device)   # target domain = 1
            labels_s = torch.zeros(B_s, 1, device=device)  # source domain = 0

            loss_domain = (
                F.binary_cross_entropy_with_logits(dom_t, labels_t) +
                F.binary_cross_entropy_with_logits(dom_s, labels_s)
            ) * 0.5

            loss = loss_task + lambda_dann * loss_domain
            loss.backward()

            optimizer.step()
            scheduler.step()

            epoch_task_loss += loss_task.item()
            epoch_domain_loss += loss_domain.item()
            steps += 1

        epoch_task_loss /= max(1, steps)
        epoch_domain_loss /= max(1, steps)

        # --- Validation on target domain ---
        model.eval()
        val_loss = 0.0
        val_metrics_acc = {}
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                logits_v, _ = process_model_output(model, val_images, conf.model)
                _, _, hv, wv = val_labels.shape
                preds_v, targets_v, buf_v = prepare_pred_and_target(logits_v, val_labels, (hv, wv))
                vl = criterion(preds_v, targets_v, buffer=buf_v)
                val_loss += vl.item()
                vm = metrics(preds_v, targets_v, buffer=buf_v)
                for k, v in vm.items():
                    val_metrics_acc[k] = val_metrics_acc.get(k, 0.0) + (v.item() if torch.is_tensor(v) else v)

        val_loss /= max(1, val_len)
        for k in val_metrics_acc:
            val_metrics_acc[k] /= max(1, val_len)

        if is_main:
            val_iou = val_metrics_acc.get("iou_segments", float("-inf"))
            logger.info(
                f"Epoch {epoch + 1}: task={epoch_task_loss:.4f}  domain={epoch_domain_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_iou={val_iou:.4f}  lambda_grl={lam:.3f}"
            )
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                raw = model.module if hasattr(model, "module") else model
                torch.save(raw.state_dict(), best_ckpt)
                logger.info(f"Saved best model → {best_ckpt}  (val_iou={val_iou:.4f})")

        if is_distributed:
            dist.barrier()

    if is_main:
        logger.info(f"DANN training completed. Best val IoU: {best_val_iou:.4f}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DANN domain-adaptation training entry point.")
    parser.add_argument("config",                 type=str, help="Path to model config (flair_unet_dann.txt)")
    parser.add_argument("--data-config",          type=str, required=True, help="Target domain data config (e.g. poland.txt)")
    parser.add_argument("--source-data-config",   type=str, default=None,  help="Source domain data config (e.g. finland.txt)")
    parser.add_argument("--verbosity",            type=str, default="info", choices=["info", "debug", "warning"])
    parser.add_argument("--eval-only",            action="store_true", help="Evaluate only (no training)")

    args = parser.parse_args()

    _ = configure_logger(verbosity=args.verbosity)

    conf = setup(args.config, data_config=args.data_config)

    source_conf = None
    if args.source_data_config:
        source_conf = setup(args.config, data_config=args.source_data_config)

    run(conf, source_conf, args.eval_only)


'''

1) Local

Usage: python3 -m treemort.main_dann <config> --data-config <target> --source-data-config <source> [--eval-only]

Example:

(Train) python3 -m treemort.main_dann $TREEMORT_REPO_PATH/configs/model/flair_unet_dann.txt \
    --data-config $TREEMORT_REPO_PATH/configs/data/poland.txt \
    --source-data-config $TREEMORT_REPO_PATH/configs/data/finland.txt

(Test)  python3 -m treemort.main_dann $TREEMORT_REPO_PATH/configs/model/flair_unet_dann.txt \
    --data-config $TREEMORT_REPO_PATH/configs/data/poland.txt \
    --eval-only

2) HPC

bash $TREEMORT_REPO_PATH/scripts/submit_treemort_dann.sh lumi poland
bash $TREEMORT_REPO_PATH/scripts/submit_treemort_dann.sh lumi poland --eval-only

'''
