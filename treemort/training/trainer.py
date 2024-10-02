import os
import torch
import logging

from treemort.training.train_loop import train_one_epoch
from treemort.training.validation_loop import validate_one_epoch
from treemort.training.callback_handler import handle_callbacks
from treemort.utils.callbacks import EarlyStopping


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def trainer(
    model,
    optimizer,
    criterion,
    metrics,
    train_loader,
    val_loader,
    conf,
    callbacks,
):
    device = next(model.parameters()).device

    #ewc_data = torch.load(os.path.join(conf.output_dir, "ewc_data.pth"))
    #optimal_parameters = {name: param.to(device) for name, param in ewc_data["optimal_parameters"].items()}
    #fisher_information = {name: fisher.to(device) for name, fisher in ewc_data["fisher_information"].items()}

    #lambda_ewc = 2000

    for epoch in range(conf.epochs):
        logger.info(f"Epoch {epoch + 1}/{conf.epochs} - Training started.")

        #train_loss, train_metrics = train_one_epoch(model, optimizer, criterion, metrics, train_loader, fisher_information, optimal_parameters, lambda_ewc, conf, device)
        train_loss, train_metrics = train_one_epoch(model, optimizer, criterion, metrics, train_loader, conf, device)

        logger.info(f"Epoch {epoch + 1} - Training completed.")
        logger.info(f"Training Loss: {train_loss:.4f}")
        logger.info(f"Training Metrics: {train_metrics}")

        val_loss, val_metrics = validate_one_epoch(model, criterion, metrics, val_loader, conf, device)

        logger.info(f"Epoch {epoch + 1} - Validation completed.")
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Metrics: {val_metrics}")

        handle_callbacks(callbacks, epoch, model, optimizer, val_loss)

        if any([isinstance(cb, EarlyStopping) and cb.stop_training for cb in callbacks]):
            logger.info("Early stopping triggered.")
            break

    logger.info("Training process completed.")
