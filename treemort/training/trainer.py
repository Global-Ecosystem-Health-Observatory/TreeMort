from tqdm import tqdm

from treemort.training.train_loop import train_one_epoch
from treemort.training.validation_loop import validate_one_epoch
from treemort.training.callback_handler import handle_callbacks

from treemort.utils.logger import get_logger
from treemort.utils.metrics import log_metrics


def trainer(
    model,
    optimizer,
    schedular,
    criterion,
    metrics,
    train_loader,
    val_loader,
    conf,
    callbacks,
    is_main=True,
    is_distributed=False,
):
    logger = get_logger()

    device = next(model.parameters()).device
    best_metric = float('inf')

    for epoch in tqdm(range(conf.epochs), desc="Epochs", unit="epoch", disable=not is_main):

        if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_metrics = train_one_epoch(
            model, optimizer, schedular, criterion, metrics,
            train_loader, conf, device,
            is_main=is_main, is_distributed=is_distributed,
        )

        val_loss, val_metrics = validate_one_epoch(
            model, criterion, metrics,
            val_loader, conf, device,
            is_main=is_main, is_distributed=is_distributed,
        )

        stop_training = handle_callbacks(
            callbacks,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            val_loss=val_loss,
            val_metrics=val_metrics
        )

        if stop_training:
            logger.info("Early stopping triggered.")
            break

    logger.info("Training completed successfully.")
    return model
