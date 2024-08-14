from treemort.utils.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def handle_callbacks(callbacks, epoch, model, optimizer, val_loss):
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            print("[INFO] Invoking ModelCheckpoint callback.")
            callback(epoch + 1, model, optimizer, val_loss)

        elif isinstance(callback, ReduceLROnPlateau):
            print("[INFO] Invoking ReduceLROnPlateau callback.")
            callback(val_loss)

        elif isinstance(callback, EarlyStopping):
            print("[INFO] Invoking EarlyStopping callback.")
            callback(epoch + 1, val_loss)
