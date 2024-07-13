import os
import math
import torch
from torch.utils.data import DataLoader

from treeseg.utils.callbacks import build_callbacks

def trainer(model, train_loader, val_loader, num_train_samples, conf):
    num_val_samples = int(conf.val_size * num_train_samples)
    num_train_batches = math.ceil((num_train_samples - num_val_samples) / conf.train_batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
    criterion = torch.nn.BCELoss()

    callbacks = build_callbacks(num_train_batches, conf.output_dir, optimizer)

    for epoch in range(conf.epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % conf.log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        train_loss /= len(train_loader.dataset)
        print(f"====> Epoch: {epoch} Average loss: {train_loss:.4f}")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()

            val_loss /= len(val_loader.dataset)
            print(f"====> Validation loss: {val_loss:.4f}")

        for callback in callbacks:
            callback.on_epoch_end(epoch, logs={'loss': train_loss, 'val_loss': val_loss if val_loader else None})



'''
import os
import math

import tensorflow as tf

from treeseg.utils.callbacks import build_callbacks


def trainer(model, train_dataset, val_dataset, num_train_samples, conf):
    num_val_samples = int(conf.val_size * num_train_samples)
    num_train_batches = math.ceil(
        (num_train_samples - num_val_samples) / conf.train_batch_size
    )

    callbacks = build_callbacks(num_train_batches, conf.output_dir)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        steps_per_epoch=num_train_batches,
        epochs=conf.epochs,
        validation_steps=num_val_samples,
        callbacks=callbacks,
    )
'''