import random

import numpy as np

from matplotlib import pyplot as plt


def plot_samples(image_paths, label_paths, num_samples=5):
    assert len(image_paths) == len(
        label_paths
    ), "The number of image and label paths should be the same."

    sample_indices = random.sample(range(len(image_paths)), num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(20, num_samples * 5))

    for i, idx in enumerate(sample_indices):

        image_np = np.load(image_paths[idx])
        label_np = np.load(label_paths[idx])

        nir = image_np[:, :, 0]
        red = image_np[:, :, 1]
        green = image_np[:, :, 2]
        blue = image_np[:, :, 3]

        true_color = np.stack((red, green, blue), axis=-1)

        false_color = np.stack((nir, red, green), axis=-1)

        axes[i, 0].imshow(true_color)
        axes[i, 0].set_title("True Color Composite (RGB)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(false_color)
        axes[i, 1].set_title("False Color Composite (NIR, R, G)")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(label_np, cmap="gray")
        axes[i, 2].set_title("Label")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def plot_augmented_samples(dataset, num_samples=3):
    iterator = iter(dataset)

    sample_count = 0

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))

    while sample_count < num_samples:

        batch_images, batch_labels = next(iterator)
        batch_size = batch_images.shape[0]

        for i in range(batch_size):
            if sample_count >= num_samples:
                break

            image = batch_images[i].numpy()
            label = batch_labels[i].numpy()

            # de-normalize image if necessary
            image = ((image + 1.0) * 127.5).astype("uint8")

            nir = image[:, :, 0]
            red = image[:, :, 1]
            green = image[:, :, 2]
            blue = image[:, :, 3]

            true_color = np.stack((red, green, blue), axis=-1)

            false_color = np.stack((nir, red, green), axis=-1)

            axes[i, 0].imshow(true_color)
            axes[i, 0].set_title("True Color Composite (RGB)")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(false_color)
            axes[i, 1].set_title("False Color Composite (NIR, R, G)")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(label.squeeze(), cmap="gray")
            axes[i, 2].set_title("Label")
            axes[i, 2].axis("off")

            sample_count += 1

    plt.tight_layout()
    plt.show()
