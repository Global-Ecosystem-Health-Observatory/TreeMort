import matplotlib.pyplot as plt
import numpy as np


def plot_examples(dataset, num_examples=5):
    for image_batch, label_batch in dataset.take(1):  # Take one batch
        for i in range(num_examples):
            image = image_batch[i].numpy()
            label = label_batch[i].numpy().squeeze()  # Remove the last dimension for plotting
            
            image_rescaled = (image * 255).astype(np.uint8)
            
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(image_rescaled[:, :, :3])  # Display only the last three channels as RGB
            plt.title('Image (RGB)')
            
            plt.subplot(1, 3, 2)
            plt.imshow(image_rescaled[:, :, 0], cmap='gray')
            plt.title('NIR Channel')
            
            plt.subplot(1, 3, 3)
            plt.imshow(label, cmap='gray')
            plt.title('Label')
            
            plt.show()

''' Usage:

import h5py

from treemort.utils.config import setup
from treemort.data.dataset import prepare_dataset
from treemort.utils.plots import plot_examples

config_file_path = "../configs/kokonet_bs8_cs256.txt"
conf = setup(config_file_path)

# Modified Config Variables for Local Execution; comment on HPC
conf.data_folder = "/Users/anisr/Documents/AerialImages"

hdf5_file_path = os.path.join(conf.data_folder, conf.hdf5_file)
with h5py.File(hdf5_file_path, 'r') as hf:
    keys = list(hf.keys())

dataset = prepare_dataset(hdf5_file_path, keys, conf.train_crop_size, conf.train_batch_size, conf.input_channels, conf.output_channels, True)

# Plot a few examples from the dataset
plot_examples(dataset, num_examples=5)

'''