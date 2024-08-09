import matplotlib.pyplot as plt
import torch

def plot_examples(data_loader, num_examples=5):
    examples_shown = 0
    
    for images, labels in data_loader:
        for i in range(images.size(0)):
            if examples_shown >= num_examples:
                return  # Stop once we've shown enough examples

            rgb_image = images[i, 1:4].permute(1, 2, 0).cpu().numpy()
            nir_image = images[i, 0].cpu().numpy()
            label_np = labels[i].cpu().numpy().squeeze()

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            
            ax[0].imshow(rgb_image)
            ax[0].set_title("RGB Image")
            ax[0].axis('off')
            
            ax[1].imshow(nir_image, cmap='gray')
            ax[1].set_title("NIR Image")
            ax[1].axis('off')
            
            ax[2].imshow(label_np, cmap='gray')
            ax[2].set_title("Label (Mask)")
            ax[2].axis('off')

            plt.show()

            examples_shown += 1
            
        if examples_shown >= num_examples:
            break

''' Usage:

import h5py

from treemort.utils.config import setup
from treemort.data.dataset import prepare_datasets
from treemort.utils.plots import plot_examples

config_file_path = "../configs/unet_bs8_cs256.txt"
conf = setup(config_file_path)

# Modified Config Variables for Local Execution; comment on HPC
conf.data_folder = "/Users/anisr/Documents/AerialImages"

hdf5_file_path = os.path.join(conf.data_folder, conf.hdf5_file)
with h5py.File(hdf5_file_path, 'r') as hf:
    keys = list(hf.keys())

train_dataset, val_dataset, test_dataset = prepare_datasets(conf)

plot_examples(train_dataset, num_examples=5)

'''