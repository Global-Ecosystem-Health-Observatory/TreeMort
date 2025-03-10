import h5py
import torch

import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset

from treemort.data.image_processing import apply_image_processor


class DeadTreeDataset(Dataset):
    def __init__(self, hdf5_file, keys, crop_size=256, transform=None, image_processor=None):
        self.hdf5_file = hdf5_file
        self.keys = keys
        self.crop_size = crop_size
        self.transform = transform
        self.image_processor = image_processor
        self._adjust_image_processor_mean_std()

    def _load_data(self, idx):
        key = self.keys[idx]
        with h5py.File(self.hdf5_file, "r") as hf:
            group = hf[key]
            
            image = group['image'][()].astype(np.float32)
            
            labels = group['labels']
            mask = labels['mask'][()]
            centroid = labels['centroid'][()]
            hybrid = labels['hybrid'][()]
            buffer_mask = labels['buffer_mask'][()]
            
            label = np.stack([mask, centroid, hybrid, buffer_mask], axis=-1)

        return image, label

    def _preprocess_image_and_label(self, image, label):
        image = torch.from_numpy(image).permute(2, 0, 1)  # [C, H, W]
        label = torch.from_numpy(label).permute(2, 0, 1)  # [C, H, W]
        
        image = image / 255.0

        image, label = self._center_crop_or_pad(image, label, self.crop_size)

        if self.image_processor:
            image = apply_image_processor(image, self.image_processor)
            
        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    def _prepare_kokonet_label(self, image, label):
        # Rescale image from [0, 1] to [-1, 1] (all channels)
        image = image * 2 - 1

        first_label = label[0]  # shape: [H, W]
        H, W = first_label.shape
        device = first_label.device

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device), 
            torch.arange(W, device=device),
            indexing='ij'
        )
        coords = torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)  # Shape: [H*W, 2]

        first_label_flat = first_label.reshape(-1)
        fg_indices = (first_label_flat == 1).nonzero(as_tuple=False).squeeze()
        bg_indices = (first_label_flat == 0).nonzero(as_tuple=False).squeeze()

        if fg_indices.numel() > 0:
            fg_coords = coords[fg_indices].float().reshape(-1, 2)  # Shape: [N_fg, 2]
        else:
            fg_coords = torch.empty((0, 2), device=device)

        if bg_indices.numel() > 0:
            bg_coords = coords[bg_indices].float().reshape(-1, 2)  # Shape: [N_bg, 2]
        else:
            bg_coords = torch.empty((0, 2), device=device)

        if bg_coords.numel() == 0:
            distance_fg = torch.ones(fg_coords.shape[0], device=device)
        elif fg_coords.numel() == 0:
            distance_fg = torch.zeros(0, device=device)  # No foreground, so no distances
        else:
            dists = torch.cdist(fg_coords.unsqueeze(0), bg_coords.unsqueeze(0), p=2)  # Ensures 2D shape
            distance_fg, _ = dists.squeeze(0).min(dim=1)  # Get minimum distance

        # Normalize distances
        if distance_fg.numel() > 0:
            max_dist = distance_fg.max()
            normalized_fg = distance_fg / max_dist if max_dist > 0 else distance_fg
        else:
            normalized_fg = distance_fg

        # Initialize all pixels to -1 (background).
        topo_flat = -1 * torch.ones_like(first_label_flat, dtype=torch.float32, device=device)
        if fg_indices.numel() > 0:
            topo_flat[fg_indices] = normalized_fg
        topo_map = topo_flat.reshape(H, W).unsqueeze(0)

        if label.shape[0] > 1:
            new_label = torch.cat([topo_map, label[1:]], dim=0)
        else:
            new_label = topo_map

        return image, new_label
    
    def _adjust_image_processor_mean_std(self):
        if self.image_processor:
            if len(self.image_processor.image_mean) == 3:
                self.image_processor.image_mean.insert(0, 0.5)
            if len(self.image_processor.image_std) == 3:
                self.image_processor.image_std.insert(0, 0.5)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):            
        image, label = self._load_data(idx)
        image, label = self._preprocess_image_and_label(image, label)

        image, label = self._prepare_kokonet_label(image, label)

        return image, label

    def _center_crop_or_pad(self, image, label, size=256):
        h, w = image.shape[1:]  # image is in (C, H, W) format

        pad_h, pad_w = max(size - h, 0), max(size - w, 0)

        if pad_h > 0 or pad_w > 0:
            image, label = self._pad_image_and_label(image, label, pad_h, pad_w)
        
        return self._crop_center(image, label, size)
        
    def _pad_image_and_label(self, image, label, pad_h, pad_w):
        image = F.pad(image, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=0)
        label = F.pad(label, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=0)

        return image, label
    
    def _crop_center(self, image, label, size):
        h, w = image.shape[1:]
        x, y = (w - size) // 2, (h - size) // 2

        return image[:, y:y + size, x:x + size], label[:, y:y + size, x:x + size]
