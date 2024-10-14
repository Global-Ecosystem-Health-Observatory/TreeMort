import random

from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

from treemort.data.dataset import DeadTreeDataset
from treemort.data.sampler import BalancedSampler
from treemort.data.image_processing import get_image_processor

from treemort.utils.augment import Augmentations
from treemort.utils.datautils import load_and_organize_data, stratify_images_by_patch_count


def prepare_datasets(conf):
    hdf5_path_finnish = Path(conf.data_folder_finnish).parent / conf.hdf5_file_finnish
    hdf5_path_us = Path(conf.data_folder_us).parent / conf.hdf5_file_us

    image_patch_map_finnish = load_and_organize_data(hdf5_path_finnish)
    image_patch_map_us = load_and_organize_data(hdf5_path_us)

    train_keys_finnish, val_keys_finnish, test_keys_finnish = stratify_images_by_patch_count(image_patch_map_finnish, conf.val_size, conf.test_size)
    train_keys_us, val_keys_us, test_keys_us = stratify_images_by_patch_count(image_patch_map_us, conf.val_size, conf.test_size)

    random.seed(None)  # Make loader non-deterministic

    train_transform = Augmentations()
    val_transform = None
    test_transform = None

    image_processor = get_image_processor(conf.model, conf.backbone)

    train_dataset_finnish = DeadTreeDataset(
        hdf5_file=hdf5_path_finnish,
        keys=train_keys_finnish,
        crop_size=conf.train_crop_size,
        transform=train_transform,
        image_processor=image_processor,
        domain_label=0  # Domain label for Finnish data
    )
    
    val_dataset_finnish = DeadTreeDataset(
        hdf5_file=hdf5_path_finnish,
        keys=val_keys_finnish,
        crop_size=conf.val_crop_size,
        transform=val_transform,
        image_processor=image_processor,
        domain_label=0  # Domain label for Finnish data
    )

    train_dataset_us = DeadTreeDataset(
        hdf5_file=hdf5_path_us,
        keys=train_keys_us,
        crop_size=conf.train_crop_size,
        transform=train_transform,
        image_processor=image_processor,
        domain_label=1  # Domain label for US data
    )

    val_dataset_us = DeadTreeDataset(
        hdf5_file=hdf5_path_us,
        keys=val_keys_us,
        crop_size=conf.val_crop_size,
        transform=val_transform,
        image_processor=image_processor,
        domain_label=1  # Domain label for US data
    )

    combined_train_dataset = ConcatDataset([train_dataset_finnish, train_dataset_us])
    combined_val_dataset = ConcatDataset([val_dataset_finnish, val_dataset_us])

    train_loader = DataLoader(
        combined_train_dataset, 
        batch_size=conf.train_batch_size, 
        shuffle=True,  # Shuffling is important to mix the domains
        drop_last=True
    )
    
    val_loader = DataLoader(
        combined_val_dataset, 
        batch_size=conf.val_batch_size, 
        shuffle=False, 
        drop_last=True
    )

    test_dataset_finnish = DeadTreeDataset(
        hdf5_file=hdf5_path_finnish,
        keys=test_keys_finnish,
        crop_size=conf.test_crop_size,
        transform=test_transform,
        image_processor=image_processor,
        domain_label=0  # Domain label for Finnish data
    )
    
    test_dataset_us = DeadTreeDataset(
        hdf5_file=hdf5_path_us,
        keys=test_keys_us,
        crop_size=conf.test_crop_size,
        transform=test_transform,
        image_processor=image_processor,
        domain_label=1  # Domain label for US data
    )

    test_loader_finnish = DataLoader(
        test_dataset_finnish, 
        batch_size=conf.test_batch_size, 
        shuffle=False, 
        drop_last=True
    )
    
    test_loader_us = DataLoader(
        test_dataset_us, 
        batch_size=conf.test_batch_size, 
        shuffle=False, 
        drop_last=True
    )
    
    return train_loader, val_loader, test_loader_finnish, test_loader_us