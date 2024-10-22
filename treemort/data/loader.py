import random
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

from treemort.data.dataset import DeadTreeDataset
from treemort.data.sampler import BalancedSampler, ClassPrioritizedSampler
from treemort.data.image_processing import get_image_processor

from treemort.utils.augment import Augmentations
from treemort.utils.datautils import load_and_organize_data, stratify_images_by_patch_count


def prepare_datasets(conf):
    hdf5_path_finnish = Path(conf.data_folder_finnish) / conf.hdf5_file_finnish
    hdf5_path_us = Path(conf.data_folder_us) / conf.hdf5_file_us

    image_patch_map_finnish = load_and_organize_data(hdf5_path_finnish)
    image_patch_map_us = load_and_organize_data(hdf5_path_us)

    train_keys_finnish, val_keys_finnish, test_keys_finnish = stratify_images_by_patch_count(image_patch_map_finnish, conf.val_size, conf.test_size)
    train_keys_us, val_keys_us, test_keys_us = stratify_images_by_patch_count(image_patch_map_us, conf.val_size, conf.test_size)

    random.seed(None)  # Non-deterministic seed

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
    )
    val_dataset_finnish = DeadTreeDataset(
        hdf5_file=hdf5_path_finnish,
        keys=val_keys_finnish,
        crop_size=conf.val_crop_size,
        transform=val_transform,
        image_processor=image_processor,
    )
    test_dataset_finnish = DeadTreeDataset(
        hdf5_file=hdf5_path_finnish,
        keys=test_keys_finnish,
        crop_size=conf.test_crop_size,
        transform=test_transform,
        image_processor=image_processor,
    )

    train_dataset_us = DeadTreeDataset(
        hdf5_file=hdf5_path_us,
        keys=train_keys_us,
        crop_size=conf.train_crop_size,
        transform=train_transform,
        image_processor=image_processor,
    )
    val_dataset_us = DeadTreeDataset(
        hdf5_file=hdf5_path_us,
        keys=val_keys_us,
        crop_size=conf.val_crop_size,
        transform=val_transform,
        image_processor=image_processor,
    )
    test_dataset_us = DeadTreeDataset(
        hdf5_file=hdf5_path_us,
        keys=test_keys_us,
        crop_size=conf.test_crop_size,
        transform=test_transform,
        image_processor=image_processor,
    )

    train_sampler_finnish = BalancedSampler(hdf5_path_finnish, train_keys_finnish)
    val_sampler_finnish = BalancedSampler(hdf5_path_finnish, val_keys_finnish)

    train_sampler_us = ClassPrioritizedSampler(hdf5_path_us, train_keys_us, prioritized_class_label=1, sample_ratio=1.0)
    val_sampler_us = ClassPrioritizedSampler(hdf5_path_us, val_keys_us, prioritized_class_label=1, sample_ratio=1.0)

    train_loader_finnish = DataLoader(
        train_dataset_finnish,
        batch_size=conf.train_batch_size,
        sampler=train_sampler_finnish,
        drop_last=True,
    )
    train_loader_us = DataLoader(
        train_dataset_us,
        batch_size=conf.train_batch_size,
        sampler=train_sampler_us,
        drop_last=True,
    )

    val_loader_finnish = DataLoader(
        val_dataset_finnish,
        batch_size=conf.val_batch_size,
        sampler=val_sampler_finnish,
        shuffle=False,
        drop_last=True,
    )
    val_loader_us = DataLoader(
        val_dataset_us,
        batch_size=conf.val_batch_size,
        sampler=val_sampler_us,
        shuffle=False,
        drop_last=True,
    )

    train_loader_combined = DataLoader(
        ConcatDataset([train_dataset_finnish, train_dataset_us]),
        batch_size=conf.train_batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader_combined = DataLoader(
        ConcatDataset([val_dataset_finnish, val_dataset_us]),
        batch_size=conf.val_batch_size,
        shuffle=False,
        drop_last=True,
    )

    test_loader_finnish = DataLoader(
        test_dataset_finnish,
        batch_size=conf.test_batch_size,
        shuffle=False,
        drop_last=True,
    )
    test_loader_us = DataLoader(
        test_dataset_us,
        batch_size=conf.test_batch_size,
        shuffle=False,
        drop_last=True,
    )

    return (
        train_loader_combined,
        val_loader_combined,
        test_loader_finnish,
        test_loader_us,
    )