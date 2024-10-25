import random

from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

from treemort.data.dataset import DeadTreeDataset
from treemort.data.sampler import BalancedSampler, DatasetAwareBalancedSampler, ClassPrioritizedSampler
from treemort.data.image_processing import get_image_processor

from treemort.utils.augment import Augmentations
from treemort.utils.datautils import load_and_organize_data, stratify_images_by_patch_count


def prepare_datasets(conf):
    hdf5_path = Path(conf.data_folder) / conf.hdf5_file

    image_patch_map = load_and_organize_data(hdf5_path)

    train_keys, val_keys, test_keys = stratify_images_by_patch_count(image_patch_map, conf.val_size, conf.test_size)

    random.seed(None) # makes loader non-deterministic

    train_transform = Augmentations()
    val_transform = None
    test_transform = None

    image_processor = get_image_processor(conf.model, conf.backbone)

    train_dataset = DeadTreeDataset(
        hdf5_file=hdf5_path,
        keys=train_keys,
        crop_size=conf.train_crop_size,
        transform=train_transform,
        image_processor=image_processor,
    )
    val_dataset = DeadTreeDataset(
        hdf5_file=hdf5_path,
        keys=val_keys,
        crop_size=conf.val_crop_size,
        transform=val_transform,
        image_processor=image_processor,
    )
    test_dataset = DeadTreeDataset(
        hdf5_file=hdf5_path,
        keys=test_keys,
        crop_size=conf.test_crop_size,
        transform=test_transform,
        image_processor=image_processor,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=conf.train_batch_size, 
        sampler=ClassPrioritizedSampler(
            hdf5_file=hdf5_path, 
            keys=train_keys, 
            prioritized_class_label=1,  # Replace 1 with the actual label of your prioritized class
            sample_ratio=1.0
        ), 
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=conf.val_batch_size, 
        sampler=ClassPrioritizedSampler(
            hdf5_file=hdf5_path, 
            keys=val_keys, 
            prioritized_class_label=1,  # Replace 1 with the actual label of your prioritized class
            sample_ratio=1.0
        ), 
        shuffle=False, 
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=conf.test_batch_size, 
        sampler=ClassPrioritizedSampler(
            hdf5_file=hdf5_path, 
            keys=test_keys, 
            prioritized_class_label=1,  # Replace 1 with the actual label of your prioritized class
            sample_ratio=1.0
        ), 
        shuffle=False, 
        drop_last=True
    )
    return train_loader, val_loader, test_loader


def prepare_datasets_mixed(conf):

    conf.hdf5_file_finnish = 'Finland_RGBNIR_25cm.h5'
    conf.hdf5_file_polish = 'Poland_RGBNIR_25cm.h5'

    hdf5_path_finnish = Path("/scratch/project_2008436/rahmanan/dead_trees/Finland/RGBNIR/25cm").parent / conf.hdf5_file_finnish
    hdf5_path_polish = Path("/scratch/project_2008436/rahmanan/dead_trees/Poland/RGBNIR/25cm").parent / conf.hdf5_file_polish

    #hdf5_path_finnish = Path(conf.data_folder).parent / conf.hdf5_file_finnish
    #hdf5_path_polish = Path(conf.data_folder).parent / conf.hdf5_file_polish

    image_patch_map_finnish = load_and_organize_data(hdf5_path_finnish)
    image_patch_map_polish = load_and_organize_data(hdf5_path_polish)

    train_keys_finnish, val_keys_finnish, test_keys_finnish = stratify_images_by_patch_count(
        image_patch_map_finnish, conf.val_size, conf.test_size
    )

    train_keys_polish, val_keys_polish, test_keys_polish = stratify_images_by_patch_count(
        image_patch_map_polish, conf.val_size, conf.test_size
    )

    random.seed(None)  # makes loader non-deterministic

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

    train_dataset_polish = DeadTreeDataset(
        hdf5_file=hdf5_path_polish,
        keys=train_keys_polish,
        crop_size=conf.train_crop_size,
        transform=train_transform,
        image_processor=image_processor,
    )
    val_dataset_polish = DeadTreeDataset(
        hdf5_file=hdf5_path_polish,
        keys=val_keys_polish,
        crop_size=conf.val_crop_size,
        transform=val_transform,
        image_processor=image_processor,
    )
    test_dataset_polish = DeadTreeDataset(
        hdf5_file=hdf5_path_polish,
        keys=test_keys_polish,
        crop_size=conf.test_crop_size,
        transform=test_transform,
        image_processor=image_processor,
    )

    mixed_train_dataset = ConcatDataset([train_dataset_finnish, train_dataset_polish])
    mixed_val_dataset = ConcatDataset([val_dataset_finnish, val_dataset_polish])
    mixed_test_dataset = ConcatDataset([test_dataset_finnish, test_dataset_polish])

    train_sampler = DatasetAwareBalancedSampler(hdf5_path_finnish, train_keys_finnish, hdf5_path_polish, train_keys_polish)
    val_sampler = DatasetAwareBalancedSampler(hdf5_path_finnish, val_keys_finnish, hdf5_path_polish, val_keys_polish)
    test_sampler = DatasetAwareBalancedSampler(hdf5_path_finnish, test_keys_finnish, hdf5_path_polish, test_keys_polish)

    train_loader = DataLoader(mixed_train_dataset, batch_size=conf.train_batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(mixed_val_dataset, batch_size=conf.val_batch_size, sampler=val_sampler, shuffle=False, drop_last=True)
    test_loader = DataLoader(mixed_test_dataset, batch_size=conf.test_batch_size, sampler=test_sampler, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader