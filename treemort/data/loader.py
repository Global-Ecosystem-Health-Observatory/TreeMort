import random

from pathlib import Path
from torch.utils.data import DataLoader

from treemort.data.dataset import DeadTreeDataset
from treemort.data.sampler import BalancedSampler
from treemort.data.image_processing import get_image_processor

from treemort.utils.augment import Augmentations
from treemort.utils.datautils import load_and_organize_data, stratify_images_by_patch_count, stratify_images_by_region


def prepare_datasets(conf):
    hdf5_path = Path(conf.data_folder).parent / conf.hdf5_file

    image_patch_map = load_and_organize_data(hdf5_path)

    train_keys, val_keys, test_keys = stratify_images_by_region(
        image_patch_map,
        val_ratio=conf.val_size,
        test_ratio=conf.test_size
    )

    random.seed(None) # makes loader non-deterministic

    aug_kwargs = {
        "brightness": getattr(conf, "augment_brightness_jitter", 0.0),
        "contrast": getattr(conf, "augment_contrast_jitter", 0.0),
        "gamma_range": tuple(getattr(conf, "augment_gamma_range", [1.0, 1.0])),
        "hue": getattr(conf, "augment_hue_jitter", 0.0),
        "saturation": getattr(conf, "augment_saturation_jitter", 0.0),
        "noise_range": tuple(getattr(conf, "augment_noise_range", [1.0, 1.0])),
        "scale_range": tuple(getattr(conf, "augment_scale_range", [1.0, 1.0])),
        "scale_prob": getattr(conf, "augment_scale_prob", 0.0),
        "blur_prob": getattr(conf, "augment_blur_prob", 0.0),
    }
    train_transform = Augmentations(**aug_kwargs)
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

    loader_kwargs = dict(
        num_workers=getattr(conf, "num_workers", 4),
        pin_memory=True,
        prefetch_factor=2
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=conf.train_batch_size,
        sampler=BalancedSampler(hdf5_path, train_keys),
        drop_last=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=conf.val_batch_size,
        sampler=BalancedSampler(hdf5_path, val_keys),
        shuffle=False,
        drop_last=True,
        **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=conf.test_batch_size,
        sampler=BalancedSampler(hdf5_path, test_keys),
        shuffle=False,
        drop_last=True,
        **loader_kwargs
    )
    
    return train_loader, val_loader, test_loader
