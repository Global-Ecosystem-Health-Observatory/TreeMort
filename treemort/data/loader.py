import random
import h5py

from pathlib import Path
from torch.utils.data import DataLoader, SequentialSampler

from treemort.data.dataset import DeadTreeDataset
from treemort.data.sampler import BalancedSampler
from treemort.data.image_processing import get_image_processor

from treemort.utils.augment import Augmentations
from treemort.utils.datautils import load_and_organize_data, stratify_images_by_patch_count, stratify_images_by_region


def prepare_datasets(conf):
    hdf5_path = Path(conf.data_folder).parent / conf.hdf5_file

    image_patch_map = load_and_organize_data(hdf5_path)

    random.seed(42)  # makes loader deterministic

    # Special test-only mode: skip splitting, use all as test set, no train/val
    if getattr(conf, "test_only", False):
        image_processor = get_image_processor(conf.model, conf.backbone)
    
        # Flatten and normalize patch-level keys from the image->patches map
        raw_patch_items = [p for patches in image_patch_map.values() for p in patches]

        normalized_keys = []
        for item in raw_patch_items:
            # If item is (key, *extra), unwrap the first element
            key = item[0] if isinstance(item, (tuple, list)) else item
            # Coerce to str if necessary
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            elif not isinstance(key, str):
                key = str(key)
            normalized_keys.append(key)

        # De-duplicate while preserving order
        seen = set()
        dedup_keys = []
        for k in normalized_keys:
            if k not in seen:
                seen.add(k)
                dedup_keys.append(k)

        # Keep only keys that exist in the HDF5 file to prevent KeyErrors
        with h5py.File(hdf5_path, "r") as hf:
            valid_keys = [k for k in dedup_keys if k in hf]
        missing = len(dedup_keys) - len(valid_keys)
        if missing > 0:
            print(f"[test_only][warn] Skipped {missing} keys not present in HDF5.")

        test_dataset = DeadTreeDataset(
            hdf5_file=hdf5_path,
            keys=valid_keys,
            crop_size=conf.test_crop_size,
            transform=None,
            image_processor=image_processor,
        )
    
        test_loader = DataLoader(
            test_dataset,
            batch_size=conf.test_batch_size,
            sampler=SequentialSampler(test_dataset),
            shuffle=False,
            drop_last=False,
        )
    
        print(f"[test_only] Using {len(valid_keys)} HDF5 patch keys for testing.")
        return None, None, test_loader

    train_keys, val_keys, test_keys = stratify_images_by_region(
        image_patch_map, val_ratio=conf.val_size, test_ratio=conf.test_size
    )

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
        train_dataset, 
        batch_size=conf.train_batch_size, 
        sampler=BalancedSampler(hdf5_path, train_keys), 
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=conf.val_batch_size,
        sampler=BalancedSampler(hdf5_path, val_keys),
        shuffle=False,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=conf.test_batch_size,
        sampler=SequentialSampler(test_dataset),  # ensures deterministic full pass
        shuffle=False,
        drop_last=False                           # keep all samples
    )

    return train_loader, val_loader, test_loader
