import random

from pathlib import Path
from torch.utils.data import DataLoader

from treemort.data.dataset import DeadTreeDataset
from treemort.data.sampler import ClassPrioritizedSampler
from treemort.data.image_processing import get_image_processor

from treemort.utils.augment import Augmentations
from treemort.utils.datautils import (
    load_and_organize_data,
    stratify_images_by_patch_count,
)


def prepare_datasets(conf):
    hdf5_path = Path(conf.data_folder) / conf.hdf5_file

    image_patch_map = load_and_organize_data(hdf5_path)

    _, _, test_keys = stratify_images_by_patch_count(image_patch_map, 0.0, 1.0)

    random.seed(None)  # makes loader non-deterministic

    image_processor = get_image_processor(conf.model, conf.backbone)

    test_dataset = DeadTreeDataset(
        hdf5_file=hdf5_path,
        keys=test_keys,
        crop_size=conf.test_crop_size,
        transform=None,
        image_processor=image_processor,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=conf.test_batch_size,
        sampler=ClassPrioritizedSampler(hdf5_path, test_keys, 1, sample_ratio=1.0),
        shuffle=False,
        drop_last=True,
    )

    return [], [], test_loader
