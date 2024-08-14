import os

from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from treemort.data.dataset import DeadTreeDataset
from treemort.utils.augment import Augmentations
from treemort.utils.datautils import load_and_organize_data, stratify_images_by_patch_count


def prepare_datasets(conf):
    hdf5_file_path = os.path.join(conf.data_folder, conf.hdf5_file)

    image_patch_map = load_and_organize_data(hdf5_file_path)

    train_keys, val_keys, test_keys = stratify_images_by_patch_count(image_patch_map, conf.val_size, conf.test_size)

    train_transform = Augmentations()
    val_transform = None
    test_transform = None

    if conf.model in ["maskformer", "detr", "beit", "dinov2"]:
        image_processor = AutoImageProcessor.from_pretrained(conf.backbone)

        if conf.model == "beit":
            image_processor.size["shortest_edge"] = min(image_processor.size["height"], image_processor.size["width"])
            image_processor.do_pad = False
        elif conf.model in ["maskformer", "dinov2"]:
            image_processor.do_pad = False
    else:
        image_processor = None

    train_dataset = DeadTreeDataset(
        hdf5_file=hdf5_file_path,
        keys=train_keys,
        crop_size=conf.train_crop_size,
        transform=train_transform,
        image_processor=image_processor,
    )
    val_dataset = DeadTreeDataset(
        hdf5_file=hdf5_file_path,
        keys=val_keys,
        crop_size=conf.val_crop_size,
        transform=val_transform,
        image_processor=image_processor,
    )
    test_dataset = DeadTreeDataset(
        hdf5_file=hdf5_file_path,
        keys=test_keys,
        crop_size=conf.test_crop_size,
        transform=test_transform,
        image_processor=image_processor,
    )

    train_loader = DataLoader(train_dataset, batch_size=conf.train_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=conf.val_batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=conf.test_batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader, image_processor
