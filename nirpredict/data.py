from torch.utils.data import Dataset, DataLoader

from treemort.data.loader import prepare_datasets


class NIRDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset  # This should be a dataset object

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        data = self.original_dataset[idx]

        if isinstance(data, tuple) and len(data) == 2:
            image, label = data
        else:
            raise ValueError(f"Expected a tuple of (image, label) at index {idx}, but got {type(data)}")

        if image.shape[0] != 4:
            raise ValueError(f"Expected image with 4 channels, but got {image.shape[0]} channels.")

        nir_target = image[0, :, :]
        rgb_input = image[1:, :, :]

        return rgb_input, nir_target


def load_data(conf):
    train_dataset, val_dataset, test_dataset = prepare_datasets(conf)

    train_nir_dataset = NIRDataset(train_dataset.dataset)
    val_nir_dataset = NIRDataset(val_dataset.dataset)
    test_nir_dataset = NIRDataset(test_dataset.dataset)

    train_nir_loader = DataLoader(train_nir_dataset, batch_size=16, shuffle=True)
    val_nir_loader = DataLoader(val_nir_dataset, batch_size=16, shuffle=False)
    test_nir_loader = DataLoader(test_nir_dataset, batch_size=16, shuffle=False)

    return train_nir_loader, val_nir_loader, test_nir_loader
