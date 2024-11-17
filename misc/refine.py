import os
import rasterio
import geopandas as gpd

import torch
import torch.nn as nn

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import jaccard_score
from rasterio.features import rasterize


def find_predictions_and_geojsons_debug(parent_folder):
    predictions_paths = []
    geojsons_paths = []

    for root, dirs, files in os.walk(parent_folder):
        if "Predictions" in dirs:
            predictions_dir = os.path.join(root, "Predictions")
            predictions_files = [
                os.path.join(predictions_dir, f)
                for f in os.listdir(predictions_dir)
                if f.endswith(".geojson")
            ]
            predictions_paths.extend(predictions_files)

        if "Geojsons" in dirs:
            geojsons_dir = os.path.join(root, "Geojsons")
            geojsons_files = [
                os.path.join(geojsons_dir, f)
                for f in os.listdir(geojsons_dir)
                if f.endswith(".geojson")
            ]
            geojsons_paths.extend(geojsons_files)

    return predictions_paths, geojsons_paths


def find_file_pairs(
    parent_folder,
    image_subdir="Images",
    predictions_subdir="Predictions",
    geojsons_subdir="Geojsons",
    file_ext=".geojson",
):
    file_pairs = []

    for root, dirs, files in os.walk(parent_folder):
        if (
            image_subdir in dirs
            and predictions_subdir in dirs
            and geojsons_subdir in dirs
        ):
            img_dir = os.path.join(root, image_subdir)
            pred_dir = os.path.join(root, predictions_subdir)
            gt_dir = os.path.join(root, geojsons_subdir)

            img_files = {
                os.path.splitext(f)[0]: os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if f.endswith((".tif", ".tiff", ".jp2"))
            }
            pred_files = {
                os.path.splitext(f)[0]: os.path.join(pred_dir, f)
                for f in os.listdir(pred_dir)
                if f.endswith(file_ext)
            }
            gt_files = {
                os.path.splitext(f)[0]: os.path.join(gt_dir, f)
                for f in os.listdir(gt_dir)
                if f.endswith(file_ext)
            }

            common_files = img_files.keys() & pred_files.keys() & gt_files.keys()
            for fname in common_files:
                img_path = img_files[fname]
                pred_path = pred_files[fname]
                gt_path = gt_files[fname]

                with rasterio.open(img_path) as src:
                    image_shape = (src.height, src.width)
                    transform = src.transform

                file_pairs.append(
                    (img_path, pred_path, gt_path, image_shape, transform)
                )

    return file_pairs


def split_image_into_patches(mask, patch_size=(256, 256), stride=(256, 256)):
    mask_patches = []
    h, w = mask.shape

    for i in range(0, h - patch_size[0] + 1, stride[0]):
        for j in range(0, w - patch_size[1] + 1, stride[1]):
            mask_patch = mask[i:i + patch_size[0], j:j + patch_size[1]]
            if mask_patch.shape == patch_size:
                mask_patches.append(mask_patch)

    return mask_patches


def add_unique_ids(gdf, id_column="id"):
    gdf[id_column] = range(1, len(gdf) + 1)
    return gdf


class SegmentationRefinementDataset(Dataset):
    def __init__(self, parent_folder, transform=None, patch_size=(256, 256), stride=(256, 256)):
        self.file_pairs = find_file_pairs(parent_folder)
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.patches = []
        self.prepare_patches()

    def prepare_patches(self):
        for idx in range(len(self.file_pairs)):
            img_path, pred_path, gt_path, image_shape, transform = self.file_pairs[idx]

            pred_mask = self.load_geojsons_to_mask(pred_path, image_shape, transform)
            gt_mask = self.load_geojsons_to_mask(gt_path, image_shape, transform)

            pred_mask_patches = split_image_into_patches(pred_mask, self.patch_size, self.stride)
            gt_mask_patches = split_image_into_patches(gt_mask, self.patch_size, self.stride)

            for pred_patch, gt_patch in zip(pred_mask_patches, gt_mask_patches):
                pred_patch_tensor = torch.from_numpy(pred_patch).float().unsqueeze(0)
                gt_patch_tensor = torch.from_numpy(gt_patch).float().unsqueeze(0)

                if self.transform:
                    pred_patch_tensor = self.transform(pred_patch_tensor)
                    gt_patch_tensor = self.transform(gt_patch_tensor)

                self.patches.append((pred_patch_tensor, gt_patch_tensor))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]

    @staticmethod
    def load_geojsons_to_mask(geojson_path, image_shape, transform):
        gdf = gpd.read_file(geojson_path)
        gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))

        gdf = add_unique_ids(gdf)

        shapes = [(geom, 1) for geom in gdf.geometry if geom.is_valid and not geom.is_empty]
        mask = rasterize(shapes, out_shape=image_shape, transform=transform, fill=0, all_touched=True)
        return mask


def pad_to_max_size(tensor, max_height, max_width):
    padded_tensor = torch.zeros((tensor.shape[0], max_height, max_width), dtype=tensor.dtype)
    padded_tensor[:, : tensor.shape[1], : tensor.shape[2]] = tensor
    return padded_tensor


def custom_collate_fn(batch):
    images, pred_masks, gt_masks = zip(*batch)

    max_height = max([img.shape[1] for img in images])
    max_width = max([img.shape[2] for img in images])

    padded_images = [pad_to_max_size(img, max_height, max_width) for img in images]
    padded_pred_masks = [pad_to_max_size(mask, max_height, max_width) for mask in pred_masks]
    padded_gt_masks = [pad_to_max_size(mask, max_height, max_width) for mask in gt_masks]

    return (
        torch.stack(padded_images),
        torch.stack(padded_pred_masks),
        torch.stack(padded_gt_masks),
    )


class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        self.final = nn.Conv2d(64, output_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Use padding to maintain size
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Use padding to maintain size
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(pool4)

        # Decoder path
        up4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))
        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        # Final output layer
        return torch.sigmoid(self.final(dec1))


def train_refinement_model(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for inputs, targets in train_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        avg_train_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with Validation Loss: {best_val_loss:.4f}")


def test_refinement_model(
    test_dataloader,
    model,
):
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    test_loss = 0.0
    all_outputs = []
    all_gt_masks = []

    with torch.no_grad():
        with tqdm(total=len(test_dataloader), desc="Testing", unit="batch") as pbar:
            for (
                images,
                pred_masks,
                gt_masks,
            ) in test_dataloader:
                images, gt_masks = images.to(device), gt_masks.to(device)

                outputs = model(images)

                loss = criterion(outputs, gt_masks)
                test_loss += loss.item()

                all_outputs.append(outputs.cpu())
                all_gt_masks.append(gt_masks.cpu())

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

    avg_test_loss = test_loss / len(test_dataloader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    outputs_bin = (torch.cat(all_outputs) > 0.5).numpy().astype(int).flatten()
    gt_masks_bin = torch.cat(all_gt_masks).numpy().astype(int).flatten()

    iou = jaccard_score(gt_masks_bin, outputs_bin, average="binary")
    print(f"IoU Score: {iou:.4f}")


if __name__ == "__main__":
    data_folder = os.getenv("DATA_PATH")
    if not data_folder:
        raise ValueError("DATA_PATH environment variable is not set. Please set it before running the script.")
    
    full_dataset = SegmentationRefinementDataset(data_folder)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    print("Train DataLoader size:", len(train_dataloader))
    print("Validation DataLoader size:", len(val_dataloader))
    print("Test DataLoader size:", len(test_dataloader))

    model = UNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    best_val_loss = float("inf")
    model_save_path = "output/refine/best_model.pth"

    train_refinement_model(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs, device)

    test_refinement_model(test_dataloader, model)
