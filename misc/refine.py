import os
import cv2
import fiona
import torch
import rasterio
import multiprocessing

import numpy as np
import torch.nn as nn
import geopandas as gpd
import albumentations as A

from tqdm import tqdm
from typing import Tuple, List, Dict

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from sklearn.metrics import jaccard_score
from shapely.geometry import shape
from rasterio.features import rasterize
from albumentations.pytorch import ToTensorV2
# from lovasz_losses import lovasz_hinge

import warnings; warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)


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
    data_folder: str,
    predictions_folder: str = None,
    image_dir_name: str = "Images",
    geojsons_dir_name: str = "Geojsons",
    predictions_dir_name: str = "Predictions",
    file_ext: str = ".geojson",
) -> List[Tuple[str, str, str]]:
    pairs = []
    pred_folder = predictions_folder if predictions_folder else os.path.join(data_folder, predictions_dir_name)
    
    for root, dirs, _ in os.walk(data_folder):
        if {image_dir_name, geojsons_dir_name}.issubset(dirs):
            image_files = {
                os.path.splitext(f)[0]: os.path.join(root, image_dir_name, f)
                for f in os.listdir(os.path.join(root, image_dir_name))
                if f.endswith(".tiff") or f.endswith(".tif")  # Assuming images are in TIFF format
            }
            gt_files = {
                os.path.splitext(f)[0]: os.path.join(root, geojsons_dir_name, f)
                for f in os.listdir(os.path.join(root, geojsons_dir_name))
                if f.endswith(file_ext)
            }
            pred_files = {
                os.path.splitext(f)[0]: os.path.join(pred_folder, f)
                for f in os.listdir(pred_folder)
                if f.endswith(file_ext)
            }
            # Match files by name
            common_files = image_files.keys() & gt_files.keys() & pred_files.keys()
            for fname in common_files:
                pairs.append((image_files[fname], gt_files[fname], pred_files[fname]))
    return pairs


def split_image_into_patches(mask, patch_size=(256, 256), stride=(128, 128)):
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


def ensure_unique_ids(gdf, id_column="id"):
    if id_column in gdf.columns:
        gdf[id_column] = range(1, len(gdf) + 1)  # Assign unique IDs
    else:
        gdf.insert(0, id_column, range(1, len(gdf) + 1))
    return gdf


def get_shape_transform(img_path):
    with rasterio.open(img_path) as src:
        image_shape = (src.height, src.width)
        transform = src.transform
    return image_shape, transform

    
def validate_geometry(geom):
    try:
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.geom_type not in ["Polygon", "MultiPolygon"]:
            return None
        return geom
    except Exception as e:
        print(f"Error validating geometry: {e}")
        return None


def validate_and_filter_geometries(gdf):
    # Apply validation
    gdf["geometry"] = gdf["geometry"].apply(validate_geometry)

    # Filter out empty or invalid geometries
    filtered_gdf = gdf[gdf["geometry"].notnull() & ~gdf["geometry"].is_empty]

    #if filtered_gdf.empty:
    #    print("Warning: No valid geometries after filtering.")
    return filtered_gdf


def load_geodata_with_unique_ids(file_path: str) -> gpd.GeoDataFrame:
    with fiona.open(file_path) as src:
        features = [
            {
                **feature,
                "id": str(idx),  # Ensure unique IDs
                "geometry": shape(feature["geometry"]) if feature["geometry"] else None,
            }
            for idx, feature in enumerate(src)
        ]
        geometries = [feature["geometry"] for feature in features]
        gdf = gpd.GeoDataFrame(features, geometry=geometries, crs=src.crs)
        gdf["geometry"] = gdf["geometry"].apply(validate_geometry)
        return gdf[gdf["geometry"].notnull()]
    

class DiceLoss(nn.Module):
    def forward(self, outputs, targets):
        smooth = 1.0  # To prevent division by zero
        outputs_flat = outputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (outputs_flat * targets_flat).sum()
        dice = (2.0 * intersection + smooth) / (outputs_flat.sum() + targets_flat.sum() + smooth)
        return 1 - dice
    

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, outputs, targets):
        bce_loss = self.bce(outputs, targets)
        dice_loss = self.dice(outputs, targets)
        return bce_loss + dice_loss


class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, weight_background=0.1, weight_segment=0.9):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.dice = DiceLoss()
        self.weight_background = weight_background
        self.weight_segment = weight_segment

    def forward(self, outputs, targets):
        bce_loss = self.bce(outputs, targets)
        class_weights = torch.where(targets > 0, self.weight_segment, self.weight_background)
        weighted_bce_loss = (class_weights * bce_loss).mean()
        dice_loss = self.dice(outputs, targets)
        return weighted_bce_loss + dice_loss
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        bce = F.binary_cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()
    

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs, targets):
        smooth = 1.0
        outputs_flat = outputs.view(-1)
        targets_flat = targets.view(-1)

        true_pos = (outputs_flat * targets_flat).sum()
        false_neg = ((1 - outputs_flat) * targets_flat).sum()
        false_pos = (outputs_flat * (1 - targets_flat)).sum()

        tversky_index = (true_pos + smooth) / (
            true_pos + self.alpha * false_neg + self.beta * false_pos + smooth
        )
        return 1 - tversky_index
    

def smooth_labels(targets, smoothing=0.1):
    return targets * (1 - smoothing) + smoothing / 2  # Smooth toward 0.5


class SegmentationRefinementDataset(Dataset):
    def __init__(self, parent_folder, predictions_folder, transform=None, patch_size=(256, 256), stride=(256, 256), augment=False):
        self.file_pairs = find_file_pairs(parent_folder, predictions_folder)
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.patches = []
        self.patch_weights = []
        self.augment = augment
        self.augmentations = self.get_augmentations() if augment else None
        self.prepare_patches()

    def prepare_patches(self):
        min_segment_threshold = 100  # Threshold for segment pixel count
        self.patches = []  # Reset patches
        self.patch_weights = []  # Reset weights

        for idx in range(len(self.file_pairs)):
            img_path, pred_path, gt_path = self.file_pairs[idx]

            image_shape, transform = get_shape_transform(img_path)
            pred_mask = self.load_geojsons_to_mask(pred_path, image_shape, transform)
            gt_mask = self.load_geojsons_to_mask(gt_path, image_shape, transform)

            pred_mask_patches = split_image_into_patches(pred_mask, self.patch_size, self.stride)
            gt_mask_patches = split_image_into_patches(gt_mask, self.patch_size, self.stride)

            for pred_patch, gt_patch in zip(pred_mask_patches, gt_mask_patches):
                segment_pixel_count = np.sum(gt_patch)

                # Add only valid patches
                if segment_pixel_count > min_segment_threshold:
                    pred_patch_tensor = torch.from_numpy(pred_patch).float().unsqueeze(0)
                    gt_patch_tensor = torch.from_numpy(gt_patch).float().unsqueeze(0)

                    # Augmentation if enabled
                    if self.augment and self.augmentations:
                        pred_patch, gt_patch = self.apply_augmentations(pred_patch, gt_patch)

                    self.patches.append((pred_patch_tensor, gt_patch_tensor))

                    # Add corresponding weight
                    self.patch_weights.append(1.0)  # High weight for valid patches
                else:
                    # Skip invalid patches
                    continue

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        pred_patch, gt_patch = self.patches[idx]

        smoothing = 0.1
        gt_patch_smoothed = smooth_labels(gt_patch, smoothing)

        return pred_patch, gt_patch_smoothed

    def get_patch_weights(self):
        return self.patch_weights
    
    def apply_augmentations(self, pred_patch, gt_patch):
        if isinstance(pred_patch, torch.Tensor):
            pred_patch = pred_patch.numpy()
        if isinstance(gt_patch, torch.Tensor):
            gt_patch = gt_patch.numpy()

        pred_patch = pred_patch.astype(np.uint8)
        gt_patch = gt_patch.astype(np.uint8)

        augmented = self.augmentations(image=pred_patch, mask=gt_patch)
        return augmented["image"], augmented["mask"]

    def get_augmentations(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),  # Horizontal flip
            A.VerticalFlip(p=0.5),  # Vertical flip
            A.RandomRotate90(p=0.5),  # Rotate by 90 degrees
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, 
                interpolation=cv2.INTER_NEAREST, p=0.5
            ),  # Shifting, scaling, and rotating
            A.ElasticTransform(
                alpha=1, sigma=50, interpolation=cv2.INTER_NEAREST, p=0.5
            ),
            A.RandomCrop(height=self.patch_size[0], width=self.patch_size[1], p=0.5),  # Random cropping
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})

    @staticmethod
    def load_geojsons_to_mask(geojson_path, image_shape, transform):
        gdf = load_geodata_with_unique_ids(geojson_path)

        gdf = validate_and_filter_geometries(gdf)

        if gdf.empty:
            # print(f"Warning: GeoJSON file {geojson_path} has no valid geometries. Returning an empty mask.")
            return np.zeros(image_shape, dtype=np.uint8)

        shapes = [(geom, 1) for geom in gdf.geometry]

        mask = rasterize(shapes, out_shape=image_shape, transform=transform, fill=0, all_touched=True)

        return mask
    

def random_erode_dilate(mask, iterations=1):
    if mask is None or mask.size == 0:
        raise ValueError("Input mask is empty.")
    
    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    
    if np.random.rand() > 0.5:
        return cv2.dilate(mask, kernel, iterations=iterations)
    else:
        return cv2.erode(mask, kernel, iterations=iterations)


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


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            SEBlock(out_channels)
        )
        # Shortcut to align input and output channels
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)  # Align dimensions if needed
        return self.relu(shortcut + self.conv(x))
    

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.final_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1x1(x))
        x2 = self.relu(self.conv3x3_1(x))
        x3 = self.relu(self.conv3x3_2(x))
        x4 = self.relu(self.conv3x3_3(x))
        
        # Global average pooling
        x5 = self.global_avg_pool(x)
        x5 = self.conv1x1_pool(x5)
        x5 = nn.functional.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate all features
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.final_conv(x)
    

class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super(DilatedResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
            SEBlock(out_channels)
        )
        # 1x1 convolution for shortcut to align input and output channels
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)  # Align dimensions if needed
        return self.relu(shortcut + self.conv(x))
    

class ComboLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, outputs, targets):
        bce_loss = self.bce(outputs, targets)
        dice_loss = self.dice(outputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
    

def compute_distance_map(mask):
    from scipy.ndimage import distance_transform_edt
    inverse_mask = 1 - mask
    return distance_transform_edt(mask) + distance_transform_edt(inverse_mask)

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        outputs = outputs.sigmoid()  # Ensure outputs are in [0, 1]
        targets = targets.float()
        
        # Compute distance maps
        distance_map = compute_distance_map(targets.cpu().numpy())
        distance_map = torch.from_numpy(distance_map).to(outputs.device)
        
        boundary_loss = (distance_map * (outputs - targets) ** 2).mean()
        return boundary_loss
    
'''
class LovaszHingeLoss(nn.Module):
    def forward(self, outputs, targets):
        outputs = outputs.squeeze(1)  # Remove channel dimension
        targets = targets.squeeze(1).float()
        return lovasz_hinge(outputs, targets)
''' 

class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = ResidualBlock(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DilatedResidualBlock(256, 512, dilation=2)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DilatedResidualBlock(512, 1024, dilation=4)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(128, 64)

        self.final = nn.Conv2d(64, output_channels, kernel_size=1)

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


class UNetWithDeepSupervision(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UNetWithDeepSupervision, self).__init__()
        # Encoder
        self.enc1 = ResidualBlock(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DilatedResidualBlock(128, 256, dilation=1)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DilatedResidualBlock(256, 512, dilation=2)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck with ASPP
        self.bottleneck = ASPP(512, 1024)

        # Decoder with auxiliary outputs
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(1024, 512)
        self.aux4 = nn.Conv2d(512, output_channels, kernel_size=1)  # Auxiliary output

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(512, 256)
        self.aux3 = nn.Conv2d(256, output_channels, kernel_size=1)  # Auxiliary output

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(256, 128)
        self.aux2 = nn.Conv2d(128, output_channels, kernel_size=1)  # Auxiliary output

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(128, 64)
        self.final = nn.Conv2d(64, output_channels, kernel_size=1)

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

        # Bottleneck with ASPP
        bottleneck = self.bottleneck(pool4)

        # Decoder path with auxiliary outputs
        up4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))
        aux4 = self.aux4(dec4)  # Auxiliary output

        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        aux3 = self.aux3(dec3)  # Auxiliary output

        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        aux2 = self.aux2(dec2)  # Auxiliary output

        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        final = self.final(dec1)

        # Interpolate auxiliary outputs to match final output resolution
        aux4_resized = nn.functional.interpolate(aux4, size=final.shape[2:], mode="bilinear", align_corners=False)
        aux3_resized = nn.functional.interpolate(aux3, size=final.shape[2:], mode="bilinear", align_corners=False)
        aux2_resized = nn.functional.interpolate(aux2, size=final.shape[2:], mode="bilinear", align_corners=False)

        return final, aux4_resized, aux3_resized, aux2_resized
    

class UNetWithASPP(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UNetWithASPP, self).__init__()
        # Encoder
        self.enc1 = ResidualBlock(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DilatedResidualBlock(128, 256, dilation=1)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DilatedResidualBlock(256, 512, dilation=2)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck (ASPP)
        self.bottleneck = ASPP(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(128, 64)

        self.final = nn.Conv2d(64, output_channels, kernel_size=1)

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

        # Bottleneck (ASPP)
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

        # Final output
        return self.final(dec1)
    

def load_model_if_exists(model, model_save_path, device):
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
        print(f"Loaded model weights from {model_save_path}")
    else:
        print("No previous model found. Starting training from scratch.")
    model.to(device)
    return model


def requires_sigmoid(criterion):
    return isinstance(criterion, (nn.BCELoss, DiceLoss, WeightedBCEDiceLoss, ComboLoss))


def train_refinement_model(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs, device):
    model = load_model_if_exists(model, model_save_path, device)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for inputs, targets in train_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                final_output, aux4, aux3, aux2 = model(inputs)

                if requires_sigmoid(criterion):
                    final_output = torch.sigmoid(final_output)
                    aux4 = torch.sigmoid(aux4)
                    aux3 = torch.sigmoid(aux3)
                    aux2 = torch.sigmoid(aux2)
                    
                final_loss = criterion(final_output, targets)
                aux_loss4 = criterion(aux4, targets)
                aux_loss3 = criterion(aux3, targets)
                aux_loss2 = criterion(aux2, targets)

                total_loss = final_loss + 0.4 * aux_loss4 + 0.3 * aux_loss3 + 0.2 * aux_loss2
                
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()
                pbar.set_postfix(loss=total_loss.item())
                pbar.update(1)

        avg_train_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass through the model
                final_output, aux4, aux3, aux2 = model(inputs)

                # Apply sigmoid if required by the criterion
                if requires_sigmoid(criterion):
                    final_output = torch.sigmoid(final_output)
                    aux4 = torch.sigmoid(aux4)
                    aux3 = torch.sigmoid(aux3)
                    aux2 = torch.sigmoid(aux2)

                # Compute losses for all outputs
                final_loss = criterion(final_output, targets)
                aux_loss4 = criterion(aux4, targets)
                aux_loss3 = criterion(aux3, targets)
                aux_loss2 = criterion(aux2, targets)

                # Combine the losses with respective weights
                loss = final_loss + 0.4 * aux_loss4 + 0.3 * aux_loss3 + 0.2 * aux_loss2
                val_loss += loss.item()

                

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            output_dir = os.path.dirname(model_save_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with Validation Loss: {best_val_loss:.4f}")

        scheduler.step(avg_val_loss)
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")


def test_refinement_model(
    test_dataloader,
    model,
    criterion,
    device,
    model_save_path="./output/refine/best_model.pth",
):
    # Load the best model
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    test_loss = 0.0
    all_final_outputs = []
    all_gt_masks = []

    with torch.no_grad():
        with tqdm(total=len(test_dataloader), desc="Testing", unit="batch") as pbar:
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass through the model
                final_output, aux4, aux3, aux2 = model(inputs)

                # Apply sigmoid if the criterion requires probabilities
                if requires_sigmoid(criterion):
                    final_output = torch.sigmoid(final_output)
                    aux4 = torch.sigmoid(aux4)
                    aux3 = torch.sigmoid(aux3)
                    aux2 = torch.sigmoid(aux2)

                # Compute individual losses
                final_loss = criterion(final_output, targets)
                aux_loss4 = criterion(aux4, targets)
                aux_loss3 = criterion(aux3, targets)
                aux_loss2 = criterion(aux2, targets)

                # Combine losses with weights
                total_loss = final_loss + 0.4 * aux_loss4 + 0.3 * aux_loss3 + 0.2 * aux_loss2
                test_loss += total_loss.item()

                # Store predictions and ground truth for evaluation
                all_final_outputs.append(final_output.cpu())
                all_gt_masks.append(targets.cpu())

                pbar.set_postfix(loss=total_loss.item())
                pbar.update(1)

    # Calculate average test loss
    avg_test_loss = test_loss / len(test_dataloader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Flatten all outputs and targets for IoU calculation
    outputs_bin = (torch.cat(all_final_outputs) > 0.5).numpy().astype(int).flatten()
    gt_masks_bin = torch.cat(all_gt_masks).numpy().astype(int).flatten()

    # Find the best threshold for IoU
    best_iou, best_threshold = 0, 0.5
    for t in np.linspace(0.1, 0.9, 9):
        iou = jaccard_score(gt_masks_bin, (outputs_bin > t).astype(int), average="binary")
        if iou > best_iou:
            best_iou, best_threshold = iou, t

    print(f"Best IoU Score: {best_iou:.4f}")
    print(f"Best Threshold: {best_threshold:.4f}")


def get_loss_function(name="Combo", **kwargs):
    if name == "Combo":
        return ComboLoss(**kwargs)
    elif name == "Boundary":
        return BoundaryLoss()
    #elif name == "Lovasz":
    #    return LovaszHingeLoss()
    elif name == "WeightedBCEDice":
        return WeightedBCEDiceLoss(weight_background=kwargs.get('weight_background', 0.1),
                                   weight_segment=kwargs.get('weight_segment', 0.9))
    elif name == "Focal":
        return FocalLoss(alpha=kwargs.get('alpha', 0.25), gamma=kwargs.get('gamma', 2))
    elif name == "Tversky":
        return TverskyLoss(alpha=kwargs.get('alpha', 0.3), beta=kwargs.get('beta', 0.7))
    else:
        raise ValueError(f"Unknown loss function: {name}")
    

if __name__ == "__main__":
    '''
    data_folder = os.getenv("DATA_PATH")
    if not data_folder:
        raise ValueError("DATA_PATH environment variable is not set. Please set it before running the script.")
    
    predictions_folder = os.path.join(data_folder, "Predictions")
    '''
    data_folder = "/Users/anisr/Documents/dead_trees/Finland"
    predictions_folder = "/Users/anisr/Documents/dead_trees/Finland/Predictions"
    
    #full_dataset = SegmentationRefinementDataset(data_folder)

    #train_size = int(0.7 * len(full_dataset))
    #val_size = int(0.15 * len(full_dataset))
    #test_size = len(full_dataset) - train_size - val_size

    # train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_dataset = SegmentationRefinementDataset(data_folder, predictions_folder, augment=True)
    val_dataset = SegmentationRefinementDataset(data_folder, predictions_folder, augment=False)
    test_dataset = SegmentationRefinementDataset(data_folder, predictions_folder, augment=False)

    patch_weights = train_dataset.get_patch_weights()

    if len(patch_weights) != len(train_dataset):
        raise ValueError(f"Patch weights ({len(patch_weights)}) and dataset length ({len(train_dataset)}) do not match!")

    sampler = WeightedRandomSampler(weights=patch_weights, num_samples=len(train_dataset), replacement=True)

    num_workers = max(1, multiprocessing.cpu_count() - 1)  # Use one less than the total CPU cores

    train_dataloader = DataLoader(train_dataset, batch_size=4, sampler=sampler, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=num_workers)

    print("Train DataLoader size:", len(train_dataloader))
    print("Validation DataLoader size:", len(val_dataloader))
    print("Test DataLoader size:", len(test_dataloader))

    model = UNetWithDeepSupervision()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_function_name = "Combo"  # Change to "Boundary", "Lovasz", etc.
    criterion = get_loss_function(loss_function_name, bce_weight=0.6, dice_weight=0.4)

    optimizer = Adam(model.parameters(), lr=0.0005)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000)

    num_epochs = 10
    model_save_path = "./output/refine/best_model.pth"

    train_refinement_model(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs, device)

    test_refinement_model(test_dataloader, model)


''' Usage:

- For Puhti

export TREEMORT_VENV_PATH="/projappl/project_2004205/rahmanan/venv"
export TREEMORT_REPO_PATH="/users/rahmanan/TreeMort"
export TREEMORT_DATA_PATH="/scratch/project_2008436/rahmanan/dead_trees"

sbatch --export=ALL,DATA_PATH="$TREEMORT_DATA_PATH/Finland" $TREEMORT_REPO_PATH/scripts/run_refine.sh

'''
