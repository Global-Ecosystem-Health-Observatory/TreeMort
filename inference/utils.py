import os
import cv2
import json
import torch
import rasterio
import rasterio.features

import numpy as np

from affine import Affine
from typing import Optional, List, Tuple, Generator, Dict

from rasterio.crs import CRS

from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, binary_dilation
from shapely.geometry import Polygon

from skimage.feature import peak_local_max
from skimage.measure import regionprops, find_contours
from skimage.morphology import erosion, disk, remove_small_objects
from skimage.segmentation import watershed

from treemort.modeling.builder import build_model
from treemort.utils.config import setup
from treemort.utils.logger import configure_logger, get_logger
from treemort.utils.metrics import apply_activation


def initialize_logger(verbosity: str) -> None:
    configure_logger(verbosity=verbosity)


def log_and_raise(logger, exception: Exception):
    logger.error(str(exception))
    raise exception


def expand_path(path):
    return os.path.expandvars(path)


def validate_path(logger, path: str, is_dir: bool = False) -> bool:
    if not os.path.exists(path):
        log_and_raise(logger, FileNotFoundError(f"Path does not exist: {path}"))
    if is_dir and not os.path.isdir(path):
        log_and_raise(logger, NotADirectoryError(f"Expected directory but got: {path}"))
    return True


def load_model(
    config_path: str,
    best_model: str,
    id2label: dict = {0: "alive", 1: "dead"},
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    logger = get_logger()

    validate_path(logger, config_path)
    validate_path(logger, best_model)

    conf = setup(config_path)
    model, *_ = build_model(conf, id2label, device)
    model = model.to(device).eval()

    try:
        model.load_state_dict(torch.load(best_model, map_location=device, weights_only=True))
    except Exception as e:
        log_and_raise(logger, RuntimeError(f"Failed to load model weights: {e}"))

    logger.debug(
        f"Model loaded successfully: {os.path.join(os.path.basename(os.path.dirname(best_model)), os.path.basename(best_model))} (Config: {os.path.basename(config_path)})"
    )
    return model


def load_and_preprocess_image(
    tiff_file: str, nir_rgb_order: Optional[List[int]] = None
) -> Tuple[torch.Tensor, Affine, CRS]:
    logger = get_logger()

    validate_path(logger, tiff_file)

    with rasterio.open(tiff_file) as src:
        image = src.read()
        transform = src.transform
        crs = src.crs
        max_pixel_value = _get_max_pixel_value(src.dtypes[0])

    _validate_image_channels(image, nir_rgb_order)
    nir_rgb_order = nir_rgb_order or list(range(image.shape[0]))

    image = image.astype(np.float32) / max_pixel_value
    image = image[nir_rgb_order] if nir_rgb_order != list(range(image.shape[0])) else image
    image_tensor = torch.tensor(image, dtype=torch.float32)

    if image_tensor.ndim != 3:
        log_and_raise(logger, ValueError(f"Invalid tensor shape: {image_tensor.shape}. Expected 3D tensor (C, H, W)."))

    return image_tensor, transform, crs


def _get_max_pixel_value(bit_depth) -> float:
    logger = get_logger()

    if np.issubdtype(bit_depth, np.integer):
        return np.iinfo(bit_depth).max
    if np.issubdtype(bit_depth, np.floating):
        return 1.0

    log_and_raise(logger, ValueError(f"Unsupported data type: {bit_depth}"))


def _validate_image_channels(image: np.ndarray, nir_rgb_order: Optional[List[int]]) -> None:
    logger = get_logger()

    if image.ndim < 3:
        log_and_raise(logger, ValueError("Image must have at least 3 dimensions (C, H, W)."))
    if nir_rgb_order and max(nir_rgb_order) >= image.shape[0]:
        log_and_raise(logger, ValueError(f"nir_rgb_order indices exceed available channels: {nir_rgb_order}"))


def sliding_window_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    window_size: int = 256,
    stride: int = 128,
    batch_size: int = 1,
    threshold: float = 0.5,
    output_channels: int = 1,
    activation: str = "sigmoid",
) -> torch.Tensor:
    _validate_inference_params(window_size, stride, threshold)

    device = next(model.parameters()).device
    padded_image = _pad_image(image, window_size)

    prediction_map, count_map = _initialize_maps(padded_image.shape[1:], output_channels, device)
    patches, coords = _generate_patches(padded_image, window_size, stride)

    for batch in _batch_patches(patches, coords, batch_size):
        prediction_map, count_map = process_batch(
            batch["patches"], batch["coords"], prediction_map, count_map, model, threshold, activation, device
        )

    return _finalize_prediction(prediction_map, count_map, image.shape, threshold)


def _validate_inference_params(window_size: int, stride: int, threshold: float) -> None:
    logger = get_logger()

    if window_size <= 0 or stride <= 0:
        log_and_raise(logger, ValueError("window_size and stride must be positive integers."))
    if not (0 <= threshold <= 1):
        log_and_raise(logger, ValueError("threshold must be between 0 and 1."))


def _initialize_maps(
    image_shape: Tuple[int, int], output_channels: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w = image_shape
    prediction_map = torch.zeros((output_channels, h, w), dtype=torch.float32, device=device)
    count_map = torch.zeros((h, w), dtype=torch.float32, device=device)
    return prediction_map, count_map


def _generate_patches(
    image: torch.Tensor, window_size: int, stride: int
) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
    h, w = image.shape[1:]
    patches, coords = [], []
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            patch = image[:, y : y + window_size, x : x + window_size].float()
            patches.append(patch)
            coords.append((y, x))
    return patches, coords


def _batch_patches(
    patches: List[torch.Tensor], coords: List[Tuple[int, int]], batch_size: int
) -> Generator[Dict[str, List], None, None]:
    for i in range(0, len(patches), batch_size):
        yield {"patches": patches[i : i + batch_size], "coords": coords[i : i + batch_size]}


def _finalize_prediction(
    prediction_map: torch.Tensor, count_map: torch.Tensor, original_shape: Tuple[int, int, int], threshold: float
) -> torch.Tensor:
    no_contribution_mask = count_map == 0
    count_map[no_contribution_mask] = 1

    final_prediction = prediction_map / count_map.unsqueeze(0)
    final_prediction[:, no_contribution_mask] = 0

    final_prediction[0] = torch.clamp(final_prediction[0], 0, 1)

    _, original_h, original_w = original_shape
    return final_prediction[:, :original_h, :original_w]


def process_batch(
    patches: list[torch.Tensor],
    coords: list[tuple[int, int]],
    prediction_map: torch.Tensor,
    count_map: torch.Tensor,
    model: torch.nn.Module,
    threshold: float,
    activation: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    logger = get_logger()

    _validate_batch_inputs(patches, coords, threshold)

    predictions = _infer_patches(patches, model, activation, device)

    for i, (y, x) in enumerate(coords):
        binary_confidence = predictions[i, 0]

        _update_maps(prediction_map, count_map, binary_confidence, threshold, y, x)

    return prediction_map, count_map


def _validate_batch_inputs(patches: list[torch.Tensor], coords: list[tuple[int, int]], threshold: float) -> None:
    logger = get_logger()

    if not patches or not coords:
        log_and_raise(logger, ValueError("Patches and coordinates cannot be empty."))
    if not (0 <= threshold <= 1):
        log_and_raise(logger, ValueError("Threshold must be between 0 and 1."))


def _infer_patches(patches: list[torch.Tensor], model: torch.nn.Module, activation: str, device: torch.device) -> torch.Tensor:
    batch_tensor = torch.stack(patches).to(device)

    with torch.no_grad():
        outputs = model(batch_tensor)
        predictions = apply_activation(outputs[:, 0:1, ...], activation=activation)
        
    return predictions


def _update_maps(
    prediction_map: torch.Tensor,
    count_map: torch.Tensor,
    binary_confidence: torch.Tensor,
    threshold: float,
    y: int,
    x: int,
) -> None:
    binary_mask = (binary_confidence >= threshold).float()

    prediction_map[:, y : y + binary_confidence.shape[0], x : x + binary_confidence.shape[1]] += binary_confidence

    count_map[y : y + binary_confidence.shape[0], x : x + binary_confidence.shape[1]] += binary_mask


def threshold_prediction_map(prediction_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    binary_mask = (prediction_map >= threshold)
    return binary_mask


def _pad_image(image: torch.Tensor, window_size: int) -> torch.Tensor:
    logger = get_logger()

    if image.ndim != 3:
        log_and_raise(logger, ValueError("Image must be a 3D tensor with shape (C, H, W)."))

    c, h, w = image.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    padded_image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0)

    logger.debug(f"Padded image from shape {(h, w)} to {(h + pad_h, w + pad_w)}.")
    return padded_image


def _apply_transform(contour: np.ndarray, transform: Affine) -> np.ndarray:
    transformed = np.array([list(transform * tuple(pt)) for pt in contour])
    return transformed


def compute_watershed(segment_map, conf):
    binary_seg = (segment_map > conf.segment_threshold).astype(np.uint8)
    binary_seg = remove_small_objects(binary_seg.astype(bool), min_size=conf.min_area_pixels).astype(np.uint8)

    smoothed_segment_map = gaussian_filter(binary_seg, sigma=conf.blur_sigma)

    mask_grown = binary_dilation(binary_seg, disk(conf.dilation_radius))

    local_max = peak_local_max(
        smoothed_segment_map,
        min_distance=conf.min_distance,
        exclude_border=False,
        labels=mask_grown
    )

    markers = np.zeros(segment_map.shape, dtype=np.int32)
    for i, (y, x) in enumerate(local_max):
        markers[y, x] = i + 1  # Ensure marker values are unique and nonzero

    labels_ws = watershed(-smoothed_segment_map, markers, mask=mask_grown)

    new_labels = _postprocess_labels(labels_ws, min_region_size=conf.min_area_pixels, dilation_radius=conf.dilation_radius)

    return new_labels


def _postprocess_labels(labels_ws, min_region_size=50, dilation_radius=1):
    unique_labels, counts = np.unique(labels_ws, return_counts=True)
    label_sizes = dict(zip(unique_labels, counts))

    new_labels = labels_ws.copy()

    for lbl in unique_labels:
        if lbl == 0:  # Skip background
            continue
        if label_sizes[lbl] < min_region_size:  # If region is too small
            mask = labels_ws == lbl
            
            dilated = binary_dilation(mask, disk(dilation_radius))
            boundary_labels = labels_ws[dilated & (labels_ws != lbl)]

            if boundary_labels.size > 0:
                unique_neighbors, neighbor_counts = np.unique(boundary_labels, return_counts=True)
                valid_neighbors = unique_neighbors[unique_neighbors != 0]
                valid_counts = neighbor_counts[unique_neighbors != 0]

                if valid_counts.size > 0:
                    target_label = valid_neighbors[np.argmax(valid_counts)]
                    new_labels[mask] = target_label

    return new_labels

def extract_ellipses(labels_ws, transform: Affine, conf, num_points=100):
    features = []

    for region in regionprops(labels_ws):
        if region.area < conf.min_area_pixels:
            continue

        mask = labels_ws == region.label
        eroded_mask = erosion(mask, disk(conf.erosion_radius))

        eroded_contours = find_contours(eroded_mask, level=0.5)
        if not eroded_contours:
            continue
        eroded_contour = max(eroded_contours, key=lambda c: c.shape[0])
        pts = np.array([[pt[1], pt[0]] for pt in eroded_contour], dtype=np.float32)
        if len(pts) < 5:
            continue

        ellipse = cv2.fitEllipse(pts)
        center = ellipse[0]  # (x0, y0)
        axes = ellipse[1]  # (major, minor)
        angle_deg = ellipse[2]
        orientation = np.deg2rad(angle_deg)

        # Compute semi-axes with tightness factor.
        a = (axes[0] / 2.0) * conf.tightness
        b = (axes[1] / 2.0) * conf.tightness

        # Generate ellipse points.
        t = np.linspace(0, 2 * np.pi, num_points)
        ellipse_x = center[0] + a * np.cos(t) * np.cos(orientation) - b * np.sin(t) * np.sin(orientation)
        ellipse_y = center[1] + a * np.cos(t) * np.sin(orientation) + b * np.sin(t) * np.cos(orientation)
        ellipse_coords = list(zip(ellipse_x.tolist(), ellipse_y.tolist()))

        # Ensure the polygon is closed.
        if ellipse_coords[0] != ellipse_coords[-1]:
            ellipse_coords.append(ellipse_coords[0])

        # Convert ellipse coordinates to spatial coordinates.
        ellipse_arr = np.array(ellipse_coords)
        transformed_ellipse = _apply_transform(ellipse_arr, transform)

        # After obtaining transformed_ellipse as a NumPy array of coordinates:
        ellipse_poly = Polygon(transformed_ellipse.tolist())

        def is_valid_geometry(geometry, area, convex_hull):
            aspect_ratio = convex_hull.length / (4 * np.sqrt(area)) if area > 0 else float("inf")
            solidity = area / convex_hull.area if convex_hull.area > 0 else 0
            return area >= conf.min_area and aspect_ratio <= conf.max_aspect_ratio and solidity >= conf.min_solidity
        
        if ellipse_poly.is_valid and not ellipse_poly.is_empty:
            convex_hull = ellipse_poly.convex_hull
            area = ellipse_poly.area
            if is_valid_geometry(ellipse_poly, area, convex_hull):
                ellipse_center_geo = list(_apply_transform(np.array([center]), transform)[0])
                feature = {
                    "type": "Feature",
                    "properties": {
                        "region_label": region.label,
                        "area": area,
                        "ellipse_center": ellipse_center_geo,
                        "ellipse_axes": axes,        # [major, minor] in pixels (or spatial units if desired)
                        "ellipse_angle_deg": angle_deg
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [transformed_ellipse.tolist()]
                    }
                }
                features.append(feature)

    return features


def extract_contours(binary_mask: np.ndarray, transform: Affine) -> List[Dict]:
    logger = get_logger()

    if isinstance(binary_mask, torch.Tensor):
        binary_mask = binary_mask.cpu().numpy()

    if binary_mask.ndim == 3:  # (C, H, W) -> Take the first channel
        binary_mask = binary_mask[0]
    elif binary_mask.ndim > 3:  # (N, C, H, W) -> Take the first image and channel
        binary_mask = binary_mask[0, 0]

    binary_mask = (binary_mask > 0).astype(np.uint8)

    if binary_mask.ndim != 2 or not np.issubdtype(binary_mask.dtype, np.integer):
        log_and_raise(logger, ValueError("binary_mask must be a 2D binary integer array."))

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    reshaped_contours = [contour.reshape(-1, 2) for contour in contours]
    print(f"Extracted {len(reshaped_contours)} contours from the binary mask.")
    
    features = []
    skipped_contours = 0
    for contour in reshaped_contours:
        if len(contour) >= 3:
            transformed_contour = _apply_transform(contour, transform)
            if not np.array_equal(transformed_contour[0], transformed_contour[-1]):
                transformed_contour = np.vstack([transformed_contour, transformed_contour[0]])

            polygon = Polygon(transformed_contour)

            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            if polygon.is_valid and not polygon.is_empty:
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [transformed_contour.tolist()]},
                        "properties": {},
                    }
                )
            else:
                skipped_contours += 1
        else:
            skipped_contours += 1

    logger.debug(f"Processed {len(features)} features, skipped {skipped_contours} contours.")
    return features


def extract_contours_from_labels(label_map: np.ndarray, transform: Affine) -> List[Dict]:
    logger = get_logger()
    
    if isinstance(label_map, torch.Tensor):
        label_map = label_map.cpu().numpy()

    # If label_map has more than 2 dimensions, assume it's in (C, H, W) or (N, C, H, W)
    if label_map.ndim == 3:
        logger.debug("Label map has 3 dimensions; selecting the first channel.")
        label_map = label_map[0]
    elif label_map.ndim > 3:
        logger.debug("Label map has more than 3 dimensions; selecting the first image and channel.")
        label_map = label_map[0, 0]

    if label_map.ndim != 2 or not np.issubdtype(label_map.dtype, np.integer):
        log_and_raise(logger, ValueError("label_map must be a 2D integer array representing labels."))

    features = []
    unique_labels = np.unique(label_map)
    skipped_labels = 0

    for label in unique_labels:
        if label == 0:
            continue

        mask = (label_map == label).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Take the largest contour as representative
            contour = max(contours, key=cv2.contourArea)
            contour = contour.reshape(-1, 2)
            
            transformed_contour = _apply_transform(contour, transform)
            
            # Ensure the contour is closed
            if not np.array_equal(transformed_contour[0], transformed_contour[-1]):
                transformed_contour = np.vstack([transformed_contour, transformed_contour[0]])
                
            polygon = Polygon(transformed_contour)

            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            if polygon.is_valid and not polygon.is_empty:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [transformed_contour.tolist()]},
                    "properties": {"label": int(label)}
                })
            else:
                skipped_labels += 1
        else:
            skipped_labels += 1

    logger.debug(f"Extracted {len(features)} features from {len(unique_labels)-1} segments, skipped {skipped_labels} labels.")
    return features


def save_geojson(features, filename, crs, transform, name="FittedEllipses"):
    logger = get_logger()

    geojson_crs = None
    if crs:
        if getattr(crs, "is_epsg_code", False):
            epsg_code = crs.to_epsg()
            geojson_crs = {"type": "name", "properties": {"name": f"EPSG:{epsg_code}"}}
        else:
            logger.warning("Warning: CRS is not in EPSG format; setting CRS to null in GeoJSON.")

    geojson = {
        "type": "FeatureCollection",
        "name": name,
        "crs": geojson_crs,
        "metadata": {"transform": tuple(transform)},
        "features": features,
    }

    with open(filename, "w") as f:
        json.dump(geojson, f, indent=2)
    logger.debug(f"GeoJSON saved to {filename}")
