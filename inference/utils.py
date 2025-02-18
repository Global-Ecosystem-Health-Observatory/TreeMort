import os
import cv2
import torch
import geojson
import rasterio
import rasterio.features

import numpy as np
import geopandas as gpd

from typing import Optional, List, Tuple, Generator, Dict

from affine import Affine
from rasterio.crs import CRS
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.feature import peak_local_max
from skimage.morphology import disk
from skimage.segmentation import watershed
from shapely.geometry import Polygon, shape

from treemort.modeling.builder import build_model
from treemort.utils.config import setup
from treemort.utils.logger import configure_logger, get_logger


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
) -> torch.Tensor:
    _validate_inference_params(window_size, stride, threshold)

    device = next(model.parameters()).device
    padded_image = pad_image(image, window_size)

    prediction_map, count_map = _initialize_maps(padded_image.shape[1:], output_channels, device)
    patches, coords = _generate_patches(padded_image, window_size, stride)

    for batch in _batch_patches(patches, coords, batch_size):
        prediction_map, count_map = process_batch(
            batch["patches"], batch["coords"], prediction_map, count_map, model, threshold, device
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
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    logger = get_logger()

    _validate_batch_inputs(patches, coords, threshold)

    predictions = _infer_patches(patches, model, device)

    for i, (y, x) in enumerate(coords):
        binary_confidence = predictions[i, 0]
    
        _update_maps(
            prediction_map, count_map, binary_confidence, threshold, y, x
        )

    return prediction_map, count_map


def _validate_batch_inputs(patches: list[torch.Tensor], coords: list[tuple[int, int]], threshold: float) -> None:
    logger = get_logger()

    if not patches or not coords:
        log_and_raise(logger, ValueError("Patches and coordinates cannot be empty."))
    if not (0 <= threshold <= 1):
        log_and_raise(logger, ValueError("Threshold must be between 0 and 1."))


def _infer_patches(patches: list[torch.Tensor], model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    batch_tensor = torch.stack(patches).to(device)

    with torch.no_grad():
        outputs = model(batch_tensor)

        predictions = torch.sigmoid(outputs[:, 0:1, ...])

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


def threshold_prediction_map(prediction_map: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    logger = get_logger()

    if not isinstance(prediction_map, torch.Tensor):
        log_and_raise(logger, ValueError("prediction_map must be a torch.Tensor"))
    if not (0 <= threshold <= 1):
        log_and_raise(logger, ValueError("threshold must be between 0 and 1"))
    if not isinstance(threshold, (float, int)):
        log_and_raise(logger, ValueError("threshold must be a float or an int"))

    binary_mask = (prediction_map >= threshold).to(dtype=torch.float32)
    return binary_mask


def extract_contours(binary_mask: np.ndarray) -> List[np.ndarray]:
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
    logger.debug(f"Extracted {len(reshaped_contours)} contours from the binary mask.")
    return reshaped_contours


def apply_transform(contour: np.ndarray, transform: Affine) -> np.ndarray:
    logger = get_logger()

    if contour.ndim != 2 or contour.shape[1] != 2:
        log_and_raise(logger, ValueError("Contour must be a 2D array with shape (N, 2)."))

    transformed_contour = np.array([transform * (x, y) for x, y in contour])
    logger.debug(f"Applied transform to contour with {len(transformed_contour)} points.")
    return transformed_contour


def contours_to_geojson(contours: List[np.ndarray], transform: Affine, crs: CRS, name: str) -> dict:
    logger = get_logger()

    if not contours:
        logger.warning("Contours list is empty. Returning an empty GeoJSON.")
        return {
            "type": "FeatureCollection",
            "name": name,
            "crs": (
                None
                if not crs
                else {"type": "name", "properties": {"name": f"EPSG:{crs.to_epsg()}" if crs.is_epsg_code else str(crs)}}
            ),
            "features": [],
        }

    geojson_crs = None
    if crs:
        if crs.is_epsg_code:  # If CRS is an EPSG code
            geojson_crs = {"type": "name", "properties": {"name": f"EPSG:{crs.to_epsg()}"}}
        else:
            logger.warning("CRS is not in EPSG format; setting CRS to null in GeoJSON.")

    geojson = {"type": "FeatureCollection", "name": name, "crs": geojson_crs, "features": []}

    skipped_contours = 0
    for contour in contours:
        if len(contour) >= 3:
            transformed_contour = apply_transform(contour, transform)
            if not np.array_equal(transformed_contour[0], transformed_contour[-1]):
                transformed_contour = np.vstack([transformed_contour, transformed_contour[0]])

            polygon = Polygon(transformed_contour)
            if polygon.is_valid and not polygon.is_empty:
                geojson["features"].append(
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

    logger.debug(f"Processed {len(geojson['features'])} features, skipped {skipped_contours} contours.")
    return geojson


def save_geojson(geojson_data: dict, output_path: str) -> None:
    logger = get_logger()

    if not isinstance(geojson_data, dict):
        log_and_raise(logger, ValueError("geojson_data must be a dictionary."))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, "w") as f:
            geojson.dump(geojson_data, f, indent=2, ensure_ascii=False)
        logger.debug(f"GeoJSON successfully saved to {os.path.basename(output_path)}.")
    except OSError as e:
        log_and_raise(logger, RuntimeError(f"Error saving GeoJSON to {os.path.basename(output_path)}: {e}"))


def save_labels_as_geojson(
    labels: np.ndarray,
    transform: Affine,
    crs: CRS,
    output_path: str,
    min_area_threshold: float = 3.0,
    max_aspect_ratio: float = 5.0,
    min_solidity: float = 0.8,
) -> None:
    logger = get_logger()

    def is_valid_geometry(geometry, area, convex_hull):
        aspect_ratio = convex_hull.length / (4 * np.sqrt(area)) if area > 0 else float("inf")
        solidity = area / convex_hull.area if convex_hull.area > 0 else 0
        return area >= min_area_threshold and aspect_ratio <= max_aspect_ratio and solidity >= min_solidity

    geometries = []
    for label_value in np.unique(labels):
        if label_value == 0:  # Skip background
            continue

        mask = labels == label_value
        shapes_gen = rasterio.features.shapes(mask.astype(np.int32), transform=transform)

        for shape_geom, shape_value in shapes_gen:
            if shape_value == 1:
                geometry = shape(shape_geom)
                if geometry.is_valid and not geometry.is_empty:
                    convex_hull = geometry.convex_hull
                    if convex_hull.is_valid and not convex_hull.is_empty:
                        area = geometry.area
                        if is_valid_geometry(geometry, area, convex_hull):
                            geometries.append({"geometry": geometry, "properties": {"label": int(label_value)}})
                        # geometries.append({"geometry": geometry, "properties": {"label": int(label_value)}})

    if not geometries:
        logger.info("No valid geometries found. Creating an empty GeoJSON.")
        empty_geojson = {"type": "FeatureCollection", "features": []}
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(output_path, "w") as f:
                geojson.dump(empty_geojson, f, indent=2, ensure_ascii=False)
            logger.info(f"Empty GeoJSON successfully created at {os.path.basename(output_path)}.")
        except Exception as e:
            log_and_raise(logger, RuntimeError(f"Error saving empty GeoJSON to {os.path.basename(output_path)}: {e}"))
        return

    gdf = gpd.GeoDataFrame.from_features(geometries, crs=crs)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        gdf.to_file(output_path, driver="GeoJSON")
        logger.debug(f"GeoJSON saved with {len(geometries)} geometries to {os.path.basename(output_path)}.")
    except Exception as e:
        log_and_raise(logger, RuntimeError(f"Error saving GeoJSON to {output_path}: {e}"))


def pad_image(image: torch.Tensor, window_size: int) -> torch.Tensor:
    logger = get_logger()

    if image.ndim != 3:
        log_and_raise(logger, ValueError("Image must be a 3D tensor with shape (C, H, W)."))

    c, h, w = image.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    padded_image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0)

    logger.debug(f"Padded image from shape {(h, w)} to {(h + pad_h, w + pad_w)}.")
    return padded_image


def generate_watershed_labels(segment_map, threshold=0.5, min_distance=4, blur_sigma=2, dilation_radius=1):
    if isinstance(segment_map, torch.Tensor):
        segment_map = segment_map.detach().cpu().numpy()
    
    if segment_map.ndim == 3 and segment_map.shape[0] == 1:
        segment_map = segment_map[0]

    binary_mask = segment_map > threshold

    smoothed_prediction_map = gaussian_filter(segment_map, sigma=blur_sigma)

    mask_grown = binary_dilation(binary_mask, disk(dilation_radius))

    local_max = peak_local_max(
        smoothed_prediction_map,
        min_distance=min_distance,
        exclude_border=False,
        labels=mask_grown
    )

    markers = np.zeros(segment_map.shape, dtype=np.int32)
    for i, (y, x) in enumerate(local_max):
        markers[y, x] = i + 1  # Ensure marker values are unique and nonzero

    if np.count_nonzero(markers) < 2:
        return np.zeros_like(segment_map, dtype=np.int32)  # Return empty segmentation if no valid markers

    labels = watershed(-smoothed_prediction_map, markers, mask=mask_grown)

    return labels