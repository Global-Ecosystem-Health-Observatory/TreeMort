import os
import gc
import cv2
import torch
import geojson
import rasterio
import rasterio.features

import numpy as np
import geopandas as gpd

from affine import Affine
from typing import Optional, List, Tuple, Generator, Dict
from scipy.ndimage import gaussian_filter, label, distance_transform_edt
from shapely.geometry import Polygon, shape
from skimage.feature import peak_local_max
from skimage.morphology import binary_dilation, disk, square
from skimage.segmentation import watershed, relabel_sequential
from rasterio.crs import CRS

from treemort.modeling.builder import build_model
from treemort.utils.config import setup
from treemort.utils.logger import configure_logger, get_logger

from misc.refine import UNetWithDeepSupervision



def initialize_logger(verbosity: str) -> None:
    configure_logger(verbosity=verbosity)


def log_and_raise(logger, exception: Exception):
    logger.error(str(exception))
    raise exception


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

    logger.info(f"Model loaded successfully: {os.path.join(os.path.basename(os.path.dirname(best_model)), os.path.basename(best_model))} (Config: {os.path.basename(config_path)})")
    return model


def load_refine_model(
    model_path: str,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    logger = get_logger()

    validate_path(logger, model_path)

    refine_model = UNetWithDeepSupervision()
    refine_model.to(device).eval()

    try:
        refine_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
        log_and_raise(logger, RuntimeError(f"Failed to load refine model weights: {e}"))

    logger.info(f"Refine model loaded successfully: {os.path.join(os.path.basename(os.path.dirname(model_path)), os.path.basename(model_path))}")
    return refine_model


def load_and_preprocess_image(
    tiff_file: str,
    nir_rgb_order: Optional[List[int]] = None
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
    threshold: float = 0.5
) -> torch.Tensor:
    _validate_inference_params(window_size, stride, threshold)

    device = next(model.parameters()).device
    padded_image = pad_image(image, window_size)

    prediction_map, count_map = _initialize_maps(padded_image.shape[1:], device)
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
    

def _initialize_maps(image_shape: Tuple[int, int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w = image_shape
    prediction_map = torch.zeros((2, h, w), dtype=torch.float32, device=device)
    count_map = torch.zeros((h, w), dtype=torch.float32, device=device)
    return prediction_map, count_map


def _generate_patches(image: torch.Tensor, window_size: int, stride: int) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
    h, w = image.shape[1:]
    patches, coords = [], []
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            patch = image[:, y:y + window_size, x:x + window_size].float()
            patches.append(patch)
            coords.append((y, x))
    return patches, coords


def _batch_patches(
    patches: List[torch.Tensor], coords: List[Tuple[int, int]], batch_size: int
) -> Generator[Dict[str, List], None, None]:
    for i in range(0, len(patches), batch_size):
        yield {"patches": patches[i:i + batch_size], "coords": coords[i:i + batch_size]}


def _finalize_prediction(
    prediction_map: torch.Tensor,
    count_map: torch.Tensor,
    original_shape: Tuple[int, int, int],
    threshold: float
) -> torch.Tensor:
    no_contribution_mask = (count_map == 0)
    count_map[no_contribution_mask] = 1

    final_prediction = prediction_map / count_map
    final_prediction[:, no_contribution_mask] = 0
    final_prediction = torch.clamp(final_prediction, 0, 1)

    _, original_h, original_w = original_shape
    return final_prediction[:, :original_h, :original_w]


def process_batch(
    patches: list[torch.Tensor],
    coords: list[tuple[int, int]],
    prediction_map: torch.Tensor,
    count_map: torch.Tensor,
    model: torch.nn.Module,
    threshold: float,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    logger = get_logger()

    _validate_batch_inputs(patches, coords, threshold)
    
    predictions = _infer_patches(patches, model, device)

    for i, (y, x) in enumerate(coords):
        binary_confidence = predictions[i, 0]
        centroid_confidence = predictions[i, 1]

        _update_maps(
            prediction_map,
            count_map,
            binary_confidence,
            centroid_confidence,
            threshold,
            y,
            x
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
        predictions = torch.sigmoid(outputs)

    return predictions


def _update_maps(
    prediction_map: torch.Tensor,
    count_map: torch.Tensor,
    binary_confidence: torch.Tensor,
    centroid_confidence: torch.Tensor,
    threshold: float,
    y: int,
    x: int
) -> None:
    binary_mask = (binary_confidence >= threshold).float()

    prediction_map[:, y:y + binary_confidence.shape[0], x:x + binary_confidence.shape[1]] += torch.stack(
        [binary_confidence, centroid_confidence]
    )
    count_map[y:y + binary_confidence.shape[0], x:x + binary_confidence.shape[1]] += binary_mask


def pad_tensor(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    logger = get_logger()

    if tensor.ndim != 2 or patch_size <= 0:
        log_and_raise(logger, ValueError("Expected a 2D tensor and a positive patch size."))
    
    height, width = tensor.shape
    pad_height = (patch_size - height % patch_size) % patch_size
    pad_width = (patch_size - width % patch_size) % patch_size
    
    return torch.nn.functional.pad(tensor, (0, pad_width, 0, pad_height))


def prepare_patches(
    mask: torch.Tensor,
    patch_size: int = 256,
    stride: int = 128,
    pad: bool = False
) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
    logger = get_logger()

    if mask.ndim != 2:
        log_and_raise(logger, ValueError("Mask must be a 2D tensor."))
    
    if pad:
        mask = pad_tensor(mask, patch_size)

    h, w = mask.shape
    patches = []
    positions = []

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = mask[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
            positions.append((i, j))

    logger.debug(f"Generated {len(patches)} patches shaped {mask.shape} with patch size {patch_size} and stride {stride}.")
    return patches, positions


def combine_patches(
    patches: List[Tuple[Tuple[int, int], torch.Tensor]],
    image_shape: Tuple[int, int],
    device: torch.device,
    patch_size: int,
    stride: int,
    threshold: float = 0.5
) -> torch.Tensor:
    logger = get_logger()

    if not patches:
        log_and_raise(logger, ValueError("No patches provided for combination."))

    for (i, j), patch in patches:
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            log_and_raise(logger, ValueError(f"Patch at ({i}, {j}) has an invalid size {patch.shape}, expected ({patch_size}, {patch_size})."))
        if i % stride != 0 or j % stride != 0:
            log_and_raise(logger, ValueError(f"Patch at ({i}, {j}) is not aligned with the stride {stride}."))

    expected_height = max(i for i, _ in (pos for pos, _ in patches)) + patch_size
    expected_width = max(j for _, j in (pos for pos, _ in patches)) + patch_size

    logger.debug(f"Expected dimensions: ({expected_height}, {expected_width})")
    logger.debug(f"Provided image_shape: {image_shape}")

    if image_shape != (expected_height, expected_width):
        log_and_raise(logger, ValueError(f"Provided image_shape {image_shape} does not match expected dimensions {(expected_height, expected_width)}."))

    combined_mask = torch.zeros(image_shape, dtype=torch.float32, device=device)
    count_map = torch.zeros(image_shape, dtype=torch.float32, device=device)

    for (i, j), patch in patches:
        combined_mask[i:i + patch.shape[0], j:j + patch.shape[1]] += patch
        count_map[i:i + patch.shape[0], j:j + patch.shape[1]] += 1

    logger.debug(f"Combining {len(patches)} patches into an image of shape {image_shape}.")
    combined_mask = combined_mask / torch.clamp(count_map, min=1.0)
    return (combined_mask > threshold).to(dtype=torch.uint8)


def threshold_prediction_map(prediction_map: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    logger = get_logger()

    if not isinstance(prediction_map, torch.Tensor):
        log_and_raise(logger, ValueError("prediction_map must be a torch.Tensor"))
    if not (0 <= threshold <= 1):
        log_and_raise(logger, ValueError("threshold must be between 0 and 1"))
    if not isinstance(threshold, (float, int)):
        log_and_raise(logger, ValueError("threshold must be a float or an int"))

    binary_mask = (prediction_map >= threshold).to(dtype=torch.uint8)
    return binary_mask


def refine_mask(
    mask: torch.Tensor,
    refine_model: torch.nn.Module,
    device: torch.device,
    patch_size: int = 64,
    stride: int = 64
) -> torch.Tensor:
    logger = get_logger()

    try:
        padded_mask = pad_tensor(mask, patch_size)
        padded_height, padded_width = padded_mask.shape

        patches, patch_positions = prepare_patches(padded_mask, patch_size, stride)
        logger.debug(f"Prepared {len(patch_positions)} patches for refinement.")

        processed_patches = []
        for (i, j), patch in zip(patch_positions, patches):
            input_tensor = patch.to(dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                pred_tensor = torch.sigmoid(refine_model(input_tensor)[0])

            pred_mask = (pred_tensor.squeeze() > 0.5).to(dtype=torch.uint8)

            processed_patches.append(((i, j), pred_mask))

        combined_mask = combine_patches(
            processed_patches,
            (padded_height, padded_width),
            device,
            patch_size,
            stride
        )

        combined_mask = combined_mask[:mask.shape[0], :mask.shape[1]]
        return combined_mask
    except Exception as e:
        logger.error(f"Error refining mask: {e}")
        raise


def generate_watershed_labels(
    prediction_map: np.ndarray, 
    mask: np.ndarray, 
    centroid_map: Optional[np.ndarray] = None, 
    min_distance: int = 4, 
    blur_sigma: float = 2, 
    dilation_radius: int = 1, 
    centroid_threshold: float = 0.7, 
    structuring_element: str = "disk"
) -> np.ndarray:
    logger = get_logger()

    logger.debug(f"Generating watershed labels with blur_sigma={blur_sigma}, min_distance={min_distance}, dilation_radius={dilation_radius}.")

    smoothed_prediction_map = gaussian_filter(prediction_map, sigma=blur_sigma)

    if structuring_element == "disk":
        selem = disk(dilation_radius)
    elif structuring_element == "square":
        selem = square(dilation_radius)
    else:
        log_and_raise(logger, ValueError("Unsupported structuring element type"))

    mask_grown = binary_dilation(mask, selem)

    if centroid_map is not None:
        # peaks = detect_peaks(centroid_map, sigma=blur_sigma, min_distance=min_distance)

        peaks = peak_local_max(
            centroid_map,
            min_distance=min_distance,
            exclude_border=False,
            labels=mask_grown,
            threshold_abs=0.1  # Adjust threshold as needed
        )

        ''' Method 1
        raw_peaks = peak_local_max(
            smoothed_prediction_map,
            min_distance=min_distance,
            exclude_border=False,
            labels=mask_grown,
            threshold_abs=0.1
        )

        peaks = [(y, x) for y, x in raw_peaks if centroid_map[y, x] > centroid_threshold]
        '''

    else:
        peaks = peak_local_max(
            smoothed_prediction_map,
            min_distance=min_distance,
            exclude_border=False,
            labels=mask_grown,
            threshold_abs=0.1  # Adjust threshold as needed
        )

    valid_markers = [(y, x) for y, x in peaks if mask_grown[y, x] == 1]
    if not valid_markers:
        logger.warning("No valid markers detected; returning original mask.")
        return mask
    
    markers = np.zeros_like(prediction_map, dtype=int)
    for i, (y, x) in enumerate(valid_markers):
        markers[y, x] = i + 1

    labels = watershed(-smoothed_prediction_map, markers, mask=mask_grown)
    labels = relabel_sequential(labels)[0]  # Ensure labels are contiguous

    return labels


def extract_contours(binary_mask: np.ndarray) -> List[np.ndarray]:
    logger = get_logger()

    if binary_mask.ndim != 2 or not np.issubdtype(binary_mask.dtype, np.integer):
        log_and_raise(logger, ValueError("binary_mask must be a 2D binary integer array."))

    binary_mask = (binary_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info(f"Extracted {len(contours)} contours from the binary mask.")
    return contours


def apply_transform(contour: np.ndarray, transform: Affine) -> np.ndarray:
    logger = get_logger()

    if contour.ndim != 2 or contour.shape[1] != 2:
        log_and_raise(logger, ValueError("Contour must be a 2D array with shape (N, 2)."))

    transformed_contour = np.array([transform * (x, y) for x, y in contour])
    logger.debug(f"Applied transform to contour with {len(transformed_contour)} points.")
    return transformed_contour


def contours_to_geojson(
    contours: List[np.ndarray], 
    transform: Affine, 
    crs: CRS, 
    name: str
) -> dict:
    logger = get_logger()

    if not contours:
        log_and_raise(logger, ValueError("Contours list is empty. Cannot create GeoJSON."))

    geojson_crs = None
    if crs:
        if crs.is_epsg_code:  # If CRS is an EPSG code
            epsg_code = crs.to_epsg()
            geojson_crs = {
                "type": "name",
                "properties": {
                    "name": f"EPSG:{epsg_code}"
                }
            }
        else:
            logger.warning("CRS is not in EPSG format; setting CRS to null in GeoJSON.")
    
    geojson = {
        "type": "FeatureCollection",
        "name": name,
        "crs": geojson_crs,
        "features": []
    }

    skipped_contours = 0
    for contour in contours:
        if len(contour) >= 3:
            transformed_contour = apply_transform(contour, transform)
            if not np.array_equal(transformed_contour[0], transformed_contour[-1]):
                transformed_contour = np.vstack([transformed_contour, transformed_contour[0]])

            polygon = Polygon(transformed_contour)
            if polygon.is_valid and not polygon.is_empty:
                geojson["features"].append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [transformed_contour.tolist()]
                    },
                    "properties": {}
                })
            else:
                skipped_contours += 1
        else:
            skipped_contours += 1

    logger.info(f"Processed {len(geojson['features'])} features, skipped {skipped_contours} contours.")
    return geojson


def save_geojson(geojson_data: dict, output_path: str) -> None:
    logger = get_logger()

    if not isinstance(geojson_data, dict):
        log_and_raise(logger, ValueError("geojson_data must be a dictionary."))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, "w") as f:
            geojson.dump(geojson_data, f, indent=2, ensure_ascii=False)
        logger.info(f"GeoJSON successfully saved to {output_path}.")
    except OSError as e:
        log_and_raise(logger, RuntimeError(f"Error saving GeoJSON to {output_path}: {e}"))


def save_labels_as_geojson(
    labels: np.ndarray,
    transform: Affine,
    crs: CRS,
    output_path: str,
    min_area_threshold: float = 3.0,
    max_aspect_ratio: float = 5.0,
    min_solidity: float = 0.8
) -> None:
    logger = get_logger()

    def is_valid_geometry(geometry, area, convex_hull):
        aspect_ratio = convex_hull.length / (4 * np.sqrt(area)) if area > 0 else float("inf")
        solidity = area / convex_hull.area if convex_hull.area > 0 else 0
        return (
            area >= min_area_threshold
            and aspect_ratio <= max_aspect_ratio
            and solidity >= min_solidity
        )

    geometries = []
    for label_value in np.unique(labels):
        if label_value == 0:  # Skip background
            continue

        mask = (labels == label_value)
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

    if not geometries:
        logger.info("No valid geometries found. GeoJSON will not be created.")
        return

    gdf = gpd.GeoDataFrame.from_features(geometries, crs=crs)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        gdf.to_file(output_path, driver="GeoJSON")
        logger.debug(f"GeoJSON saved with {len(geometries)} geometries to {os.path.basename(output_path)}.")
    except Exception as e:
        log_and_raise(logger, RuntimeError(f"Error saving GeoJSON to {output_path}: {e}"))


def detect_peaks(
    centroid_map: np.ndarray, 
    sigma: float = 2, 
    min_distance: int = 5
) -> np.ndarray:
    logger = get_logger()

    if centroid_map.ndim != 2:
        log_and_raise(logger, ValueError("Centroid map must be a 2D array."))

    smoothed = gaussian_filter(centroid_map, sigma=sigma)
    peaks = peak_local_max(smoothed, min_distance=min_distance, threshold_abs=0.5)
    logger.info(f"Detected {len(peaks)} peaks in the centroid map.")
    return peaks


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


def calculate_iou(geojson_path: str, predictions_path: str) -> Optional[float]:
    logger = get_logger()

    validate_path(logger, geojson_path)
    validate_path(logger, predictions_path)

    try:
        true_gdf = gpd.read_file(geojson_path)
        pred_gdf = gpd.read_file(predictions_path)

        if true_gdf.empty or pred_gdf.empty:
            logger.warning("One or both GeoJSON files have no geometries.")
            return None

        true_gdf["geometry"] = true_gdf["geometry"].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
        pred_gdf["geometry"] = pred_gdf["geometry"].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)

        true_union = true_gdf.geometry.unary_union
        pred_union = pred_gdf.geometry.unary_union

        intersection = true_union.intersection(pred_union)
        union = true_union.union(pred_union)

        intersection_area = intersection.area
        union_area = union.area

        if union_area == 0:
            logger.info("Union area is zero. IoU cannot be calculated.")
            return None

        iou = intersection_area / union_area
        logger.info(f"IoU calculated: {iou:.4f}")
        return iou
    except Exception as e:
        log_and_raise(logger, RuntimeError(f"Error calculating IoU: {e}"))
