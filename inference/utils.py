import os
import cv2
import torch
import geojson
import rasterio
import rasterio.features

import numpy as np
import geopandas as gpd

from affine import Affine
from typing import Optional, List, Tuple, Generator, Dict

from rasterio.crs import CRS
from shapely.geometry import Polygon, shape

from sklearn.decomposition import PCA

from scipy import ndimage
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation, binary_erosion

from skimage import exposure
from skimage.draw import ellipse
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.morphology import disk, square, remove_small_objects
from skimage.segmentation import watershed, relabel_sequential, find_boundaries

from treemort.modeling.builder import build_model
from treemort.utils.config import setup
from treemort.utils.logger import configure_logger, get_logger

from misc.refine import UNetWithDeepSupervision


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

    logger.debug(
        f"Refine model loaded successfully: {os.path.join(os.path.basename(os.path.dirname(model_path)), os.path.basename(model_path))}"
    )
    return refine_model


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
    prediction_map = torch.zeros((3, h, w), dtype=torch.float32, device=device) # 3 channels for binary, centroid, hybrid
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
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    logger = get_logger()

    _validate_batch_inputs(patches, coords, threshold)

    predictions = _infer_patches(patches, model, device)

    for i, (y, x) in enumerate(coords):
        binary_confidence = predictions[i, 0]
        centroid_confidence = predictions[i, 1]
        hybrid_confidence = predictions[i, 1]

        _update_maps(prediction_map, count_map, binary_confidence, centroid_confidence, hybrid_confidence, threshold, y, x)

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
    hybrid_confidence: torch.Tensor,
    threshold: float,
    y: int,
    x: int,
) -> None:
    binary_mask = (binary_confidence >= threshold).float()

    prediction_map[:, y : y + binary_confidence.shape[0], x : x + binary_confidence.shape[1]] += torch.stack(
        [binary_confidence, centroid_confidence, hybrid_confidence]
    )
    
    count_map[y : y + binary_confidence.shape[0], x : x + binary_confidence.shape[1]] += binary_mask


def pad_tensor(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    logger = get_logger()

    if tensor.ndim != 2 or patch_size <= 0:
        log_and_raise(logger, ValueError("Expected a 2D tensor and a positive patch size."))

    height, width = tensor.shape
    pad_height = (patch_size - height % patch_size) % patch_size
    pad_width = (patch_size - width % patch_size) % patch_size

    return torch.nn.functional.pad(tensor, (0, pad_width, 0, pad_height))


def prepare_patches(
    mask: torch.Tensor, patch_size: int = 256, stride: int = 128, pad: bool = False
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
            patch = mask[i : i + patch_size, j : j + patch_size]
            patches.append(patch)
            positions.append((i, j))

    logger.debug(
        f"Generated {len(patches)} patches shaped {mask.shape} with patch size {patch_size} and stride {stride}."
    )
    return patches, positions


def combine_patches(
    patches: List[Tuple[Tuple[int, int], torch.Tensor]],
    image_shape: Tuple[int, int],
    device: torch.device,
    patch_size: int,
    stride: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    logger = get_logger()

    if not patches:
        log_and_raise(logger, ValueError("No patches provided for combination."))

    for (i, j), patch in patches:
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            log_and_raise(
                logger,
                ValueError(
                    f"Patch at ({i}, {j}) has an invalid size {patch.shape}, expected ({patch_size}, {patch_size})."
                ),
            )
        if i % stride != 0 or j % stride != 0:
            log_and_raise(logger, ValueError(f"Patch at ({i}, {j}) is not aligned with the stride {stride}."))

    expected_height = max(i for i, _ in (pos for pos, _ in patches)) + patch_size
    expected_width = max(j for _, j in (pos for pos, _ in patches)) + patch_size

    logger.debug(f"Expected dimensions: ({expected_height}, {expected_width})")
    logger.debug(f"Provided image_shape: {image_shape}")

    if image_shape != (expected_height, expected_width):
        log_and_raise(
            logger,
            ValueError(
                f"Provided image_shape {image_shape} does not match expected dimensions {(expected_height, expected_width)}."
            ),
        )

    combined_mask = torch.zeros(image_shape, dtype=torch.float32, device=device)
    count_map = torch.zeros(image_shape, dtype=torch.float32, device=device)

    for (i, j), patch in patches:
        combined_mask[i : i + patch.shape[0], j : j + patch.shape[1]] += patch
        count_map[i : i + patch.shape[0], j : j + patch.shape[1]] += 1

    logger.debug(f"Combining {len(patches)} patches into an image of shape {image_shape}.")
    combined_mask = combined_mask / torch.clamp(count_map, min=1.0)
    return (combined_mask > threshold).to(dtype=torch.float32)


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


def refine_mask(
    mask: torch.Tensor, refine_model: torch.nn.Module, device: torch.device, patch_size: int = 64, stride: int = 64
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

            pred_mask = (pred_tensor.squeeze() > 0.5).to(dtype=torch.float32)

            processed_patches.append(((i, j), pred_mask))

        combined_mask = combine_patches(processed_patches, (padded_height, padded_width), device, patch_size, stride)

        combined_mask = combined_mask[: mask.shape[0], : mask.shape[1]]
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
    structuring_element: str = "disk",
) -> np.ndarray:
    logger = get_logger()

    logger.debug(
        f"Generating watershed labels with blur_sigma={blur_sigma}, min_distance={min_distance}, dilation_radius={dilation_radius}."
    )

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
            threshold_abs=0.1,  # Adjust threshold as needed
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
            threshold_abs=0.1,  # Adjust threshold as needed
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
    logger.info(f"Extracted {len(reshaped_contours)} contours from the binary mask.")
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


def detect_peaks(centroid_map: np.ndarray, sigma: float = 2, min_distance: int = 5) -> np.ndarray:
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


def decompose_elliptical_regions(
    binary_mask, intensity_image, centroid_map, min_distance=5, sigma=2, peak_threshold=0.7
):
    distance = ndimage.distance_transform_edt(binary_mask)

    smoothed_distance = gaussian(distance, sigma=sigma)
    smoothed_intensity = gaussian(intensity_image, sigma=sigma)
    smoothed_centroid_map = gaussian(centroid_map, sigma=sigma)

    thresholded_centroid_map = np.where(smoothed_centroid_map >= peak_threshold, smoothed_centroid_map, 0).astype(bool)

    combined = smoothed_distance * smoothed_intensity

    if combined.ndim > 2:
        combined = combined.mean(axis=0)

    if combined.shape != binary_mask.shape:
        raise ValueError(f"Shape mismatch: combined {combined.shape} vs binary_mask {binary_mask.shape}")

    local_maxi = peak_local_max(
        combined, min_distance=min_distance, labels=thresholded_centroid_map, exclude_border=False
    )

    markers = np.zeros_like(binary_mask, dtype=np.int32)
    for idx, (row, col) in enumerate(local_maxi):
        markers[row, col] = idx + 1

    refined_labels = watershed(-combined, markers, mask=binary_mask)
    return refined_labels


def refine_segments(
    partitioned_labels, intensity_image, max_dilation=10, max_erosion=10, erosion_threshold=0.4, dilation_threshold=0.1
):
    refined_labels = np.zeros_like(partitioned_labels, dtype=np.int32)
    intensity_image_2d = intensity_image.mean(axis=0) if intensity_image.ndim == 3 else intensity_image

    for label in np.unique(partitioned_labels):
        if label == 0:  # Skip background
            continue

        mask = partitioned_labels == label
        mean_intensity = intensity_image_2d[mask].mean()

        refined_mask = mask.copy()

        for _ in range(max_erosion):
            contour = find_boundaries(refined_mask, mode="inner")
            contour_intensities = intensity_image_2d[contour]

            if abs(np.mean(contour_intensities) - mean_intensity) > erosion_threshold:
                refined_mask = binary_erosion(refined_mask, disk(1))
            else:
                break

        for _ in range(max_dilation):
            contour = find_boundaries(refined_mask, mode="outer")
            contour_intensities = intensity_image_2d[contour]

            if abs(np.mean(contour_intensities) - mean_intensity) <= dilation_threshold:
                refined_mask = binary_dilation(refined_mask, disk(1))
            else:
                break

        refined_labels[refined_mask] = label

    return refined_labels


def filter_segment_map(segment_map, threshold=0.5, min_size=50):
    binary_segment_map = segment_map > threshold

    filtered_binary_map = remove_small_objects(binary_segment_map, min_size=min_size)
    filtered_segment_map = np.where(filtered_binary_map, segment_map, 0.0)

    return filtered_segment_map


def preprocess_centroid_map(centroid_map, segment_map, sigma=5):
    normalized_centroid_map = centroid_map / centroid_map.max()
    
    binary_mask = segment_map > 0  # Mask for valid regions
    masked_centroid_map = normalized_centroid_map * binary_mask
    
    return masked_centroid_map


def contrast_stretch_segment(segment_mask, intensity_image, p_low=10, p_high=60):
    segment_pixels = intensity_image[segment_mask == 1]

    if segment_pixels.size == 0:
        return intensity_image  # Return original image if no segment pixels

    p_min, p_max = np.percentile(segment_pixels, (p_low, p_high))

    stretched = exposure.rescale_intensity(intensity_image, in_range=(p_min, p_max), out_range=(0, 1))

    return stretched


def refine_centroid_map(segment_map, centroid_map, threshold=0.5):
    distance_map = distance_transform_edt(centroid_map > threshold)

    refined_centroid_map = distance_map * centroid_map    
    refined_centroid_map = refined_centroid_map / refined_centroid_map.max()

    return refined_centroid_map


def detect_multiple_peaks(segment_map, min_distance=5, threshold_abs=0.1):
    coordinates = peak_local_max(
        segment_map,
        min_distance=min_distance,
        threshold_abs=threshold_abs
    )
    return coordinates


def segment_using_watershed(segment_map, refined_centroid_map, peak_coords, threshold=0.5):
    binary_mask = segment_map > threshold

    markers = np.zeros_like(segment_map, dtype=int)
    for i, (row, col) in enumerate(peak_coords, start=1):
        markers[row, col] = i
    
    segmented_map = watershed(-refined_centroid_map, markers, mask=binary_mask)
    
    return segmented_map


def compute_orientation_with_pca(contour_points):
    contour_points = np.squeeze(contour_points)
    if len(contour_points.shape) < 2 or contour_points.shape[0] < 2:
        return None  # Insufficient points for PCA
    
    pca = PCA(n_components=2)
    pca.fit(contour_points)

    orientation_angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    return orientation_angle


def smooth_segment_contours(segment_map, dilation_size=2, min_area=28, min_axes_length=3):
    refined_segment_map = np.zeros_like(segment_map, dtype=np.int32)
    unique_labels = np.unique(segment_map)

    for label in unique_labels:
        if label == 0:  # Background
            continue

        segment_mask = segment_map == label
        
        if segment_mask.sum() < min_area:
            segment_mask = binary_dilation(
                segment_mask,
                structure=np.ones((dilation_size, dilation_size))
            )
        
        contours, _ = cv2.findContours(
            segment_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours and len(contours[0]) >= 5:  # Minimum points for ellipse fitting
            orientation_angle_pca = compute_orientation_with_pca(contours[0])

            ellipse_params = cv2.fitEllipse(contours[0])

            ellipse_angle_cv = np.deg2rad(ellipse_params[2])
            if abs(orientation_angle_pca - ellipse_angle_cv) > np.pi / 4:
                corrected_angle = orientation_angle_pca
            else:
                corrected_angle = ellipse_angle_cv

            axes = (
                max(ellipse_params[1][0], min_axes_length * 2),  # Semi-minor axis
                max(ellipse_params[1][1], min_axes_length * 2),  # Semi-major axis
            )
            if axes[0] > axes[1]:
                axes = (axes[1], axes[0])
                corrected_angle += np.pi / 2

            rr, cc = ellipse(
                int(ellipse_params[0][1]), int(ellipse_params[0][0]),  # Center
                int(axes[1] / 2), int(axes[0] / 2),  # Axes
                rotation=corrected_angle, shape=segment_map.shape
            )
            refined_segment_map[rr, cc] = label
        else:
            refined_segment_map[segment_mask] = label

    final_segment_map = np.zeros_like(segment_map, dtype=np.int32)
    for label in unique_labels:
        if label == 0:  # Background
            continue
        dilated_label = binary_dilation(
            refined_segment_map == label,
            structure=np.ones((dilation_size, dilation_size))
        )
        final_segment_map[dilated_label] = label
    
    return final_segment_map


def refine_dead_tree_segments_adaptive(segment_map, intensity_image, intensity_threshold=0.1, erosion_iterations=10, dilation_iterations=10):
    refined_segment_map = np.copy(segment_map)
    unique_labels = np.unique(segment_map)

    # Calculate NDVI
    # nir = intensity_image[3, :, :]  # Assuming NIR is the 4th channel
    # red = intensity_image[2, :, :]  # Assuming Red is the 3rd channel
    # ndvi = (nir - red) / (nir + red + 1e-6)  # Adding a small epsilon to avoid division by zero
    # intensity_image_2d = ndvi
    intensity_image_2d = intensity_image.mean(axis=0) if intensity_image.ndim == 3 else intensity_image

    for label in unique_labels:
        if label == 0:  # Skip background
            continue

        segment_mask = (segment_map == label).astype(np.uint8)

        segment_intensity = intensity_image_2d[segment_mask == 1]
        mean_intensity = float(np.mean(segment_intensity))

        erosion_kernel = np.ones((3, 3), np.uint8)
        segment_core = cv2.erode(segment_mask, erosion_kernel, iterations=2)

        core_intensity = intensity_image_2d[segment_core == 1]

        if core_intensity.size == 0:
            core_intensity = segment_intensity  # Fallback to entire segment intensity
            print(f"Warning: Erosion removed segment {label}, using full segment intensity.")

        lower_bound = np.percentile(core_intensity, 20)
        upper_bound = np.percentile(core_intensity, 90)

        filtered_intensity = core_intensity[
            (core_intensity >= lower_bound) & (core_intensity <= upper_bound)
        ]
        core_mean_intensity = float(np.mean(filtered_intensity))

        upper_threshold = mean_intensity + intensity_threshold  # More tolerant range
        lower_threshold = mean_intensity - intensity_threshold

        for _ in range(erosion_iterations):
            contours, _ = cv2.findContours(segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                for point in contour:
                    x, y = point[0]

                    contour_intensity = intensity_image_2d[y, x]

                    if contour_intensity > upper_threshold or contour_intensity < lower_threshold:
                        segment_mask[y, x] = 0
        
        for _ in range(dilation_iterations):
            contours, _ = cv2.findContours(segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                for point in contour:
                    x, y = point[0]

                    contour_intensity = intensity_image_2d[y, x]

                    if abs(contour_intensity - mean_intensity) <= intensity_threshold:
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-connected neighbors
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < intensity_image_2d.shape[1] and 0 <= ny < intensity_image_2d.shape[0]:
                                if segment_mask[ny, nx] == 0 and abs(intensity_image_2d[ny, nx] - mean_intensity) <= intensity_threshold:
                                    segment_mask[ny, nx] = 1
                                        
        refined_segment_map[segment_mask == 1] = label

    return refined_segment_map


def refine_segments_irregular_shape(segment_map, intensity_image, intensity_threshold=0.1, max_growth_radius=50):
    refined_segment_map = np.zeros_like(segment_map)
    unique_labels = np.unique(segment_map)

    intensity_image_2d = intensity_image.mean(axis=0) if intensity_image.ndim == 3 else intensity_image

    for label in unique_labels:
        if label == 0:
            continue

        segment_mask = (segment_map == label).astype(np.uint8)

        region = regionprops(segment_mask.astype(int))[0]
        centroid_y, centroid_x = region.centroid

        erosion_kernel = np.ones((3, 3), np.uint8)
        segment_core = cv2.erode(segment_mask, erosion_kernel, iterations=2)

        core_intensity = intensity_image_2d[segment_core == 1]
        if core_intensity.size == 0:
            core_intensity = intensity_image_2d[segment_mask == 1]  # Fallback to full segment

        core_mean_intensity = np.mean(core_intensity)

        convex_hull = convex_hull_image(segment_mask)
        refined_segment_map[convex_hull] = label

        visited = np.zeros_like(segment_mask, dtype=bool)
        to_explore = [(int(centroid_x), int(centroid_y))]
        growth_radius = 0

        while to_explore and growth_radius < max_growth_radius:
            new_pixels = []
            for x, y in to_explore:
                if visited[y, x]:
                    continue
                visited[y, x] = True

                if abs(intensity_image_2d[y, x] - core_mean_intensity) > intensity_threshold:
                    continue

                refined_segment_map[y, x] = label

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < segment_map.shape[1] and 0 <= ny < segment_map.shape[0]:
                        if not visited[ny, nx] and (refined_segment_map[ny, nx] == label or refined_segment_map[ny, nx] == 0):
                            new_pixels.append((nx, ny))

            to_explore = new_pixels
            growth_radius += 1

    return refined_segment_map
