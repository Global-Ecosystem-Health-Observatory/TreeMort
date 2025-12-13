import os
import cv2
import json
import torch
import rasterio
import rasterio.features

import numpy as np

from scipy import ndimage as ndi
from affine import Affine
from typing import Optional, List, Tuple, Generator, Dict
from shapely.geometry import Polygon, MultiPolygon

from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.measure import regionprops, find_contours
from skimage.morphology import erosion, disk, remove_small_objects, binary_dilation, binary_opening, remove_small_holes
from skimage.segmentation import watershed

from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject

from treemort.modeling.builder import build_model
from treemort.utils.config import setup
from treemort.utils.logger import configure_logger, get_logger


def initialize_logger(verbosity: str) -> None:
    configure_logger(verbosity=verbosity)


def log_and_raise(logger, exception: Exception):
    logger.error(str(exception))
    raise exception


def _safe_polygon(coords) -> Optional[Polygon]:
    """
    Build a polygon defensively:
    - ensure at least 3 points
    - close the ring
    - buffer(0) to fix minor invalidities
    - if MultiPolygon, keep the largest
    Returns None on failure.
    """
    try:
        coords_arr = np.asarray(coords)
    except Exception:
        return None

    if coords_arr.shape[0] < 3:
        return None

    # Close ring if needed
    if not np.array_equal(coords_arr[0], coords_arr[-1]):
        coords_arr = np.vstack([coords_arr, coords_arr[0]])

    try:
        poly = Polygon(coords_arr)
    except Exception:
        return None

    if not poly.is_valid:
        try:
            poly = poly.buffer(0)
        except Exception:
            return None

    if poly.is_empty:
        return None

    if isinstance(poly, MultiPolygon):
        if len(poly.geoms) == 0:
            return None
        poly = max(poly.geoms, key=lambda g: g.area)

    return poly


def expand_path(path):
    return os.path.expandvars(path)


def validate_path(logger, path: str, is_dir: bool = False) -> bool:
    if not os.path.exists(path):
        log_and_raise(logger, FileNotFoundError(f"Path does not exist: {path}"))
    if is_dir and not os.path.isdir(path):
        log_and_raise(logger, NotADirectoryError(f"Expected directory but got: {path}"))
    return True


def load_model(
    conf,
    id2label: dict = {0: "alive", 1: "dead"},
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    logger = get_logger()

    best_model_path = os.path.join(conf.output_dir, conf.model, conf.best_model)
    validate_path(logger, best_model_path)

    model, *_ = build_model(conf, id2label, device)
    model = model.to(device).eval()

    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    except Exception as e:
        log_and_raise(logger, RuntimeError(f"Failed to load model weights: {e}"))

    logger.debug(f"Model loaded successfully: {best_model_path}")
    return model


def load_and_preprocess_image(
    tiff_file: str,
    nir_rgb_order: Optional[List[int]] = None,
    target_resolution: float = 0.25,
) -> Tuple[torch.Tensor, Affine, CRS]:
    logger = get_logger()

    validate_path(logger, tiff_file)

    with rasterio.open(tiff_file) as src:
        image = src.read()
        transform = src.transform
        crs = src.crs

        # Resample to target resolution if needed
        current_res_x, current_res_y = src.res
        if (current_res_x != target_resolution) or (current_res_y != target_resolution):
            scale_x = current_res_x / target_resolution
            scale_y = current_res_y / target_resolution
            new_width = int(src.width * scale_x)
            new_height = int(src.height * scale_y)
            dst_transform = Affine(
                target_resolution,
                transform.b,
                transform.c,
                transform.d,
                -target_resolution,
                transform.f,
            )
            resampled = np.empty((image.shape[0], new_height, new_width), dtype=image.dtype)
            for i in range(image.shape[0]):
                reproject(
                    source=image[i],
                    destination=resampled[i],
                    src_transform=transform,
                    src_crs=crs,
                    dst_transform=dst_transform,
                    dst_crs=crs,
                    resampling=Resampling.bilinear,
                )
            image = resampled
            transform = dst_transform

        max_pixel_value = _get_max_pixel_value(src.dtypes[0])

    _validate_image_channels(image, nir_rgb_order)
    nir_rgb_order = nir_rgb_order or list(range(image.shape[0]))

    image = image.astype(np.float32) / max_pixel_value
    image = image[nir_rgb_order] if nir_rgb_order != list(range(image.shape[0])) else image
    image_tensor = torch.tensor(image, dtype=torch.float32)

    if image_tensor.ndim != 3:
        log_and_raise(
            logger,
            ValueError(f"Invalid tensor shape: {image_tensor.shape}. Expected 3D tensor (C, H, W)."),
        )

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
        log_and_raise(
            logger,
            ValueError(f"nir_rgb_order indices exceed available channels: {nir_rgb_order}"),
        )


def sliding_window_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    window_size: int = 256,
    stride: int = 128,
    batch_size: int = 1,
    threshold: float = 0.5,
    output_channels: int = 3,
) -> torch.Tensor:
    _validate_inference_params(window_size, stride, threshold)

    device = next(model.parameters()).device
    padded_image = _pad_image(image, window_size)

    prediction_map, count_map = _initialize_maps(padded_image.shape[1:], output_channels=output_channels, device=device)
    patches, coords = _generate_patches(padded_image, window_size, stride)
    blend_w = _make_blend_weights(window_size, device=device)

    for batch in _batch_patches(patches, coords, batch_size):
        prediction_map, count_map = process_batch(
            batch["patches"],
            batch["coords"],
            prediction_map,
            count_map,
            model,
            threshold,
            device,
            blend_w,
        )

    return _finalize_prediction(prediction_map, count_map, image.shape, threshold)


def _validate_inference_params(window_size: int, stride: int, threshold: float) -> None:
    logger = get_logger()

    if window_size <= 0 or stride <= 0:
        log_and_raise(logger, ValueError("window_size and stride must be positive integers."))
    if not (0 <= threshold <= 1):
        log_and_raise(logger, ValueError("threshold must be between 0 and 1."))


def _make_blend_weights(window_size: int, device: torch.device) -> torch.Tensor:
    """
    Create a smooth 2D blending window to reduce visible seams.

    UNet-style models often have border effects. Weighting patch centers higher than
    patch borders reduces stride-grid artifacts in smooth regression maps (centroid/hybrid).

    Uses a Hann (raised cosine) window in each dimension. Hann is zero at the ends,
    so we clamp to a small epsilon to keep borders contributing non-zero weight.

    Returns a tensor of shape (window_size, window_size).
    """
    w1 = torch.hann_window(window_size, periodic=False, dtype=torch.float32, device=device)
    w1 = w1.clamp_min(1e-3)
    return torch.outer(w1, w1)

def _initialize_maps(
    image_shape: Tuple[int, int],
    output_channels: int = 3,
    device: torch.device = torch.device("cpu"),
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

    ys = list(range(0, h - window_size + 1, stride))
    xs = list(range(0, w - window_size + 1, stride))

    # Ensure we always include the last possible start so borders are covered
    if not ys or ys[-1] != h - window_size:
        ys.append(h - window_size)
    if not xs or xs[-1] != w - window_size:
        xs.append(w - window_size)

    for y in ys:
        for x in xs:
            patch = image[:, y : y + window_size, x : x + window_size].float()
            patches.append(patch)
            coords.append((y, x))

    return patches, coords

def _batch_patches(
    patches: List[torch.Tensor], coords: List[Tuple[int, int]], batch_size: int
) -> Generator[Dict[str, List], None, None]:
    for i in range(0, len(patches), batch_size):
        yield {
            "patches": patches[i : i + batch_size],
            "coords": coords[i : i + batch_size],
        }


def _finalize_prediction(
    prediction_map: torch.Tensor,
    count_map: torch.Tensor,
    original_shape: Tuple[int, int, int],
    threshold: float,
) -> torch.Tensor:
    no_contribution_mask = count_map == 0
    count_map[no_contribution_mask] = 1

    final_prediction = prediction_map / count_map.unsqueeze(0)
    final_prediction[:, no_contribution_mask] = 0

    final_prediction[0] = torch.clamp(final_prediction[0], 0, 1)
    # final_prediction[1] = torch.clamp(final_prediction[1], 0, 1)
    final_prediction[2] = torch.clamp(final_prediction[2], -1, 1)

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
    blend_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logger = get_logger()

    _validate_batch_inputs(patches, coords, threshold)

    predictions = _infer_patches(patches, model, device)

    for i, (y, x) in enumerate(coords):
        binary_confidence = predictions[i, 0]
        centroid_confidence = predictions[i, 1]
        hybrid_confidence = predictions[i, 2]

        _update_maps(
            prediction_map,
            count_map,
            binary_confidence,
            centroid_confidence,
            hybrid_confidence,
            threshold,
            y,
            x,
            blend_w,
        )

    return prediction_map, count_map


def _validate_batch_inputs(patches: list[torch.Tensor], coords: list[tuple[int, int]], threshold: float) -> None:
    logger = get_logger()

    if not patches or not coords:
        log_and_raise(logger, ValueError("Patches and coordinates cannot be empty."))
    if not (0 <= threshold <= 1):
        log_and_raise(logger, ValueError("Threshold must be between 0 and 1."))


def _infer_patches(patches: list[torch.Tensor], model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    logger = get_logger()

    batch_tensor = torch.stack(patches).to(device)

    with torch.no_grad():
        outputs = model(batch_tensor)

        # Normalize to 4D: (B, C, H, W)
        if outputs.ndim == 3:  # (B, H, W) -> (B, 1, H, W)
            outputs = outputs.unsqueeze(1)
        elif outputs.ndim != 4:
            raise RuntimeError(f"Unexpected model output shape: {tuple(outputs.shape)}")

        outputs = outputs.to(dtype=torch.float32)

        # Pad or truncate to exactly 3 channels: [seg, centroid, hybrid]
        C = outputs.shape[1]
        if C < 3:
            pad = torch.zeros(
                (outputs.shape[0], 3 - C, outputs.shape[2], outputs.shape[3]),
                device=outputs.device,
                dtype=outputs.dtype,
            )
            outputs = torch.cat([outputs, pad], dim=1)
        elif C > 3:
            outputs = outputs[:, :3, ...]

        # Apply activations according to target semantics:
        # - segmentation: probability -> sigmoid
        # - centroid: use raw logits for peak detection (sigmoid collapses contrast)
        # - hybrid SDT+boundary: regression target (inside (0,1], boundary=-1, background=0) -> keep raw
        seg_predictions = torch.sigmoid(outputs[:, 0:1, ...])
        centroid_predictions = outputs[:, 1:2, ...]  # logits
        hybrid_predictions = outputs[:, 2:3, ...]

        predictions = torch.cat([seg_predictions, centroid_predictions, hybrid_predictions], dim=1)

        logger.debug(f"Predictions shape: {tuple(predictions.shape)}")

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
    blend_w: torch.Tensor,
) -> None:
    # Weighted blending reduces patch-border seams (esp. for centroid/hybrid regression maps).
    stacked = torch.stack([binary_confidence, centroid_confidence, hybrid_confidence])  # (3, H, W)

    w = blend_w.to(device=stacked.device, dtype=stacked.dtype)  # (H, W)

    prediction_map[:, y : y + stacked.shape[1], x : x + stacked.shape[2]] += stacked * w
    count_map[y : y + stacked.shape[1], x : x + stacked.shape[2]] += w


def _binary_cleanup(mask: np.ndarray, conf) -> np.ndarray:
    """Light mask cleanup to prevent thin bridges merging nearby circular crowns."""
    mask = mask.astype(bool)

    # Fill tiny holes inside crowns (helps circular objects)
    holes_area = getattr(conf, "holes_area", 32)
    if holes_area and holes_area > 0:
        mask = remove_small_holes(mask, area_threshold=int(holes_area))

    # Light opening breaks 1-2px bridges between adjacent crowns
    opening_radius = getattr(conf, "opening_radius", 1)
    if opening_radius and opening_radius > 0:
        mask = binary_opening(mask, disk(int(opening_radius)))

    return mask.astype(np.uint8)


def _markers_from_centroids(centroid_map: np.ndarray, mask: np.ndarray, conf) -> np.ndarray:
    """
    Create labeled markers from centroid logits using adaptive (percentile) thresholding
    within the segmentation mask. This is robust to logit scale drift across tiles/domains.
    """
    mask_bool = mask.astype(bool)
    markers = np.zeros_like(mask, dtype=np.int32)
    if mask_bool.sum() == 0:
        return markers

    # Smooth logits lightly (helps stabilize local maxima)
    cen_sm = gaussian(centroid_map.astype(np.float32), sigma=float(getattr(conf, "blur_sigma", 1.5)))

    # Adaptive threshold inside the mask
    vals = cen_sm[mask_bool]
    if vals.size == 0:
        return markers

    # User-configurable knobs with safe defaults
    pct = float(getattr(conf, "centroid_peak_percentile", 95))

    # Percentile threshold is the primary mechanism (robust to logit scale drift).
    thr_pct = float(np.percentile(vals, pct))
    thr = thr_pct

    # Optional absolute floor: disabled by default because it can easily suppress peaks
    # when centroid logits are not calibrated.
    if bool(getattr(conf, "use_centroid_abs_floor", False)):
        abs_floor = float(getattr(conf, "centroid_threshold", -np.inf))
        thr = max(thr, abs_floor)

    # Guard against flat maps (percentile == max => may return 0 peaks)
    if np.isclose(thr, float(vals.max())):
        thr = float(np.percentile(vals, max(90.0, pct - 5.0)))

    if os.getenv("TREEMORT_DEBUG_MARKERS", "0") == "1":
        print(
            f"[MARKERS] mask_pixels={int(mask_bool.sum())} pct={pct} thr={thr:.6f} thr_pct={thr_pct:.6f} "
            f"use_abs_floor={bool(getattr(conf,'use_centroid_abs_floor', False))}",
            flush=True,
        )

    coords = peak_local_max(
        cen_sm,
        min_distance=int(getattr(conf, "min_distance", 3)),
        threshold_abs=float(thr),
        exclude_border=False,
        labels=mask_bool.astype(np.uint8),
    )

    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i

    # Ensure proper connected-component labeling of seed points
    markers = ndi.label(markers > 0)[0].astype(np.int32)
    return markers


def threshold_prediction_map(prediction_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    binary_mask = prediction_map >= threshold
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


def compute_watershed(segment_map, centroid_map, hybrid_map, conf):

    logger = get_logger()

    binary_seg = (segment_map > conf.segment_threshold).astype(np.uint8)
    binary_seg = remove_small_objects(binary_seg.astype(bool), min_size=conf.min_area_pixels).astype(np.uint8)

    if conf.output_channels == 1:
        smoothed_segment_map = gaussian(binary_seg, sigma=conf.blur_sigma)

        mask_grown = binary_dilation(binary_seg, disk(conf.dilation_radius))

        local_max = peak_local_max(
            smoothed_segment_map, min_distance=conf.min_distance, exclude_border=False, labels=mask_grown
        )

        markers = np.zeros(segment_map.shape, dtype=np.int32)
        for i, (y, x) in enumerate(local_max, 1):
            markers[y, x] = i  # Ensure marker values are unique and nonzero

        labels_ws = watershed(-smoothed_segment_map, markers, mask=mask_grown)

    elif conf.output_channels == 3:

        # Base segmentation mask
        binary_seg = (segment_map > conf.segment_threshold).astype(np.uint8)
        binary_seg = remove_small_objects(binary_seg.astype(bool), min_size=conf.min_area_pixels).astype(np.uint8)

        # Light morphological cleanup to break thin bridges between adjacent circular crowns.
        binary_seg = _binary_cleanup(binary_seg, conf)

        # Marker-controlled watershed using distance transform is much better at splitting touching objects
        # than using -centroid as the watershed surface.
        mask_bool = binary_seg.astype(bool)
        if mask_bool.sum() == 0:
            labels_ws = np.zeros_like(binary_seg, dtype=np.int32)
        else:
            dist = ndi.distance_transform_edt(mask_bool).astype(np.float32)

            # Use centroid map peaks as seeds (constrained to mask).
            markers = _markers_from_centroids(centroid_map, binary_seg, conf)

            num_markers = int(markers.max())
            logger.info(
                f"[WS] centroid markers: {num_markers} | "
                f"seg_pixels={int(binary_seg.sum())} | "
                f"centroid stats in seg-mask: "
                f"min={centroid_map[binary_seg>0].min():.4f}, "
                f"med={np.median(centroid_map[binary_seg>0]):.4f}, "
                f"p95={np.percentile(centroid_map[binary_seg>0],95):.4f}, "
                f"max={centroid_map[binary_seg>0].max():.4f}"
            )

            # Fallback: if centroid peaks are missing, seed from distance peaks.
            if markers.max() == 0:
                fallback_thr = float(getattr(conf, "dist_peak_rel", 0.35))
                dist_thr = dist.max() * fallback_thr
                coords = peak_local_max(
                    dist,
                    min_distance=conf.min_distance,
                    threshold_abs=dist_thr,
                    exclude_border=False,
                    labels=mask_bool,
                )
                markers = np.zeros_like(binary_seg, dtype=np.int32)
                for i, (r, c) in enumerate(coords, 1):
                    markers[r, c] = i
                markers = ndi.label(markers > 0)[0]

            labels_ws = watershed(-dist, markers, mask=mask_bool)

            # Optional post-watershed hybrid refinement:
            # Cut boundary-like pixels from each instance and relabel connected components.
            # This is safer than pre-watershed gating because it cannot erase the whole foreground mask.
            if bool(getattr(conf, "use_hybrid_refine", False)):
                boundary_thr = float(getattr(conf, "hybrid_boundary_threshold", -0.5))
                boundary_mask = (hybrid_map < boundary_thr) & (labels_ws > 0)

                before_px = int((labels_ws > 0).sum())
                labels_ws_cut = labels_ws.copy()
                labels_ws_cut[boundary_mask] = 0
                after_px = int((labels_ws_cut > 0).sum())

                # If cutting is too destructive, skip for this tile
                min_keep_frac = float(getattr(conf, "hybrid_min_keep_frac", 0.40))
                keep_frac = after_px / max(before_px, 1)
                if before_px > 0 and keep_frac < min_keep_frac:
                    logger.warning(
                        f"Post-watershed hybrid cutting too aggressive (kept {after_px}/{before_px}={keep_frac:.2%}); skipping hybrid refine."
                    )
                else:
                    # Relabel connected components after boundary removal
                    cc = ndi.label(labels_ws_cut > 0)[0].astype(np.int32)
                    # Preserve original labels where possible by assigning component ids
                    # as new instance ids.
                    labels_ws = cc

    else:
        log_and_raise(logger, ValueError(f"Unsupported number of output channels: {conf.output_channels}"))

    new_labels = _postprocess_labels(
        labels_ws,
        min_region_size=conf.min_area_pixels,
        dilation_radius=conf.dilation_radius,
    )

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
    for region in regionprops(labels_ws):
        if region.area < conf.min_area_pixels:
            continue

        # Calculate extended bounding box with padding for erosion
        extended_min_row = max(0, region.bbox[0] - conf.erosion_radius)
        extended_max_row = min(labels_ws.shape[0], region.bbox[2] + conf.erosion_radius)
        extended_min_col = max(0, region.bbox[1] - conf.erosion_radius)
        extended_max_col = min(labels_ws.shape[1], region.bbox[3] + conf.erosion_radius)

        # Crop labels_ws to the extended bounding box
        cropped_labels = labels_ws[extended_min_row:extended_max_row, extended_min_col:extended_max_col]

        # Create mask within the cropped area
        mask = cropped_labels == region.label

        # Erode the mask
        eroded_mask = erosion(mask, disk(conf.erosion_radius))

        # Find contours in the eroded mask
        eroded_contours = find_contours(eroded_mask, level=0.5)
        if not eroded_contours:
            continue
        eroded_contour = max(eroded_contours, key=lambda c: c.shape[0])

        # Adjust contour points to original image coordinates
        pts = np.array(
            [[pt[1] + extended_min_col, pt[0] + extended_min_row] for pt in eroded_contour],
            dtype=np.float32,
        )
        if len(pts) < 5:
            continue

        # Sample contour points if there are too many
        if len(pts) > 200:
            indices = np.linspace(0, len(pts) - 1, 200, dtype=int)
            pts = pts[indices]

        # Fit ellipse to the contour points
        ellipse = cv2.fitEllipse(pts)
        center = ellipse[0]
        axes = ellipse[1]
        angle_deg = ellipse[2]
        orientation = np.deg2rad(angle_deg)

        # Compute semi-axes with tightness factor
        a = (axes[0] / 2.0) * conf.tightness
        b = (axes[1] / 2.0) * conf.tightness

        # Generate ellipse points
        t = np.linspace(0, 2 * np.pi, num_points)
        ellipse_x = center[0] + a * np.cos(t) * np.cos(orientation) - b * np.sin(t) * np.sin(orientation)
        ellipse_y = center[1] + a * np.cos(t) * np.sin(orientation) + b * np.sin(t) * np.cos(orientation)
        ellipse_coords = list(zip(ellipse_x.tolist(), ellipse_y.tolist()))

        # Ensure the polygon is closed
        if ellipse_coords[0] != ellipse_coords[-1]:
            ellipse_coords.append(ellipse_coords[0])

        # Apply transform to ellipse coordinates
        ellipse_arr = np.array(ellipse_coords)
        transformed_ellipse = _apply_transform(ellipse_arr, transform)

        # Ensure the polygon is closed after transform
        if not np.array_equal(transformed_ellipse[0], transformed_ellipse[-1]):
            transformed_ellipse = np.vstack([transformed_ellipse, transformed_ellipse[0]])

        # Create and validate polygon
        ellipse_poly = _safe_polygon(transformed_ellipse)
        if ellipse_poly and ellipse_poly.is_valid and not ellipse_poly.is_empty:
            convex_hull = ellipse_poly.convex_hull
            area = ellipse_poly.area
            aspect_ratio = convex_hull.length / (4 * np.sqrt(area)) if area > 0 else float("inf")
            solidity = area / convex_hull.area if convex_hull.area > 0 else 0
            if area >= conf.min_area and aspect_ratio <= conf.max_aspect_ratio and solidity >= conf.min_solidity:
                ellipse_center_geo = list(_apply_transform(np.array([center]), transform)[0])
                feature = {
                    "type": "Feature",
                    "properties": {
                        "region_label": region.label,
                        "area": area,
                        "ellipse_center": ellipse_center_geo,
                        "ellipse_axes": axes,
                        "ellipse_angle_deg": angle_deg,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [transformed_ellipse.tolist()],
                    },
                }
                yield feature


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
    logger.debug(f"Extracted {len(reshaped_contours)} contours from the binary mask.")

    features = []
    skipped_contours = 0
    for contour in reshaped_contours:
        if len(contour) >= 3:
            transformed_contour = _apply_transform(contour, transform)
            if not np.array_equal(transformed_contour[0], transformed_contour[-1]):
                transformed_contour = np.vstack([transformed_contour, transformed_contour[0]])

            polygon = _safe_polygon(transformed_contour)

            if polygon:
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [np.asarray(polygon.exterior.coords)[:, :2].tolist()],
                        },
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
        log_and_raise(
            logger,
            ValueError("label_map must be a 2D integer array representing labels."),
        )

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

            polygon = _safe_polygon(transformed_contour)

            if polygon:
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [np.asarray(polygon.exterior.coords)[:, :2].tolist()],
                        },
                        "properties": {"label": int(label)},
                    }
                )
            else:
                skipped_labels += 1
        else:
            skipped_labels += 1

    logger.debug(
        f"Extracted {len(features)} features from {len(unique_labels)-1} segments, skipped {skipped_labels} labels."
    )
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


"""
Functions for Abulation study:
"""


def segment_filtering_only(segment_map: np.ndarray, conf) -> np.ndarray:
    binary_mask = (segment_map > conf.segment_threshold).astype(np.uint8)

    binary_mask = remove_small_objects(binary_mask.astype(bool), min_size=conf.min_area_pixels)
    binary_mask = binary_mask.astype(np.uint8)

    return binary_mask


# Example usage:
# Assuming 'segment_map' is a numpy array extracted from the model's segmentation output,
# and 'conf' is a configuration object with required thresholds:
# filtered_mask = segment_filtering_only(segment_map, conf)


def watershed_segmentation_only(
    segment_map: np.ndarray, centroid_map: np.ndarray, hybrid_map: np.ndarray, conf
) -> np.ndarray:
    logger = get_logger()
    
    binary_seg = (segment_map > conf.segment_threshold).astype(np.uint8)
    binary_seg = remove_small_objects(binary_seg.astype(bool), min_size=conf.min_area_pixels).astype(np.uint8)

    centroid_map_smoothed = gaussian(centroid_map, sigma=conf.blur_sigma)

    local_max_coords = peak_local_max(
        centroid_map_smoothed,
        min_distance=conf.min_distance,
        threshold_abs=conf.centroid_threshold,
        exclude_border=False,
        labels=binary_seg.astype(bool),
    )
    markers = np.zeros_like(centroid_map, dtype=np.int32)
    for i, (row, col) in enumerate(local_max_coords, 1):
        markers[row, col] = i
    markers = ndi.label(markers)[0]

    labels_ws = watershed(-ndi.distance_transform_edt(binary_seg.astype(bool)).astype(np.float32), markers, mask=binary_seg.astype(bool))

    # Optional post-watershed hybrid refinement:
    # Cut boundary-like pixels from each instance and relabel connected components.
    # This is safer than pre-watershed gating because it cannot erase the whole foreground mask.
    if bool(getattr(conf, "use_hybrid_refine", False)):
        boundary_thr = float(getattr(conf, "hybrid_boundary_threshold", -0.5))
        boundary_mask = (hybrid_map < boundary_thr) & (labels_ws > 0)

        before_px = int((labels_ws > 0).sum())
        labels_ws_cut = labels_ws.copy()
        labels_ws_cut[boundary_mask] = 0
        after_px = int((labels_ws_cut > 0).sum())

        # If cutting is too destructive, skip for this tile
        min_keep_frac = float(getattr(conf, "hybrid_min_keep_frac", 0.40))
        keep_frac = after_px / max(before_px, 1)
        if before_px > 0 and keep_frac < min_keep_frac:
            logger.warning(
                f"Post-watershed hybrid cutting too aggressive (kept {after_px}/{before_px}={keep_frac:.2%}); skipping hybrid refine."
            )
        else:
            # Relabel connected components after boundary removal
            cc = ndi.label(labels_ws_cut > 0)[0].astype(np.int32)
            # Preserve original labels where possible by assigning component ids
            # as new instance ids.
            labels_ws = cc

    return labels_ws
