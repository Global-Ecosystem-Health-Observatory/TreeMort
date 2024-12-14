import json

import cv2
import numpy as np
import rasterio
from tqdm import tqdm


def create_label_mask(img_arr: np.ndarray, polys: list[np.ndarray]):

    image_h, image_w = img_arr.shape[:2]
    label_mask = np.zeros((image_h, image_w), dtype=np.float32)

    for poly in tqdm(polys, desc="Creating label masks"):
        cv2.fillPoly(label_mask, pts=[poly], color=1)

    return label_mask


def create_label_mask_with_centroids(img_arr: np.ndarray, polys: list[np.ndarray]):
    image_h, image_w = img_arr.shape[:2]

    label_mask = np.zeros((image_h, image_w), dtype=np.float32)
    centroid_mask = np.zeros((image_h, image_w), dtype=np.float32)

    for poly in tqdm(polys, desc="Creating label masks and centroids"):
        cv2.fillPoly(label_mask, pts=[poly], color=1)

        moments = cv2.moments(poly)
        if moments["m00"] != 0:  # Avoid division by zero
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])

            centroid_mask[centroid_y, centroid_x] = 1

    return label_mask, centroid_mask


def segmap_to_topo(img_arr: np.ndarray, contours: list) -> np.ndarray:
    image_h, image_w = img_arr.shape[:2]
    topolabel = np.zeros((image_h, image_w), dtype=np.float32)

    for tree in tqdm(contours, desc="Processing contours"):
        mask = np.zeros((image_h, image_w), dtype=np.uint8)
        cv2.fillPoly(mask, pts=[tree], color=1)
        transfer_layer = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        maxpoint = transfer_layer.max()

        if maxpoint > 0:
            transfer_layer = 255 * (transfer_layer / maxpoint)
        
        topolabel += np.where(mask == 1, transfer_layer, 0)

    topolabel = np.clip(topolabel, 0, 255).astype(np.uint8)
    return topolabel


def get_image_and_polygons(
    image_filepath: str,
    geojson_filepath: str,
    nir_rgb_order: list[int],
    normalize_channelwise: bool,
    normalize_imagewise: bool,
) -> tuple[np.ndarray, list[np.ndarray]]:

    img_arr, bounds, resolution = load_geotiff(
        image_filepath,
        nir_rgb_order,
        normalize_channelwise,
        normalize_imagewise
    )

    min_xy = bounds[:2]
    n_rows = img_arr.shape[0]
    
    # Polygons in georeferenced coordinates
    label_polygons = load_geojson_labels(geojson_filepath)

    # Transform polygons into image coordinates
    adjusted_polygons = []
    for polygon in label_polygons:
        adjusted_polygon = []
        for coordinate in polygon:
            img_coords = geo_to_img_coords(coordinate, min_xy, resolution, n_rows)
            adjusted_polygon.append(img_coords)

        adjusted_polygons.append(np.int32(np.array(adjusted_polygon).reshape((-1, 1, 2))))

    return img_arr, adjusted_polygons


def geo_to_img_coords(
    geocoord: list,
    min_xy: tuple,
    resolution: tuple,
    n_rows: int) -> tuple[int]:

    x,y = geocoord
    min_x, min_y = min_xy
    res_x, res_y = resolution

    col = np.int32((x - min_x) / res_x) 
    row = np.int32(n_rows + (min_y - y) / res_y)

    return col, row


def load_geotiff(
    filename: str,
    nir_rgb_order: list[int],
    normalize_channelwise: bool = False,
    normalize_imagewise: bool = False,
) -> tuple[np.ndarray, tuple[float], tuple[float]]:
           
    with rasterio.open(filename) as img:

        # Read pixel values and move channels last
        img_arr = np.moveaxis(img.read(), 0, -1).astype(np.float32)

        # Reorder channels
        img_arr = img_arr[:,:,nir_rgb_order]

        # Set missing values to zero
        if img.nodata is not None:
            img_arr[img.nodata] = 0
        else:
            img_arr[img_arr < -3e38] = 0

        if normalize_channelwise:
            img_arr = normalize_channelwise_to_uint8(img_arr)
        elif normalize_imagewise:
            img_arr = normalize_imagewise_to_uint8(img_arr)
        else:
            img_arr = np.uint8(np.clip(img_arr, 0, 255))

        bounds = tuple(img.bounds)
        resolution = img.res

    return img_arr, bounds, resolution


def load_geojson_labels(geojson_path: str) -> list[list[list[float]]]:

    with open(geojson_path,"r") as f:
        samples_json = json.load(f)

    coord_list = []
    for multipoly in samples_json.get("features", []):
        geometry = multipoly.get("geometry")
        if geometry and geometry.get("coordinates"):
            # Inner loop ensures that all polygons in single multipolygon are
            # stored separately
            for poly in geometry["coordinates"]:
                try:
                    coordinates = poly[0]
                    if len(coordinates) > 2:
                        coord_list.append(coordinates)
                except (IndexError, TypeError) as e:
                    print(f"Error processing polygon coordinates: {e}")

    return coord_list


def normalize_channelwise_to_uint8(img_arr: np.ndarray) -> np.ndarray:

    scaled_channels = [
        normalize_imagewise_to_uint8(img_arr[:,:,i]) 
        for i 
        in range(img_arr.shape[-1])
    ]

    return np.stack(scaled_channels, axis=-1)


def normalize_imagewise_to_uint8(img_arr: np.ndarray) -> np.ndarray:

    min_val, max_val = get_nonzero_percentiles(img_arr, [1,99])

    if max_val != min_val:
        img_arr_scaled = (img_arr - min_val) / (max_val - min_val) * 255
        img_arr_scaled = np.clip(img_arr_scaled, 0, 255)
    else:
        img_arr_scaled = (
            np.zeros_like(img_arr)
            if min_val == 0
            else np.ones_like(img_arr) * 255
        )

    return img_arr_scaled.astype(np.uint8)


def get_nonzero_percentiles(input_array: np.ndarray, percentages: float) -> float:

    all_nonzero_values = input_array[input_array > 0]
    if len(all_nonzero_values) == 0:
        return 0

    return np.percentile(all_nonzero_values, percentages)
