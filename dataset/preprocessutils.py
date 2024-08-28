import cv2
import json
import rasterio

import numpy as np

from tqdm import tqdm


def get_image_and_polygons_reorder(
    image_filepath,
    geojson_filepath,
    nir_rgb_order,
    normalize_channelwise,
    normalize_imagewise,
):
    exim_np, x_min, y_min, x_max, y_max, pixel_step_meters = load_geotiff_reorder(image_filepath, nir_rgb_order, normalize_channelwise, normalize_imagewise)
    label_polygons = load_geojson_labels(geojson_filepath)
    
    k_y = (y_max - y_min) / exim_np.shape[0]
    k_x = (x_max - x_min) / exim_np.shape[1]

    adjusted_polygons = []
    for polygon in label_polygons:
        adjusted_polygon = []
        for coordinate in polygon:
            newx = np.int32((coordinate[0] - (x_min)) / k_x)
            newy = np.int32(exim_np.shape[0] + ((y_min) - coordinate[1]) / k_y)  # y-coord grow up -- cv2 grows down
            adjusted_polygon.append(np.array([newx, newy]))

        adjusted_polygons.append(np.int32(np.array(adjusted_polygon).reshape((-1, 1, 2))))

    return exim_np, adjusted_polygons


def get_image_and_polygons_normalize(image_filepath, geojson_filepath, normalize_channelwise, normalize_imagewise):
    exim_np, x_min, y_min, x_max, y_max, _ = load_tiff_normalize(image_filepath, None, normalize_channelwise, normalize_imagewise)
    label_polygons = load_geojson_labels(geojson_filepath)
    
    k_y = (y_max - y_min) / exim_np.shape[0]
    k_x = (x_max - x_min) / exim_np.shape[1]

    adjusted_polygons = []
    for polygon in label_polygons:
        adjusted_polygon = []
        for coordinate in polygon:
            newx = np.int32((coordinate[0] - (x_min)) / k_x)
            newy = np.int32(exim_np.shape[0] + ((y_min) - coordinate[1]) / k_y)  # y-coord grow up -- cv2 grows down
            adjusted_polygon.append(np.array([newx, newy]))
            
        adjusted_polygons.append(np.int32(np.array(adjusted_polygon).reshape((-1, 1, 2))))

    return exim_np, adjusted_polygons


def load_tiff_normalize(
    filename: str,
    pixel_step_meters: float,
    normalize_channelwise: bool,
    normalize_imagewise: bool,
) -> tuple:
    with rasterio.open(filename) as img:
        exim = img.read()
        exim[exim < -3e38] = 0
        exim_np = exim.astype(np.float32)

        if exim.shape[0] < 30:
            exim_np = np.moveaxis(exim_np, 0, -1)

        if normalize_channelwise:
            exim_np = normalize_channelwise_to_uint8(exim_np)
        elif normalize_imagewise:
            exim_np = normalize_imagewise_to_uint8(exim_np)
        else:
            exim_np = np.uint8(np.clip(exim_np, 0, 255))

        x_min, y_min, x_max, y_max = img.bounds

    return exim_np, x_min, y_min, x_max, y_max, pixel_step_meters


def segmap_to_topo(image_np: np.ndarray, contours: list) -> np.ndarray:
    image_h, image_w = image_np.shape[:2]
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


def load_geotiff_reorder(
    filename: str,
    nir_rgb_order: list,
    normalize_channelwise: bool = False,
    normalize_imagewise: bool = False,
) -> tuple:
    with rasterio.open(filename) as img:
        exim = np.moveaxis(img.read(), 0, -1)
        exim[exim < -3e38] = 0  # set missing values to zero
        exim_np = np.float32(exim)
        if exim.shape[0] < 30:  # if channels are first, move them last
            exim_np = np.moveaxis(exim_np, 0, -1)

        exim_np = exim_np[:, :, nir_rgb_order]

        if normalize_channelwise:
            exim_np = normalize_channelwise_to_uint8(exim_np)
        elif normalize_imagewise:
            exim_np = normalize_imagewise_to_uint8(exim_np)
        else:
            exim_np = np.uint8(np.clip(exim_np, 0, 255))

        x_min, y_min, x_max, y_max = img.bounds

    pixel_step_meters = 0.5 # TODO. undefined
    return exim_np, x_min, y_min, x_max, y_max, pixel_step_meters


def load_geojson_labels(geojson_path: str) -> list:
    with open(geojson_path) as f:
        samples_json = json.load(f)

    coord_list = []
    for polygon in samples_json.get("features", []):
        geometry = polygon.get("geometry")
        if geometry and geometry.get("coordinates"):
            try:
                coordinates = geometry["coordinates"][0][0]
                if len(coordinates) > 2:
                    coord_list.append(coordinates)
            except (IndexError, TypeError) as e:
                print(f"Error processing polygon coordinates: {e}")

    return coord_list


def normalize_channelwise_to_uint8(exim_np: np.ndarray) -> np.ndarray:

    def normalize_channel(channel: np.ndarray) -> np.ndarray:
        min_val = getTopXValue(channel, 1)
        max_val = getTopXValue(channel, 99)
        if max_val != min_val:
            channel_scaled = (channel - min_val) / (max_val - min_val) * 255
            return np.clip(channel_scaled, 0, 255)
        else:
            return (np.zeros_like(channel) if min_val == 0 else np.ones_like(channel) * 255)

    exim_np = np.stack([normalize_channel(exim_np[:, :, i]) for i in range(exim_np.shape[-1])], axis=-1)
    return exim_np.astype(np.uint8)


def normalize_imagewise_to_uint8(exim_np: np.ndarray) -> np.ndarray:
    min_val = getTopXValue(exim_np, 1)
    max_val = getTopXValue(exim_np, 99)

    if max_val != min_val:
        exim_np_scaled = (exim_np - min_val) / (max_val - min_val) * 255
        exim_np_scaled = np.clip(exim_np_scaled, 0, 255)
    else:
        exim_np_scaled = np.zeros_like(exim_np)

    return exim_np_scaled.astype(np.uint8)


def getTopXValue(input_array: np.ndarray, percentage: float) -> float:
    all_nonzero_values = input_array[input_array > 0]
    if len(all_nonzero_values) == 0:
        return 0

    sorted_array = np.sort(all_nonzero_values)
    index = int(len(sorted_array) * percentage / 100)
    return sorted_array[index]