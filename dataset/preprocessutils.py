import json

import cv2
import numpy as np
import rasterio
from tqdm import tqdm


def create_label_mask(img_arr: np.ndarray, polys: list[np.ndarray]):
    """Creates a binary label mask from annotated polygons

    Parameters
    ----------
    img_arr : np.ndarray
        image for which the annotations have been extracted
    polys : list[np.ndarray]
        annotated polygons

    Returns
    -------
    np.ndarray
        binary label mask (1=foreground, 0=background)
    """

    image_h, image_w = img_arr.shape[:2]
    label_mask = np.zeros((image_h, image_w), dtype=np.float32)

    for poly in tqdm(polys, desc="Creating label masks"):
        cv2.fillPoly(label_mask, pts=[poly], color=1)

    return label_mask


def segmap_to_topo(img_arr: np.ndarray, contours: list) -> np.ndarray:
    """Creates a topographic label mask

    Parameters
    ----------
    img_arr : np.ndarray
        array of image values
    contours : list[np.ndarray]
        coordinates of vertices of polygons in image coordinates

    Returns
    -------
    np.ndarray
        topographic label mask

    Notes
    -----
    A topographic label mask is a mask in which the center of each polygon gets
    the maximum value (255) and the values decrease towards the polygon borders.
    Zeros represent background pixels.

    """
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
    nir_r_g_b_order: list[int],
    normalize_channelwise: bool,
    normalize_imagewise: bool,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Reads and preprocesses an image and corresponding annotations

    Parameters
    ----------
    image_filepath : str
        filepath to the image file
    geojson_filepath : str
        filepath to the annotation file
    nir_r_g_b_order : list[int]
        order of nir, r, g, and b channels in the image
    normalize_channelwise : bool
        whether to apply channelwise normalization
    normalize_imagewise : bool
        whether to apply imagewise normalization

    Returns
    -------
    img_arr : np.ndarray
        array of image values. Dimensions are (rows, cols, channels)
    adjusted_polygons : list[np.ndarray]
        coordinates of vertices of annotated polygons in image coordinates
    """

    img_arr, bounds, resolution = load_geotiff(
        image_filepath,
        nir_r_g_b_order,
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
    """Converts georeferenced coordinates into image coordinates

    Parameters
    ----------
    geocoord : list
        x and y geocoordinates
    min_xy : tuple
        left and bottom bounds of image in geocoordinates
    resolution : tuple
        pixel resolution in x and y directions
    n_rows : int
        number of rows in the image

    Returns
    -------
    tuple[int]
        col and row index of input geocoordinate
    """

    x,y = geocoord
    min_x, min_y = min_xy
    res_x, res_y = resolution

    col = np.int32((x - min_x) / res_x) 
    row = np.int32(n_rows + (min_y - y) / res_y)

    return col, row


def load_geotiff(
    filename: str,
    nir_r_g_b_order: list = None,
    normalize_channelwise: bool = False,
    normalize_imagewise: bool = False,
) -> tuple[np.ndarray, tuple[float], tuple[float]]:
    """Reads image, reorders channels, and optionally scales image

    Parameters
    ----------
    filename : str
        filepath to image file
    nir_r_g_b_order : list
        indices of nir, r, g, and b channels, respectively, by default [3,0,1,2]
    normalize_channelwise : bool, optional
        whether to perform channelwise normalization, by default False
    normalize_imagewise : bool, optional
        whether to perform imagewise normalization, by default False

    Returns
    -------
    img_arr : np.ndarray
        array of image values. Dimensions are (rows, cols, channels)
    bounds : tuple
        left, bottom, right, and top of image in georeferenced coordinates
    res : tuple
        x and y resolution of image in meters
    """

    if nir_r_g_b_order is None:
        nir_r_g_b_order = [3,0,1,2]
           
    with rasterio.open(filename) as img:

        # Read pixel values and move channels last
        img_arr = np.moveaxis(img.read(), 0, -1).astype(np.float32)

        # Reorder channels
        img_arr = img_arr[:,:,nir_r_g_b_order]

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
    """Extracts coordinates of label polygon vertices

    Parameters
    ----------
    geojson_path : str
        filepath to label geojson file

    Returns
    -------
    list[list[list[float]]]
        coordinates of vertices of polygons in georeferenced coordinates

    Notes
    -----
    Each polygon in a multipolygon is stored separately
        
    """
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
    """Scales channels to range 0-255 and switches data type to uint8

    Parameters
    ----------
    img_arr : np.ndarray
        multi-channel image

    Returns
    -------
    np.ndarray
        input image scaled channelwise
    """

    scaled_channels = [
        normalize_imagewise_to_uint8(img_arr[:,:,i]) 
        for i 
        in range(img_arr.shape[-1])
    ]

    return np.stack(scaled_channels, axis=-1)


def normalize_imagewise_to_uint8(img_arr: np.ndarray) -> np.ndarray:
    """Scales image to range 0-255 and switches data type to uint8

    Parameters
    ----------
    img_arr : np.ndarray
        multi-channel image or single channel of image

    Returns
    -------
    np.ndarray
        scaled image
    """

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
    """Gets specified percentiles of values in input array excluding zeros

    Parameters
    ----------
    input_array : np.ndarray
        array from which the percentile is extracted
    percentages : array_like of float
        percentages to extract

    Returns
    -------
    float
        specified percentiles of array from which zeros have been excluded
    """

    all_nonzero_values = input_array[input_array > 0]
    if len(all_nonzero_values) == 0:
        return 0

    return np.percentile(all_nonzero_values, percentages)
