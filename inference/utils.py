import cv2
import geojson
import rasterio

import numpy as np
import geopandas as gpd

from shapely.geometry import Polygon, shape, mapping

from treemort.utils.config import setup
from treemort.modeling.builder import build_model


def load_and_preprocess_image(tiff_file, nir_rgb_order):
    with rasterio.open(tiff_file) as src:
        image = src.read()
        image = image.astype(np.float32) / 255.0
        transform = src.transform
        crs = src.crs

    image = image[nir_rgb_order, :, :]
    return image, transform, crs


def pad_image(image, window_size):
    c, h, w = image.shape

    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size

    padded_image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

    return padded_image


def threshold_prediction_map(prediction_map, threshold=0.5):
    binary_mask = (prediction_map >= threshold).astype(np.uint8)
    return binary_mask


def extract_contours(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def apply_transform(contour, transform):
    transformed_contour = np.array([transform * (x, y) for x, y in contour])
    return transformed_contour


def contours_to_geojson(contours, transform, crs, name):
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
            print("CRS is not in EPSG format; setting CRS to null in GeoJSON.")
    
    geojson = {
        "type": "FeatureCollection",
        "name": name,
        "crs": geojson_crs,
        "features": []
    }

    for contour in contours:
        if len(contour) >= 3:  # Ensure valid contour (at least 3 points)
            contour = contour.reshape(-1, 2)
            contour = apply_transform(contour, transform)

            # Ensure the polygon is closed
            if not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack([contour, contour[0]])

            polygon = Polygon(contour)
            new_feature = {
                "type": "Feature",
                "properties": {},  # Initialize with empty properties or add relevant properties
                "geometry": {"type": "Polygon", "coordinates": [contour.tolist()]},
            }
            geojson["features"].append(new_feature)
        else:
            print(f"Skipped contour with {len(contour)} points")

    return geojson


def save_geojson(geojson_data, output_path):
    with open(output_path, "w") as f:
        geojson.dump(geojson_data, f, indent=2, ensure_ascii=False)


def calculate_iou(geojson_path, predictions_path):
    true_gdf = gpd.read_file(geojson_path)
    pred_gdf = gpd.read_file(predictions_path)

    # TODO. find the root cause
    # Fix invalid geometries
    true_gdf["geometry"] = true_gdf["geometry"].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
    pred_gdf["geometry"] = pred_gdf["geometry"].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)

    true_union = true_gdf.geometry.union_all()
    pred_union = pred_gdf.geometry.union_all()

    intersection = true_union.intersection(pred_union)
    union = true_union.union(pred_union)

    intersection_area = intersection.area
    union_area = union.area

    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return iou
