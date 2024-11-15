import cv2
import geojson
import rasterio
import rasterio.features

import numpy as np
import geopandas as gpd

from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon, shape
from skimage.feature import peak_local_max
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import watershed


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
    binary_mask = (binary_mask > 0).astype(np.uint8)
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


def save_labels_as_geojson(labels, transform, crs, output_path, min_area_threshold=1.0):
    geometries = []
    for label_value in np.unique(labels):
        if label_value == 0:
            continue  # Skip the background

        mask = (labels == label_value)

        shapes_gen = rasterio.features.shapes(mask.astype(np.int32), transform=transform)

        for shape_geom, shape_value in shapes_gen:
            if shape_value == 1:  # Ensure we only include shapes matching the label mask
                geometry = shape(shape_geom)
                if geometry.is_valid and not geometry.is_empty:
                    convex_hull = geometry.convex_hull
                    if convex_hull.is_valid and not convex_hull.is_empty:
                        geometries.append({'geometry': convex_hull, 'properties': {'label': int(label_value)}})
                    else:
                        print(f"Warning: Convex hull for label {label_value} is invalid or empty.")
                else:
                    print(f"Warning: Geometry for label {label_value} is invalid or empty.")

    if not geometries:
        print("No valid geometries found. GeoJSON will be empty.")
        return

    gdf = gpd.GeoDataFrame.from_features(geometries, crs=crs)

    if gdf.crs.is_geographic:
        projected_crs = gdf.estimate_utm_crs()
        print(f"Reprojecting to {projected_crs} for accurate area calculations...")
        gdf = gdf.to_crs(projected_crs)

    gdf = gdf[gdf['geometry'].area >= min_area_threshold]
    
    if gdf.empty:
        print("No valid geometries meet the area threshold. GeoJSON will be empty.")
        return

    gdf.to_file(output_path, driver="GeoJSON")
    print(f"GeoJSON saved to {output_path}")


def generate_watershed_labels(prediction_map, mask, min_distance=4, blur_sigma=2, dilation_radius=1):
    smoothed_prediction_map = gaussian_filter(prediction_map, sigma=blur_sigma)

    mask_grown = binary_dilation(mask, disk(dilation_radius))

    local_max = peak_local_max(
        smoothed_prediction_map,
        min_distance=min_distance,
        exclude_border=False,
        labels=mask_grown
    )

    markers = np.zeros_like(prediction_map, dtype=int)
    for i, (y, x) in enumerate(local_max):
        markers[y, x] = i + 1

    num_markers = np.count_nonzero(markers)
    if num_markers <= 1:
        return None

    labels = watershed(-smoothed_prediction_map, markers, mask=mask_grown)
    
    return labels


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
