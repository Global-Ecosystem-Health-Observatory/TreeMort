import geopandas as gpd
from shapely.ops import polygonize
from shapely.geometry import MultiPolygon
import os

def fix_geometry(geometry):
    """
    Fix an invalid geometry by attempting to repair self-intersections.
    
    Parameters:
    - geometry (shapely.geometry): The input geometry to be fixed.

    Returns:
    - shapely.geometry: A valid geometry or the original if it cannot be fixed.
    """
    if not geometry.is_valid:
        try:
            # Buffer with zero distance can sometimes fix minor self-intersections
            fixed_geometry = geometry.buffer(0)
            
            # If the buffer operation results in an invalid or empty geometry, use polygonize
            if not fixed_geometry.is_valid or fixed_geometry.is_empty:
                fixed_geometry = MultiPolygon(polygonize(geometry))
                
            if fixed_geometry.is_valid:
                return fixed_geometry
            else:
                return geometry  # Return the original if the fix fails
        except Exception:
            return geometry  # Return the original geometry if an error occurs
    else:
        return geometry

def fix_geometries_in_folder(input_folder, output_folder):
    """
    Fix geometries in all GeoJSON files in a folder and save them in a new folder.

    Parameters:
    - input_folder (str): Path to the folder containing the original GeoJSON files.
    - output_folder (str): Path to the folder where corrected GeoJSON files will be saved.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.geojson'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                # Load the GeoJSON file
                gdf = gpd.read_file(input_path)
                
                # Keep track of whether any geometry has been fixed
                original_gdf = gdf.copy()
                gdf['geometry'] = gdf['geometry'].apply(fix_geometry)
                
                # Check if any geometries were changed
                if not gdf.equals(original_gdf):
                    # Save the corrected GeoDataFrame to the output folder
                    gdf.to_file(output_path, driver='GeoJSON')
                    print(f"Corrected file saved: {output_path}")
                else:
                    # Copy the original file without modifications if no changes were made
                    gdf.to_file(output_path, driver='GeoJSON')
                    print(f"No changes needed; file saved as is: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage:
input_folder = '/Users/anisr/Documents/copenhagen_data/Predictions'
output_folder = '/Users/anisr/Documents/copenhagen_data/Corrected_Predictions'
fix_geometries_in_folder(input_folder, output_folder)


import geopandas as gpd
from shapely.validation import make_valid
from shapely.geometry import box
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def validate_and_fix_geometry(geom):
    """
    Validate and fix geometry. Return None if the geometry cannot be fixed.
    """
    try:
        # Apply make_valid to fix the geometry
        valid_geom = make_valid(geom)
        if valid_geom.is_valid and not valid_geom.is_empty:
            return valid_geom
        else:
            return None
    except Exception:
        return None

def process_prediction_file(prediction_path, ground_truth_segments):
    """
    Process a single prediction file and calculate IoU values against the ground truth.

    Parameters:
    - prediction_path (str): Path to the prediction GeoJSON file.
    - ground_truth_segments (GeoDataFrame): The ground truth GeoDataFrame.

    Returns:
    - list: IoU values for the prediction file.
    """
    try:
        predicted_segments = gpd.read_file(prediction_path)

        # Check if the GeoDataFrame is empty
        if predicted_segments.empty:
            print(f"File {prediction_path} is empty. Skipping...")
            return []

        # Ensure both GeoDataFrames use the same CRS
        if predicted_segments.crs != ground_truth_segments.crs:
            ground_truth_segments = ground_truth_segments.to_crs(predicted_segments.crs)

        # Clean and validate predicted geometries
        predicted_segments['geometry'] = predicted_segments['geometry'].apply(validate_and_fix_geometry)
        predicted_segments = predicted_segments[predicted_segments['geometry'].notnull()]

        # Check again if the GeoDataFrame is empty after cleaning
        if predicted_segments.empty:
            print(f"File {prediction_path} has no valid geometries after cleaning. Skipping...")
            return []

        # Get the bounding box of the predicted segments
        bounds = predicted_segments.total_bounds  # [minx, miny, maxx, maxy]
        bounding_box = box(*bounds)

        # Filter ground truth segments to those within the bounds of the predicted geometries
        filtered_ground_truth = ground_truth_segments[ground_truth_segments.intersects(bounding_box)]

        # Calculate IoU for overlapping segments
        iou_values = []

        for pred_geom in predicted_segments.geometry:
            for truth_geom in filtered_ground_truth.geometry:
                if pred_geom.is_valid and truth_geom.is_valid:
                    if pred_geom.intersects(truth_geom):
                        intersection_area = pred_geom.intersection(truth_geom).area
                        union_area = pred_geom.union(truth_geom).area
                        iou = intersection_area / union_area if union_area != 0 else 0
                        iou_values.append(iou)

        return iou_values

    except Exception as e:
        print(f"Error processing {prediction_path}: {e}")
        return []

def calculate_mean_iou_multithreaded(predictions_dir, ground_truth_path):
    """
    Calculate the mean IoU for all GeoJSON predictions in a directory against a ground truth GPKG file using multithreading.

    Parameters:
    - predictions_dir (str): Path to the directory containing prediction GeoJSON files.
    - ground_truth_path (str): Path to the ground truth GPKG file.

    Returns:
    - float: The mean IoU for all predictions.
    """
    # Load the ground truth segments
    ground_truth_segments = gpd.read_file(ground_truth_path)

    # Check if the ground truth is empty
    if ground_truth_segments.empty:
        print("The ground truth GeoDataFrame is empty. Exiting...")
        return 0

    # Ensure ground truth geometries are valid
    ground_truth_segments['geometry'] = ground_truth_segments['geometry'].apply(validate_and_fix_geometry)
    ground_truth_segments = ground_truth_segments[ground_truth_segments['geometry'].notnull()]

    all_iou_values = []

    # Get a list of GeoJSON files in the directory
    prediction_files = [os.path.join(predictions_dir, filename) for filename in os.listdir(predictions_dir) if filename.endswith('.geojson')]

    # Use ThreadPoolExecutor for multithreading and tqdm for progress
    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_prediction_file, prediction_path, ground_truth_segments): prediction_path for prediction_path in prediction_files}
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing Files"):
            try:
                iou_values = future.result()
                all_iou_values.extend(iou_values)
            except Exception as e:
                print(f"Error processing file {future_to_file[future]}: {e}")

    # Calculate and return the mean IoU
    mean_iou = np.mean(all_iou_values) if all_iou_values else 0
    return mean_iou

# Example usage:
predictions_dir = "/Users/anisr/Documents/copenhagen_data/Corrected_Predictions"
ground_truth_path = "/Users/anisr/Documents/copenhagen_data/labels/target_features_20241031.gpkg"
mean_iou = calculate_mean_iou_multithreaded(predictions_dir, ground_truth_path)
print(f"Mean IoU for all predictions: {mean_iou:.4f}")