import os
import geopandas as gpd
import rasterio

def summarize_geojsons(root_dir, _):
    total_files = 0
    total_area_km2 = 0.0
    total_dead_trees = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.geojson'):
                geojson_path = os.path.join(dirpath, filename)
                image_base = os.path.join(
                    os.path.dirname(os.path.dirname(geojson_path)),
                    "Images",
                    os.path.splitext(os.path.basename(geojson_path))[0]
                )
                image_path = None
                for ext in ['.tif', '.tiff']:
                    candidate = image_base + ext
                    if os.path.exists(candidate):
                        image_path = candidate
                        break

                try:
                    gdf = gpd.read_file(geojson_path)
                    total_files += 1

                    # Ensure geometries are valid
                    gdf = gdf[gdf.geometry.notnull()]
                    gdf = gdf[gdf.geometry.is_valid]

                    # Count dead trees
                    total_dead_trees += len(gdf)

                    # Get image bounds and resolution to calculate area
                    if image_path is not None:
                        with rasterio.open(image_path) as src:
                            bounds = src.bounds
                            res = src.res
                            width_m = abs(bounds.right - bounds.left)
                            height_m = abs(bounds.top - bounds.bottom)
                            area_km2 = (width_m * height_m) / 1e6
                            total_area_km2 += area_km2
                    else:
                        print(f"Image not found for: {geojson_path}")

                except Exception as e:
                    print(f"Error processing {geojson_path}: {e}")

    avg_density_per_ha = (total_dead_trees / (total_area_km2 * 100)) if total_area_km2 > 0 else 0

    summary = {
        "Image Count": total_files,
        "Total Area Covered (km²)": round(total_area_km2, 2),
        "Annotated Dead Trees": total_dead_trees,
        "Avg. Tree Density (per ha)": round(avg_density_per_ha, 2)
    }

    return summary

# Example usage:
if __name__ == "__main__":
    geojson_dir = "/Users/anisr/Documents/dead_trees/Poland/RGBNIR/25cm"
    stats = summarize_geojsons(geojson_dir, None)
    for key, value in stats.items():
        print(f"{key}: {value}")
