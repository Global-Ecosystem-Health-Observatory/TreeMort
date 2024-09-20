import os
import rasterio

import geopandas as gpd

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse


app = FastAPI()


def get_bounds(tiff_file):
    with rasterio.open(tiff_file) as src:
        bounds = src.bounds
    return bounds


@app.get("/map_data/{filename}")
async def map_data(filename: str):
    try:
        # Assume TIFF and GeoJSON files are in a specific directory
        data_folder = "/Users/anisr/Documents/AerialImages"

        tiff_path = os.path.join(data_folder, "4band_25cm", filename)
        geojson_path = os.path.join(data_folder, "Geojsons", os.path.splitext(filename)[0] + ".geojson")
        predictions_path = os.path.join(data_folder, "predictions", os.path.splitext(filename)[0] + ".geojson")
        
        print(tiff_path, geojson_path, predictions_path)

        if not os.path.exists(tiff_path):
            raise HTTPException(
                status_code=404, detail=f"TIFF file not found: {tiff_path}"
            )

        if not os.path.exists(geojson_path):
            raise HTTPException(
                status_code=404, detail=f"GeoJSON file not found: {geojson_path}"
            )
        
        if not os.path.exists(predictions_path):
            raise HTTPException(
                status_code=404, detail=f"GeoJSON file with predictions not found: {predictions_path}"
            )

        bounds = get_bounds(tiff_path)

        gdf = gpd.read_file(geojson_path)
        gdf = gdf.to_crs(epsg=4326)

        gdf_pred = gpd.read_file(predictions_path)
        gdf_pred = gdf_pred.to_crs(epsg=4326)

        centroid = gdf.geometry.centroid.iloc[0]
        latitude = centroid.y
        longitude = centroid.x

        print(f"Calculated centroid - Latitude: {latitude}, Longitude: {longitude}")

        return {
            "bounds": [bounds.bottom, bounds.left, bounds.top, bounds.right],
            "latitude": latitude,
            "longitude": longitude,
            "geojson": gdf.to_json(),
            "geojson_pred": gdf_pred.to_json(),
            "tiff_url": f"/tiff/{filename}",  # Serve TIFF file through a URL
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# N4212G_2013_1.tiff

@app.get("/tiff/{filename}")
async def get_tiff(filename: str):
    data_folder = "/Users/anisr/Documents/AerialImages"

    tiff_path = os.path.join(data_folder, "4band_25cm", filename)

    if os.path.exists(tiff_path):
        return FileResponse(tiff_path, media_type="image/tiff")
    else:
        raise HTTPException(status_code=404, detail=f"TIFF file not found: {tiff_path}")
