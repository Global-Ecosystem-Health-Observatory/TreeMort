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
        tiff_path = f"files/{filename}.tiff"
        geojson_path = f"files/{filename}.geojson"

        if not os.path.exists(tiff_path):
            raise HTTPException(
                status_code=404, detail=f"TIFF file not found: {tiff_path}"
            )

        if not os.path.exists(geojson_path):
            raise HTTPException(
                status_code=404, detail=f"GeoJSON file not found: {geojson_path}"
            )

        bounds = get_bounds(tiff_path)

        gdf = gpd.read_file(geojson_path)

        gdf = gdf.to_crs(epsg=4326)

        centroid = gdf.geometry.centroid.iloc[0]
        latitude = centroid.y
        longitude = centroid.x

        print(f"Calculated centroid - Latitude: {latitude}, Longitude: {longitude}")

        return {
            "bounds": [bounds.bottom, bounds.left, bounds.top, bounds.right],
            "latitude": latitude,
            "longitude": longitude,
            "geojson": gdf.to_json(),
            "tiff_url": f"/tiff/{filename}.tiff",  # Serve TIFF file through a URL
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tiff/{filename}")
async def get_tiff(filename: str):
    tiff_path = f"files/{filename}"
    if os.path.exists(tiff_path):
        return FileResponse(tiff_path, media_type="image/tiff")
    else:
        raise HTTPException(status_code=404, detail=f"TIFF file not found: {tiff_path}")
