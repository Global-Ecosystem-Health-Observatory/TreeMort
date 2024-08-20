import io
import folium
import requests
import rasterio

import numpy as np
import streamlit as st

from rasterio.io import MemoryFile


# Check if rasterio.warp is available for reprojection
try:
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    WARP_AVAILABLE = True
except ImportError:
    WARP_AVAILABLE = False
    st.warning("Rasterio warp module is not available. Reprojection will not be possible.")


def fetch_and_verify_tiff(tiff_url):
    response = requests.get(tiff_url)
    if response.status_code == 200:
        try:
            with MemoryFile(io.BytesIO(response.content)) as memfile:
                with memfile.open() as dataset:
                    print(f"Original CRS: {dataset.crs}")

                    if WARP_AVAILABLE and dataset.crs.to_string() != "EPSG:4326":
                        dst_crs = "EPSG:4326"
                        transform, width, height = calculate_default_transform(
                            dataset.crs,
                            dst_crs,
                            dataset.width,
                            dataset.height,
                            *dataset.bounds,
                        )
                        kwargs = dataset.meta.copy()
                        kwargs.update(
                            {
                                "crs": dst_crs,
                                "transform": transform,
                                "width": width,
                                "height": height,
                            }
                        )

                        with MemoryFile() as memfile_reproj:
                            with memfile_reproj.open(**kwargs) as dst:
                                for i in range(1, dataset.count + 1):
                                    reproject(
                                        source=rasterio.band(dataset, i),
                                        destination=rasterio.band(dst, i),
                                        src_transform=dataset.transform,
                                        src_crs=dataset.crs,
                                        dst_transform=transform,
                                        dst_crs=dst_crs,
                                        resampling=Resampling.nearest,
                                    )
                                image = dst.read([1, 2, 3])
                                bounds = dst.bounds
                    else:
                        image = dataset.read([1, 2, 3])
                        bounds = dataset.bounds

                    print(f"Bounds after reprojecting (if needed): {bounds}")
                    return image, bounds
        except Exception as e:
            st.error(f"Error processing TIFF file with Rasterio: {e}")
            return None, None
    else:
        st.error(f"Error fetching TIFF file: {response.status_code}")
        return None, None


def normalize_image(image):
    image = np.moveaxis(image, 0, -1)  # Move channels to the last dimension
    image = (image / np.max(image) * 255).astype(np.uint8)  # Normalize to 0-255 and convert to uint8
    return image


def overlay_image_on_folium_map(image, bounds, geojson):
    map_center = [(bounds.top + bounds.bottom) / 2, (bounds.left + bounds.right) / 2]
    m = folium.Map(location=map_center, zoom_start=15)
 
    overlay_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

    folium.raster_layers.ImageOverlay(
        name="TIFF Overlay",
        image=image,
        bounds=overlay_bounds,
        opacity=0.6,
        interactive=True,
        cross_origin=False,
    ).add_to(m)

    folium.GeoJson(geojson, name="GeoJSON Overlay").add_to(m)
    folium.LayerControl().add_to(m)

    return m


def main():
    st.title("TIFF and GeoJSON Overlay Viewer")

    filename = st.text_input("Enter the filename (without extension)")

    if filename:
        response = requests.get(f"http://127.0.0.1:8000/map_data/{filename}")

        if response.status_code == 200:
            data = response.json()
            tiff_url = f"http://127.0.0.1:8000{data.get('tiff_url')}"
            geojson_url = f"http://127.0.0.1:8000{data.get('geojson_url')}"

            image, bounds = fetch_and_verify_tiff(tiff_url)
            if image is not None:
                normalized_image = normalize_image(image)

                geojson = data.get("geojson")

                map_overlay = overlay_image_on_folium_map(normalized_image, bounds, geojson)

                # Render the map in the Streamlit app
                st.components.v1.html(map_overlay._repr_html_(), height=600)
            else:
                st.error("Failed to process the TIFF image.")
        else:
            st.error(f"Error: {response.json().get('detail')}")
    else:
        st.error("Please enter a valid filename.")


if __name__ == "__main__":
    main()

''' Usage:

1) Start Uvicorn FastAPI server

    uvicorn treemort_api:app --reload

2) Start the application

    streamlit run treemort_app.py

''' 