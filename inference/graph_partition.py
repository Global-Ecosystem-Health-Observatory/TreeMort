import os
import torch
import rasterio

import numpy as np
import networkx as nx
import geopandas as gpd

from scipy import ndimage
from skimage import measure
from skimage.feature import peak_local_max
from skimage import segmentation, filters
from sklearn.cluster import SpectralClustering
from shapely.geometry import Polygon
from rasterio.features import rasterize
from networkx.algorithms import community
from sklearn.cluster import SpectralClustering

from treemort.utils.logger import get_logger

logger = get_logger(__name__)


def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()
        transform = src.transform
        image_shape = (src.height, src.width)
    return image, transform, image_shape


def load_geodataframes(prediction_path, ground_truth_path):
    predicted_gdf = gpd.read_file(prediction_path)
    groundtruth_gdf = gpd.read_file(ground_truth_path)
    return predicted_gdf, groundtruth_gdf


def polygons_to_mask(geo_df, image_shape, transform):
    if geo_df.crs is None:
        raise ValueError("GeoDataFrame does not have a CRS defined.")

    if geo_df.empty:
        logger.warning("The GeoDataFrame is empty.")
        return np.zeros(image_shape, dtype=np.uint8)

    shapes = [(geom, 1) for geom in geo_df.geometry if geom.is_valid]

    if not shapes:
        logger.warning("No valid geometries found in the GeoDataFrame.")
        return np.zeros(image_shape, dtype=np.uint8)

    mask = rasterize(shapes, out_shape=image_shape, transform=transform, fill=0, default_value=1, all_touched=True)
    return mask


def create_pixel_graph(image, mask):
    height, width = mask.shape
    G = nx.grid_2d_graph(height, width, periodic=False)

    for r, c in list(G.nodes):
        if mask[r, c] == 1:
            G.nodes[(r, c)]['rgb'] = image[:3, r, c]
            G.nodes[(r, c)]['nir'] = image[3, r, c]

    nodes_to_remove = [(r, c) for (r, c) in G.nodes if mask[r, c] == 0]
    G.remove_nodes_from(nodes_to_remove)

    return G


def create_graph(predicted_mask, image):
    labels = measure.label(predicted_mask)
    G = nx.Graph()
    regions = measure.regionprops(labels, intensity_image=image.transpose(1, 2, 0))

    for region in regions:
        if region.area >= 4:
            mean_color = region.mean_intensity
            G.add_node(region.label, mean_color=mean_color)

    for i, region1 in enumerate(regions):
        for j, region2 in enumerate(regions):
            if i != j:
                # Weighted edges based on spatial overlap or proximity
                min_row1, min_col1, max_row1, max_col1 = region1.bbox
                min_row2, min_col2, max_row2, max_col2 = region2.bbox
                overlap = min_row1 < max_row2 and max_row1 > min_row2 and min_col1 < max_col2 and max_col1 > min_col2
                if overlap:
                    distance = np.linalg.norm(np.array(region1.centroid) - np.array(region2.centroid))
                    weight = 1 / (distance + 1e-5)  # Inverse distance as weight
                    G.add_edge(region1.label, region2.label, weight=weight)

    return G, labels


import warnings

def process_connected_components(G, n_clusters=None):
    components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    partitioned_labels = {}

    for subgraph in components:
        num_nodes = len(subgraph.nodes)
        if num_nodes > 1:
            clusters = min(n_clusters or num_nodes, num_nodes)
            adj_matrix = nx.to_numpy_array(subgraph)
            if clusters >= num_nodes:
                warnings.warn(f"Adjusting clusters ({clusters}) to fewer than nodes ({num_nodes}).")
            logger.info(f"SpectralClustering: clusters={clusters}, num_nodes={num_nodes}")
            clustering = SpectralClustering(n_clusters=clusters, affinity='precomputed').fit(adj_matrix)
            for i, node in enumerate(subgraph.nodes):
                partitioned_labels[node] = clustering.labels_[i]
        else:
            for node in subgraph.nodes:
                partitioned_labels[node] = len(partitioned_labels) + 1
    
    return partitioned_labels


def label_graph_nodes(G):
    communities = community.greedy_modularity_communities(G)
    community_dict = {}
    for i, community_nodes in enumerate(communities):
        for node in community_nodes:
            community_dict[node] = i
    return community_dict


def spectral_clustering(G, n_clusters=None):
    connected_components = list(nx.connected_components(G))
    if len(connected_components) > 1:
        cluster_labels = process_connected_components(G, n_clusters=n_clusters)
    else:
        adj_matrix = nx.to_numpy_array(G)
        clustering = SpectralClustering(n_clusters=n_clusters or len(G.nodes), affinity='precomputed').fit(adj_matrix)
        cluster_labels = {node: clustering.labels_[i] for i, node in enumerate(G.nodes)}

    for node, label in cluster_labels.items():
        G.nodes[node]['cluster'] = label

    return G


def save_image(image_data, image_path, metadata, nchannels=1):
    metadata.update(
        {
            "dtype": "uint8",
            "count": nchannels,
            "driver": "GTiff",
        }
    )
    clipped_image_data = np.clip(image_data, 0, 255).astype(np.uint8)

    with rasterio.open(image_path, "w", **metadata) as dst:
        if nchannels == 3:
            for i in range(3):
                dst.write(clipped_image_data[:, :, i], i + 1)
        else:
            dst.write(clipped_image_data, 1)
    logger.info(f"Image saved as TIFF at: {image_path}")


def decompose_large_segments(labels, min_distance=2, sigma=2):
    distance = ndimage.distance_transform_edt(labels > 0)
    smoothed_distance = filters.gaussian(distance, sigma=sigma)
    local_maxi = peak_local_max(smoothed_distance, min_distance=min_distance, labels=labels, exclude_border=False)
    markers = np.zeros_like(labels, dtype=np.int32)

    for idx, (row, col) in enumerate(local_maxi):
        markers[row, col] = idx + 1  # Use non-zero values as marker labels

    refined_labels = segmentation.watershed(-smoothed_distance, markers, mask=labels > 0)
    return refined_labels


def save_polygons_as_geojson(labels, transform, output_geojson_path):
    contours = measure.find_contours(labels, level=0.5)
    polygons = [Polygon([(transform * (c[1], c[0])) for c in contour]) for contour in contours]

    gdf = gpd.GeoDataFrame(geometry=polygons)
    gdf.to_file(output_geojson_path, driver='GeoJSON')
    logger.info(f"GeoJSON saved at: {output_geojson_path}")


def perform_graph_partitioning(image, predicted_mask, min_distance=2, sigma=2):
    G, labels = create_graph(predicted_mask, image)

    partitioned_labels = torch.zeros_like(predicted_mask, dtype=torch.int32)
    component_id = 1  # Start component IDs from 1

    for component in nx.connected_components(G):
        subgraph = G.subgraph(component).copy()
        n_nodes = len(subgraph.nodes)

        if n_nodes < 2:
            for node in subgraph.nodes:
                partitioned_labels[labels == node] = component_id
            component_id += 1
            continue

        adj_matrix = nx.to_numpy_array(subgraph)

        try:
            n_clusters = min(len(subgraph.nodes), 48, adj_matrix.shape[0] - 1)  # Combine the constraints
            if n_clusters < 1:
                logger.warning(f"Skipping clustering: n_clusters={n_clusters}, adj_matrix.shape={adj_matrix.shape}")
                continue  # Skip clustering if not feasible

            clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42).fit(adj_matrix)

            for i, node in enumerate(subgraph.nodes):
                partitioned_labels[labels == node] = component_id + clustering.labels_[i]
            component_id += n_clusters
        except Exception as e:
            logger.error(f"Clustering failed for component with {n_nodes} nodes. Error: {e}")

    return partitioned_labels


def refine_elliptical_regions_with_graph(labels, intensity_image):
    intensity_image = intensity_image.mean(axis=0) if intensity_image.ndim == 3 else intensity_image

    if labels.shape != intensity_image.shape:
        raise ValueError(f"Shape mismatch: labels {labels.shape} and intensity_image {intensity_image.shape}")

    refined_labels = labels.copy()

    regions = measure.regionprops(labels, intensity_image=intensity_image)
    G = nx.Graph()

    for region in regions:
        G.add_node(region.label, centroid=(0, 0), mean_intensity=0)  # Default attributes
        if region.area >= 10:
            G.nodes[region.label]['centroid'] = region.centroid
            G.nodes[region.label]['mean_intensity'] = region.mean_intensity

    # Add edges based on proximity and intensity difference
    for region1 in regions:
        for region2 in regions:
            if region1.label != region2.label:
                dist = np.linalg.norm(np.array(region1.centroid) - np.array(region2.centroid))
                intensity_diff = abs(region1.mean_intensity - region2.mean_intensity)
                if dist < 15 and intensity_diff < 0.2:  # Thresholds for proximity and intensity
                    G.add_edge(region1.label, region2.label, weight=1 / (dist + 1e-5))

    # Add edges between disconnected components
    components = list(nx.connected_components(G))
    for i in range(len(components) - 1):
        node1 = list(components[i])[0]
        node2 = list(components[i + 1])[0]
        centroid1 = G.nodes[node1]['centroid']
        centroid2 = G.nodes[node2]['centroid']
        dist = np.linalg.norm(np.array(centroid1) - np.array(centroid2))
        G.add_edge(node1, node2, weight=1 / (dist + 1e-5))

    num_nodes = len(G.nodes)
    n_clusters = min(num_nodes - 1, 10)

    if num_nodes < 2:
        for node in G.nodes:
            refined_labels[labels == node] = node  # Assign unique label
        return refined_labels

    adjacency_matrix = nx.to_numpy_array(G)
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed').fit(adjacency_matrix)

    for i, region in enumerate(regions):
        refined_labels[labels == region.label] = clustering.labels_[i] + 1

    return refined_labels


def main(image_path, prediction_path, ground_truth_path, output_folder, output_image_name="partitioned_labels.tif"):
    image, transform, _ = load_image(image_path)
    predicted_gdf, _ = load_geodataframes(prediction_path, ground_truth_path)

    with rasterio.open(image_path) as src:
        if predicted_gdf.crs != src.crs:
            predicted_gdf = predicted_gdf.to_crs(src.crs)
        predicted_mask = polygons_to_mask(predicted_gdf, (src.height, src.width), src.transform)

    refined_labels = perform_graph_partitioning(image, predicted_mask, min_distance=3, sigma=2)

    output_image_path = os.path.join(output_folder, output_image_name)
    save_image(refined_labels, output_image_path, src.meta, nchannels=1)

    output_geojson_path = os.path.join(output_folder, "partitioned_labels.geojson")
    save_polygons_as_geojson(refined_labels, transform, output_geojson_path)


if __name__ == "__main__":
    image_path = "/Users/anisr/Documents/copenhagen_data/Images/patches_3095_377.tif"
    prediction_path = "/Users/anisr/Documents/copenhagen_data/Predictions/patches_3095_377.geojson"
    ground_truth_path = "/Users/anisr/Documents/copenhagen_data/labels/target_features_20241031.gpkg"
    output_folder = "/Users/anisr/Documents/copenhagen_data/Output"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    main(image_path, prediction_path, ground_truth_path, output_folder)
