import os
import cv2
import rasterio
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import optimize


def read_image(path, bands=None):
    with rasterio.open(path) as src:
        image_data = src.read(bands) if bands else src.read(1)
        metadata = src.meta.copy()
    return image_data, metadata


def save_tiff(image_data, image_path, metadata, nchannels=1):
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

    print(f"Image saved as TIFF at: {image_path}")


def grayscale_conversion(red_band, green_band, blue_band):
    return (0.2989 * red_band + 0.5870 * green_band + 0.1140 * blue_band).astype(
        np.uint8
    )


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def normalize_image(image, norm_min=0, norm_max=255):
    return cv2.normalize(image, None, norm_min, norm_max, cv2.NORM_MINMAX).astype(
        "uint8"
    )


def resize_image(image, size=(12000, 12000)):
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def mutual_information(hist_2d):
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = np.outer(px, py)
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def calculate_mutual_information(image1, image2, bins=256):
    hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=bins)
    return mutual_information(hist_2d)


def align_images_using_mutual_information(
    aerial_image, reference_image, initial_shift=(0, 0)
):
    def objective_function(shift):
        translated_image = np.roll(aerial_image, int(shift[0]), axis=0)
        translated_image = np.roll(translated_image, int(shift[1]), axis=1)
        return -calculate_mutual_information(translated_image, reference_image)

    result = optimize.minimize(objective_function, initial_shift, method="Powell")
    return result.x


def generate_edges(image, threshold1, threshold2):
    return cv2.Canny(image, threshold1, threshold2)


def apply_sobel(image):
    sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return normalize_image(gradient_magnitude)


def detect_contours(image, threshold=30):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, 255, thickness=1)
    return contour_image


def apply_difference_of_gaussians(image, kernel_size1=(5, 5), kernel_size2=(11, 11)):
    blurred1 = cv2.GaussianBlur(image, kernel_size1, 0)
    blurred2 = cv2.GaussianBlur(image, kernel_size2, 0)
    dog_image = cv2.absdiff(blurred1, blurred2)
    return normalize_image(dog_image)


def process_image_pair(image_path, chm_path, output_folder, method, save_intermediate):
    aerial_img, aerial_metadata = read_image(image_path, [1, 2, 3])
    chm_data, _ = read_image(chm_path)

    grayscale_img = grayscale_conversion(*aerial_img)
    enhanced_img = apply_clahe(grayscale_img)
    rescaled_chm_data = resize_image(chm_data)
    rescaled_chm_data_8u = normalize_image(rescaled_chm_data)
    edges_aerial = generate_edges(enhanced_img, 100, 200)

    if method == "edges":
        reference_image = generate_edges(rescaled_chm_data_8u, 50, 150)
    elif method == "sobel":
        reference_image = apply_sobel(rescaled_chm_data)
    elif method == "contour":
        reference_image = detect_contours(rescaled_chm_data)
    elif method == "dog":
        reference_image = apply_difference_of_gaussians(rescaled_chm_data)
    else:
        raise ValueError("Invalid method. Choose 'edges', 'sobel', or 'dog'.")

    optimal_shift = align_images_using_mutual_information(edges_aerial, reference_image)
    aligned_aerial_img = np.roll(grayscale_img, int(optimal_shift[0]), axis=0)
    aligned_aerial_img = np.roll(aligned_aerial_img, int(optimal_shift[1]), axis=1)

    aligned_image_output_path = os.path.join(output_folder, os.path.basename(image_path))
    save_tiff(aligned_aerial_img, aligned_image_output_path, aerial_metadata)

    print(f"Optimal Shift for {os.path.basename(image_path)} (using {method}): {optimal_shift}")
    print(f"Aligned image saved as TIFF at: {aligned_image_output_path}")


from tqdm import tqdm

def main(data_folder, method="dog", save_intermediate=False):
    image_folder = os.path.join(data_folder, 'Images')
    chm_folder = os.path.join(data_folder, 'CHM')

    if not os.path.exists(image_folder) or not os.path.exists(chm_folder):
        raise FileNotFoundError("One or both of the specified folders do not exist. Please check the paths.")

    output_folder = os.path.join(data_folder, 'Aligned_Images')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jp2')]
    chm_files = [f for f in os.listdir(chm_folder) if f.endswith('.tif')]

    image_paths = []
    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]
        chm_name = f"CHM_{image_name}_2016.tif"
        if chm_name in chm_files:
            image_paths.append((
                os.path.join(image_folder, image_file),
                os.path.join(chm_folder, chm_name)
            ))
    
    batch_size = 2  # Adjust based on available RAM
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    process_image_pair, image_path, chm_path, output_folder, method, save_intermediate
                ): (image_path, chm_path) for image_path, chm_path in batch
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
                image_path, chm_path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {os.path.basename(image_path)}: {e}")


if __name__ == "__main__":
    data_folder = os.getenv('DATA_PATH')
    if not data_folder:
        raise ValueError("DATA_PATH environment variable is not set. Please set it before running the script.")

    main(data_folder=data_folder, method="dog")