import os
import glob


def check_missing_labels(label_paths):
    missing_labels = [label for label in label_paths if not os.path.exists(label)]
    if missing_labels:
        print(f"Warning: The following label files are missing: {missing_labels}")


def get_image_label_paths_by_type(data_folder, dataset_type):
    image_path_pattern = os.path.join(data_folder, dataset_type, "Images", "*.npy")
    image_paths = glob.glob(image_path_pattern)
    label_paths = [
        os.path.join(data_folder, dataset_type, "Labels", os.path.basename(x))
        for x in image_paths
    ]

    check_missing_labels(image_paths)

    return image_paths, label_paths


def get_image_label_paths(data_folder):
    train_images, train_labels = get_image_label_paths_by_type(data_folder, "Train")
    test_images, test_labels = get_image_label_paths_by_type(data_folder, "Test")

    return train_images, train_labels, test_images, test_labels
