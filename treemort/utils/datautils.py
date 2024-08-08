import h5py

from sklearn.model_selection import train_test_split


def load_patch_keys(hdf5_file_path):
    with h5py.File(hdf5_file_path, "r") as hf:
        keys_with_dead_tree = []
        keys_without_dead_tree = []
        for key in hf.keys():
            if "contains_dead_tree" in hf[key].attrs:
                if hf[key].attrs["contains_dead_tree"] == 1:
                    keys_with_dead_tree.append(key)
                else:
                    keys_without_dead_tree.append(key)
    return keys_with_dead_tree, keys_without_dead_tree


def stratified_split(hdf5_file_path, val_ratio, test_ratio):
    keys_with_dead_tree, keys_without_dead_tree = load_patch_keys(hdf5_file_path)

    total_with = len(keys_with_dead_tree)
    total_without = len(keys_without_dead_tree)

    val_size_with = int(val_ratio * total_with)
    test_size_with = int(test_ratio * total_with)

    val_size_without = int(val_ratio * total_without)
    test_size_without = int(test_ratio * total_without)

    train_keys_with, val_test_keys_with = train_test_split(keys_with_dead_tree, test_size=(val_size_with + test_size_with), random_state=42)
    val_keys_with, test_keys_with = train_test_split(val_test_keys_with, test_size=test_size_with, random_state=42)

    train_keys_without, val_test_keys_without = train_test_split(keys_without_dead_tree, test_size=(val_size_without + test_size_without), random_state=42,)
    val_keys_without, test_keys_without = train_test_split(val_test_keys_without, test_size=test_size_without, random_state=42)

    train_keys = train_keys_with + train_keys_without
    val_keys = val_keys_with + val_keys_without
    test_keys = test_keys_with + test_keys_without

    return train_keys, val_keys, test_keys
