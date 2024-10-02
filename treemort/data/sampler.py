import h5py
import random

from torch.utils.data import Sampler


class BalancedSampler(Sampler):
    def __init__(self, hdf5_file, keys):
        self.hdf5_file = hdf5_file
        self.keys = keys
        self.dead_tree_indices = []
        self.no_dead_tree_indices = []

        with h5py.File(self.hdf5_file, "r") as hf:
            for idx, key in enumerate(self.keys):
                contains_dead_tree = hf[key].attrs.get("contains_dead_tree", 0)
                if contains_dead_tree:
                    self.dead_tree_indices.append(idx)
                else:
                    self.no_dead_tree_indices.append(idx)

    def __iter__(self):
        min_count = min(len(self.dead_tree_indices), len(self.no_dead_tree_indices))

        dead_tree_sample = random.sample(self.dead_tree_indices, min_count)
        no_dead_tree_sample = random.sample(self.no_dead_tree_indices, min_count)

        balanced_indices = dead_tree_sample + no_dead_tree_sample
        random.shuffle(balanced_indices)

        return iter(balanced_indices)

    def __len__(self):
        return 2 * min(len(self.dead_tree_indices), len(self.no_dead_tree_indices))


class DatasetAwareBalancedSampler(Sampler):
    def __init__(self, hdf5_file_finnish, keys_finnish, hdf5_file_polish, keys_polish):
        self.hdf5_file_finnish = hdf5_file_finnish
        self.keys_finnish = keys_finnish
        self.dead_tree_indices_finnish = []
        self.no_dead_tree_indices_finnish = []

        self.hdf5_file_polish = hdf5_file_polish
        self.keys_polish = keys_polish
        self.dead_tree_indices_polish = []
        self.no_dead_tree_indices_polish = []

        with h5py.File(self.hdf5_file_finnish, "r") as hf_finnish:
            for idx, key in enumerate(self.keys_finnish):
                contains_dead_tree = hf_finnish[key].attrs.get("contains_dead_tree", 0)
                if contains_dead_tree:
                    self.dead_tree_indices_finnish.append(idx)
                else:
                    self.no_dead_tree_indices_finnish.append(idx)

        with h5py.File(self.hdf5_file_polish, "r") as hf_polish:
            for idx, key in enumerate(self.keys_polish):
                contains_dead_tree = hf_polish[key].attrs.get("contains_dead_tree", 0)
                if contains_dead_tree:
                    self.dead_tree_indices_polish.append(idx)
                else:
                    self.no_dead_tree_indices_polish.append(idx)

    def __iter__(self):
        min_dead_tree_count = min(len(self.dead_tree_indices_finnish), len(self.dead_tree_indices_polish))
        min_no_dead_tree_count = min(len(self.no_dead_tree_indices_finnish), len(self.no_dead_tree_indices_polish))

        dead_tree_sample_finnish = random.sample(self.dead_tree_indices_finnish, min_dead_tree_count)
        dead_tree_sample_polish = random.sample(self.dead_tree_indices_polish, min_dead_tree_count)

        no_dead_tree_sample_finnish = random.sample(self.no_dead_tree_indices_finnish, min_no_dead_tree_count)
        no_dead_tree_sample_polish = random.sample(self.no_dead_tree_indices_polish, min_no_dead_tree_count)

        balanced_indices_finnish = dead_tree_sample_finnish + no_dead_tree_sample_finnish
        balanced_indices_polish = dead_tree_sample_polish + no_dead_tree_sample_polish

        balanced_indices = balanced_indices_finnish + balanced_indices_polish
        random.shuffle(balanced_indices)

        return iter(balanced_indices)

    def __len__(self):
        return 2 * (min(len(self.dead_tree_indices_finnish), len(self.dead_tree_indices_polish)) +
                    min(len(self.no_dead_tree_indices_finnish), len(self.no_dead_tree_indices_polish)))