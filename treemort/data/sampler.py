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
                num_trees = hf[key].attrs.get("num_trees", 0)
                if num_trees > 0:
                    self.dead_tree_indices.append(idx)
                else:
                    self.no_dead_tree_indices.append(idx)

    def __iter__(self):
        min_count = min(len(self.dead_tree_indices), 
                       len(self.no_dead_tree_indices))
        
        dead_sample = random.sample(self.dead_tree_indices, min_count)
        no_dead_sample = random.sample(self.no_dead_tree_indices, min_count)
        
        balanced = dead_sample + no_dead_sample
        random.shuffle(balanced)
        
        return iter(balanced)

    def __len__(self):
        return 2 * min(len(self.dead_tree_indices),
                      len(self.no_dead_tree_indices))
