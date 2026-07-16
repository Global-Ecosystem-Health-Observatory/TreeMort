import h5py
import random

from torch.utils.data import Sampler


class BalancedSampler(Sampler):
    def __init__(self, hdf5_file, keys, rank=0, world_size=1, seed=0):
        self.hdf5_file = hdf5_file
        self.keys = keys
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.dead_tree_indices = []
        self.no_dead_tree_indices = []

        with h5py.File(self.hdf5_file, "r") as hf:
            for idx, key in enumerate(self.keys):
                num_trees = hf[key].attrs.get("num_trees", 0)
                if num_trees > 0:
                    self.dead_tree_indices.append(idx)
                else:
                    self.no_dead_tree_indices.append(idx)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        min_count = min(len(self.dead_tree_indices), len(self.no_dead_tree_indices))
        # Deterministic per epoch so all ranks see the same global order before slicing
        rng = random.Random(self.seed + self.epoch)
        dead_sample = rng.sample(self.dead_tree_indices, min_count)
        no_dead_sample = rng.sample(self.no_dead_tree_indices, min_count)
        balanced = dead_sample + no_dead_sample
        rng.shuffle(balanced)

        per_rank = len(balanced) // self.world_size
        start = self.rank * per_rank
        return iter(balanced[start:start + per_rank])

    def __len__(self):
        total = 2 * min(len(self.dead_tree_indices), len(self.no_dead_tree_indices))
        return total // self.world_size
