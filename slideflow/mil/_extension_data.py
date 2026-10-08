"""Bag loading and sampling used only by the added MIL models."""

import numpy as np
import numpy.typing as npt
import torch
from typing import List
from slideflow.mil.data import BagDataset


class InstanceBagDataset(BagDataset):
    """Read feature or RGB tensor bags, including grouped slide paths."""

    def _load(self, index):
        bag = self.bags[index]
        if isinstance(bag, str):
            return torch.load(bag, map_location='cpu', weights_only=True).to(self.dtype)
        if isinstance(bag, (list, tuple)) or (isinstance(bag, np.ndarray)
                                            and bag.dtype.kind in ('O', 'U', 'S')):
            return torch.cat([torch.load(str(path), map_location='cpu', weights_only=True)
                              .to(self.dtype) for path in bag])
        return torch.as_tensor(bag, dtype=self.dtype)

    def __getitem__(self, index):
        bag = self._load(index)
        if not len(bag):
            raise ValueError('bags cannot be empty')
        if self.bag_size:
            samples = bag[torch.randperm(len(bag))[:self.bag_size]]
            padded = torch.cat([samples, bag.new_zeros((self.bag_size - len(samples),
                                                        *bag.shape[1:]))])
            return padded, len(samples)
        if self.max_bag_size and len(bag) > self.max_bag_size:
            return bag[torch.randperm(len(bag))[:self.max_bag_size]], self.max_bag_size
        return bag, len(bag)


class StratifiedShuffle:
    def __init__(self, strata: npt.NDArray) -> None:
        """Epoch ordering that approximately preserves the overall stratum mix.

        Used as the ``shuffle_fn`` of a FastAI DataLoader. Items are shuffled
        within each stratum and then interleaved at evenly spaced positions
        (with jitter), so any run of consecutive items, and therefore any
        batch, contains each stratum in close to its overall proportion.
        Every item is used exactly once per epoch.

        Args:
            strata (np.ndarray): Integer stratum for each item in the dataset.

        """
        self.strata = np.asarray(strata)

    def __call__(self, idxs: List[int]) -> List[int]:
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        idxs = np.asarray(idxs, dtype=int)
        if not len(idxs):
            return []
        keys, items = [], []
        for s in np.unique(self.strata[idxs]):
            members = rng.permutation(idxs[self.strata[idxs] == s])
            keys.append((np.arange(len(members)) + rng.uniform(size=len(members))) / len(members))
            items.append(members)
        order = np.argsort(np.concatenate(keys), kind='stable')
        return np.concatenate(items)[order].tolist()
