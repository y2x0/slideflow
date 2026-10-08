"""Bag loading and sampling used only by the added MIL models."""

import numpy as np
import numpy.typing as npt
import torch
from os import PathLike
from pathlib import Path
from typing import List
from slideflow.mil.data import BagDataset


def resolve_tile_bags(bags):
    """Expand tile bag directories without changing Slideflow's feature discovery."""
    items = [bags] if isinstance(bags, (str, PathLike)) else bags
    paths = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            records = sorted(p for p in path.iterdir()
                             if p.is_file() and p.suffix in ('.tfrecord', '.tfrecords'))
            tensors = sorted(path.glob('*.pt'))
            if records and tensors:
                raise ValueError('Bag directory contains both TFRecords and .pt files; '
                                 'provide an explicit list of tile bag paths')
            paths.extend(records or tensors)
        elif path.is_file():
            if path.suffix not in ('.pt', '.tfrecord', '.tfrecords'):
                raise ValueError(f'Unsupported tile bag file: {path}')
            paths.append(path)
        else:
            raise FileNotFoundError(path)
    paths = list(dict.fromkeys(str(p) for p in paths))
    if not paths:
        raise ValueError('No .tfrecord, .tfrecords or .pt tile bags found')
    return paths


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


class TileBagDataset(InstanceBagDataset):
    """Sample TFRecord tile indices before decoding RGB images."""

    def __init__(self, *args, **kwargs):
        if kwargs.get('preload'):
            raise ValueError('Tile bags are loaded on demand; preload is unsupported')
        super().__init__(*args, **kwargs)
        self._records = {}

    def _reader(self, path):
        import slideflow as sf
        from slideflow.util import tfrecord2idx

        path = str(path)
        if path not in self._records:
            reader = sf.TFRecord(path, create_index=False, decode_images=True)
            if reader.index is None:
                # keep unindexed resources read-only
                reader.index, _ = tfrecord2idx._build_index_from_tfrecord(path)
            self._records[path] = reader
        return self._records[path]

    def __getitem__(self, index):
        bag = self.bags[index]
        paths = [bag] if isinstance(bag, (str, PathLike)) else bag
        if not isinstance(paths, (list, tuple, np.ndarray)) or not len(paths):
            return super().__getitem__(index)
        if not all(isinstance(p, (str, PathLike)) for p in paths):
            return super().__getitem__(index)
        is_record = [Path(p).suffix in ('.tfrecord', '.tfrecords') for p in paths]
        if not any(is_record):
            return super().__getitem__(index)
        if not all(is_record):
            raise ValueError('A grouped tile bag cannot mix TFRecords and .pt files')
        readers = [self._reader(p) for p in paths]
        ends = np.cumsum([len(reader) for reader in readers])
        total = int(ends[-1])
        if not total:
            raise ValueError('bags cannot be empty')
        cap = self.bag_size or self.max_bag_size
        indices = (torch.randperm(total)[:cap].numpy()
                   if self.bag_size or (cap and total > cap) else np.arange(total))
        starts = np.r_[0, ends[:-1]]
        images = []
        for tile in indices:
            slide = int(np.searchsorted(ends, tile, side='right'))
            image = torch.as_tensor(readers[slide][int(tile - starts[slide])]['image_raw'])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != torch.uint8:
                raise ValueError('TFRecord tiles must decode to uint8 RGB images')
            images.append(image)
        if any(image.shape != images[0].shape for image in images):
            raise ValueError('Tiles within a bag must have the same image dimensions')
        tiles = torch.stack(images).to(self.dtype)
        length = len(tiles)
        if self.bag_size and length < self.bag_size:
            tiles = torch.cat([tiles, tiles.new_zeros((self.bag_size - length, *tiles.shape[1:]))])
        return tiles, length


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
