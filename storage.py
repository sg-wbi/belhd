#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interfaces to write and read HDF5 file.

See: https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
"""

import itertools
import os
from typing import Iterable, Iterator, Optional, Union

import h5py
import numcodecs
import numpy as np
import zarr

from utils import get_naturally_sorted_files, natural_sorted

ANN_PAD_TOKEN_ID = -1

KB_GROUP: dict = {
    "name": {"dtype": "int64"},
    "uid": {"dtype": "int64", "size": 1},
}

EMBEDDING_GROUP: dict = {
    "embedding": {"dtype": "float32"},
    "uid": {"dtype": "int64", "size": 1},
}

QUERY_GROUP: dict = {
    "eid": {"size": 1, "dtype": "int64"},
    "passage_ids": {"dtype": "int64"},
    "input_ids": {"dtype": "int64", "pad": 0},
    "input_ids_shape": {"size": 2, "dtype": "int64"},
    "annotation_identifiers": {"dtype": "int64", "pad": ANN_PAD_TOKEN_ID},
    "annotation_identifiers_shape": {"size": 3, "dtype": "int64"},
    "annotation_ids": {"dtype": "int64", "pad": ANN_PAD_TOKEN_ID},
    "annotation_ids_shape": {"size": 3, "dtype": "int64"},
    "annotation_offsets": {"dtype": "int64", "pad": -1},
    "annotation_offsets_shape": {"size": 3, "dtype": "int64"},
}

KILT_QUERY_GROUP: dict = {
    "eid": {"size": 1, "dtype": "int64"},
    "annotation_id": {"size": 1, "dtype": "int64"},
    "input_ids": {"dtype": "int64", "pad": 0},
    "input_ids_shape": {"size": 2, "dtype": "int64"},
    "annotation_identifiers": {"dtype": "int64", "pad": ANN_PAD_TOKEN_ID},
    "annotation_identifiers_shape": {"size": 2, "dtype": "int64"},
}


class ZarrWriter:
    """
    Zarr file writer
    The `group` attribute must return a dictionary in the form:
        {
            "image": {"dtype": "float32"},
            "image_shape": {"size": 2, "dtype": "int64"},
        },
    If `size` is not specified we assume variable length input.
    Store `<group>_shape` to companion `ZarrReader`
    """

    def __init__(self, group: dict, file_size: Optional[int] = None):
        """ """
        self.group = group
        self.file_size = file_size

    def get_pad_value(self, key: str, fallback: Optional[int] = None) -> int:
        """Get key pad value"""

        specs = self.group.get(key)

        if specs is None:
            raise ValueError(
                f"Key `{key}` not found in `groups`! Available keys: `{list(self.group.keys())}`"
            )

        pad = specs.get("pad", fallback)

        return pad

    def get_shape_array(self, key: str) -> str:
        """Get shape key"""

        return key + "_shape"

    def has_shape(self, key: str) -> bool:
        """Get shape key name"""

        return key + "_shape" in self.group

    def is_shape_array(self, key: str) -> bool:
        """Check if key stores shapes"""
        return "_shape" in key

    def sanity_check(self, data: dict, is_last: bool = False):
        """
        See companion Hdf5Reader
        """

        lengths = {k: len(v) for k, v in data.items()}

        num_rows = list(set(lengths.values()))

        assert len(num_rows) == 1, f"All groups must have same length! Found: {lengths}"

        if self.file_size is not None and not is_last:
            assert (
                num_rows == self.file_size
            ), f"Available arrays {num_rows} != {self.file_size} specified file size!"

    def write(
        self,
        path: str,
        batch: dict[str, list[np.ndarray]],
        is_last: bool = False,
    ):
        """
        Write list of items in a HDF5 file.
        """

        self.sanity_check(data=batch, is_last=is_last)

        store = zarr.storage.FSStore(path)
        group = zarr.open_group(store, mode="w")
        for array_name, data in batch.items():
            elem = self.group[array_name]
            size = elem.get("size")
            dtype = elem.get("dtype")
            assert dtype is not None, f"Arrray {array_name} does not specify `dtype`"
            shape = (len(data), size) if size is not None else (len(data),)

            array_data = np.asarray(
                [a.flatten() for a in data],
                dtype=dtype if size is not None else "object",
            )
            object_codec = None if size is not None else numcodecs.VLenArray(dtype)

            kwargs: dict = {
                "name": array_name,
                "data": array_data,
                "shape": shape,
                "object_codec": object_codec,
            }

            group.create_dataset(**kwargs)

            if self.has_shape(array_name):
                array_name_shape = self.get_shape_array(array_name)
                array_shape_data = np.asarray(
                    [a.shape for a in data],
                    dtype=self.group[array_name_shape]["dtype"],
                )

                size = self.group[array_name_shape]["size"]
                array_shape_kwargs: dict = {
                    "name": array_name_shape,
                    "data": array_shape_data,
                    "shape": (len(data), size),
                }
                group.create_dataset(**array_shape_kwargs)
        zarr.consolidate_metadata(store)


class ZarrReader:
    """
    Read single zarr file written by ZarrWriter
    """

    def __init__(self, path: str, group: dict):
        self.path = path
        self.group = group
        self.loader = zarr.load(self.path)
        self.shape_keys = [k for k in self.loader if k.endswith("_shape")]
        self._counter = 0

    def __len__(self):
        lengths = {k: len(self.loader[k]) for k in self.loader}
        assert (
            len(set(lengths.values())) == 1
        ), f"All arrays should have same size. Found: {lengths}"
        return next(iter(lengths.values()))

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self._counter < len(self):
            item = self[self._counter]
        else:
            raise StopIteration
        self._counter += 1
        return item

    def __getitem__(self, idx: int) -> dict:
        """
        Load example according to dataset index
        """

        example: dict = {}

        for k in self.group:
            if k in self.shape_keys:
                continue

            # Check if key needs reshaping
            if k + "_shape" in self.shape_keys:
                array = self.loader[k][idx]
                shape = self.loader[k + "_shape"][idx]
                item = array.reshape(shape)
            else:
                item = self.loader[k][idx]

            example[k] = item

        return example

    def to_numpy(self, key: str) -> list[np.ndarray]:
        """
        Stack data into numpy array
        """

        assert (
            key in self.group
        ), f"Group `{key}` does not exists! Available are: {self.group}"

        return self.loader[key]

    def load(self):
        """
        Cache data
        """

        for key in self.group:
            _ = self.loader[key]


class ShardedZarrReader:
    """
    Dataset to read zarr files written by ZarrWriter.

    It assumes:
        - zarr files in a folder (and/or subfolders)
        - all files must be numbered, e.g. file1,file2,file3,...
        - all files have same amount of examples: only exception is the 'last' one (due to original # of examples)

    Original implementation used h5py.
    See : https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
    """

    def __init__(
        self,
        source: Union[list, str],
        group: dict,
        extension: str = "zarr",
        recursive: bool = False,
    ):
        """
        Instantiate object.

        As it all arrays in HDF5 must be 1d, it is necessary to store the original shape.
        For each key in `group` it is possible to specify a `_shape` pointing to the dataset that
        contains the shapes of the specified key. These will be used to reshape the array into its original shape.
        """

        self.source = source
        self.extension = extension
        self.recursive = recursive
        self.group = group

        self._files: Optional[list] = None
        self._file_sizes: Optional[dict] = None
        self._shard_size: Optional[int] = None
        self._path: Optional[str] = None
        self._loader: Optional[ZarrReader] = None

    @property
    def files(self) -> list[str]:
        """
        Load list of files
        """
        if self._files is None:
            if isinstance(self.source, list):
                self._files = natural_sorted(self.source)
            else:
                self._files = list(
                    get_naturally_sorted_files(
                        directory=self.source,
                        extension=self.extension,
                        recursive=self.recursive,
                    )
                )
            assert (
                len(self._files) > 0
            ), "You must provide a list of paths to `zarr` files (if `source` is a folder, check that it's not empty)"

        return self._files

    def iter_shards(self) -> Iterator[ZarrReader]:
        """
        Iterate over files
        """

        for file in self.files:
            yield ZarrReader(path=file, group=self.group)

    @property
    def shard_sizes(self) -> list:
        """
        Get length of each shard
        """

        if self._file_sizes is None:
            self._file_sizes = {}
            for file in self.files:
                loader = ZarrReader(path=file, group=self.group)
                self._file_sizes[file] = len(loader)
        return list(self._file_sizes.values())

    @property
    def shard_size(self) -> int:
        """
        Size of all shards (besides one)
        """

        if self._shard_size is None:
            # tell me which file is problematic
            unique_sizes = set(self.shard_sizes)
            assert self._file_sizes is not None
            pointers = {os.path.basename(k): v for k, v in self._file_sizes.items()}
            assert (
                len(unique_sizes) <= 2
            ), f"All files must be of same length (except of one)! Check input files : {pointers}"
            self._shard_size = max(unique_sizes)

        return self._shard_size

    def __len__(self):
        return sum(self.shard_sizes)

    def __getitem__(self, idx: int):
        # Get path to file according to global idx (w.r.t. total # of examples)
        file_idx = idx // self.shard_size
        path = self.files[file_idx]

        if path != self._path:
            # set current file
            self._path = path
            self._loader = ZarrReader(path=path, group=self.group)

        # Get relative idx in dataset file according to global idx(w.r.t. total  # of examples)
        dataset_idx = idx % self.shard_size

        assert self._loader is not None

        return self._loader[dataset_idx]


def pairwise(iterable: Iterable) -> Iterable:
    """
    Return successive overlapping pairs taken from the input iterable.
    The number of 2-tuples in the output iterator will be one fewer than the number of inputs.
    It will be empty if the input iterable has fewer than two values.

    Built-in in python3.10
    """
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class Hdf5Writer:
    """
    HDF5 file writer
    The `groups` attribute must return a dictionary in the form:
        {
            "image": {"dtype": "float32"},
            "image_shape": {"size": 2, "dtype": "int64"},
        },
    HDF5 can only store 1d arrays.
    If `size` is not specified we assume variable length input.
    Store `<group>_shape` to companion `Hdf5Reader`
    """

    def __init__(
        self, group: dict, scope: str = "dataset", file_size: Optional[int] = None
    ):
        """ """
        self.group = group
        self.scope = scope
        self.file_size = file_size

    def get_pad_value(self, key: str, fallback: Optional[int] = None) -> int:
        """Get key pad value"""

        specs = self.group.get(key)

        if specs is None:
            raise ValueError(
                f"Key `{key}` not found in `groups`! Available keys: `{list(self.group.keys())}`"
            )

        pad = specs.get("pad", fallback)

        return pad

    def get_shape_key(self, key: str) -> str:
        """Get shape key"""

        return key + "_shape"

    def key_has_shape(self, key: str) -> bool:
        """Get shape key name"""

        return key + "_shape" in self.group

    def is_shape_key(self, key: str) -> bool:
        """Check if key stores shapes"""
        return "_shape" in key

    def create_groups(self, group_ds: h5py.Group, num_rows: int):
        """
        Create groups under given scope
        """

        for key in self.group:
            group = self.group[key]
            size = group.get("size")
            dtype = group.get("dtype")
            assert dtype is not None, f"Groupt {key} does not specify `dtype`"

            shape = (num_rows, size) if size is not None else (num_rows,)
            maxshape = tuple([None] * len(shape))

            if size is not None:
                dtype = np.dtype(dtype)
            else:
                vlen = np.dtype(dtype)
                dtype = h5py.special_dtype(vlen=vlen)

            group_ds.create_dataset(
                key, shape=shape, compression="gzip", dtype=dtype, maxshape=maxshape
            )

    def resize_groups(self, file: h5py.File, chunksize: int):
        """
        Extend size (length) of a given group
        """

        for key in self.group:
            orig_size = file[self.scope][key].shape[0]
            file[self.scope][key].resize(orig_size + chunksize, axis=0)

    def sanity_check(self, num_rows: int, is_last: bool = False):
        """
        See companion Hdf5Reader
        """

        if self.file_size is not None and not is_last:
            assert (
                num_rows == self.file_size
            ), f"Available arrays {num_rows} != {self.file_size} specified file size!"

    def get_num_rows(self, shard: dict[str, list[np.ndarray]]):
        """
        Find # of examples per group
        """

        lengths = {k: len(v) for k, v in shard.items()}

        num_rows = list(set(lengths.values()))

        assert len(num_rows) == 1, f"All groups must have same length! Found: {lengths}"

        return num_rows[0]

    def write(
        self,
        path: str,
        batch: dict[str, list[np.ndarray]],
        block_size: int = 1000,
        is_last: bool = False,
    ):
        """
        Write list of items in a HDF5 file.
        """

        num_rows = self.get_num_rows(batch)

        self.sanity_check(num_rows=num_rows, is_last=is_last)

        with h5py.File(path, "w") as outfile:
            group_ds = outfile.create_group(self.scope)
            self.create_groups(group_ds=group_ds, num_rows=num_rows)

            blocks = (
                [[0, num_rows]]
                if num_rows <= block_size
                else [list(p) for p in pairwise(range(0, num_rows, block_size))]
            )

            for idx, block in enumerate(blocks):
                if idx == len(blocks) - 1:
                    block[1] = num_rows

                for key in batch:
                    group_ds[key][block[0] : block[1]] = [
                        a.flatten() for a in batch[key][block[0] : block[1]]
                    ]
                    if self.key_has_shape(key):
                        shape_key = self.get_shape_key(key)
                        group_ds[shape_key][block[0] : block[1]] = [
                            a.shape for a in batch[key][block[0] : block[1]]
                        ]


class Hdf5Reader:
    """
    Read single HDF5 file written by Hdf5Writer
    """

    def __init__(self, path: str, group: dict, scope: str = "dataset"):
        self.path = path
        self.scope = scope
        self.group = group
        self.file = h5py.File(self.path, "r")
        self.shape_keys = [k for k in self.dataset if k.endswith("_shape")]
        self._size: Optional[int] = None
        self._counter = 0

    @property
    def dataset(self) -> h5py.Dataset:
        """
        Shortcut to data
        """
        return self.file.get(self.scope)

    @property
    def size(self) -> int:
        """
        File length w/ sanity check
        """
        if self._size is None:
            sizes = set(self.dataset.get(k).shape[0] for k in self.group)
            assert (
                len(sizes) == 1
            ), f"# examples must be equal in all groups! In `{self.path}` found {sizes}"

            self._size = next(iter(sizes))

        return self._size

    def __len__(self):
        return self.size

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self._counter < len(self):
            item = self[self._counter]
        else:
            raise StopIteration
        self._counter += 1
        return item

    def __getitem__(self, idx: int) -> dict:
        """
        Load example according to dataset index
        """

        example: dict = {}

        for k, ds in self.dataset.items():
            if k in self.shape_keys:
                continue

            # Check if key needs reshaping
            if k + "_shape" in self.shape_keys:
                array = ds[idx]
                shape = self.dataset[k + "_shape"][idx]
                item = array.reshape(shape)
            else:
                item = ds[idx]

            example[k] = item

        return example

    def close(self):
        """
        Close handle to file
        """

        self.file.close()

    def to_numpy(self, key: str) -> list[np.ndarray]:
        """
        Stack data into numpy array
        """

        assert (
            key in self.group
        ), f"Group `{key}` does not exists! Available are: {self.dataset.keys()}"

        return self.dataset[key][:]


class ShardedHdf5Reader:
    """
    Dataset to read HDF5 files written by BaseHdf5Writer.

    It assumes:
        - HDF5 files in a folder (and/or subfolders)
        - all files must be numbered, e.g. file1,file2,file3,...
        - all files have same amount of examples: only exception is the 'last' one (due to original # of examples)
        - HDF5 files have depth == 2, i.e all arrays are reachable by `ds.get(level1/level2)`

    See : https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
    """

    def __init__(
        self,
        source: Union[list, str],
        group: dict,
        scope: str = "dataset",
        extension: str = "h5",
        recursive: bool = False,
    ):
        """
        Instantiate object.

        As it all arrays in HDF5 must be 1d, it is necessary to store the original shape.

        For each key in `groups` it is possible to specify a `_shape` pointing to the dataset that
        contains the shapes of the specified key. These will be used to reshape the array into its original shape.
        """

        self.scope = scope
        self.group = group

        if isinstance(source, list):
            self.files = natural_sorted(source)
        else:
            self.files = list(
                get_naturally_sorted_files(
                    directory=source, extension=extension, recursive=recursive
                )
            )
        assert (
            len(self.files) > 0
        ), "You must provide a list of paths to `HDF5` files (if `source` is a folder, check that it's not empty)"

        self._file_sizes: Optional[dict] = None
        self._shard_size: Optional[int] = None
        self._path: Optional[str] = None
        self._reader: Optional[Hdf5Reader] = None

    def iter_shards(self) -> Iterator[Hdf5Reader]:
        """
        Iterate over files
        """

        for file in self.files:
            yield Hdf5Reader(
                path=file,
                group=self.group,
                scope=self.scope,
            )

    @property
    def shard_sizes(self) -> list:
        """
        Get length of each shard
        """
        if self._file_sizes is None:
            self._file_sizes = {}
            for file in self.files:
                reader = Hdf5Reader(path=file, group=self.group, scope=self.scope)
                self._file_sizes[file] = len(reader)
                reader.close()
        return list(self._file_sizes.values())

    @property
    def shard_size(self) -> int:
        """
        Size of all shards (besides one)
        """

        if self._shard_size is None:
            # tell me which file is problematic
            unique_sizes = set(self.shard_sizes)
            assert self._file_sizes is not None
            pointers = {os.path.basename(k): v for k, v in self._file_sizes.items()}
            assert (
                len(unique_sizes) <= 2
            ), f"All files must be of same length (except of one)! Check input files : {pointers}"
            self._shard_size = max(unique_sizes)

        return self._shard_size

    def __len__(self):
        return sum(self.shard_sizes)

    def __getitem__(self, idx: int):
        # Get path to file according to global idx (w.r.t. total # of examples)
        file_idx = idx // self.shard_size
        path = self.files[file_idx]

        if path != self._path:
            # set current file
            self._path = path
            if self._reader is not None:
                self._reader.close()
            self._reader = Hdf5Reader(path=path, group=self.group, scope=self.scope)

        # Get relative idx in dataset file according to global idx(w.r.t. total  # of examples)
        dataset_idx = idx % self.shard_size

        assert self._reader is not None

        return self._reader[dataset_idx]
