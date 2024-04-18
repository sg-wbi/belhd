#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Index
"""
import os
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import faiss
import numpy as np
import torch
from loguru import logger

from model import BiEncoderModel
from storage import EMBEDDING_GROUP, ShardedZarrReader
from utils import StrEnum

SIMILARITY_METRICS = [faiss.METRIC_INNER_PRODUCT, faiss.METRIC_L2]


class IndexFactory(StrEnum):
    """
    Wrapper for available indices
    """

    IVF = "ivf"
    IVFPQ = "ivfpf"
    PQ = "pq"
    FLAT = "flat"


@dataclass
class IndexConfig:
    """
    Options for faiss index (PQ, IVF)
    """

    dim: int
    index_type: str = IndexFactory.FLAT
    metric: int = faiss.METRIC_INNER_PRODUCT

    # PQ options
    # number of sub-vectors to split original ones into
    code_size: int = 16
    # we can translate this into the number of centroids assigned to each subspace as k_ = 2**n_bits.
    # An n_bits of 11 leaves us with 2048 centroids per subspace.
    nbits: int = 8
    # OPQ works by rotating vectors to flatten the distribution of values across the subvectors used in PQ.
    # This is particularly beneficial for unbalanced vectors with uneven data distributions.
    opq: bool = True

    # IVF options
    # how many Voronoi cells (must be >= k* which is 2**n_bits when combined w/ PQ)
    nlist: Optional[int] = None
    # which tells us how many of the nearest Voronoi cells to include in our search scope.
    nprobe: Optional[int] = None
    probe_factor: int = 1

    def __post_init__(self):
        assert self.index_type in tuple(
            IndexFactory
        ), f"Invalid index type {self.index_type}! Must be one of {tuple(IndexFactory)}"
        assert self.metric in SIMILARITY_METRICS
        self.keep_max = self.metric == faiss.METRIC_INNER_PRODUCT

        # if self.index_type in [IndexFactory.PQ, IndexFactory.IVFPQ]:
        #     assert (
        #         self.dim % self.code_size == 0
        #     ), f"Cannot use {self.code_size} codes (subvectors) for vectors w/ {self.dim} dimensions!"

    def to_index_factory_string(self, ntotal: Optional[int] = None) -> str:
        """
        Create index factory string
        """

        if self.index_type == IndexFactory.FLAT:
            desc = "Flat"
        else:
            if "ivf" in self.index_type and ntotal is None:
                assert (
                    self.nlist is not None
                ), "Either provide `ntotal` or specify `nlist`!"

            if self.nlist is None and ntotal is not None:
                self.nlist = int(np.floor(np.sqrt(ntotal)))

            if self.index_type == IndexFactory.PQ:
                desc = f"OPQ{self.code_size}," if self.opq else ""
                desc += f"PQ{self.code_size}x{self.nbits}"
            elif self.index_type == IndexFactory.IVF:
                desc = f"IVF{self.nlist},Flat"
            elif self.index_type == IndexFactory.IVFPQ:
                desc = f"OPQ{self.code_size}," if self.opq else ""
                desc += f"IVF{self.nlist},"
                desc += f"PQ{self.code_size}x{self.nbits}"

        return desc


class BaseShardedIndex:
    """
    Basic sharded index
    """

    def __init__(
        self,
        process_id: int,
        directory: str,
        config: IndexConfig,
        debug: bool = False,
    ):
        self.process_id = process_id
        self.directory = directory
        self.config = config
        self.debug = debug
        EMBEDDING_GROUP["embedding"]["size"] = self.config.dim
        self.reader = ShardedZarrReader(source=directory, group=EMBEDDING_GROUP)
        self._shards: list = []

    @abstractmethod
    def iter_shards(self):
        """
        Iterate over index shards
        """

    @property
    def shards_loaded(self):
        """
        Check if shards are loaded into memory
        """

        return len(self._shards) > 0

    def reset(self):
        """
        Reload shards: use when index data has changed and `preload_shards=True`
        """

        self._shards = []


class FlatShardedIndex(BaseShardedIndex):
    """
    Flat index: brute-force search
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iter_shards(self):
        """
        Pre-load shards
        """
        if not self.shards_loaded:
            if self.debug:
                logger.debug("Index: load shards")
            self._shards = []
            for shard in self.reader.iter_shards():
                shard.load()
                self._shards.append(shard)
            if self.debug:
                logger.debug("Index: shards loaded")

        for shard in self._shards:
            yield shard

    def build(self):
        """
        No-op: just reset shard data
        """
        self.reset()
        if self.debug:
            logger.debug("Index: reload shards")

    def search(self, xq: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Use brute force search
        """

        heap = faiss.ResultHeap(nq=xq.shape[0], k=topk, keep_max=self.config.keep_max)

        for shard in self.iter_shards():
            xb = shard.to_numpy("embedding")
            xi = shard.to_numpy("uid")

            id_to_uid = {idx: uid.item() for idx, uid in enumerate(xi)}

            dists, idxs = faiss.knn(
                xq=xq,
                xb=xb,
                k=topk,
                metric=self.config.metric,
            )

            uids = np.vectorize(id_to_uid.get)(idxs)

            heap.add_result(dists, uids)

        heap.finalize()

        return heap.D, heap.I


class FaissShardedIndex(BaseShardedIndex):
    """
    Faiss index:
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index_directory = None

    def iter_shards(self):
        """
        Pre-load shards
        """

        assert self.index_directory is not None, "You must call `.build` first!"

        if not self.shards_loaded:
            if self.debug:
                logger.debug("Index: load shards")
            self._shards = [
                faiss.read_index(os.path.join(self.index_directory, f))
                for f in os.listdir(self.index_directory)
            ]
            if self.debug:
                logger.debug("Index: shards loaded")
            for shard in self._shards:
                yield shard

    def get_index_directory(self, ntotal: Optional[int] = None):
        """
        Get directory where to store index shards
        """

        index_factory_string = self.config.to_index_factory_string(ntotal=ntotal)
        name = index_factory_string.replace(",", "-").lower()
        index_dir = os.path.join(self.directory, name)

        return index_dir

    def get_index(
        self, vectors: np.ndarray, ids: np.ndarray, debug: bool = False
    ) -> faiss.Index:
        """
        Build the index, i.e. : train and add
        """

        index_factory_string = self.config.to_index_factory_string(
            ntotal=vectors.shape[0]
        )

        index = faiss.index_factory(
            self.config.dim, index_factory_string, self.config.metric
        )

        start = time.time()
        index.train(vectors)
        elapsed = round(time.time() - start, 4)
        if debug:
            logger.info(
                "Training index `{}` on {} took {} seconds",
                index_factory_string,
                vectors.shape[0],
                elapsed,
            )
        index.add(vectors)

        if hasattr(index, "nprobe") and self.config.nprobe is None:
            self.config.nprobe = int(
                np.floor(np.sqrt(self.config.nlist) * self.config.probe_factor)
            )
            index.nprobe = self.config.nprobe

        # IVF subclasses natively support `add_with_ids`
        # https://github.com/facebookresearch/faiss/wiki/Pre--and-post-processing#the-indexidmap
        if "ivf" not in self.config.index_type:
            index = faiss.IndexIDMap(index)

        index.add_with_ids(vectors, ids)

        return index

    def build(self, debug: bool = False):
        """
        Build index or load existing one
        """

        vectors_list = []
        uids_list = []
        for shard in self.reader.iter_shards():
            vectors_list.append(shard.to_numpy("embedding"))
            uids_list.append(shard.to_numpy("uid"))

        vectors = np.vstack(vectors_list)
        uids = np.vstack(uids_list)

        index_dir = self.get_index_directory(ntotal=vectors.shape[0])
        os.makedirs(index_dir, exist_ok=True)
        self.index_directory = index_dir

        path = os.path.join(index_dir, f"p{self.process_id}_shard")

        if not os.path.exists(path):
            index = self.get_index(vectors=vectors, ids=uids, debug=debug)
            faiss.write_index(index, path)

        self.reset()

    def search(self, xq: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Search w/ built index
        """

        heap = faiss.ResultHeap(nq=xq.shape[0], k=topk, keep_max=self.config.keep_max)

        for index_shard in self.iter_shards():
            dists, uids = index_shard.search(xq, topk)

            heap.add_result(dists, uids)

        heap.finalize()

        return heap.D, heap.I


class ShardedIndex:
    """
    Wrapper for faiss index: index shards
    Only supports CPU.
    """

    def __init__(
        self,
        process_id: int,
        directory: str,
        config: IndexConfig,
        # preload_shards: Optional[bool] = None,
        debug: bool = False,
    ):
        self._index: Union[FlatShardedIndex, FaissShardedIndex]

        if config.index_type == IndexFactory.FLAT:
            self._index = FlatShardedIndex(
                process_id=process_id,
                directory=directory,
                config=config,
                debug=debug,
            )
        else:
            self._index = FaissShardedIndex(
                process_id=process_id,
                directory=directory,
                config=config,
                debug=debug,
            )

    @property
    def directory(self):
        """
        Directory
        """

        return self._index.directory

    def build(self):
        """
        Build index
        """

        self._index.build()

    def search(self, xq: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Search embeddings
        """

        return self._index.search(xq=xq, topk=topk)


def search_index(
    batch: dict,
    model: BiEncoderModel,
    index: ShardedIndex,
    mixed_precision: bool = False,
    topk: int = 64,
    debug: bool = False,
) -> np.ndarray:
    """
    Fetch candidates from index and convert position to kb identifier
    """

    with torch.no_grad():
        queries_embd = model(forward="queries", queries=batch)["mentions"]

    if mixed_precision:
        queries_embd = queries_embd.float()

    queries_embd = queries_embd.detach().cpu().numpy()

    start = time.time()
    _, idxs = index.search(xq=queries_embd, topk=topk)
    elapsed = round(time.time() - start, 4)

    if debug:
        logger.debug(
            "Searching {} mentions took {} seconds", queries_embd.shape[0], elapsed
        )

    return idxs
