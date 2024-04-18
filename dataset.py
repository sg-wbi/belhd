#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities to load data
"""
import copy
import itertools
import os
from dataclasses import asdict, dataclass
from typing import Iterator, Optional, Union

import numpy as np
from belb import AutoBelbCorpusConfig
from belb.kbs import ENTITY_TO_KB_NAME
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from storage import KB_GROUP, QUERY_GROUP, ShardedZarrReader, ZarrReader
from utils import load_json


class Collator:
    """
    Padding and batching for 1d arrays. It supports left padding.
    """

    def __init__(self, pad_token_id: int = 0, pad_left: bool = False):
        self.pad_left = pad_left
        self.pad_token_id = pad_token_id

        self.get_pad_width = lambda x: (x, 0) if self.pad_left else (0, x)
        self.get_pad_len = lambda x, y: x - y

    def pad(self, array: np.ndarray, maxlen: tuple, pad_value: Optional[int] = None):
        """
        Pad array.

        Parameters
        - ---------
        array: np.ndarray
            Array to be padded
        maxlen: tuple
            Maximum size for each dimension
        pad_value: Optional[int]
            Value to use for padding
        """

        pad_value = pad_value if pad_value is not None else self.pad_token_id

        # assert hasattr(maxlen, '__getitem__'), "Argument `maxlen` must indexble with length for each dimension"

        pad_width = [
            self.get_pad_width(self.get_pad_len(maxlen[i], array.shape[i]))
            for i in range(len(maxlen))
        ]

        padded = np.pad(
            array, pad_width=pad_width, mode="constant", constant_values=pad_value
        )

        return padded

    def batch(
        self,
        arrays: list[Union[list, np.ndarray]],
        axis: int = 0,
        pad_value: Optional[int] = None,
    ):
        """
        Batchify with padding collection of arrays/lists.

        Parameters
        - ---------
        arrays: list[Union[list, np.ndarray]]
            Arrays to be padded
        axis: int
            Axis for stakcing
        pad_value: Optional[int]
            Padding value

        Raises
        - -----
        ValueError:
            list of arrays is empty
        ValueError:
            list of arrays contains ragged nested sequence
        ValueError:
            list of arrays have different shapes
        """

        if not len(arrays) > 0:
            raise ValueError("Cannot batch emtpy list of arrays!")

        if not isinstance(arrays, np.ndarray):
            np_arrays: list[np.ndarray] = [np.asanyarray(a) for a in arrays]
        else:
            np_arrays = arrays

        try:
            arrays_dims = len(np_arrays[0].shape)
        except AttributeError as error:
            error_msg = (
                "Probably converted into numpy ragged nested sequences"
                "(which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes)"
            )
            raise ValueError(
                f"Cannot infer shape of {arrays[0]}! \n {error_msg}"
            ) from error

        if not all(arrays_dims == len(a.shape) for a in np_arrays):
            raise ValueError("All examples must have same number of dimensions!")

        if pad_value is None:
            pad_value = self.pad_token_id

        maxlen = tuple(
            list(max(a.shape[i] for a in np_arrays) for i in range(arrays_dims))
        )
        np_padded_arrays = [
            self.pad(array=a, maxlen=maxlen, pad_value=pad_value) for a in np_arrays
        ]

        out = np.stack(np_padded_arrays, axis=axis)

        return out

    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizerBase, **kwargs):
        """
        Instantite collator from tokenizer
        """
        kwargs["pad_token_id"] = tokenizer.pad_token_id
        kwargs["pad_left"] = tokenizer.padding_side == "left"
        return cls(**kwargs)


class KbDataset(IterableDataset):
    """
    Pytorch iterable dataset to stream over dictionary data
    """

    def __init__(
        self,
        source: Union[list, str],
    ):
        super().__init__()
        self.reader = ShardedZarrReader(
            source=source,
            group=KB_GROUP,
        )

    def __getitem__(self, index: int) -> dict:
        return self.reader[index]

    def __iter__(self) -> Iterator[dict]:
        """
        Iterable
        """
        return (self.reader[i] for i in range(len(self.reader)))


class KbCollator(Collator):
    """
    Collator for dictionary input
    """

    def __init__(
        self,
        *args,
        cls_token_id: int,
        sep_token_id: int,
        max_len: int = 512,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_len = max_len
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

    def add_special_tokens(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Add special tokens
        """

        input_ids = np.insert(input_ids, 0, self.cls_token_id)
        input_ids = np.insert(input_ids, len(input_ids), self.sep_token_id)

        return input_ids

    def collate_fn(self, batch: list[dict]) -> dict[str, np.ndarray]:
        """
        Create batch from list
        """

        names = self.batch(
            [
                self.add_special_tokens(e["name"].tolist()[: self.max_len - 2])
                for e in batch
            ]
        )

        uids = self.batch([e["uid"] for e in batch]).flatten()

        return {"input_ids": names, "uids": uids}

    @classmethod
    def from_tokenizer(
        cls, tokenizer: PreTrainedTokenizerBase, max_len: int = 512, **kwargs
    ):
        """
        Instantite collator from tokenizer
        """
        kwargs["cls_token_id"] = tokenizer.cls_token_id
        kwargs["sep_token_id"] = tokenizer.sep_token_id
        kwargs["max_len"] = max_len
        return super().from_tokenizer(tokenizer=tokenizer, **kwargs)


@dataclass
class QuerySplit:
    """
    Query dataset split
    """

    corpus: str
    name: str
    entity_type: str
    sentences: bool = True
    mention_markers: bool = True
    add_foreign_annotations: bool = False
    max_mentions: int = -1

    @classmethod
    def from_string(
        cls,
        string: str,
        sentences: bool = False,
        mention_markers: bool = False,
        add_foreign_annotations: bool = False,
        max_mentions: int = -1,
    ):
        """
        Instantiate from string
        """

        items = string.split(".")
        assert (
            len(items) == 3
        ), "Training data must be specified in the form: <corpus name>.<split>.<entity type>"
        kwargs: dict = {}
        kwargs["corpus"] = items[0]
        kwargs["name"] = items[1]
        kwargs["entity_type"] = items[2]
        kwargs["sentences"] = sentences
        kwargs["mention_markers"] = mention_markers
        kwargs["add_foreign_annotations"] = add_foreign_annotations
        kwargs["max_mentions"] = max_mentions

        return cls(**kwargs)


def squeeze_array(
    array: np.ndarray,
    pad_token_id: int,
) -> np.ndarray:
    """
    `mentions_ids`: NumSentences x NumMentions x NumIDs
    -->
    `mentions_ids`: NumMentions x NumIDs
    """

    def is_empty(v):
        return v.size == 0

    num_sents = array.shape[0]
    num_mentions = array.shape[1]
    idxs = itertools.product(range(num_sents), range(num_mentions))

    row_list = []

    for i, j in idxs:
        m_tids = array[i, j, :]

        if is_empty(m_tids[m_tids != pad_token_id]):
            continue

        row_list.append(m_tids)

    squeezed_array = np.vstack(row_list)

    return squeezed_array


def get_tokenized_kb(
    directory: str,
    tokenizers_dir: str,
    tokenizer_name: str,
    split: str,
    subset: Optional[str] = None,
) -> str:
    """
    Get path to dictionary
    """

    tokenizers_cache = {
        str(v): k
        for k, v in load_json(os.path.join(tokenizers_dir, "cache.json")).items()
    }
    tokenizer_full_name = tokenizers_cache[tokenizer_name]

    query_split = QuerySplit.from_string(split)
    kb_name = ENTITY_TO_KB_NAME[query_split.entity_type]

    parts = [directory, kb_name]
    if subset is not None:
        parts.append(subset)
    parts.append(tokenizer_name)

    path = os.path.join(*parts)

    if not os.path.exists(path):
        raise ValueError(
            f"Belb KB `{kb_name}` was never tokenized "
            f"with tokenizer `{tokenizer_full_name}`"
        )

    return path


def get_queries(
    directory: str,
    tokenizer_name: str,
    tokenizers_dir: str,
    train: Optional[str] = None,
    dev: Optional[str] = None,
    test: Optional[str] = None,
    sentences: bool = True,
    mention_markers: bool = True,
    add_foreign_annotations: bool = False,
    max_mentions: int = -1,
    abbres: bool = False,
) -> dict:
    """
    Get path to query data
    """

    assert (
        train is not None or test is not None
    ), "You must specify either `train` or `test` corpus!"

    tokenizers_cache = {
        str(v): k
        for k, v in load_json(os.path.join(tokenizers_dir, "cache.json")).items()
    }
    tokenizer_full_name = tokenizers_cache[tokenizer_name]

    kwargs: dict = {
        "sentences": sentences,
        "mention_markers": mention_markers,
        "add_foreign_annotations": add_foreign_annotations,
        "max_mentions": max_mentions,
    }

    splits = []

    if train is not None:
        kwargs["string"] = train
        train_split = QuerySplit.from_string(**kwargs)
        splits.append(train_split)
        kwargs["string"] = dev if dev is not None else train.replace("train", "dev")
        dev_split = QuerySplit.from_string(**kwargs)
        assert train_split.entity_type == dev_split.entity_type, (
            f"Entity type of train and dev data must be the same!"
            f"Found: {train_split.entity_type} and {dev_split.entity_type}"
        )

        splits.append(dev_split)

    if test is None:
        assert train is not None, "If `test` is not specified you must provide `train`!"
        test = train.replace("train", "test")
    else:
        kwargs["string"] = test
        test_split = QuerySplit.from_string(**kwargs)
        splits.append(test_split)

    queries: dict = {}
    for split in splits:
        queries[split.name] = {}

        config = AutoBelbCorpusConfig.from_name(
            name=split.corpus,
            entity_type=split.entity_type,
            sentences=split.sentences,
            mention_markers=split.mention_markers,
            add_foreign_annotations=split.add_foreign_annotations,
            max_mentions=split.max_mentions,
        )

        hexdigest = config.to_hexdigest()

        parts = [directory, split.corpus]
        if abbres:
            parts.append("abbres")
        parts += [hexdigest, tokenizer_name]
        folder = os.path.join(*parts)
        if abbres and not os.path.exists(folder):
            breakpoint()
            raise ValueError(
                (
                    f"Corpus `{asdict(split)}` "
                    + f"was never tokenized with tokenizer `{tokenizer_full_name}`",
                )
            )

        queries[split.name]["dir"] = folder
        queries[split.name]["path"] = os.path.join(folder, f"{split.name}.zarr")
        queries[split.name]["entity_type"] = split.entity_type

    return queries


class QueryDataset(Dataset):
    """
    Pytorch iterable dataset to stream over corpus data
    """

    def __init__(self, path: str, group: Optional[dict] = None):
        super().__init__()
        group = group if group is not None else QUERY_GROUP
        reader = ZarrReader(path=path, group=group)
        self.data = [item.copy() for item in reader]

    def concat(self, other: "QueryDataset"):
        """
        Concatenate other dataset (axis=1, i.e. append to data)
        """
        self.data += other.data

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[dict]:
        return (i for i in self.data)


class QueryCollator(Collator):
    """
    Collator for dictionary input
    """

    def __init__(
        self,
        *args,
        cls_token_id: int,
        sep_token_id: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

    def add_special_tokens(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Add special tokens
        """

        input_ids = np.insert(input_ids, 0, self.cls_token_id, axis=1)
        indices = np.sum(input_ids != self.pad_token_id, axis=1)

        expanded = []
        for idx, array in zip(indices, input_ids):
            array = np.insert(array, idx, self.sep_token_id)
            expanded.append(array)

        return self.batch(expanded, pad_value=self.pad_token_id)

    def collate_fn(self, batch: list[dict]) -> dict[str, np.ndarray]:
        """
        Create batch from list
        """

        passages = copy.deepcopy(batch[0])
        annotation_identifiers = passages["annotation_identifiers"]
        annotation_ids = passages["annotation_ids"]

        squeezed_annotation_identifiers = squeeze_array(
            annotation_identifiers, QUERY_GROUP["annotation_identifiers"]["pad"]
        )
        squeezed_annotation_ids = squeeze_array(
            annotation_ids, QUERY_GROUP["annotation_ids"]["pad"]
        )

        inputs: dict = {
            "eid": passages["eid"][0],
            "passage_ids": passages["passage_ids"],
            "annotation_offsets": passages["annotation_offsets"],
            "input_ids": self.add_special_tokens(passages["input_ids"]),
            "annotation_identifiers": squeezed_annotation_identifiers,
            "annotation_ids": squeezed_annotation_ids,
            "by_sent_annotation_identifiers": annotation_identifiers,
            "by_sent_annotation_ids": annotation_ids,
        }

        return inputs

    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizerBase, **kwargs):
        """
        Instantite collator from tokenizer
        """
        kwargs["cls_token_id"] = tokenizer.cls_token_id
        kwargs["sep_token_id"] = tokenizer.sep_token_id
        return super().from_tokenizer(tokenizer=tokenizer, **kwargs)


def get_queries_dataset(
    queries: dict,
    train_on_dev: bool = False,
    train_with_dev: bool = False,
    load_test: bool = False,
    group: Optional[dict] = None,
):
    """
    Get dataloders of queries
    """

    splits: dict = {}

    if "train" in queries:
        if train_on_dev:
            splits["train"] = QueryDataset(path=queries["train"]["path"], group=group)

        elif train_with_dev:
            splits["train"] = QueryDataset(path=queries["train"]["path"], group=group)
            splits["train"].concat(
                QueryDataset(path=queries["dev"]["path"]), group=group
            )

        else:
            splits["train"] = QueryDataset(path=queries["train"]["path"], group=group)
            splits["dev"] = QueryDataset(path=queries["dev"]["path"], group=group)

    if load_test:
        assert "test" in queries, "Test split not specified!"
        splits["test"] = QueryDataset(path=queries["test"]["path"], group=group)

    return splits
