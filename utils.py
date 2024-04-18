#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for training bi-encoder retrieval model
"""
import glob
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import random
import re
from enum import Enum, EnumMeta
from typing import Any, Iterator, Optional, Union

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Metric
from transformers import get_scheduler


class MetaEnum(EnumMeta):
    """
    https://stackoverflow.com/questions/63335753/how-to-check-if-string-exists-in-enum-of-strings/63336176
    >>> 2.3 in Stuff
    False
    >>> 'zero' in Stuff
    False
    """

    def __contains__(cls, item):
        try:
            cls(item)  # pylint: disable=E
        except ValueError:
            return False
        return True


class StrEnum(str, Enum, metaclass=MetaEnum):
    """
    String Enum
    >>> class Foo(Enum):
           TEST = 'test'
    >>> print(Foo.TEST == "test")
    False
    >>> class Bar(StrEnum):
           TEST = 'test'
    >>> print(Bar.TEST == "test")
    True
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self):
        return str(self)


def compute_hexdigest(message: str, hasher: str = "sha256"):
    """
    Compute hexdigest
    """

    hash_func = getattr(hashlib, hasher, None)

    if hash_func is None:
        raise ValueError(f"Hash function `{hasher}` not found in hashlib!")

    h = hash_func()

    if isinstance(message, str):
        h.update(message.encode())
    else:
        for e in message:
            h.update(str(e).encode())

    return h.hexdigest()


def save_json(path: str, item: dict, *args, **kwargs):
    """
    Save dict to JSON file
    """
    with open(path, "w") as fp:
        json.dump(item, fp, *args, **kwargs)


def load_json(path: str, *args, **kwargs) -> dict:
    """
    Load dict from JSON
    """
    with open(path) as fp:
        item = json.load(fp, *args, **kwargs)

    return item


def load_pickle(path: str) -> dict:
    """
    Load pickle
    """

    with open(str(path), mode="rb") as fp:
        item = pickle.load(fp)
    return item


def save_pickle(path: str, item: Any):
    """
    Save python object to pickle file.
    """

    with open(str(path), mode="wb") as outfile:
        pickle.dump(item, outfile, pickle.HIGHEST_PROTOCOL)


class RunTracker:
    """
    Track runs w/ hyperparameters
    """

    def __init__(
        self,
        directory: str,
        cfg: Union[DictConfig, OmegaConf, dict],
        run_id: Optional[str] = None,
    ):
        self.file = os.path.join(directory, "runs", "experiments.jsonl")
        self.cfg = dict(cfg.items())
        self._run_id = run_id

    def to_string(self):
        """
        Convert to string
        """
        return json.dumps(
            {k: str(v) for k, v in self.cfg.items()},
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )

    @property
    def run_id(self) -> int:
        """
        Get run id: multi-process friendly
        """

        if self._run_id is None:
            run_id = 0
            hd = compute_hexdigest(self.to_string())

            if os.path.exists(self.file):
                df = pd.read_json(self.file, lines=True)
                if hd in set(df["hexdigest"]):
                    run_id = int(df[df["hexdigest"] == hd]["run_id"].values)
                else:
                    run_id = max(df["run_id"]) + 1

            self.cfg["run_id"] = run_id
            self.cfg["hexdigest"] = hd
        else:
            self.cfg["run_id"] = self._run_id

        return self.cfg["run_id"]

    def track(self, items: dict):
        """
        Track items
        """
        self.cfg.update(items)

    def end_training(self):
        """
        Save at the end of training
        """
        with open(self.file, "a") as fp:
            fp.write(f"{self.to_string()}\n")


def set_seed(seed: int):
    """
    Seed
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_run(
    seed: int,
    cores: Optional[int] = None,
    cuda_available: bool = False,
):
    """
    Initialize experiment environment:
    """

    set_seed(seed=seed)

    cores = cores if cores is not None else min(20, mp.cpu_count())
    # cores
    os.environ["OMP_NUM_THREADS"] = str(cores)
    torch.set_num_threads(cores)

    if cuda_available:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            raise ValueError(
                "CUDA available: launch script with: CUDA_VISIBLE_DEVICES="
            )

    return cuda_available


def is_mixed_precision(accelerator: Accelerator) -> bool:
    """
    Check if accelerator is in mixed precision mode
    """

    return accelerator.mixed_precision in ("fp16", "bf16")


def row_wise_isin(src: np.ndarray, trg: np.ndarray) -> np.ndarray:
    """
    Given two 2d arryas compute row-wise isin
    See comment here:
    https://stackoverflow.com/questions/67870579/rowwise-numpy-isin-for-2d-arrays

    >>> y_true = np.asarray([ [11457], [8740], [2779] ])
    >>> y_pred = np.asarray([ [6791, 8742], [8735, 5054], [ 299, 2779] ])
    >>> row_wise_isin(y_pred, y_true)
    array([[False, False],
           [False, False],
           [False,  True]])
    """
    return (src[:, :, None] == trg[:, None, :]).any(-1)


def chunkize_list(a: list, n: int):
    """
    Divide list in `n` (almost) equally sized parts
    """

    if not len(a) >= n:
        raise ValueError(f"Cannot split a list of length {len(a)} into {n} chunks!")
    # n = min(n, len(a))  # don't create empty buckets
    k, m = divmod(len(a), n)
    chunks = (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))
    return chunks


def ddp_getattr(model: Union[Any, DistributedDataParallel], attr_name: str) -> Any:
    """
    Helper to get attribute from distributed model
    https://github.com/deepset-ai/FARM/blob/master/farm/modeling/optimization.py#L44
    """

    attr = (
        getattr(model.module, attr_name)
        if isinstance(model, DistributedDataParallel)
        else getattr(model, attr_name)
    )

    return attr


def natural_sorted(items: list[str]) -> list[str]:
    """
    Natural sort, i.e.
    >>> l = ['5', '11', '1']
    >>> sorted(l) == ['1', '11', '5']
    >>> natural_sort(l) == ['1','5', '11']
    """

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    return sorted(
        items, key=lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    )


def get_naturally_sorted_files(
    directory: str, extension: str, recursive: bool = False
) -> Iterator[str]:
    """
    Get files in directory (and subdirectories if `recursive=True`) naturally sorted
    """

    path = os.path.expanduser(directory)

    extension = f"*.{extension}" if extension is not None else "*"

    splits = [path, extension]

    if recursive:
        splits.insert(1, "**")

    pregex = os.path.join(*splits)

    for p in natural_sorted(list(glob.iglob(pregex, recursive=recursive))):
        yield p


def convert_to_disambiguated_name(row: dict, connector: str) -> dict:
    """
    Create unique name from: `name`, `disambiguation`
    """

    name = row["name"]

    if len(row["disambiguation"]) > 0:
        disambiguation: dict = dict(
            d.split(":", maxsplit=1)
            for d in row["disambiguation"].split(connector)
            if len(d) > 0
        )

        disambiguation_values = [
            disambiguation[k]
            for k in ["D", "A", "F"]
            if disambiguation.get(k) is not None
        ]
        if len(disambiguation_values) > 0:
            name += " (" + ",".join(disambiguation_values) + ")"

    row["unique_name"] = name

    return row


def get_optimizer(
    lr: float,
    epsilon: float,
    weight_decay: float,
    model: torch.nn.Module,
):
    """
    Instantiate optimizer
    """

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=lr,
        eps=epsilon,
    )

    return optimizer


def get_lr_scheduler(
    lr_scheduler: str,
    optimizer: torch.optim.Optimizer,
    warmup_steps: str = "0",
    num_training_steps: Optional[int] = None,
):
    """
    Learning rate scheduler
    """

    try:
        num_warmup_steps = int(warmup_steps)
    except ValueError:
        assert (
            num_training_steps is not None
        ), "Need `num_training_steps` to compute `warmup_steps` relative to number of steps"
        relative_warmup_steps = float(warmup_steps)
        num_warmup_steps = int(num_training_steps * relative_warmup_steps)

    scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return scheduler


def get_gradients_norm(
    model: torch.nn.Module, norm_type: float = 2, skip_bias: bool = True
) -> dict[str, torch.Tensor]:
    """
    Compute norm of the gradients of each model parameters (except biases)
    """

    grads_norm: dict = {}
    all_norms = []

    for name, param in model.named_parameters():
        if skip_bias and ("bias" in name or param.grad is None):
            continue

        param_grad_norm = round(float(param.grad.data.norm(norm_type)), 4)
        grad_name = f"gn/{name}"
        grads_norm[grad_name] = param_grad_norm
        all_norms.append(param_grad_norm)

    grad_norm_global = round(float(torch.tensor(all_norms).norm(norm_type)), 4)
    grads_norm["gn/ggn"] = grad_norm_global

    return grads_norm


class MultiLabelRecall(Metric):
    """
    Compute recall when the ground truth contains multiple values,
    e.g. composite mentions "breast and ovarian cancer".

    As the models can predict multiple values only in form of a ranking
    we use a relaxed version of the metric.
    If any of the top-k prediction is in the groud truth set we count this as a hit.

    >>> import numpy as np
    >>> y_pred, y_true = np.asarray([8,5,2,7,9]), np.asarray([1,5,6,4,3])

    >>> mlr = MultiLabelRecall(k=1)
    >>> mlr.add(y_pred=y_pred[None, :], y_true=y_true[None, :]) # mlr expects 2d arrays
    >> assert mlr.compute() == 0

    >>> mlr = MultiLabelRecall(k=3)
    >>> mlr.add(y_pred=y_pred[None, :], y_true=y_true[None, :]) # mlr expects 2d arrays
    >> assert mlr.compute() == 1
    """

    def __init__(self, k: int = 1):
        super().__init__()
        self.k = k
        self.add_state("score", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Add batch
        """

        assert (
            y_pred.shape[0] == y_true.shape[0]
        ), f"Shape mismatch between y_pred ({y_pred.shape[0]}) and y_true ({y_true.shape[0]})"

        self.total += y_pred.shape[0]
        self.score += row_wise_isin(y_pred[:, : self.k], y_true).any(-1).sum()

    def compute(self):
        """
        Final metric
        """
        return self.score / self.total


class Accuracy(Metric):
    """"""

    def __init__(self):
        super().__init__()
        self.add_state("score", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Add batch
        """
        self.score += int(y_true[np.arange(y_pred.shape[0]), y_pred].sum())
        self.total += y_pred.shape[0]

    def compute(self):
        """
        Final metric
        """
        return self.score / self.total
