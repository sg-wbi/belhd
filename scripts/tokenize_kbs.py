#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encode KBs into zarr files
"""
import os
import pickle
from typing import Optional

import hydra
import numpy as np
from belb import AutoBelbKb, BelbKb, Tables
from belb.resources import Kbs
from loguru import logger
from omegaconf import DictConfig
from sqlalchemy import select
from transformers import PreTrainedTokenizer

from storage import KB_GROUP, ZarrWriter
from tokenizer import batch_encode_name, load_tokenizer


class KbTokenizer:
    """Convert KBs into zarr file format"""

    def __init__(
        self,
        directory: str,
        tokenizer: PreTrainedTokenizer,
        kb: BelbKb,
    ):
        self.directory = directory
        self.tokenizer = tokenizer
        self.writer = ZarrWriter(group=KB_GROUP)
        self.kb = kb
        self.total = 0

    @property
    def tokenized_directory(self) -> str:
        """
        Directory where tokenized data is stored
        """
        parts = [
            self.directory,
            self.kb.schema.kb_config.name,
        ]

        if self.kb.kb_config.subset is not None:
            parts.append(self.kb.kb_config.subset)

        parts.append(self.tokenizer._codename)

        return os.path.join(*parts)

    def write_batch(self, batch: list, idx: int, is_last: bool = False):
        """Write batch of processed data"""

        self.writer.write(
            path=os.path.join(self.tokenized_directory, f"shard{idx}.zarr"),
            batch=self.process_batch(batch),
            is_last=is_last,
        )
        self.total += len(batch)
        if not is_last:
            logger.info("#PROGRESS : processed {} dictionary entries", self.total)

    def process_batch(self, batch: list):
        uids = [np.asarray(r["uid"]) for r in batch]
        names = [
            np.asarray(n)
            for n in batch_encode_name(
                tokenizer=self.tokenizer, batch_names=[r["name"] for r in batch]
            )
        ]

        return {"name": names, "uid": uids}

    def encode(self, chunksize: Optional[int] = None):
        """Convert KB into zarr file format"""

        chunksize = chunksize if chunksize is not None else 1000000

        logger.info("Start tokenizing kb dictionary `{}`...", self.kb.kb_config.name)

        os.makedirs(self.tokenized_directory, exist_ok=True)

        subset = None
        if self.kb.kb_config.subset is not None:
            assert self.kb.kb_config.subsets is not None
            subset = self.kb.kb_config.subsets[self.kb.kb_config.subset]

        table = self.kb.schema.get(Tables.KB)
        query = select(
            table.c.name,
            self.kb.queries.aggregate(table.c.uid).label("uids"),
            self.kb.queries.aggregate(table.c.identifier).label("identifiers"),
        ).group_by(table.c.name)
        if subset is not None:
            query = query.where(table.c.foreign_identifier.in_(subset))

        shard_idx = 0
        batch: list = []
        uid_to_identifiers = {}

        with self.kb as handle:
            for row in handle.query(query):
                uids = [
                    int(i)
                    for i in self.kb.queries.unpack_aggregated_values(row.pop("uids"))
                ]

                row["uid"] = next(iter(sorted(uids)))

                identifiers = [
                    int(i)
                    for i in self.kb.queries.unpack_aggregated_values(
                        row.pop("identifiers")
                    )
                ]

                uid_to_identifiers[row["uid"]] = identifiers

                if len(batch) < chunksize:
                    batch.append(row)
                else:
                    self.write_batch(batch=batch, idx=shard_idx)
                    batch = []
                    batch.append(row)
                    shard_idx += 1

        if len(batch) > 0:
            self.write_batch(batch=batch, idx=shard_idx, is_last=True)

        with open(
            os.path.join(self.tokenized_directory, "uid_to_identifiers.pkl"), "wb"
        ) as fp:
            pickle.dump(uid_to_identifiers, fp, pickle.HIGHEST_PROTOCOL)

        logger.info("Completed processing {} dictionary entries.", self.total)


@hydra.main(version_base=None, config_path="../data/configs", config_name="config_nohd")
def main(cfg: DictConfig):
    """
    Main
    """

    logger.info("Start tokenizing dictionaries of {} kbs...", len(Kbs))

    CONFIG_DIR = os.path.join(os.getcwd(), "data", "configs")

    # NOTE: ensure tokenized dictionary is split into mulitple files
    # to allow multi-gpu processing
    kbs = [
        # {"name": Kbs.CTD_DISEASES.name, "chunksize": 10000},
        # {"name": Kbs.CTD_CHEMICALS.name, "chunksize": 10000},
        # {"name": Kbs.CELLOSAURUS.name, "chunksize": 10000},
        # {"name": Kbs.NCBI_TAXONOMY.name, "chunksize": 100000},
        # {"name": Kbs.UMLS.name, "chunksize": 100000},
        # {"name": Kbs.NCBI_GENE.name, "subset": "gnormplus", "chunksize": 100000},
        {"name": Kbs.NCBI_GENE.name, "subset": "nlm_gene", "chunksize": 100000},
        # {"name": Kbs.NCBI_GENE.name},
    ]

    tokenizer = load_tokenizer(directory=cfg.tokenizers_dir, name_or_path=cfg.lm)

    for kb in kbs:
        belb_kb = AutoBelbKb.from_name(
            directory=cfg.belb_dir,
            name=kb["name"],
            db_config=os.path.join(CONFIG_DIR, "db.yaml"),
            debug=True,
            subset=kb.get("subset"),
        )
        converter = KbTokenizer(
            directory=cfg.kbs_dir,
            tokenizer=tokenizer,
            kb=belb_kb,
        )
        converter.encode(kb.get("chunksize"))

    logger.info("Completed!")


if __name__ == "__main__":
    main()
