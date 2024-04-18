#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encode **DISAMBIGUATED** KBs into zarr files
"""
import os
from typing import Optional

import hydra
import numpy as np
from belb.kbs import AutoBelbKb, BelbKb
from belb.kbs.query import Queries

# from belb.kbs.ncbi_gene import NCBI_GENE_SUBSETS
from belb.resources import Kbs
from loguru import logger
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from storage import KB_GROUP, ZarrWriter
from tokenizer import batch_encode_name, load_tokenizer
from utils import convert_to_disambiguated_name

# NOTE: maximum length
# {'ctd_diseases': 55,
#  'ctd_chemicals': 290,
#  'cellosaurus': 33,
#  'ncbi_taxonomy': 74,
#  'ncbi_gene': 182}


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
            "kbs",
            self.kb.schema.kb_config.name,
        ]

        if self.kb.kb_config.subset is not None:
            parts.append(f"{self.kb.kb_config.subset}_gnormplus")

        parts.append(self.tokenizer._codename)

        return os.path.join(*parts)

    def process_batch(self, batch: list[dict]) -> dict:
        """
        Encode batch of rows
        """

        batch = [
            convert_to_disambiguated_name(row=r, connector=self.kb.queries.connector)
            for r in batch
        ]

        uids = [np.asarray(r["uid"]) for r in batch]
        names = [
            np.asarray(n)
            for n in batch_encode_name(
                tokenizer=self.tokenizer, batch_names=[r["unique_name"] for r in batch]
            )
        ]

        return {
            "name": names,
            "uid": uids,
        }

    def write_batch(self, batch: list, idx: int, is_last: bool = False):
        """
        Write batch of processed data
        """

        self.writer.write(
            path=os.path.join(self.tokenized_directory, f"shard{idx}.zarr"),
            batch=self.process_batch(batch),
            is_last=is_last,
        )
        self.total += len(batch)
        if not is_last:
            logger.info("#PROGRESS : processed {} dictionary entries", self.total)

    def encode(self, chunksize: Optional[int] = None):
        """Convert KB into zarr file format"""

        chunksize = chunksize if chunksize is not None else 1000000

        logger.info("Start tokenizing kb dictionary `{}`...", self.kb.kb_config.name)

        os.makedirs(self.tokenized_directory, exist_ok=True)

        subset = None
        if self.kb.kb_config.subset is not None:
            assert self.kb.kb_config.subsets is not None
            subset = self.kb.kb_config.subsets[self.kb.kb_config.subset]
            # subset += self.kb.kb_config.subsets["gnormplus"]
            # subset = set(subset)

        query = self.kb.queries.get(Queries.DICTIONARY_ENTRIES, subset=subset)

        idx = 0
        batch: list = []

        with self.kb as handle:
            for row in handle.query(query):
                row.pop("identifier")
                if len(batch) < chunksize:
                    batch.append(row)
                else:
                    self.write_batch(batch=batch, idx=idx)
                    batch = []
                    batch.append(row)
                    idx += 1

        if len(batch) > 0:
            self.write_batch(batch=batch, idx=idx, is_last=True)
        logger.info("Completed processing {} dictionary entries.", self.total)


@hydra.main(version_base=None, config_path="../data/configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main
    """

    logger.info("Start tokenizing dictionaries of {} kbs...", len(Kbs))

    CONFIG_DIR = os.path.join(os.getcwd(), "data", "configs")

    # NOTE: ensure tokenized dictionary is split into mulitple files
    # to allow multi-gpu processing
    kbs = [
        {
            "name": Kbs.NCBI_GENE.name,
            "subset": "nlm_gene",
            "chunksize": 100000,
        },
        # {"name": Kbs.CTD_DISEASES.name, "chunksize": 10000},
        # {"name": Kbs.CTD_CHEMICALS.name, "chunksize": 10000},
        # {"name": Kbs.CELLOSAURUS.name, "chunksize": 10000},
        # {"name": Kbs.NCBI_TAXONOMY.name, "chunksize": 100000},
        # {"name": Kbs.UMLS.name, "chunksize": 100000},
        # {"name": Kbs.NCBI_GENE.name, "subset": "gnormplus", "chunksize": 100000},
        # {"name": Kbs.NCBI_GENE.name, "subset": "nlm_gene", "chunksize": 100000},
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
