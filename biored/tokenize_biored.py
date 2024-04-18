#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenize BioRED corpus annotated by AIONER (no identifiers) and save to zarr files
"""
import os
from collections import defaultdict
from bioc import biocjson

import hydra
import numpy as np
from belb import Example
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizer

from dataset import Collator
from storage import QUERY_GROUP, ZarrWriter
from tokenizer import batch_encode_text, load_tokenizer
from utils import save_json

QUERY_GROUP.pop("annotation_identifiers")
QUERY_GROUP.pop("annotation_identifiers_shape")


class CorpusTokenizer:
    """Convert Corpus into zarr file format"""

    def __init__(
        self,
        directory: str,
        hexdigest: str,
        config: OmegaConf,
        corpus: dict[str, list[Example]],
        tokenizer: PreTrainedTokenizer,
    ):
        self.directory = directory
        self.config = config
        self.writer = ZarrWriter(group=QUERY_GROUP)
        self.tokenizer = tokenizer
        self.corpus = corpus

        self.tokenized_directory = os.path.join(
            directory, hexdigest, self.tokenizer._codename
        )

        os.makedirs(self.tokenized_directory, exist_ok=True)
        self.collator = Collator()
        self.annotation_id_to_hexdigest: dict = {}

    def get_pad_value(self, key: str) -> int:
        """
        Get padding value for given key
        """

        assert (
            key in self.writer.group
        ), f"Unknown key `{key}`! Valid keys are {tuple(self.writer.group.keys())}"

        pad_value = self.writer.group[key].get("pad")

        assert pad_value is not None, f"Pad value for key `{key}` was not specified!"

        return pad_value

    def set_offsets_relative_to_passage(self):
        for _, examples in self.corpus.items():
            for e in examples:
                for p in e.passages:
                    # Make offsets relative to passage
                    for a in p.annotations:
                        a.start = a.start - p.offset
                        a.end = a.end - p.offset

    def update_tokenized_example(self, example: dict, items: dict):
        """
        Reconstruct tokenized example
        """

        if example.get("eid") is None:
            example["eid"] = items["eid"]
        else:
            eid = example["eid"]
            items_eid = items["eid"]
            assert (
                eid == items_eid
            ), f"Trying to add items from example `{eid}` to example `{items_eid}`"

        num_mentions = len(items["annotation_ids"])
        # Example contains mentions
        if num_mentions > 0:
            # sanity check offsets and padding
            num_offsets = len(items["annotation_offsets"])
            if num_offsets != num_mentions:
                raise RuntimeError(
                    f"EID:{eid} -  # offsets {num_offsets} != # mentions {num_mentions}"
                )
            items["annotation_ids"] = self.collator.batch(
                items["annotation_ids"],
                pad_value=self.get_pad_value("annotation_ids"),
            )
        # example has no mentions: add padding
        else:
            pad_value = self.get_pad_value("annotation_offsets")
            items["annotation_offsets"] = [[pad_value, pad_value]]
            items["annotation_identifiers"] = [
                [self.get_pad_value("annotation_identifiers")]
            ]
            items["annotation_ids"] = [[self.get_pad_value("annotation_ids")]]

        example["passage_ids"].append(items["passage_id"])
        example["input_ids"].append(items["input_ids"])
        example["annotation_offsets"].append(items["annotation_offsets"])
        example["annotation_ids"].append(items["annotation_ids"])

    def tokenize_examples(self, examples: list[Example]):
        """
        Tokenize corpus and store to zarr file
        """

        batch = [
            p.to_tuple() + (idx,) for idx, e in enumerate(examples) for p in e.passages
        ]

        pids, _, texts, annotations, eids, idxs = zip(*batch)

        input_ids, offsets = batch_encode_text(
            batch_texts=list(texts),
            batch_annotations=list(annotations),
            tokenizer=self.tokenizer,
        )

        tokenized_examples = []
        tokenized_example: dict = defaultdict(list)
        for elems in zip(input_ids, offsets, annotations, eids, pids, idxs):
            items = dict(
                zip(
                    [
                        "input_ids",
                        "annotation_offsets",
                        "annotations",
                        "eid",
                        "passage_id",
                        "idx",
                    ],
                    elems,
                )
            )

            eid = items["eid"]
            idx = items["idx"]

            items.update({"annotation_ids": []})
            for a in items.pop("annotations"):
                items["annotation_ids"].append([int(a.id)])
                self.annotation_id_to_hexdigest[f"{eid}.{a.id}"] = a.infons["hexdigest"]

            if tokenized_example.get("idx") is None:
                tokenized_example["idx"] = idx

            # reconstructed example or finished processing batch
            if idx != tokenized_example["idx"]:
                tokenized_examples.append(dict(tokenized_example.copy()))
                tokenized_example.clear()

            if tokenized_example.get("idx") is None:
                tokenized_example["idx"] = idx

            self.update_tokenized_example(example=tokenized_example, items=items)

        if tokenized_example["idx"] != tokenized_examples[-1]["idx"]:
            tokenized_examples.append(tokenized_example)

        for t in tokenized_examples:
            t.pop("idx")

        return tokenized_examples

    def aggregate_sentences_into_passages(self, example: dict) -> list:
        """
        Aggregate sentences of full text example into passages
        Aggregate title and abstract (passage ids 0 and 1).
        """

        bucket: list = []
        buckets: list = []
        current = 0
        for idx, pid in enumerate(example["passage_ids"]):
            # Aggregate title and abstract (passage ids 0 and 1).
            # Unless corpus does not have that format,
            # e.g. bioid has figure captions
            if self.config.title_abstract and pid == 1:
                pid = 0

            if pid != current:
                if len(bucket) > 0:
                    buckets.append(bucket)
                bucket = []
                current = pid
            bucket.append(idx)

        if len(bucket) > 0:
            buckets.append(bucket)

        passages = []
        for bucket in buckets:
            passage = {"eid": example["eid"]}
            for key, values in example.items():
                if isinstance(values, list):
                    try:
                        passage[key] = values[bucket[0] : bucket[-1] + 1]
                    except IndexError:
                        breakpoint()
            passages.append(passage)

        assert len(example["passage_ids"]) == sum(
            len(p["passage_ids"]) for p in passages
        ), "Aggregating sentences into passages failed!"

        return passages

    def aggregate_title_abstract(self, example: dict) -> dict:
        """
        Aggreate title and abstract into single text unit
        """

        for k, v in example.items():
            if isinstance(v, list):
                if k in ["passage_ids"]:
                    example[k][0:2] = [-1]
                else:
                    t, a = v[0], v[1]
                    if all(isinstance(x, list) for x in [t, a]):
                        example[k][0:2] = [t + a]
                    else:
                        t = np.asarray(t)
                        a = np.asarray(a)
                        maxlen = max([t.shape[1], a.shape[1]])
                        t = self.collator.pad(
                            t,
                            maxlen=(t.shape[0], maxlen),
                            pad_value=QUERY_GROUP[k].get("pad"),
                        )
                        a = self.collator.pad(
                            a,
                            maxlen=(a.shape[0], maxlen),
                            pad_value=QUERY_GROUP[k].get("pad"),
                        )
                        example[k][0:2] = [np.vstack([t, a])]

        return example

    def split_example_into_passages(self, e: dict) -> list[dict]:
        """Split full text articles into passages"""

        eid = e.pop("eid")
        num_passages = len(e["input_ids"])

        assert all(
            len(v) == num_passages for k, v in e.items() if k != "eid"
        ), f"EID:{eid} | # of passages must be the same for each item!"

        passages = []
        for i in range(num_passages):
            p = {"eid": eid}
            for k, v in e.items():
                p[k] = v[i]
            passages.append(p)

        return passages

    def filter_passages(self):
        for _, examples in self.corpus.items():
            for e in examples:
                e.passages = [p for p in e.passages if len(p.annotations) > 0]

        # # filter out passages w/o annotations
        # filtered_examples = [
        #     e
        #     for e in examples
        #     if not (
        #         e["annotation_identifiers"].flatten()
        #         == QUERY_GROUP["annotation_identifiers"]["pad"]
        #     ).all()
        # ]
        #

    def process_batch(self, examples: list):
        """
        Adjust tokenized examples for writing
        """

        ndarrays = [
            "input_ids",
            "annotation_ids",
            "annotation_offsets",
        ]

        for e in examples:
            e["eid"] = int(e["eid"])

        if self.config.sentences:
            if self.config.pmc:
                examples = [
                    p
                    for e in examples
                    for p in self.aggregate_sentences_into_passages(e)
                ]
        else:
            ndarrays.remove("input_ids")
            if self.config.title_abstract:
                examples = [self.aggregate_title_abstract(e) for e in examples]
            if self.config.pmc:
                examples = [
                    p for e in examples for p in self.split_example_into_passages(e)
                ]

        # batch 2/3d arrays
        for e in examples:
            for k in ndarrays:
                e[k] = self.collator.batch(e[k], pad_value=self.writer.group[k]["pad"])
                # try:
                #     e[k] = self.collator.batch(
                #         e[k], pad_value=self.writer.group[k]["pad"]
                #     )
                # except ValueError:
                #     breakpoint()

        # filtered_examples = self.filter_examples(examples=examples)

        # convert lists to numpy arrays
        batch = defaultdict(list)
        for e in examples:
            for k, v in e.items():
                v = v if isinstance(v, np.ndarray) else np.asarray(v)

                batch[k].append(v)

        return batch

    def tokenize_corpus(self):
        """Convert KB into zarr file format"""

        self.set_offsets_relative_to_passage()
        self.filter_passages()

        for split, examples in self.corpus.items():
            tokenized_examples = self.tokenize_examples(examples=examples)

            assert len(examples) == len(
                tokenized_examples
            ), f"Tokenization failed! Loaded {len(examples)} examples but {len(tokenized_examples)} after tokenization"

            path = os.path.join(self.tokenized_directory, f"{split}.zarr")
            self.writer.write(path=path, batch=self.process_batch(tokenized_examples))

            logger.debug(
                "Completed tokenizing `{}` split ({} examples)",
                split,
                len(tokenized_examples),
            )

        save_json(
            path=os.path.join(
                self.tokenized_directory, "annotation_id_to_hexdigest.json"
            ),
            item=self.annotation_id_to_hexdigest,
            indent=1,
        )


@hydra.main(version_base=None, config_path="../data/configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main
    """

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    CORPORA: list = [
        "4da1866bda6dd310a00d48c06db6beca",
        "79353782f04960cc902bd73ccd3ce447",
        "8c039885c70985722316bc60b048b805",
        "97842f6cebed290d040e404ae247a70a",
        "dac3563dfb7ea3af6be0357c4f0a7211",
    ]

    os.path.join(os.getcwd(), "data", "biored", "aioner_pred")

    tokenizer = load_tokenizer(directory=cfg.tokenizers_dir, name_or_path=cfg.lm)

    logger.info("Start tokenizing {} corpora...", len(CORPORA))

    for hexdigest in CORPORA:
        corpus_dir = os.path.join(DIR, hexdigest)

        config = OmegaConf.load(os.path.join(corpus_dir, "conf.yaml"))

        corpus = {}
        with open(os.path.join(corpus_dir, "test.bioc.json")) as fp:
            collection = biocjson.load(fp)
        corpus["test"] = [Example.from_bioc(d) for d in collection.documents]

        converter = CorpusTokenizer(
            directory=os.path.join(cfg.corpora_dir, "biored"),
            config=config,
            corpus=corpus,
            hexdigest=hexdigest,
            tokenizer=tokenizer,
        )

        converter.tokenize_corpus()


if __name__ == "__main__":
    main()
