#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenize corpora and save to zarr files
"""
import os
from collections import defaultdict

import hydra
import numpy as np
from belb import (
    ENTITY_TO_KB_NAME,
    Annotation,
    AutoBelbCorpus,
    AutoBelbKb,
    BelbCorpus,
    BelbKb,
    Entities,
    Example,
)
from belb.resources import Corpora
from loguru import logger
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from dataset import Collator
from storage import QUERY_GROUP, ZarrWriter
from tokenizer import batch_encode_text, load_tokenizer
from utils import load_json, save_json


class CorpusTokenizer:
    """Convert Corpus into zarr file format"""

    def __init__(
        self,
        directory: str,
        tokenizer: PreTrainedTokenizer,
        kb: BelbKb,
        corpora: list[BelbCorpus],
    ):
        self.directory = directory
        self.kb = kb
        self.writer = ZarrWriter(group=QUERY_GROUP)
        self.tokenizer = tokenizer
        self.tokenized_directory = os.path.join(
            directory, "_".join(c.config.name for c in corpora), tokenizer._codename
        )
        os.makedirs(self.tokenized_directory, exist_ok=True)
        self.corpora = corpora
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

    def load_mappings_for_identifiers(self, identifiers: set) -> dict:
        """
        Load mappings for homogeneization of identifiers
        """

        path_ih = os.path.join(self.tokenized_directory, "identifier_homonyms.json")
        path_im = os.path.join(self.tokenized_directory, "identifier_mapping.json")

        out = {}

        if not os.path.exists(path_ih):
            with self.kb as handle:
                identifier_homonyms = handle.get_identifier_homonyms(
                    identifiers=identifiers
                )
            save_json(item=identifier_homonyms, path=path_ih)
        else:
            identifier_homonyms = load_json(path_ih)

        out["ih"] = identifier_homonyms

        if self.kb.kb_config.string_identifier:
            if not os.path.exists(path_im):
                with self.kb as handle:
                    identifier_mapping = handle.get_identifier_mapping(
                        identifiers=identifiers
                    )
                save_json(item=identifier_mapping, path=path_im)
            else:
                identifier_mapping = load_json(path_im)

            out["im"] = identifier_mapping

        return out

    def get_identifiers(self, corpora: list[BelbCorpus]) -> set:
        identifiers = set()
        for corpus in corpora:
            for _, examples in corpus.items():
                for e in examples:
                    for p in e.passages:
                        # Make offsets relative to passage
                        for a in p.annotations:
                            identifiers.update(a.identifiers)

        return identifiers

    def set_offsets_relative_to_passage(self, corpus: BelbCorpus):
        for _, examples in corpus.items():
            for e in examples:
                for p in e.passages:
                    # Make offsets relative to passage
                    for a in p.annotations:
                        a.start = a.start - p.offset
                        a.end = a.end - p.offset

    def preprocess_identifiers(self, corpora: list[BelbCorpus]):
        """
        map indetifiers to integers
        """

        identifiers = self.get_identifiers(corpora)

        maps = self.load_mappings_for_identifiers(identifiers=identifiers)

        def _pp_ids(a: Annotation, maps: dict):
            if a.foreign:
                a.identifiers = [-1]
            else:
                if maps.get("im") is not None:
                    a.identifiers = [maps["im"][i] for i in a.identifiers]
                a.identifiers = [int(maps["ih"].get(i, i)) for i in a.identifiers]

        for corpus in corpora:
            for _, examples in corpus.items():
                for e in examples:
                    for p in e.passages:
                        for a in p.annotations:
                            _pp_ids(a=a, maps=maps)

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

        num_mentions = len(items["annotation_identifiers"])
        # Example contains mentions
        if num_mentions > 0:
            # sanity check offsets and padding
            num_offsets = len(items["annotation_offsets"])
            if num_offsets != num_mentions:
                raise RuntimeError(
                    f"EID:{eid} -  # offsets {num_offsets} != # mentions {num_mentions}"
                )

            # sanity check mentions ids and padding
            items["annotation_identifiers"] = self.collator.batch(
                items["annotation_identifiers"],
                pad_value=self.get_pad_value("annotation_identifiers"),
            )
            items["annotation_ids"] = self.collator.batch(
                items["annotation_ids"],
                pad_value=self.get_pad_value("annotation_ids"),
            )
            num_annotations_identifiers = items["annotation_identifiers"].shape[0]
            if num_annotations_identifiers != num_mentions:
                raise RuntimeError(
                    f"EID:{eid} - # annotations_identifiers {num_annotations_identifiers} != # mentions {num_mentions}"
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
        example["annotation_identifiers"].append(items["annotation_identifiers"])
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

            items.update({"annotation_identifiers": [], "annotation_ids": []})
            for a in items.pop("annotations"):
                items["annotation_identifiers"].append(a.identifiers)
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

    def aggregate_sentences_into_passages(
        self, example: dict, title_abstract: bool
    ) -> list:
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
            if title_abstract and pid == 1:
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

    def filter_passages(self, corpus: BelbCorpus):
        for _, examples in corpus.items():
            for e in examples:
                e.passages = [p for p in e.passages if len(p.annotations) > 0]

    def process_batch(
        self, examples: list, sentences: bool, pmc: bool, title_abstract: bool
    ):
        """
        Adjust tokenized examples for writing
        """

        ndarrays = [
            "input_ids",
            "annotation_ids",
            "annotation_offsets",
            "annotation_identifiers",
        ]

        for e in examples:
            e["eid"] = int(e["eid"])

        if sentences:
            if pmc:
                examples = [
                    p
                    for e in examples
                    for p in self.aggregate_sentences_into_passages(
                        e, title_abstract=title_abstract
                    )
                ]
        else:
            ndarrays.remove("input_ids")
            if title_abstract:
                examples = [self.aggregate_title_abstract(e) for e in examples]
            if pmc:
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

    def tokenize(self):
        """Convert KB into zarr file format"""

        self.preprocess_identifiers(self.corpora)

        corpora_examples = defaultdict(list)
        for corpus in self.corpora:
            self.set_offsets_relative_to_passage(corpus)
            self.filter_passages(corpus)

            for split, examples in corpus.items():
                if split == "test" and corpus.config.name not in [
                    Corpora.BIOID.name,
                    Corpora.NLM_CHEM.name,
                    Corpora.LINNAEUS.name,
                    Corpora.S800.name,
                ]:
                    continue

                tokenized_examples = self.tokenize_examples(examples=examples)
                assert len(examples) == len(
                    tokenized_examples
                ), f"Tokenization failed! Loaded {len(examples)} examples but {len(tokenized_examples)} after tokenization"
                processed_examples = self.process_batch(
                    tokenized_examples,
                    sentences=corpus.config.sentences,
                    pmc=corpus.config.pmc,
                    title_abstract=corpus.config.title_abstract,
                )

                for k, values in processed_examples.items():
                    corpora_examples[k] += values

        path = os.path.join(self.tokenized_directory, "train.zarr")
        self.writer.write(path=path, batch=corpora_examples)

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

    CONFIG_DIR = os.path.join(os.getcwd(), "data", "configs")

    opts = {
        "sentences": cfg.sentences,
        "mention_markers": True,
        "add_foreign_annotations": False,
    }

    CORPORA = [
        ([Corpora.NLM_GENE.name, Corpora.GNORMPLUS.name], Entities.GENE),
        ([Corpora.NCBI_DISEASE.name, Corpora.BC5CDR.name], Entities.DISEASE),
        ([Corpora.NLM_CHEM.name], Entities.CHEMICAL),
        ([Corpora.BIOID.name], Entities.CELL_LINE),
        ([Corpora.LINNAEUS.name, Corpora.S800.name], Entities.SPECIES),
    ]

    tokenizer = load_tokenizer(directory=cfg.tokenizers_dir, name_or_path=cfg.lm)

    logger.info("Start tokenizing corpora...")

    for names, entity_type in CORPORA:
        kb = AutoBelbKb.from_name(
            directory=cfg.belb_dir,
            name=ENTITY_TO_KB_NAME[entity_type],
            db_config=os.path.join(CONFIG_DIR, "db.yaml"),
            debug=False,
        )

        corpora = []
        for name in names:
            corpora.append(
                AutoBelbCorpus.from_name(directory=cfg.belb_dir, name=name, **opts)
            )

        converter = CorpusTokenizer(
            directory=cfg.corpora_dir,
            tokenizer=tokenizer,
            corpora=corpora,
            kb=kb,
        )

        converter.tokenize()


if __name__ == "__main__":
    main()
