#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenize corpora and save to zarr files
"""
import copy
import os
import re
import subprocess
import tempfile
from collections import defaultdict

import hydra
import numpy as np
from belb import (ENTITY_TO_KB_NAME, NAME_TO_CORPUS_CONFIG, Annotation,
                  AutoBelbCorpus, AutoBelbCorpusConfig, AutoBelbKb, BelbCorpus,
                  BelbKb, Entities, Example)
from belb.preprocessing.mark import AddMentionMarkers
from belb.resources import Corpora
from loguru import logger
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from dataset import Collator
from storage import QUERY_GROUP, ZarrWriter
from tokenizer import batch_encode_text, load_tokenizer
from utils import load_json, save_json


def run_ab3p(ab3p_path: str, texts: list[str]) -> dict:
    """Use Ab3p to resolve abbreviations"""
    abbreviations: dict = {}

    full_ab3p_path = os.path.expanduser(ab3p_path)
    word_data_dir = os.path.join(full_ab3p_path, "WordData")

    # Temporarily create path file in the current working directory for Ab3P
    with open(os.path.join(os.getcwd(), "path_Ab3P"), "w") as path_file:
        path_file.write(f"{word_data_dir}{os.path.sep}\n")

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as temp_file:
        for text in texts:
            temp_file.write(f"{text}\n")

        executable = os.path.join(full_ab3p_path, "identify_abbr")

        # Run Ab3P with the temp file containing the dataset
        # https://pylint.pycqa.org/en/latest/user_guide/messages/warning/subprocess-run-check.html
        try:
            out = subprocess.run(
                [executable, temp_file.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError:
            logger.error(
                (
                    "The Ab3P could not be run on your system.",
                    "To ensure maximum accuracy, please install Ab3P yourself.",
                    "See https://github.com/ncbi-nlp/Ab3P",
                )
            )
            out = None

        if out is not None:
            result = out.stdout.decode("utf-8")
            if "Path file for type cshset does not exist!" in result:
                logger.error(
                    (
                        "A file path_Ab3p needs to exist in your current directory",
                        "with the path to the WordData directory",
                    )
                )
            elif "Cannot open" in result or "failed to open" in result:
                logger.error("Could not open the WordData directory for Ab3P!")

            lines = result.split("\n")

            current = None

            for line in lines:
                elems = line.split("|")

                if len(elems) == 2:
                    eid, _ = elems
                    if current != eid:
                        current = eid

                if current not in abbreviations:
                    abbreviations[current] = {}

                elif len(elems) == 3:
                    sf, lf, _ = elems
                    sf = sf.strip()
                    lf = lf.strip()
                    abbreviations[current][sf] = lf

    return abbreviations


def build_abbreviation_dictionary(corpus: BelbCorpus, ab3p_path: str) -> dict:
    abbreviations = {}
    for _, examples in corpus.items():
        abbr = run_ab3p(
            ab3p_path=ab3p_path,
            texts=[f"{e.id}|{' '.join(p.text for p in e.passages)}" for e in examples],
        )
        abbreviations.update(abbr)

    return abbreviations


def clean_abbreviation_dictionary(abbreviations: dict) -> dict:
    cleand_abbreviations: dict = {}

    for pmid, sf_lf in abbreviations.items():
        if pmid not in cleand_abbreviations:
            cleand_abbreviations[pmid] = {}

        for sf, lf in sf_lf.items():
            cleand_abbreviations[pmid][sf] = lf

            if " -" in sf:
                clean_sf = sf.replace("-", "").strip()
                clean_lf = lf.replace("-", "").strip()
                cleand_abbreviations[pmid][clean_sf] = clean_lf

    return cleand_abbreviations


def example_expand_abbreviations(
    example: Example, abbreviations: dict
) -> list[Example]:
    # original = copy.deepcopy(example)
    # expanded = False

    offset = 0
    for p in example.passages:
        annotations = [a for a in p.annotations if not a.foreign]

        expanded_annotations = copy.deepcopy(annotations)

        for sf, lf in abbreviations.items():
            p.text = p.text.replace(sf, lf)

            for a in expanded_annotations:
                if a.text == sf:
                    a.text = lf
                elif sf in a.text:
                    a.text = a.text.replace(sf, lf)

        remapped = []
        annotation_match_checks = [False] * len(expanded_annotations)
        last_match = 0
        for idx, a in enumerate(sorted(expanded_annotations, key=lambda x: x.start)):
            # print(f"Query: {a.text}")
            pattern_str = re.escape(a.text)
            pattern_str = rf"(?<!\w){pattern_str}(?!\w)"
            pattern = re.compile(pattern_str)
            # from last match found in sentence
            # check for exact match of # of sentinel tokens
            # print(f"Search: {text[last_match:]}")
            match = re.search(pattern, p.text[last_match:])

            if match is not None:
                # print(f"Found: {a.text}\n")
                text_offset = len(p.text[:last_match])

                last_match = match.end() + text_offset

                a.start = match.start() + text_offset + offset
                a.end = match.end() + text_offset + offset

                remapped.append(a)

                annotation_match_checks[idx] = True
            # else:
            # print(f"Not found: {a.text}\n")

        if not all(annotation_match_checks):
            unmatched = [
                a.text
                for idx, a in enumerate(expanded_annotations)
                if not annotation_match_checks[idx]
            ]
            logger.debug(f"EID:{example.id}| Could not remap: `{unmatched}`")

        p.annotations = remapped
        p.offset = offset
        offset += len(p.text)

    return example


def expand_abbreviations(corpus: BelbCorpus, ab3p_path: str) -> BelbCorpus:
    abbreviations = build_abbreviation_dictionary(corpus=corpus, ab3p_path=ab3p_path)

    abbreviations = clean_abbreviation_dictionary(abbreviations)

    for _, examples in corpus.items():
        for i in range(len(examples)):
            example = examples[i]
            if example.id in abbreviations:
                example.drop_nested_annotations()
                example.drop_overlapping_annotations()
                examples[i] = example_expand_abbreviations(
                    example=example,
                    abbreviations=abbreviations[example.id],
                )

    return corpus


def add_mention_markers(corpus: BelbCorpus) -> BelbCorpus:
    marker = AddMentionMarkers(corpus.config.entity_type)

    for split, examples in corpus.items():
        for e in examples:
            try:
                # original = copy.deepcopy(e)
                e = marker.safe_apply(e)
                # if int(e.id) == 27635144:
                #     breakpoint()
            except RuntimeError:
                logger.debug(f"EID:{e.id}: Failed adding mention markers")

    return corpus


class CorpusTokenizer:
    """Convert Corpus into zarr file format"""

    def __init__(
        self,
        directory: str,
        belb_directory: str,
        tokenizer: PreTrainedTokenizer,
        corpus: BelbCorpus,
        kb: BelbKb,
        abbres: bool = False,
    ):
        self.directory = directory
        self.belb_directory = belb_directory
        self.kb = kb
        self.corpus = corpus
        self.writer = ZarrWriter(group=QUERY_GROUP)
        self.tokenizer = tokenizer

        self.corpus_directory = os.path.join(directory, self.corpus.config.name)

        self.abbres = abbres
        parts = [self.corpus_directory]
        if self.abbres:
            parts.append("abbres")
        parts += [self.corpus.hexdigest, self.tokenizer._codename]

        self.tokenized_directory = os.path.join(*parts)

        os.makedirs(self.tokenized_directory, exist_ok=True)
        self.collator = Collator()
        self.annotation_id_to_hexdigest: dict = {}

    @property
    def cache_dir(self):
        """
        Store KB outputs here ONCE per corpus
        """

        cache_dir = [self.corpus_directory]

        if len(self.corpus.config.entity_types) > 1:
            cache_dir.append(self.corpus.config.entity_type)

        if self.corpus.config.subsets is not None:
            cache_dir.append(self.corpus.config.subset)

        path = os.path.join(*cache_dir)

        os.makedirs(path, exist_ok=True)

        return path

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

        path_ih = os.path.join(self.cache_dir, "identifier_homonyms.json")
        path_im = os.path.join(self.cache_dir, "identifier_mapping.json")

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

    def get_corpus_identifiers(self) -> set:
        identifiers = set()
        for _, examples in self.corpus.items():
            for e in examples:
                for p in e.passages:
                    # Make offsets relative to passage
                    for a in p.annotations:
                        identifiers.update(a.identifiers)

        return identifiers

    def set_offsets_relative_to_passage(self):
        for _, examples in self.corpus.items():
            for e in examples:
                for p in e.passages:
                    # Make offsets relative to passage
                    for a in p.annotations:
                        a.start = a.start - p.offset
                        a.end = a.end - p.offset

    def preprocess_identifiers(self):
        """
        map indetifiers to integers
        """

        identifiers = self.get_corpus_identifiers()

        maps = self.load_mappings_for_identifiers(identifiers=identifiers)

        def _pp_ids(a: Annotation, maps: dict):
            if a.foreign:
                a.identifiers = [-1]
            else:
                if maps.get("im") is not None:
                    a.identifiers = [maps["im"][i] for i in a.identifiers]
                a.identifiers = [int(maps["ih"].get(i, i)) for i in a.identifiers]

        for _, examples in self.corpus.items():
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
            if self.corpus.config.title_abstract and pid == 1:
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
            "annotation_identifiers",
        ]

        for e in examples:
            e["eid"] = int(e["eid"])

        if self.corpus.config.sentences:
            if self.corpus.config.pmc:
                examples = [
                    p
                    for e in examples
                    for p in self.aggregate_sentences_into_passages(e)
                ]
        else:
            ndarrays.remove("input_ids")
            if self.corpus.config.title_abstract:
                examples = [self.aggregate_title_abstract(e) for e in examples]
            if self.corpus.config.pmc:
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

        logger.info(
            "Tokenize corpus `{}` ({})",
            self.corpus.config.name,
            self.corpus.hexdigest,
        )

        self.set_offsets_relative_to_passage()
        self.preprocess_identifiers()
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

    CONFIG_DIR = os.path.join(os.getcwd(), "data", "configs")

    OPTIONS = {
        "sentences": cfg.sentences,
        "mention_markers": True,
        "add_foreign_annotations": False,
    }

    CORPORA = [
        # (Corpora.GNORMPLUS.name, Entities.GENE),
        (Corpora.NLM_GENE.name, Entities.GENE),
        # (Corpora.NCBI_DISEASE.name, Entities.DISEASE),
        # (Corpora.BC5CDR.name, Entities.DISEASE),
        # (Corpora.BC5CDR.name, Entities.CHEMICAL),
        # (Corpora.NLM_CHEM.name, Entities.CHEMICAL),
        # (Corpora.LINNAEUS.name, Entities.SPECIES),
        # (Corpora.S800.name, Entities.SPECIES),
        # (Corpora.BIOID.name, Entities.CELL_LINE),
        # (Corpora.MEDMENTIONS.name, Entities.UMLS),
    ]

    tokenizer = load_tokenizer(directory=cfg.tokenizers_dir, name_or_path=cfg.lm)

    logger.info("Start tokenizing {} corpora...", len(CORPORA))

    for name, entity_type in CORPORA:
        base_config = NAME_TO_CORPUS_CONFIG[name]

        opts = copy.deepcopy(OPTIONS)

        opts["entity_type"] = entity_type

        opts_list = [opts]

        if base_config.foreign_entity_types is not None:
            opts_list.append(
                {
                    "sentences": cfg.sentences,
                    "mention_markers": True,
                    "add_foreign_annotations": True,
                    "entity_type": entity_type,
                }
            )

        if name == Corpora.MEDMENTIONS.name:
            opts_list.extend(
                [copy.deepcopy(OPTIONS) | {"max_mentions": mm} for mm in [20, 15]]
            )

        for opts in opts_list:
            kb = AutoBelbKb.from_name(
                directory=cfg.belb_dir,
                name=ENTITY_TO_KB_NAME[entity_type],
                db_config=os.path.join(CONFIG_DIR, "db.yaml"),
                debug=False,
            )

            if cfg.abbres:
                no_markers_opts = copy.deepcopy(opts)
                no_markers_opts["mention_markers"] = False
                corpus = AutoBelbCorpus.from_name(
                    directory=cfg.belb_dir, name=name, **no_markers_opts
                )
                # corpus = set_offsets_relative_to_passage(corpus)
                corpus = expand_abbreviations(corpus=corpus, ab3p_path=cfg.ab3p_path)
                corpus = add_mention_markers(corpus=corpus)
                config = AutoBelbCorpusConfig.from_name(name=name, **opts)
                corpus.config = config
                corpus.hexdigest = config.to_hexdigest()

            else:
                corpus = AutoBelbCorpus.from_name(
                    directory=cfg.belb_dir, name=name, **opts
                )

            converter = CorpusTokenizer(
                directory=cfg.corpora_dir,
                belb_directory=cfg.belb_dir,
                tokenizer=tokenizer,
                corpus=corpus,
                kb=kb,
                abbres=cfg.abbres,
            )

            converter.tokenize_corpus()


if __name__ == "__main__":
    main()
