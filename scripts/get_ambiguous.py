#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze corpora mentions"""
import os
from collections import Counter, defaultdict
from typing import Optional

from belb import (ENTITY_TO_KB_NAME, AutoBelbCorpus, AutoBelbKb, BelbCorpus,
                  BelbKb, Entities, Tables)
from belb.kbs.ncbi_gene import NCBI_GENE_SUBSETS
from belb.resources import Corpora, Kbs
from loguru import logger
from rapidfuzz import fuzz
from rapidfuzz import process as fuzzy_process
from rapidfuzz import utils as fuzzy_utils
from sqlalchemy import select

from utils import save_json

BELB_DIR = "/vol/home-vol3/wbi/gardasam/data/belb"
DB_CONFIG = "/vol/home-vol3/wbi/gardasam/projects/belbert/data/configs/db.yaml"
CORPORA = [
    (Corpora.NCBI_DISEASE.name, Entities.DISEASE),
    (Corpora.BC5CDR.name, Entities.DISEASE),
    (Corpora.BC5CDR.name, Entities.CHEMICAL),
    (Corpora.NLM_CHEM.name, Entities.CHEMICAL),
    (Corpora.BIOID.name, Entities.CELL_LINE),
    (Corpora.LINNAEUS.name, Entities.SPECIES),
    (Corpora.S800.name, Entities.SPECIES),
    (Corpora.GNORMPLUS.name, Entities.GENE),
    (Corpora.NLM_GENE.name, Entities.GENE),
    (Corpora.MEDMENTIONS.name, Entities.UMLS),
]


def preprocess(n):
    return " ".join(fuzzy_utils.default_process(n).split())


def chunks(xs, n):
    n = max(1, n)
    return (xs[i : i + n] for i in range(0, len(xs), n))


def get_homonyms(kb: BelbKb, subset: Optional[list] = None) -> dict:
    """
    Collect counts of homonyms
    """

    logger.debug("{}: homonyms", kb.kb_config.name)

    if kb.kb_config.name == Kbs.NCBI_GENE.name:
        subset = NCBI_GENE_SUBSETS[subset]

    homonyms: dict = {"h": list(), "fh": list(), "names": {}}

    nh = kb.schema.get(Tables.NAME_HOMONYMS)
    query = (
        select(nh.c.uid).where(nh.c.foreign_identifier.in_(subset))
        if subset is not None
        else select(nh.c.uid)
    )
    for r in kb.query(query):
        homonyms["h"].append(r["uid"])

    if kb.kb_config.foreign_identifier:
        fnh = kb.schema.get(Tables.FOREIGN_NAME_HOMONYMS)
        query = (
            select(fnh.c.uid).where(fnh.c.identifier.in_(subset))
            if subset is not None
            else select(fnh.c.uid)
        )
        for r in kb.query(query):
            homonyms["fh"].append(r["uid"])

    table = kb.schema.get(Tables.KB)
    uids = list(set(uid for _, uids in homonyms.items() for uid in uids))

    for chunk in chunks(uids, 10000):
        query = select(table.c.name, table.c.description).where(table.c.uid.in_(chunk))
        for r in kb.query(query):
            homonyms["names"][r["name"]] = r["description"]

    return homonyms


def get_identifier_to_names(kb: BelbKb, ids: set) -> dict:
    table = kb.schema.get(Tables.KB)
    query = select(table.c.identifier, table.c.name).where(table.c.identifier.in_(ids))

    identifier_to_names = defaultdict(set)
    for r in kb.query(query):
        identifier_to_names[str(r["identifier"])].add(r["name"])

    return identifier_to_names


# def get_identifier_to_names(kb: BelbKb, subset: Optional[str] = None) -> dict:
#     if kb.kb_config.name == Kbs.NCBI_GENE.name:
#         subset = NCBI_GENE_SUBSETS[subset]
#
#     query = kb.queries.get(Queries.SYNSET, subset=subset)
#     identifier_to_names = {}
#     for r in kb.query(query):
#         pr = kb.queries.parse_result(name=Queries.SYNSET, row=r)
#         identifier_to_names[str(pr["identifier"])] = pr["names"]
#
#     return identifier_to_names


def get_mentions(corpus: BelbCorpus, kb: BelbKb) -> dict:
    mentions = {}
    for e in corpus["test"]:
        for p in e.passages:
            for a in p.annotations:
                if a.foreign:
                    continue
                mentions[a.infons["hexdigest"]] = {
                    "text": a.text,
                    "ids": a.identifiers,
                }

    if kb.kb_config.string_identifier:
        identifiers = set(i for _, q in mentions.items() for i in q["ids"])
        mapping = kb.get_identifier_mapping(identifiers)
        for h in mentions:
            mentions[h]["ids"] = [str(mapping[i]) for i in mentions[h]["ids"]]

    return mentions


def get_potentially_ambiguous_mentions():
    stats = {}
    current_entity_type = None
    for corpus_name, entity_type in CORPORA:
        if (
            current_entity_type is None
            or current_entity_type != entity_type
            or entity_type == Entities.GENE
        ):
            kb_name = ENTITY_TO_KB_NAME[entity_type]
            logger.info("Entity type: {} - KB: {}", entity_type, kb_name)
            subset = corpus_name if entity_type == Entities.GENE else None
            kb = AutoBelbKb.from_name(
                directory=BELB_DIR,
                name=kb_name,
                db_config=DB_CONFIG,
                subset=corpus_name if entity_type == Entities.GENE else None,
                debug=False,
            )
            current_entity_type = entity_type

        corpus = AutoBelbCorpus.from_name(
            name=corpus_name,
            directory=BELB_DIR,
            entity_type=entity_type,
            add_foreign_annotations=False,
        )

        key = f"{corpus_name}-{entity_type}" if corpus_name == "bc5cdr" else corpus_name
        stats[key] = {}

        logger.info("Corpus: {}", key)

        with kb as handle:
            mentions = get_mentions(corpus=corpus, kb=handle)
            ids = set(int(i) for _, m in mentions.items() for i in m["ids"])
            identifier_to_names = get_identifier_to_names(handle, ids=ids)
            homonyms = get_homonyms(kb=handle, subset=subset)

        homonyms_names = defaultdict(list)
        for n, d in homonyms["names"].items():
            homonyms_names[preprocess(n)].append(d)

        identifier_to_names = {
            str(i): [preprocess(n) for n in names]
            for i, names in identifier_to_names.items()
        }

        stats[key]["counts"] = {}
        stats[key]["counts"]["total"] = len(mentions)
        stats[key]["counts"]["ambiguous"] = Counter()
        stats[key]["homonyms"] = homonyms
        for _, m in mentions.items():
            m["text"] = preprocess(m["text"])
            choices = [preprocess(n) for i in m["ids"] for n in identifier_to_names[i]]
            ranked_choices = [
                (n, s)
                for n, s, _ in fuzzy_process.extract(
                    m["text"], choices, scorer=fuzz.ratio, limit=None, processor=None
                )
            ]
            for k in [80, 85, 95, 100]:
                k_ranked_choices = [(n, s) for n, s in ranked_choices if s >= k]
                if any(n in homonyms_names for n, s in k_ranked_choices):
                    stats[key]["counts"]["ambiguous"][k] += 1

    save_json(
        item=stats,
        path="./data/ambiguous_mentions.json",
    )


def main():
    get_potentially_ambiguous_mentions()


if __name__ == "__main__":
    main()
