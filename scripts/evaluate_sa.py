#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate SpeciesAssignment vs disambiguated KB"""
import os

import pandas as pd
from belb import AutoBelbCorpus, AutoBelbKb, BelbKb, Entities, Splits, Tables
from belb.resources import Corpora, Kbs
from bioc import pubtator
from sqlalchemy import select

from scripts.evaluate import multi_label_recall, parse_args
from utils import load_json, set_seed

CORPORA = [
    (Corpora.GNORMPLUS.name, Entities.GENE),
    (Corpora.NLM_GENE.name, Entities.GENE),
]


def load_species_assignment(path: str) -> dict:
    species_assignment: dict = {}

    with open(path) as fp:
        documents = pubtator.load(fp)

    for d in documents:
        for a in d.annotations:
            if a.type == "Gene":
                key = (a.pmid, a.start, a.end)
                species_assignment[key] = a.id.replace("Tax:", "")

    return species_assignment


def get_id_to_species(kb: BelbKb, ids: list) -> dict:
    table = kb.schema.get(Tables.KB)

    n = 10000
    id_to_fid = {}
    for chunk in [ids[i : i + n] for i in range(0, len(ids), n)]:
        query = select(table.c.identifier, table.c.foreign_identifier).where(
            table.c.identifier.in_(ids)
        )
        for r in kb.query(query):
            id_to_fid[str(r["identifier"])] = str(r["foreign_identifier"])

    return id_to_fid


def filter_by_species(
    hexdigest: str, y_pred: list, id_to_species: dict, species_assignment: dict
):
    filtered = [
        [y for y in yps if id_to_species[y] == species_assignment[hexdigest]]
        for yps in y_pred
    ]

    for i, f in enumerate(filtered):
        if len(f) == 0:
            filtered[i] = y_pred[i]

    return filtered


def build_corpus_species_assignment(belb_dir: str):
    DIR = os.getcwd()
    SPECIES_ASSIGN_DIR = os.path.join(
        DIR, "data", "species_assign", "text_species_gene_assign"
    )
    corpus_species_assignment = {}
    for corpus_name, entity_type in CORPORA:
        corpus = AutoBelbCorpus.from_name(
            name=corpus_name,
            directory=belb_dir,
            entity_type=entity_type,
            sentences=False,
            mention_markers=False,
            add_foreign_annotations=False,
        )

        corpus_gold = {}
        position_to_hexdigest = {}
        for e in corpus[Splits.TEST]:
            for p in e.passages:
                for a in p.annotations:
                    hexdigest = a.infons["hexdigest"]
                    corpus_gold[hexdigest] = a.identifiers
                    position_to_hexdigest[(e.id, a.start, a.end)] = hexdigest

        species_assignment = load_species_assignment(
            path=os.path.join(SPECIES_ASSIGN_DIR, f"{corpus_name}_test.PubTator")
        )
        species_assignment = {
            position_to_hexdigest[k]: v for k, v in species_assignment.items()
        }

        corpus_species_assignment[corpus_name] = species_assignment

    return corpus_species_assignment


def build_model_id_to_species(belb_dir: str):
    DIR = os.getcwd()
    CONFIG_DIR = os.path.join(DIR, "data", "configs")
    RESULTS_DIR = os.path.join(os.getcwd(), "data", "belb", "belhd_pred")
    kb = AutoBelbKb.from_name(
        directory=belb_dir,
        name=Kbs.NCBI_GENE.name,
        db_config=os.path.join(CONFIG_DIR, "db.yaml"),
        debug=False,
    )

    ids = set()
    for corpus_name, entity_type in CORPORA:
        for model in ["genbioel", "belhd_nohd"]:
            pred_path = os.path.join(
                RESULTS_DIR,
                corpus_name,
                model,
                "predictions.json",
            )
            corpus_pred = {p["hexdigest"]: p["y_pred"] for p in load_json(pred_path)}
            ids.update(
                [
                    int(i)
                    for y_pred in corpus_pred.values()
                    for yps in y_pred
                    for i in yps
                ]
            )

    with kb as handle:
        id_to_species = get_id_to_species(kb=handle, ids=list(ids))

    return id_to_species


def main():
    set_seed(72)
    args = parse_args()
    args.mode = "strict"
    args.topk = 1

    RESULTS_DIR = os.path.join(os.getcwd(), "data", "belb", "belhd_pred")
    GOLD = os.path.join(os.getcwd(), "data", "belb", "gold", "belb_gold.json")
    PRED_ID_TO_SPECIES = load_json(
        os.path.join(
            os.getcwd(), "data", "belb", "species_assign", "pred_id_to_species.json"
        )
    )
    CORPUS_SPECIES_ASSIGNMENT = load_json(
        os.path.join(
            os.getcwd(),
            "data",
            "belb",
            "species_assign",
            "corpus_species_assignment.json",
        )
    )

    gold = load_json(GOLD)
    gold = {
        k: v for k, v in gold.items() if k in [name for (name, entity_type) in CORPORA]
    }

    data = {}
    for corpus_name, entity_type in CORPORA:
        data[corpus_name] = {}
        corpus_gold = gold[corpus_name]
        species_assignment = CORPUS_SPECIES_ASSIGNMENT[corpus_name]

        for model in [
            "arboel_species",
            "genbioel",
            "genbioel_hd",
            "belhd",
            "belhd_nohd",
        ]:
            pred_path = os.path.join(
                RESULTS_DIR,
                corpus_name,
                model,
                "predictions.json",
            )

            if not os.path.exists(pred_path):
                continue

            corpus_pred = {p["hexdigest"]: p["y_pred"] for p in load_json(pred_path)}

            data[corpus_name][model] = multi_label_recall(
                gold=corpus_gold, pred=corpus_pred, mode=args.mode, topk=args.topk
            )

            if model in ["genbioel", "belhd_nohd"]:
                try:
                    corpus_pred = {
                        h: filter_by_species(
                            hexdigest=h,
                            y_pred=[[str(i) for i in yp] for yp in y_pred],
                            id_to_species=PRED_ID_TO_SPECIES,
                            species_assignment=species_assignment,
                        )
                        for h, y_pred in corpus_pred.items()
                    }
                except KeyError:
                    breakpoint()

                data[corpus_name][f"{model}_sa"] = multi_label_recall(
                    gold=corpus_gold,
                    pred=corpus_pred,
                    mode=args.mode,
                    topk=args.topk,
                )

    df = pd.DataFrame(data)

    print(df)


if __name__ == "__main__":
    main()
