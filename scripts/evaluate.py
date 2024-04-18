#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Get results"""

import argparse
import os
import random
import pandas as pd
from belb import ENTITY_TO_KB_NAME, AutoBelbCorpus, AutoBelbKb, BelbKb, Entities, Splits
from belb.resources import Corpora

from utils import load_json, set_seed, save_json

NIL = "NIL"

EVAL_MODES = ("std", "strict", "lenient")
CORPORA_MULTI_ENTITY_TYPES = [Corpora.BC5CDR.name, Corpora.BIOID.name]


CORPORA = [
    (Corpora.NCBI_DISEASE.name, Entities.DISEASE),
    (Corpora.BC5CDR.name, Entities.DISEASE),
    (Corpora.BC5CDR.name, Entities.CHEMICAL),
    (Corpora.NLM_CHEM.name, Entities.CHEMICAL),
    (Corpora.BIOID.name, Entities.CELL_LINE),
    (Corpora.GNORMPLUS.name, Entities.GENE),
    (Corpora.NLM_GENE.name, Entities.GENE),
    (Corpora.S800.name, Entities.SPECIES),
    (Corpora.LINNAEUS.name, Entities.SPECIES),
    (Corpora.MEDMENTIONS.name, Entities.UMLS),
]

ENTITY_TYPE_STRING_IDENTIFIERS = [
    Entities.DISEASE,
    Entities.CHEMICAL,
    Entities.CELL_LINE,
    Entities.UMLS,
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate results")
    parser.add_argument(
        "--belb_dir",
        type=str,
        default=None,
        help="Directory where all BELB data is stored",
    )
    # parser.add_argument(
    #     "--topk",
    #     type=int,
    #     default=1,
    #     help="Ranks to consider in prediction",
    # )
    # parser.add_argument(
    #     "--mode",
    #     type=str,
    #     choices=EVAL_MODES,
    #     default="std",
    #     help="If multiple predictions are return consider it wrong",
    # )
    return parser.parse_args()


def get_integer_identifiers(kb: BelbKb, gold: dict) -> dict:
    identifiers = set(i for ids in gold.values() for i in ids)

    with kb as handle:
        map = handle.get_identifier_mapping(identifiers)

    gold = {h: list(set([map[i] for i in ids])) for h, ids in gold.items()}

    return gold


def load_gold(
    corpus_name: str,
    entity_type: str,
    belb_dir: str,
    db_config,
) -> dict:
    corpus = AutoBelbCorpus.from_name(
        name=corpus_name,
        directory=belb_dir,
        entity_type=entity_type,
        sentences=False,
        mention_markers=False,
        add_foreign_annotations=False,
    )

    gold = {
        a.infons["hexdigest"]: a.identifiers
        for e in corpus[Splits.TEST]
        for p in e.passages
        for a in p.annotations
    }

    if entity_type in ENTITY_TYPE_STRING_IDENTIFIERS:
        kb = AutoBelbKb.from_name(
            directory=belb_dir,
            name=ENTITY_TO_KB_NAME[entity_type],
            db_config=db_config,
            debug=False,
        )

        gold = get_integer_identifiers(kb=kb, gold=gold)

    return gold


def multi_label_recall(
    gold: dict, pred: dict, topk: int = 1, mode: str = "std"
) -> float:
    """
    Labels can have multiple valid ids,
    e.g. composite mentions "breast and ovarian cancer".
    Consider correct if predicted id is in the gold label set.
    >>> y_true = [21, 53, 45]
    >>> y_pred = 21
    >>> assert multi_label_recall(y_true, y_pred) == 1
    """

    hits = 0

    for h, y_true in gold.items():
        # int
        y_true = set(int(y) for y in y_true)

        # get topk predictions
        y_preds = [list(set(yp)) for yp in pred[h][:topk]]

        if mode in ["std", "strict"]:
            # get single prediction
            if mode == "strict":
                # in strict mode default wrong if multiple predictions
                y_pred = [NIL if len(y) > 1 else y[0] for y in y_preds]
            elif mode == "std":
                # sample if multiple predictions
                y_pred = [random.sample(y, 1)[0] for y in y_preds]

            # go over k predicitons
            for y in y_pred:
                # int
                y = -1 if y == NIL else int(y)
                if y in y_true:
                    hits += 1
                    # if you get a hit stop
                    break
        else:
            for ys in y_preds:
                ys = [-1 if y == NIL else int(y) for y in ys]
                if any(y in y_true for y in ys):
                    hits += 1
                    # if you get a hit stop
                    break

    return round(hits / len(gold) * 100, 2)


def get_main_table(
    gold: dict,
    models: list,
    directory: str,
    corpora: list,
    mode: str = "std",
    topk: int = 1,
) -> pd.DataFrame:
    data: dict = {}
    for corpus, entity_type in corpora:
        for model in models:
            corpus_name = (
                f"{corpus}_{entity_type}"
                if corpus in CORPORA_MULTI_ENTITY_TYPES
                else corpus
            )

            pred_path = os.path.join(
                directory,
                corpus_name,
                model,
                "predictions.json",
            )

            if not os.path.exists(pred_path):
                continue

            corpus_pred = {p["hexdigest"]: p["y_pred"] for p in load_json(pred_path)}

            corpus_gold = gold[
                f"{corpus}_{entity_type}"
                if corpus in CORPORA_MULTI_ENTITY_TYPES
                else corpus
            ]

            if corpus_name not in data:
                data[corpus_name] = {}

            try:
                data[corpus_name][model] = multi_label_recall(
                    gold=corpus_gold, pred=corpus_pred, mode=mode, topk=topk
                )
            except TypeError:
                print(f"ERROR with  model `{model}` and corpus `{corpus_name}`")

    return pd.DataFrame(data)


def main():
    set_seed(72)
    args = parse_args()

    RESULTS_DIR = os.path.join(os.getcwd(), "data", "belb", "belhd_pred")
    GOLD = os.path.join(os.getcwd(), "data", "belb", "gold", "belb_gold.json")
    DB_CONFIG = os.path.join(os.getcwd(), "data", "configs", "db.yaml")
    MODELS = [
        "arboel",
        "biosyn",
        "biosyn_hd",
        "genbioel",
        "genbioel_hd",
        "belhd",
    ]

    if not os.path.exists(GOLD):
        assert args.belb_dir is not None, "Nedd path to BELB directory!"
        gold = {}
        for corpus_name, entity_type in CORPORA:
            full_name = (
                f"{corpus_name}_{entity_type}"
                if corpus_name in CORPORA_MULTI_ENTITY_TYPES
                else corpus_name
            )

            gold[full_name] = load_gold(
                corpus_name=corpus_name,
                entity_type=entity_type,
                belb_dir=args.belb_dir,
                db_config=DB_CONFIG,
            )
        save_json(item=gold, path=GOLD, indent=1)
    else:
        gold = load_json(GOLD)

    df = get_main_table(
        models=MODELS,
        gold=gold,
        directory=RESULTS_DIR,
        mode="strict",
        topk=1,
        corpora=CORPORA,
    )

    df.replace(pd.NA, "-", inplace=True)

    print(df)

    # print(df.to_latex(float_format="%.2f"))


if __name__ == "__main__":
    main()
