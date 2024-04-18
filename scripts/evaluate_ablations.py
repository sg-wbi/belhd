#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate BELHD ablations"""
import os

import pandas as pd
from belb import Entities
from belb.resources import Corpora

from scripts.evaluate import (
    multi_label_recall,
    parse_args,
)
from utils import load_json, set_seed


def main():
    set_seed(72)
    args = parse_args()
    args.mode = "strict"
    args.topk = 1

    RESULTS_DIR = os.path.join(os.getcwd(), "data", "belb", "belhd_pred")
    GOLD = os.path.join(os.getcwd(), "data", "belb", "gold", "belb_gold.json")

    CORPORA = [
        (Corpora.NLM_GENE.name, Entities.GENE),
    ]
    MODELS = ["belhd", "belhd_nohd", "belhd_nocs", "belhd_noph", "belhd_noctx"]

    gold = load_json(GOLD)

    data = {}
    for corpus_name, entity_type in CORPORA:
        if corpus_name not in data:
            data[corpus_name] = {}

        corpus_gold = gold[corpus_name]

        for model in MODELS:
            pred_path = os.path.join(
                RESULTS_DIR,
                corpus_name,
                model,
                "predictions.json",
            )

            if not os.path.exists(pred_path):
                continue

            corpus_pred = {p["hexdigest"]: p["y_pred"] for p in load_json(pred_path)}

            try:
                data[corpus_name][model] = multi_label_recall(
                    gold=corpus_gold,
                    pred=corpus_pred,
                    mode=args.mode,
                    topk=args.topk,
                )
            except TypeError:
                print(f"Formatting error with `{model}` and `{corpus_name}`")

    print(pd.DataFrame(data))


if __name__ == "__main__":
    main()
