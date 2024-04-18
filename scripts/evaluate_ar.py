#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate SpeciesAssignment vs disambiguated KB"""
import os

import pandas as pd
from belb import Entities
from belb.resources import Corpora

from scripts.evaluate import (
    CORPORA_MULTI_ENTITY_TYPES,
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
        (Corpora.GNORMPLUS.name, Entities.GENE),
        (Corpora.NLM_GENE.name, Entities.GENE),
        (Corpora.MEDMENTIONS.name, Entities.UMLS),
    ]
    MODELS = [
        "arboel_ar",
        "genbioel_ar",
        "belhd_nohd_ar",
    ]

    gold = load_json(GOLD)
    gold = {
        k: v for k, v in gold.items() if k in [name for (name, entity_type) in CORPORA]
    }

    data = {}
    preds = {}
    for corpus_name, entity_type in CORPORA:
        full_name = (
            f"{corpus_name}_{entity_type}"
            if corpus_name in CORPORA_MULTI_ENTITY_TYPES
            else corpus_name
        )
        for model in MODELS:
            if model not in preds:
                preds[model] = {}

            pred_path = os.path.join(
                RESULTS_DIR,
                corpus_name,
                model,
                "predictions.json",
            )

            if not os.path.exists(pred_path):
                continue

            preds[model][full_name] = {
                p["hexdigest"]: p["y_pred"] for p in load_json(pred_path)
            }

    data = {}
    for full_name, corpus_gold in gold.items():
        if full_name not in data:
            data[full_name] = {}

        ##################################################################
        # Use only predictions avilable for ALL models
        # arboEL and BELHD skip some mentions
        # because they need to repalce abbr. with long forms
        # WITHIN the text,
        # this means we need to modify all offsets which is a NIGHTMARE
        # so we just skip them instead (they are only a minimal fraction)
        ##################################################################
        intersection = set.intersection(
            *[
                set(model_pred[full_name].keys())
                for _, model_pred in preds.items()
                if full_name in model_pred
            ]
        )
        corpus_gold = {
            h: y_true for h, y_true in corpus_gold.items() if h in intersection
        }
        ##################################################################

        for model, model_pred in preds.items():
            if full_name in model_pred:
                corpus_pred = model_pred[full_name]
                try:
                    data[full_name][model] = multi_label_recall(
                        gold=corpus_gold,
                        pred=corpus_pred,
                        mode=args.mode,
                        topk=args.topk,
                    )
                except TypeError:
                    print(f"Formatting error with `{model}` and `{full_name}`")

    print(pd.DataFrame(data))


if __name__ == "__main__":
    main()
