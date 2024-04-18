import os
from collections import defaultdict
from bioc import pubtator
from belb import Entities
from utils import load_json
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BELHD on BioRED")
    parser.add_argument(
        "--biored",
        type=str,
        default=os.path.join(os.getcwd(), "data", "biored", "gold", "Test.PubTator"),
        help="Test split (PubTator format) of BioRED corpus",
    )
    parser.add_argument(
        "--pred",
        type=str,
        default=os.path.join(os.getcwd(), "data", "biored", "belhd_pred"),
        help="Directory with BELHD predictions on BioRED tagged by AIONER",
    )
    return parser.parse_args()


def document_level_normalization(gold, pred):
    p, r = 0, 0
    total = 0
    for eid, y_true in gold.items():
        y_pred = set(pred.get(eid, []))
        tps = set(y_true).intersection(y_pred)
        p += len(tps) / len(y_pred) if len(y_pred) > 0 else len(y_pred)
        r += len(tps) / len(y_true)
        total += 1

    p = p / total
    r = r / total
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else (p + r)

    out = {"p": p, "r": r, "f1": f1}

    return {k: round(v * 100, 2) for k, v in out.items()}


def main(args: argparse.Namespace):
    entity_type_map = {
        "DiseaseOrPhenotypicFeature": Entities.DISEASE,
        "ChemicalEntity": Entities.CHEMICAL,
        "OrganismTaxon": Entities.SPECIES,
        "GeneOrGeneProduct": Entities.GENE,
        "CellLine": Entities.CELL_LINE,
    }

    gold = {v: defaultdict(set) for v in entity_type_map.values()}

    with open(args.biored) as fp:
        for d in pubtator.load(fp):
            for a in d.annotations:
                if a.type == "SequenceVariant":
                    continue

                ids = a.id.replace("|", ",").split(",")
                ids = [i for i in ids if i != "-"]

                entity_type = entity_type_map[a.type]
                gold[entity_type][d.pmid].update(ids)

    results = {}
    for entity_type in entity_type_map.values():
        pred_path = os.path.join(args.pred, f"aioner_belhd_{entity_type}.json")
        pred = {
            k: [str(v) for v in values] for k, values in load_json(pred_path).items()
        }
        if entity_type in [Entities.CHEMICAL, Entities.DISEASE]:
            pred = {
                k: [v.replace("MESH:", "") for v in values]
                for k, values in pred.items()
            }
        s = document_level_normalization(gold=gold[entity_type], pred=pred)
        results[entity_type] = s

    print(pd.DataFrame(results))


if __name__ == "__main__":
    main(parse_args())
