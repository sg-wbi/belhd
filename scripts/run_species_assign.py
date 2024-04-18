#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert BELB corpora into PubTator format"""
import argparse
import os

from belb import AutoBelbCorpus, Entities
from belb.resources import Corpora
from bioc import pubtator

CHOICES = ("input", "append")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate results")
    parser.add_argument(
        "--belb_dir",
        type=str,
        required=True,
        help="Directory where all BELB data is stored",
    )
    parser.add_argument(
        "--run",
        choices=CHOICES,
        type=str,
        default="input",
        help="Mode. `input`: export text to PubTator"
        + ", `append`: append annotations to file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    CORPORA = [
        (Corpora.GNORMPLUS.name, Entities.GENE),
        (Corpora.NLM_GENE.name, Entities.GENE),
    ]

    for corpus_name, entity_type in CORPORA:
        corpus = AutoBelbCorpus.from_name(
            name=corpus_name,
            directory=args.belb_dir,
            entity_type=entity_type,
            sentences=False,
            mention_markers=False,
            add_foreign_annotations=False,
        )

        if args.run == "input":
            documents = []
            for e in corpus["test"]:
                title = e.passages[0].text
                abstract = " ".join([p.text for p in e.passages[1:]])

                d = pubtator.PubTator(pmid=e.id, title=title, abstract=abstract)
                documents.append(d)

                folder = os.path.join(os.getcwd(), "data", "species_assign", "text")
                os.makedirs(folder, exist_ok=True)

                with open(
                    os.path.join(folder, f"{corpus_name}_test.PubTator"), "w"
                ) as fp:
                    # NOTE: GNorm2 complains for missing `\n` at the end of the file
                    # pubtator.dump(documents, fp)
                    for d in documents:
                        fp.write(str(d))
                        fp.write("\n")

        elif args.run == "append":
            eid_to_annotations = {
                e.id: [
                    pubtator.PubTatorAnn(
                        pmid=e.id,
                        start=a.start,
                        end=a.end,
                        type=a.entity_type.capitalize(),
                        text=a.text,
                        id="",
                    )
                    for p in e.passages
                    for a in p.annotations
                ]
                for e in corpus["test"]
            }

            in_dir = os.path.join(os.getcwd(), "data", "species_assign", "text_species")
            assert os.path.exists(
                in_dir
            ), f"Folder should contain output of GNorm2 species recognition: {in_dir}"
            assert (
                len(os.listdir(in_dir)) > 0
            ), f"Folder should contain output of GNorm2 species recognition: {in_dir}"

            with open(os.path.join(in_dir, f"{corpus_name}_test.PubTator")) as fp:
                documents = pubtator.load(fp)

            for d in documents:
                d.annotations = eid_to_annotations[d.pmid] + d.annotations

            out_dir = os.path.join(
                os.getcwd(), "data", "species_assign", "text_species_gene"
            )
            os.makedirs(out_dir, exist_ok=True)

            with open(os.path.join(out_dir, f"{corpus_name}_.PubTator"), "w") as fp:
                # NOTE: GNorm2 complains for missing `\n` at the end of the file
                # pubtator.dump(documents, fp)
                for d in documents:
                    fp.write(str(d))
                    fp.write("\n")


if __name__ == "__main__":
    main()
