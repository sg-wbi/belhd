#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create dictionaries from KBs in belb format
"""
import argparse

from loguru import logger

from belb.kbs import AutoBelbKb
from disambiguation import DisambiguationModule
from belb.resources import Kbs
from belb.utils import set_logging


def parse_args():
    """
    Parse CLI
    """
    parser = argparse.ArgumentParser(
        description="Convert KBs into dictionaries (build disambiguation tables)"
    )
    parser.add_argument(
        "--dir",
        required=True,
        type=str,
        help="Directory where all BELB data is stored",
    )
    parser.add_argument(
        "--db",
        required=True,
        type=str,
        help="Database configuration",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log level to DEBUG",
    )
    return parser.parse_args()


def main():
    """
    Standalone
    """
    args = parse_args()

    set_logging(
        logger=logger,
        directory=args.dir,
        logfile="dictionaries.log",
        level="DEBUG" if args.debug else "INFO",
    )

    ready = [
        Kbs.CTD_DISEASES.name,
        Kbs.CTD_CHEMICALS.name,
        Kbs.NCBI_TAXONOMY.name,
        Kbs.CELLOSAURUS.name,
        Kbs.UMLS.name,
        Kbs.NCBI_GENE.name,
    ]

    for resource in Kbs:

        if resource.name not in ready:
            continue

        logger.info("Adding disambigation data to kb: {}", resource.name)

        kb = AutoBelbKb.from_name(
            name=resource.name,
            directory=args.dir,
            db_config=args.db,
            debug=args.debug,
        )

        foreign_kb = None
        if kb.kb_config.foreign_kb is not None:
            foreign_kb = AutoBelbKb.from_name(
                name=kb.kb_config.foreign_kb,
                directory=args.dir,
                db_config=args.db,
                debug=args.debug,
            )

        disambiguation = DisambiguationModule(kb=kb, foreign_kb=foreign_kb)

        disambiguation.disambiguate()


if __name__ == "__main__":
    main()
