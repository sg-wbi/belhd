#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface to add disambiguation tables
"""

from typing import Optional

import pandas as pd
from loguru import logger
from sqlalchemy.schema import AddConstraint, UniqueConstraint
from sqlalchemy.sql.expression import bindparam, select, text, update

from belb.kbs.db import SqlDialects
from belb.kbs.kb import BelbKb
from belb.kbs.query import Queries
from belb.kbs.schema import Tables
from belb.utils import CompressedFileWriter, chunkize, save_json


class DisambiguationModule(CompressedFileWriter):
    """
    Interface to add disambiguation tables
    """

    def __init__(self, kb: BelbKb, foreign_kb: Optional[BelbKb] = None):

        self.kb = kb
        self.kb_config = self.kb.schema.kb_config
        self.foreign_kb = foreign_kb

    def _iter_find_homonyms(
        self,
        kb: BelbKb,
        query_name: Queries,
        table_name: Tables,
        find_batch_size: int = 1000,
        insert_batch_size: int = 10000,
    ):

        assert query_name in [
            Queries.IDENTIFIER_HOMONYMS,
            Queries.NAME_HOMONYMS,
        ], "Can use this only with quereis IDENTIFIER_HOMONYMS or NAME_HOMONYMS"

        assert table_name in [
            Tables.IDENTIFIER_HOMONYMS,
            Tables.NAME_HOMONYMS,
        ], "Can use this only with tables IDENTIFIER_HOMONYMS or NAME_HOMONYMS"

        fi_table = kb.schema.get(Tables.FOREIGN_IDENTIFIERS)
        for rows in chunkize(kb.query(select(fi_table.c.identifier)), find_batch_size):
            logger.debug(
                "Query `{}` for {} `foreign identifier`s", query_name, len(
                    rows)
            )
            query = self.kb.queries.get(
                query_name, foreign_identifiers=[r["identifier"] for r in rows]
            )
            kb.populate_table_from_query(
                query=query,
                query_name=query_name,
                table_name=table_name,
                chunksize=insert_batch_size,
            )

    def disambiguate_foreign_name_homonyms(self, kb: BelbKb, chunksize: int = 10000):
        """
        Handle disambiguation for name homonyms by foreign identifier
        """

        logger.info(
            "Disambiguating foreign name homonyms (same name, different foreign identifier)..."
        )

        kb.populate_table_from_query(
            query_name=Queries.FOREIGN_NAME_HOMONYMS,
            table_name=Tables.FOREIGN_NAME_HOMONYMS,
        )

        table = self.kb.schema.get(Tables.KB)
        stmt = (
            update(table)
            .where(table.c.uid == bindparam("_uid"))
            .values(
                {
                    "disambiguation": table.c.disambiguation
                    + "F:"
                    + bindparam("_name")
                    + self.kb.queries.connector
                }
            )
        )

        fi = self.kb.schema.get(Tables.FOREIGN_IDENTIFIERS)
        fnh = self.kb.schema.get(Tables.FOREIGN_NAME_HOMONYMS)
        query = select(fnh.c.uid.label("uid"), fi.c.name.label("name")).join(
            fi, fi.c.identifier == fnh.c.identifier
        )
        # https://groups.google.com/g/sqlalchemy/c/xue9j34XSzk
        for rows in chunkize(kb.query(query), chunksize):
            for r in rows:
                r["_uid"] = r.pop("uid")
                r["_name"] = r.pop("name")
                if r["_name"] is None:
                    logger.warning(
                        "Could not find foreign name for uid: {}", r["_uid"])
                    r["_name"] = ""

            kb.connection.execute(stmt, rows)

        kb.connection.commit()

    def update_name_homonyms(self, kb: BelbKb, chunksize: int = 10000):
        """
        Add disambiguation names to table `NAME_HOMONYMS`
        """

        table = self.kb.schema.get(Tables.NAME_HOMONYMS)
        stmt = (
            update(table)
            .where(table.c.uid == bindparam("_uid"))
            .values(disambiguation_name=bindparam("disambiguation_name"))
        )
        for query_name in [
            Queries.NON_SYMBOL_DISAMBIGUATION,
            Queries.SYMBOL_DISAMBIGUATION,
        ]:

            logger.info("Query: `{}`...", query_name)

            query = self.kb.queries.get(query_name)
            for rows in chunkize(kb.query(query), chunksize):
                parsed_rows = [
                    self.kb.queries.parse_result(name=query_name, row=r) for r in rows
                ]
                for r in parsed_rows:
                    assert isinstance(r, dict)
                    # https://docs.sqlalchemy.org/en/14/tutorial/data_update.html#updating-and-deleting-rows-with-core
                    r["_uid"] = r.pop("uid")
                kb.connection.execute(stmt, parsed_rows)
            kb.connection.commit()

    def disambiguate_name_homonyms(
        self,
        kb: BelbKb,
        find_batch_size: int = 1000,
        insert_batch_size: int = 10000,
    ):
        """
        Find name homonyms and create disambiguation names
        """

        logger.info(
            "Disambiguating name homonyms (same name, different identifier[+foreign identifier])..."
        )

        query_name = Queries.NAME_HOMONYMS
        table_name = Tables.NAME_HOMONYMS

        if self.kb_config.foreign_identifier:
            self._iter_find_homonyms(
                query_name=query_name,
                table_name=table_name,
                kb=kb,
                find_batch_size=find_batch_size,
                insert_batch_size=insert_batch_size,
            )
        else:
            kb.populate_table_from_query(
                query_name=query_name,
                table_name=table_name,
                chunksize=insert_batch_size,
            )

        self.update_name_homonyms(kb=kb, chunksize=insert_batch_size)

        table = self.kb.schema.get(Tables.KB)
        nh = self.kb.schema.get(Tables.NAME_HOMONYMS)
        stmt = (
            update(table)
            .where(table.c.uid == bindparam("_uid"))
            .values(
                {
                    "disambiguation": table.c.disambiguation
                    + "D:"
                    + bindparam("disambiguation_name")
                    + self.kb.queries.connector
                }
            )
        )
        query = select(nh.c.uid, nh.c.disambiguation_name)
        for rows in chunkize(kb.query(query), insert_batch_size):
            # https://docs.sqlalchemy.org/en/14/tutorial/data_update.html#updating-and-deleting-rows-with-core
            for r in rows:
                r["_uid"] = r.pop("uid")

            # for N ambiguous names, we only need N-1 disambiguation names.
            # If one ambiguous name does not provide a disambiguation name
            # (e.g. it is the only name associated to the given identifier)
            # we can still guarantee uniqueness with `disambiguation`.
            rows = [r for r in rows if r["disambiguation_name"] is not None]

            kb.connection.execute(stmt, rows)

        kb.connection.commit()

    def add_attribute_disambiguation(self, kb: BelbKb, chunksize: int = 10000):
        """
        Add `attribute_name` to those entries which are still ambiguous
        aftern `disambiguation_name` and `foreign_name` have been added
        """

        table = kb.schema.get(Tables.KB)

        stmt = (
            update(table)
            .where(table.c.uid == bindparam("_uid"))
            .values(
                {
                    "disambiguation": table.c.disambiguation
                    + "A:"
                    + bindparam("_attribute")
                }
            )
        )

        query = kb.queries.get(Queries.ATTRIBUTE_DISAMBIGUATION)
        for rows in chunkize(kb.query(query), chunksize):

            parsed_rows = [
                kb.queries.parse_result(
                    name=Queries.ATTRIBUTE_DISAMBIGUATION, row=row)
                for row in rows
            ]
            parsed_rows = [r for rows in parsed_rows for r in rows]

            for r in parsed_rows:
                assert isinstance(r, dict)
                r["_uid"] = r.pop("uid")
                r["_attribute"] = r.pop("attribute")

            kb.connection.execute(stmt, parsed_rows)

        kb.connection.commit()

    def _sqlite_update_contraint(self, kb: BelbKb):
        """
        Sqlite does not support UPDATE CONSTRAINT:
            1. Modify table name
            2. Create new table w/ new/modified constraint
            3. Copy data from original table
            4. Delete original table
        """

        logger.info(
            "SQLite does not support `UPDATE` constraint: copying data to new table..."
        )

        table = self.kb.schema.get(Tables.KB)

        # 1. rename the old table and remove associated stuff
        kb.connection.execute(
            text(f"ALTER TABLE {table.name} RENAME TO {table.name}_backup")
        )
        for index in table.indexes:
            kb.connection.execute(text(f"DROP INDEX {index.name}"))

        # # 2. create the new table
        kb.schema.metadata.clear()
        _ = kb.schema.get(Tables.KB, disambiguation=True)
        kb.schema.metadata.create_all(kb.engine)

        # 3. copy the data back
        kb.connection.execute(
            text(f"INSERT INTO {table.name} SELECT * FROM {table.name}_backup")
        )

        # from sqlalchemy.exc import IntegrityError
        #
        # kb.schema.metadata.clear()
        # updated_table = self.kb.schema.get(Tables.KB, disambiguation=True)
        # print(updated_table)
        # kb.schema.metadata.create_all(kb.engine)
        # for row in kb.query(text(f"select * from {table.name}_backup")):
        #     try:
        #         kb.connection.execute(updated_table.insert(), row)
        #         kb.connection.commit()
        #     except IntegrityError as error:
        #         raise ValueError(f"Duplicate disambiguation row: {row}") from error

        # # 4. drop old table
        kb.connection.execute(text(f"DROP TABLE {table.name}_backup"))

    def update_constraint(self, kb: BelbKb):
        """
        Update constraint on KB table to check disambiguation consistency
        """

        logger.info(
            "Add constraint for disambiguation: UNIQUE(name,disambiguation)")

        if kb.db_config.dialect == SqlDialects.SQLITE:

            self._sqlite_update_contraint(kb=kb)

        else:

            table = self.kb.schema.get(Tables.KB)
            columns = [table.c.name, table.c.disambiguation]
            unique_constraint = UniqueConstraint(*columns, name="unique_name")
            kb.connection.execute(AddConstraint(unique_constraint))

        kb.connection.commit()

    def disambiguate(self, find_batch_size: int = 1000, insert_batch_size: int = 10000):
        """
        Add disambiguation tables to create VALID (name,identifier) pairs, i.e. UNIQUE(name, disambiguation):
            1. Find and disambiguate homonyms by foreign identifier: same name, different foreign identifier
            2. Find and disambiguate homonyms: same name[+foreign identifier], different identifier
        """

        with self.kb as handle:

            if self.kb_config.foreign_identifier:
                self.disambiguate_foreign_name_homonyms(
                    kb=handle, chunksize=insert_batch_size
                )

            self.disambiguate_name_homonyms(
                kb=handle,
                find_batch_size=find_batch_size,
                insert_batch_size=insert_batch_size,
            )

            self.update_constraint(kb=handle)

        save_json(
            path=self.kb.sentinel_file, item={
                "status": "up", "disambiguation": True}
        )
