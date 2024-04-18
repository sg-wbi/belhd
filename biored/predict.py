#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate BiEncoder on BioRED corpus annotated by AIONER
"""
import copy
import multiprocessing as mp
import os

import faiss
import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import is_bf16_available
from belb import Entities, Tables
from belb.kbs import ENTITY_TO_KB_NAME, AutoBelbKb, BelbKb, Kbs
from loguru import logger
import numpy as np
from omegaconf import DictConfig
from sqlalchemy import select
from torch.utils.data import DataLoader
from transformers import BertModel, LongformerModel

import utils
from dataset import QueryCollator, QueryDataset, squeeze_array
from index import IndexConfig, ShardedIndex, search_index
from model import BiEncoderModel
from tokenizer import load_tokenizer
from storage import QUERY_GROUP

QUERY_GROUP.pop("annotation_identifiers")
QUERY_GROUP.pop("annotation_identifiers_shape")


def map_uids_to_identifiers(
    uids: np.ndarray,
    kb: BelbKb,
    train: bool = True,
):
    """
    Given a 2d array of KB uid(s):
        1. retrive the corresponding dictionary name
        2. tokenize and batch
    """

    batch_uids = sorted(set(int(i) for i in uids.flatten()))
    table = kb.schema.get(Tables.KB)

    query = select(
        table.c.uid, table.c.identifier, table.c.name, table.c.disambiguation
    ).where(table.c.uid.in_(batch_uids))

    uid_to_identifier = {}
    for row in kb.query(query):
        uid_to_identifier[row["uid"]] = row["identifier"]

    identifiers = np.vectorize(uid_to_identifier.get)(uids)

    return identifiers


class NerQueryCollator(QueryCollator):
    def collate_fn(self, batch: list[dict]) -> dict[str, np.ndarray]:
        """
        Create batch from list
        """

        passages = copy.deepcopy(batch[0])
        annotation_ids = passages["annotation_ids"]
        squeezed_annotation_ids = squeeze_array(
            annotation_ids, QUERY_GROUP["annotation_ids"]["pad"]
        )

        inputs: dict = {
            "eid": passages["eid"][0],
            "passage_ids": passages["passage_ids"],
            "annotation_offsets": passages["annotation_offsets"],
            "input_ids": self.add_special_tokens(passages["input_ids"]),
            "annotation_ids": squeezed_annotation_ids,
            "by_sent_annotation_ids": annotation_ids,
        }

        return inputs


@hydra.main(version_base=None, config_path="../data/configs", config_name="config")
def main(cfg: DictConfig):
    """
    Run training w/ evaluateuation at the end of each epoch
    """

    # path to tokenized BioRED corpus (see tokenize_biored.py)
    ZARR_DIR = ""
    # path to model trained on BELB (see train.py)
    MODELS_DIR = ""

    hexdigest, entity_type = ("97842f6cebed290d040e404ae247a70a", Entities.CHEMICAL)
    # hexdigest, entity_type = ("8c039885c70985722316bc60b048b805", Entities.DISEASE)
    # hexdigest, entity_type = ("4da1866bda6dd310a00d48c06db6beca", Entities.CELL_LINE)
    # hexdigest, entity_type = ("dac3563dfb7ea3af6be0357c4f0a7211", Entities.SPECIES)
    # hexdigest, entity_type = ("79353782f04960cc902bd73ccd3ce447", Entities.GENE)

    cfg.run_dir = os.path.join(MODELS_DIR, entity_type)

    assert cfg.predict in [
        "best",
        "last",
    ], f"Unknown checkpoint `{cfg.predict}`: must be either `last` or `best`"

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cfg.mixed_precision = "bf16" if is_bf16_available() else "fp16"

    cores = min(30, mp.cpu_count())
    utils.init_run(seed=cfg.seed, cores=cores, cuda_available=cuda_available)
    faiss_cores = 1 if cfg.index_type == "flat" else max(20, cores)
    faiss.omp_set_num_threads(faiss_cores)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        device_placement=False,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )

    logger.info("Run: `{}`", cfg.run_dir)

    tokenizer = load_tokenizer(directory=cfg.tokenizers_dir, name_or_path=cfg.lm)

    query_collator = NerQueryCollator.from_tokenizer(tokenizer)
    query_dataset = QueryDataset(
        path=os.path.join(ZARR_DIR, hexdigest, tokenizer._codename, "test.zarr"),
        group=QUERY_GROUP,
    )
    test_dl = DataLoader(
        dataset=query_dataset,
        shuffle=False,
        batch_size=1,
        pin_memory=cuda_available,
        collate_fn=query_collator.collate_fn,
    )
    kb = AutoBelbKb.from_name(
        directory=cfg.belb_dir,
        name=ENTITY_TO_KB_NAME[entity_type],
        db_config=os.path.join(os.getcwd(), "data", "configs", "db.yaml"),
        debug=False,
    )

    if cfg.model_type == "bert":
        encoder = BertModel.from_pretrained(cfg.lm, add_pooling_layer=False)
    elif cfg.model_type == "longformer":
        encoder = LongformerModel.from_pretrained(
            cfg.lm,
            num_hidden_layers=12,
            attention_window=[64] * 12,
            add_pooling_layer=False,
        )

    model = BiEncoderModel(
        tokenizer=tokenizer,
        encoder=encoder,
        project=cfg.project,
        device=accelerator.device,
        query_side_ft=cfg.query_side_ft,
        global_candidates=cfg.global_candidates,
        max_global_candidates=cfg.max_global_candidates,
        foreign_attention=cfg.foreign_attention,
        exclude_context=cfg.exclude_context,
    )

    assert os.path.exists(
        os.path.join(cfg.run_dir, f"{cfg.predict}.pt")
    ), f"Run `{cfg.run_id} has no `{cfg.predict}.pt` model saved!"

    logger.debug("Load model from: `{}`", cfg.run_dir)

    model.load_state_dict(
        torch.load(os.path.join(cfg.run_dir, f"{cfg.predict}.pt")), strict=False
    )

    model.eval()

    model.to(accelerator.device)

    logger.info("Start evaluation process")

    index_config = IndexConfig(
        dim=utils.ddp_getattr(model, "embedding_size"),
        index_type=cfg.index_type,
    )

    index_dir = "index" if cfg.predict == "last" else "index_best"
    index = ShardedIndex(
        process_id=accelerator.local_process_index,
        directory=os.path.join(cfg.run_dir, index_dir),
        config=index_config,
    )

    predictions = {}
    with kb as handle:
        for batch in test_dl:
            uids = search_index(
                batch=batch,
                model=model,
                index=index,
                mixed_precision=utils.is_mixed_precision(accelerator),
                topk=cfg.eval_topk,
                debug=False,
            )
            identifiers = map_uids_to_identifiers(uids=uids, kb=handle)
            predictions[batch["eid"]] = set(
                int(i) for i in identifiers[:, 0].flatten().tolist()
            )

        if kb.kb_config.string_identifier:
            identifiers = set(
                i for eid, identifiers in predictions.items() for i in identifiers
            )
            mapping = kb.get_reverse_identifier_mapping(identifiers)
            predictions = {
                eid: {m for i in identifiers for m in mapping[i]}
                for eid, identifiers in predictions.items()
            }
            if kb.kb_config.name == Kbs.CTD_DISEASES.name:
                predictions = {
                    eid: {i for i in identifiers if i.startswith("MESH:")}
                    for eid, identifiers in predictions.items()
                }

    out = {int(eid): list(identifiers) for eid, identifiers in predictions.items()}
    utils.save_json(
        item=out,  # type: ignore
        path=os.path.join(f"./data/biored/aioner_belhd_{entity_type}.json"),
        indent=1,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
