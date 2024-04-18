#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate BiEncoder model
"""
import multiprocessing as mp
import os
import pickle
import random

import faiss
import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import is_bf16_available
from belb import BelbKb, Tables
from belb.kbs import ENTITY_TO_KB_NAME, AutoBelbKb
from loguru import logger
from omegaconf import DictConfig
from sqlalchemy import select
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from transformers import BertModel, LongformerModel, PreTrainedTokenizerBase

import utils
from dataset import (KbCollator, QueryCollator, QuerySplit, get_queries,
                     get_queries_dataset, get_tokenized_kb)
from index import IndexConfig, ShardedIndex, search_index
from model import BiEncoderModel
from tokenizer import batch_encode_name, load_tokenizer

# from train import build_index
# from utils import RunTracker

# search failed somehow and -1 is already taken (pad annotation_ids)
DUMMY_IDENTIFIER = -100


def get_candidates(
    uids: np.ndarray,
    kb: BelbKb,
    tokenizer: PreTrainedTokenizerBase,
    collator: KbCollator,
    uid_to_identifier: dict,
):
    batch_uids = sorted(set(int(i) for i in uids.flatten()))
    table = kb.schema.get(Tables.KB)

    query = select(table.c.uid, table.c.name).where(table.c.uid.in_(batch_uids))

    batch = list(kb.query(query))

    for i, n in enumerate(
        batch_encode_name(tokenizer=tokenizer, batch_names=[r["name"] for r in batch])
    ):
        batch[i]["name"] = np.asarray(n)
        batch[i]["uid"] = np.asarray(batch[i]["uid"])[..., np.newaxis]

    candidates = collator.collate_fn(batch)

    candidates["identifiers"] = [
        [uid_to_identifier.get(uid, DUMMY_IDENTIFIER) for uid in cuids]
        for cuids in uids
    ]

    uid_not_inkb = [
        uid for uid in set(uids.flatten().tolist()) if uid not in uid_to_identifier
    ]

    if len(uid_not_inkb) > 0:
        logger.warning(
            "Mismatch between KB and index:"
            + "retrieval returned uid `{}` which is not in KB",
            uid_not_inkb,
        )

    uid_to_eidx = {uid: i for i, uid in enumerate(candidates["uids"])}
    candidates["embd_idxs"] = np.vectorize(uid_to_eidx.get)(uids)
    candidates["eidx_to_identifiers"] = {
        i: uid_to_identifier[uid] for uid, i in uid_to_eidx.items()
    }

    return candidates


@hydra.main(version_base=None, config_path="data/configs", config_name="config_nohd")
def main(cfg: DictConfig):
    """
    Run training w/ evaluateuation at the end of each epoch
    """

    assert cfg.test is not None, "You must specify `test` split!"

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

    # hps = OmegaConf.load(os.path.join(os.getcwd(), "data", "configs", "train.yaml"))
    # tracker = RunTracker(
    #     directory=cfg.exp_dir,
    #     cfg={k: v for k, v in cfg.items() if k in hps},
    #     run_id=cfg.run_id,
    # )
    assert cfg.run_id is not None, "You must specify `run_id`!"
    cfg.run_dir = os.path.join(cfg.exp_dir, "runs", str(cfg.run_id))

    assert os.path.exists(cfg.run_dir), f"Run `{cfg.run_id}` does not exists!"

    logger.info("Run: `{}`", cfg.run_dir)

    tokenizer = load_tokenizer(directory=cfg.tokenizers_dir, name_or_path=cfg.lm)

    queries = get_queries(
        directory=cfg.corpora_dir,
        tokenizers_dir=cfg.tokenizers_dir,
        tokenizer_name=tokenizer._codename,
        train=cfg.train,
        dev=cfg.dev,
        test=cfg.test,
        mention_markers=cfg.mention_markers,
        sentences=cfg.sentences,
        add_foreign_annotations=cfg.add_foreign_annotations or cfg.foreign_attention,
        max_mentions=cfg.max_mentions,
        abbres=cfg.abbres,
    )

    query_collator = QueryCollator.from_tokenizer(tokenizer)
    queries_dataset = get_queries_dataset(queries=queries, load_test=True)

    test_dl = DataLoader(
        dataset=queries_dataset["test"],
        shuffle=False,
        # one abstract at a time: use gradient accumulation for bigger batch size
        batch_size=1,
        pin_memory=cuda_available,
        collate_fn=query_collator.collate_fn,
    )

    kb = AutoBelbKb.from_name(
        directory=cfg.belb_dir,
        name=ENTITY_TO_KB_NAME[queries["test"]["entity_type"]],
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
        os.path.join(cfg.run_dir, "last.pt")
    ), f"Run `{cfg.run_id} has no model saved!"

    # On some machine CUDA is too old so
    # we need to use older version of transformers
    # with 'position_ids' stored as parameters
    # Missing key(s) in state_dict: "query_encoder.embeddings.position_ids"

    logger.debug("Load model from: `{}`", cfg.run_dir)
    model.load_state_dict(
        torch.load(os.path.join(cfg.run_dir, "best.pt")), strict=False
    )

    model.eval()

    model.to(accelerator.device)

    metrics = MetricCollection(
        {
            "r@1": utils.MultiLabelRecall(k=1),
            "r@10": utils.MultiLabelRecall(k=10),
            f"r@{cfg.eval_topk}": utils.MultiLabelRecall(k=cfg.eval_topk),
        }
    )

    metrics.to(accelerator.device)

    index_config = IndexConfig(
        dim=utils.ddp_getattr(model, "embedding_size"),
        index_type=cfg.index_type,
    )

    index = ShardedIndex(
        process_id=accelerator.local_process_index,
        directory=os.path.join(cfg.run_dir, "index"),
        config=index_config,
    )

    kb_collator = KbCollator.from_tokenizer(tokenizer, max_len=cfg.max_len_name)

    tokenized_kb = get_tokenized_kb(
        directory=cfg.kbs_dir,
        tokenizers_dir=cfg.tokenizers_dir,
        tokenizer_name=tokenizer._codename,
        split=cfg.test,
        subset=cfg.kb_subset,
    )

    if accelerator.is_local_main_process:
        logger.debug("Tokenized KB: {}", tokenized_kb)

    with open(os.path.join(tokenized_kb, "uid_to_identifiers.pkl"), "rb") as fp:
        uid_to_identifier = pickle.load(fp)

    # index = build_index(
    #     epoch=-1,
    #     step=-1,
    #     kb_collator=kb_collator,
    #     index=index,
    #     accelerator=accelerator,
    #     model=model,
    #     tokenized_kb=tokenized_kb,
    #     shard_size=int(cfg.index_shard_size),
    #     overwrite=False,
    #     index_batch_size=cfg.index_batch_size,
    #     cuda_available=cuda_available,
    #     max_len_name=cfg.max_len_name,
    # )

    logger.info("Start evaluation process")

    predictions = {}

    with kb as handle:
        for batch in test_dl:
            uids = search_index(
                batch=batch,
                model=model,
                index=index,
                mixed_precision=utils.is_mixed_precision(accelerator),
                topk=cfg.eval_topk,
                debug=True,
            )

            candidates = get_candidates(
                uids=uids,
                kb=handle,
                tokenizer=tokenizer,
                collator=kb_collator,
                uid_to_identifier=uid_to_identifier,
            )

            eid = batch["eid"]
            annotation_ids = batch["annotation_ids"].flatten().tolist()
            identifiers = candidates["identifiers"]

            for i, aid in enumerate(annotation_ids):
                predictions[f"{eid}.{aid}"] = identifiers[i]

            metrics(
                y_pred=np.asarray(
                    [[random.choice(y) for y in yp] for yp in candidates["identifiers"]]
                ),
                y_true=batch["annotation_identifiers"],
            )

        epoch_metrics = {
            k: round(v.detach().float().item(), 5) for k, v in metrics.compute().items()
        }

        logger.info("Test: {} | Metrics: {}", cfg.test, epoch_metrics)

    aid_to_hexdigest = utils.load_json(
        os.path.join(queries["test"]["dir"], "annotation_id_to_hexdigest.json")
    )

    query_split = QuerySplit.from_string(string=cfg.test)
    name = "belhd_nohd"
    if cfg.abbres:
        name += "_ar"
    parts = [cfg.results_dir, query_split.corpus, name]
    out_dir = os.path.join(*parts)
    os.makedirs(out_dir, exist_ok=True)

    item = [
        {"hexdigest": aid_to_hexdigest[aid], "y_pred": y_pred}
        for aid, y_pred in predictions.items()
    ]

    utils.save_json(
        item=item,  # type: ignore
        path=os.path.join(out_dir, "predictions.json"),
        indent=1,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
