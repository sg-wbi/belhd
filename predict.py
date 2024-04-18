#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate BiEncoder model
"""
import multiprocessing as mp
import os
import time
from datetime import datetime, timedelta
from typing import Optional

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
from dataset import (
    KbCollator,
    KbDataset,
    QueryCollator,
    QuerySplit,
    get_queries,
    get_queries_dataset,
    get_tokenized_kb,
)
from index import IndexConfig, ShardedIndex, search_index
from model import BiEncoderModel, embed_tokenized_kb
from tokenizer import batch_encode_name, load_tokenizer

# search failed somehow and -1 is already taken (pad annotation_ids)
DUMMY_IDENTIFIER = -100


def get_candidates(
    uids: np.ndarray,
    kb: BelbKb,
    tokenizer: PreTrainedTokenizerBase,
    collator: KbCollator,
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
    batch = []
    for row in kb.query(query):
        uid_to_identifier[row["uid"]] = row["identifier"]
        row = utils.convert_to_disambiguated_name(
            row=row, connector=kb.queries.connector
        )
        row["original_name"] = row.pop("name")
        batch.append(row)

    for i, n in enumerate(
        batch_encode_name(
            tokenizer=tokenizer, batch_names=[r["unique_name"] for r in batch]
        )
    ):
        batch[i]["name"] = np.asarray(n)
        batch[i]["uid"] = np.asarray(batch[i]["uid"])[..., np.newaxis]

    candidates = collator.collate_fn(batch)

    candidates["identifiers"] = np.vectorize(
        lambda x: uid_to_identifier.get(x, DUMMY_IDENTIFIER), otypes=["int"]
    )(uids)

    uid_not_inkb = [
        uid for uid in set(uids.flatten().tolist()) if uid not in uid_to_identifier
    ]

    if len(uid_not_inkb) > 0:
        logger.warning(
            "Mismatch between KB and index:  uid `{}` is not in KB",
            uid_not_inkb,
        )

    uid_to_eidx = {uid: i for i, uid in enumerate(candidates["uids"])}
    if train:
        candidates["embd_idxs"] = np.vectorize(uid_to_eidx.get)(uids)
        candidates["eidx_to_identifier"] = {
            i: uid_to_identifier[uid] for uid, i in uid_to_eidx.items()
        }

    return candidates


def build_index(
    epoch: int,
    step: int,
    kb_collator: KbCollator,
    index: ShardedIndex,
    accelerator: Accelerator,
    model: BiEncoderModel,
    tokenized_kb: str,
    index_batch_size: int = 1024,
    shard_size: Optional[int] = None,
    overwrite: bool = False,
    cuda_available: bool = False,
    max_len_name: int = 512,
) -> ShardedIndex:
    """
    Accelerated version of `embed_dictionary`
    Every GPU process a shard of the dictionary
    """

    if (
        os.path.exists(index.directory)
        and len(os.listdir(index.directory))
        and not overwrite
    ):
        if accelerator.is_local_main_process:
            logger.info(
                "EPOCH:{} | Step:{} | Use pre-computed embedings from: `{}`",
                epoch,
                step,
                index.directory,
            )
    else:
        os.makedirs(index.directory, exist_ok=True)

        if accelerator.is_local_main_process:
            logger.info("EPOCH:{} | Step:{} | Start embedding KB", epoch, step)

        model.eval()

        if accelerator.is_local_main_process:
            start = time.time()

        # get all shards
        files = list(
            utils.get_naturally_sorted_files(directory=tokenized_kb, extension="zarr")
        )
        try:
            shards = list(utils.chunkize_list(files, accelerator.num_processes))
        except ValueError as error:
            raise RuntimeError(
                f"KB shards {len(files)}",
                f"cannot be distributed among {accelerator.num_processes} processes!"
                "Try to reduce the # of processes or increase the number of shards...",
            ) from error

        # assign shard to process
        process_shards = shards[accelerator.local_process_index]

        assert len(process_shards) > 0, "Each process must encode at least one KB shard"

        tokenized_kb_dataset = KbDataset(source=process_shards)

        tokenized_kb_dl = DataLoader(
            dataset=tokenized_kb_dataset,
            collate_fn=kb_collator.collate_fn,
            batch_size=index_batch_size,
            shuffle=False,
            pin_memory=cuda_available,
        )

        embed_tokenized_kb(
            process_id=accelerator.local_process_index,
            model=model,
            directory=index.directory,
            dataloader=tokenized_kb_dl,
            mixed_precision=utils.is_mixed_precision(accelerator),
            shard_size=shard_size,
        )

        accelerator.wait_for_everyone()

        if accelerator.is_local_main_process:
            elapsed = datetime(1, 1, 1) + timedelta(seconds=int(time.time() - start))
            logger.info(
                "EPOCH:{} | Step:{} | Embedding KB took: {}h:{}m:{}s",
                epoch,
                step,
                elapsed.hour,
                elapsed.minute,
                elapsed.second,
            )

        index.build()

    return index


@hydra.main(version_base=None, config_path="data/configs", config_name="config")
def main(cfg: DictConfig):
    """
    Run training w/ evaluateuation at the end of each epoch
    """

    assert cfg.predict in [
        "best",
        "last",
    ], f"Unknown checkpoint `{cfg.predict}`: must be either `last` or `best`"

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

    logger.debug("Tokenized corpus: {}", queries["test"]["dir"])

    dictionary_collator = KbCollator.from_tokenizer(tokenizer, max_len=cfg.max_len_name)
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
        os.path.join(cfg.run_dir, f"{cfg.predict}.pt")
    ), f"Run `{cfg.run_id} has no `{cfg.predict}.pt` model saved!"

    logger.debug("Load model from: `{}`", cfg.run_dir)

    model.load_state_dict(
        torch.load(os.path.join(cfg.run_dir, f"{cfg.predict}.pt")), strict=False
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

    if cfg.predict == "best":
        index = build_index(
            epoch=-1,
            step=-1,
            kb_collator=kb_collator,
            index=index,
            accelerator=accelerator,
            model=model,
            tokenized_kb=tokenized_kb,
            shard_size=int(cfg.index_shard_size),
            overwrite=False,
            index_batch_size=cfg.index_batch_size,
            cuda_available=cuda_available,
            max_len_name=cfg.max_len_name,
        )

    predictions = {}

    breakpoint()

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
                collator=dictionary_collator,
                train=False,
            )

            eid = batch["eid"]
            annotation_ids = batch["annotation_ids"].flatten().tolist()
            identifiers = candidates["identifiers"]

            for idx, aid in enumerate(annotation_ids):
                predictions[f"{eid}.{aid}"] = [[i] for i in identifiers[idx].tolist()]

            metrics(
                y_pred=candidates["identifiers"], y_true=batch["annotation_identifiers"]
            )

        epoch_metrics = {
            k: round(v.detach().float().item(), 5) for k, v in metrics.compute().items()
        }

        logger.info("Test: {} | Metrics: {}", cfg.test, epoch_metrics)

    aid_to_hexdigest = utils.load_json(
        os.path.join(queries["test"]["dir"], "annotation_id_to_hexdigest.json")
    )

    query_split = QuerySplit.from_string(string=cfg.test)
    name = "belhd"
    if cfg.abbres:
        name += "_ar"
    if cfg.project < 0:
        name += "_noph"
    if not cfg.global_candidates:
        name += "_nocs"
    if cfg.exclude_context:
        name += "_noctx"
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
