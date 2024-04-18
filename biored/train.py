#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train BiEncoder model
"""
import multiprocessing as mp
import os

import faiss
import hydra
import torch
from accelerate import Accelerator

# from accelerate import DistributedDataParallelKwargs
from belb import ENTITY_TO_KB_NAME, AutoBelbKb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from transformers import BertModel, LongformerModel

import utils
from dataset import (
    KbCollator,
    QueryCollator,
    QueryDataset,
)
from index import IndexConfig, ShardedIndex, search_index
from model import BiEncoderModel
from predict import build_index, get_candidates
from tokenizer import load_tokenizer

# NOTE: can't use longformer and bf16
# RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'
# from accelerate.utils import is_bf16_available


@hydra.main(version_base=None, config_path="../data/configs", config_name="config")
def main(cfg: DictConfig):
    """
    Run training w/ evaluateuation at the end of each epoch
    """

    # path to tokenized BELB corpus (see tokenize_corpora.py)
    QUERIES_PATH = ""
    # path to tokenized BELB kb (see tokenize_dkbs.py)
    KB_PATH = ""
    # entity type of corpus
    ENTITY_TYPE = "chemical"

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cfg.mixed_precision = "fp16"
        # cfg.mixed_precision = "bf16" if is_bf16_available() else "fp16"

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        device_placement=False,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        # kwargs_handlers=[ddp_kwargs]
        # log_with=LoggerType.TENSORBOARD,
    )

    DEBUG = accelerator.is_local_main_process and cfg.get("debug", False)

    cores = min(30, mp.cpu_count())

    utils.init_run(
        seed=cfg.seed,
        cores=cores,
        cuda_available=cuda_available,
    )

    faiss_cores = 1 if cfg.index_type == "flat" else max(20, cores)
    faiss.omp_set_num_threads(faiss_cores)
    if DEBUG:
        logger.debug("faiss: OMP threads: {}", faiss_cores)

    with accelerator.main_process_first():
        hps = OmegaConf.load(os.path.join(os.getcwd(), "data", "configs", "train.yaml"))
        tracker = utils.RunTracker(
            directory=cfg.exp_dir,
            cfg={k: v for k, v in cfg.items() if k in hps},
            run_id=cfg.run_id,
        )
        cfg.run_dir = os.path.join(cfg.exp_dir, "runs", str(tracker.run_id))
        os.makedirs(cfg.run_dir, exist_ok=True)

    logger.add(os.path.join(cfg.run_dir, "train.log"))

    if accelerator.is_local_main_process:
        logger.info("Run: `{}`", cfg.run_dir)
        OmegaConf.save(cfg, os.path.join(cfg.run_dir, "train.yaml"))

    tokenizer = load_tokenizer(directory=cfg.tokenizers_dir, name_or_path=cfg.lm)

    if accelerator.is_local_main_process:
        logger.debug("Tokenized corpus: {}", QUERIES_PATH)

    query_collator = QueryCollator.from_tokenizer(tokenizer)
    queries_dataset = QueryDataset(
        path=os.path.join(QUERIES_PATH, tokenizer._codename, "train.zarr")
    )
    train_dl = DataLoader(
        dataset=queries_dataset,
        shuffle=True,
        batch_size=1,  # use gradient accumulation for bigger batch size
        pin_memory=cuda_available,
        collate_fn=query_collator.collate_fn,
    )

    tokenized_kb = os.path.join(KB_PATH, tokenizer._codename)

    if accelerator.is_local_main_process:
        logger.debug("Tokenized KB: {}", tokenized_kb)

    kb_collator = KbCollator.from_tokenizer(tokenizer, max_len=cfg.max_len_name)
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

    if cfg.load_last_model:
        if accelerator.is_local_main_process:
            logger.info(
                "Loading model weights from: {}", os.path.join(cfg.run_dir, "last.pt")
            )
        model.load_state_dict(
            torch.load(os.path.join(cfg.run_dir, "last.pt")), strict=False
        )

    model.to(accelerator.device)

    optimizer = utils.get_optimizer(
        model=model, lr=cfg.lr, epsilon=cfg.epsilon, weight_decay=cfg.weight_decay
    )

    lr_scheduler = utils.get_lr_scheduler(
        lr_scheduler=cfg.lr_scheduler,
        optimizer=optimizer,
        warmup_steps=cfg.warmup_steps,
        num_training_steps=len(queries_dataset)
        * cfg.max_epochs
        // cfg.gradient_accumulation_steps,
    )

    belb_kb = AutoBelbKb.from_name(
        directory=cfg.belb_dir,
        name=ENTITY_TO_KB_NAME[ENTITY_TYPE],
        db_config=os.path.join(os.getcwd(), "data", "configs", "db.yaml"),
        debug=False,
    )

    train_dl, model, optimizer, lr_scheduler = accelerator.prepare(
        train_dl, model, optimizer, lr_scheduler
    )

    metrics = MetricCollection(
        {
            "r@1": utils.MultiLabelRecall(k=1),
            "r@10": utils.MultiLabelRecall(k=10),
            f"r@{cfg.eval_topk}": utils.MultiLabelRecall(k=cfg.eval_topk),
        }
    )

    metrics.to(accelerator.device)

    if accelerator.is_local_main_process:
        logger.info("Start training process")

    index_config = IndexConfig(
        dim=utils.ddp_getattr(model, "embedding_size"),
        index_type=cfg.index_type,
    )

    index = ShardedIndex(
        process_id=accelerator.local_process_index,
        directory=os.path.join(cfg.run_dir, "index"),
        config=index_config,
    )

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

    logs = {"best": 0}
    with belb_kb as handle:
        for epoch in range(cfg.max_epochs):
            if accelerator.is_local_main_process:
                logger.info("EPOCH:{} | Start train", epoch)

            logs["total_loss"] = 0
            for idx, batch in enumerate(train_dl):
                model.eval()

                uids = search_index(
                    batch=batch,
                    model=model,
                    index=index,
                    mixed_precision=utils.is_mixed_precision(accelerator),
                    topk=cfg.train_topk,
                    debug=DEBUG,
                )

                model.train()

                candidates = get_candidates(
                    uids=uids,
                    kb=handle,
                    tokenizer=tokenizer,
                    collator=kb_collator,
                )

                # something went wrong with search...
                if candidates.get("identifiers") is None:
                    logger.warning("EID:{} | Search failed: skip batch", batch["eid"])
                    continue

                with accelerator.accumulate(model):
                    loss = model(forward="loss", queries=batch, candidates=candidates)
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    # gradient_norms = get_gradients_norm(model)
                    optimizer.zero_grad()
                    logs["total_loss"] += loss.detach().float()

                if (idx + 1) % 100 == 0 and accelerator.is_local_main_process:
                    logger.info(
                        "EPOCH:{} | Step {} | average loss: {}",
                        epoch,
                        idx + 1,
                        round(float(logs["total_loss"]) / idx, 4),
                    )

                if (
                    cfg.refresh_index_every_n is not None
                    and ((idx + 1) % cfg.refresh_index_every_n) == 0
                    and not cfg.query_side_ft
                ):
                    index = build_index(
                        epoch=epoch,
                        step=idx + 1,
                        kb_collator=kb_collator,
                        index=index,
                        accelerator=accelerator,
                        model=model,
                        tokenized_kb=tokenized_kb,
                        shard_size=int(cfg.index_shard_size),
                        overwrite=True,
                        index_batch_size=cfg.index_batch_size,
                        cuda_available=cuda_available,
                        max_len_name=cfg.max_len_name,
                    )

            if accelerator.is_local_main_process:
                logger.info(
                    "EPOCH:{} | Step:{} | Average loss: {} | Save model",
                    epoch,
                    idx + 1,
                    round(float(logs["total_loss"]) / len(train_dl), 4),
                )
            accelerator.save(
                accelerator.unwrap_model(model).state_dict(),
                os.path.join(cfg.run_dir, "last.pt"),
            )

            if not cfg.query_side_ft:
                index = build_index(
                    epoch=epoch,
                    step=-1,
                    kb_collator=kb_collator,
                    index=index,
                    accelerator=accelerator,
                    model=model,
                    tokenized_kb=tokenized_kb,
                    shard_size=int(cfg.index_shard_size),
                    overwrite=True,
                    index_batch_size=cfg.index_batch_size,
                    cuda_available=cuda_available,
                    max_len_name=cfg.max_len_name,
                )

    if accelerator.is_local_main_process:
        logger.info("Training completed")
        tracker.end_training()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
