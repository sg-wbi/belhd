#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train BiEncoder model with AMBIGUOUS KB
"""
import multiprocessing as mp
import os
import pickle
import random
from typing import Optional, Union

import faiss
import hydra
import numpy as np
import torch
from accelerate import Accelerator
from belb import AutoBelbKb, BelbKb
from belb.kbs import ENTITY_TO_KB_NAME
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from transformers import BertModel, LongformerModel, PreTrainedTokenizerBase

import utils
from dataset import (KbCollator, QueryCollator, get_queries,
                     get_queries_dataset, get_tokenized_kb)
from index import IndexConfig, ShardedIndex, search_index
from model import BiEncoderModel
from predict import build_index
from predict_nohd import get_candidates
from tokenizer import load_tokenizer

# NOTE: can't use longformer and bf16
# RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'
# from accelerate.utils import is_bf16_available


# search failed somehow and -1 is already taken (pad annotation_ids)
DUMMY_IDENTIFIER = -100


class AblationBiEncoderModel(BiEncoderModel):
    def _get_labels(
        self, y_pred: Union[np.ndarray, list], y_true: Union[np.ndarray, list]
    ):
        labels = []
        for i, mention_ypred in enumerate(y_pred):
            ml = [
                int(any(y in y_true[i] for y in candidates))
                for candidates in mention_ypred
            ]
            labels.append(ml)

        return torch.as_tensor(labels, device=self.device)

    def _add_global_candidates(
        self,
        candidates: dict,
        candidates_embd: torch.Tensor,
        max_candidates: Optional[int] = None,
    ) -> dict:
        """
        Get extra positive/negative candidates
        by looking at what other mentions have retrieved.
        """

        global_indices = self._get_global_indices(
            candidates_embd=candidates_embd,
            max_candidates=max_candidates,
            embedding_indices=candidates["embd_idxs"],
        )

        if global_indices.size != 0:
            candidates["global_embd_idxs"] = global_indices

            global_identifiers = [
                [candidates["eidx_to_identifiers"][i] for i in mgi]
                for mgi in global_indices
            ]

            candidates["global_identifiers"] = global_identifiers

        return candidates


def validation(
    epoch: int,
    step: int,
    monitor: str,
    logs: dict,
    run_dir: str,
    tracker: utils.RunTracker,
    accelerator: Accelerator,
    tokenizer: PreTrainedTokenizerBase,
    model: BiEncoderModel,
    dev_dl: DataLoader,
    index: ShardedIndex,
    kb: BelbKb,
    kb_collator: KbCollator,
    metrics: MetricCollection,
    uid_to_identifier: dict,
    topk: int = 64,
    debug: bool = False,
) -> dict:
    if accelerator.is_local_main_process:
        logger.info("EPOCH:{} | Step:{} | Start validation", epoch, step)

    model.eval()
    for batch in dev_dl:
        uids = search_index(
            batch=batch,
            model=model,
            index=index,
            mixed_precision=utils.is_mixed_precision(accelerator),
            topk=topk,
            debug=debug,
        )

        candidates = get_candidates(
            uids=uids,
            kb=kb,
            tokenizer=tokenizer,
            collator=kb_collator,
            uid_to_identifier=uid_to_identifier,
        )

        # something went wrong with search...
        if candidates.get("identifiers") is None:
            logger.warning("EID:{} | Search failed: skip batch", batch["eid"])

        y_pred = np.asarray(
            [[random.choice(y) for y in yp] for yp in candidates["identifiers"]]
        )
        y_true = batch["annotation_identifiers"]
        metrics(y_pred=y_pred, y_true=y_true)

    val_metrics = {
        k: round(v.detach().float().item(), 5) for k, v in metrics.compute().items()
    }

    if accelerator.is_local_main_process:
        logger.info(
            "EPOCH:{} | Step:{} | Eval metrics: {}",
            epoch,
            step,
            val_metrics,
        )

    if val_metrics[monitor] > logs["best"]:
        if accelerator.is_local_main_process:
            logger.info("EPOCH:{} | Step: {} | Save best model", epoch, step)
            tracker.track({f"best/{k}": v for k, v in val_metrics.items()})

        accelerator.save(
            accelerator.unwrap_model(model).state_dict(),
            os.path.join(run_dir, "best.pt"),
        )

        logs["best"] = val_metrics[monitor]

    metrics.reset()

    return val_metrics


@hydra.main(version_base=None, config_path="data/configs", config_name="config_nohd")
def main(cfg: DictConfig):
    """
    Run training w/ evaluateuation at the end of each epoch
    """

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

    # https://github.com/facebookresearch/faiss/issues/53#issuecomment-288351188
    # multi-thrading is SIGNIFICANTLY slower for batched queries
    # In [14]: faiss.omp_set_num_threads(1)
    # In [15]: t0 = time.time(); index.search(X[:20], 20); print time.time() - t0
    # 0.331252098083
    # In [22]: faiss.omp_set_num_threads(40)
    # In [23]: t0 = time.time(); index.search(X[:20], 20); print time.time() - t0
    # 5.00787210464
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

    if accelerator.is_local_main_process:
        logger.debug("Tokenized corpus: {}", queries["train"]["dir"])

    tracker.track({"corpus": queries})

    query_collator = QueryCollator.from_tokenizer(tokenizer)
    queries_dataset = get_queries_dataset(
        queries=queries,
        train_on_dev=cfg.train_on_dev,
        train_with_dev=cfg.train_with_dev,
        load_test=False,
    )
    train_dl = DataLoader(
        dataset=queries_dataset["train"],
        shuffle=True,
        batch_size=1,  # use gradient accumulation for bigger batch size
        pin_memory=cuda_available,
        collate_fn=query_collator.collate_fn,
    )
    dev_dl = DataLoader(
        dataset=queries_dataset["dev"],
        shuffle=False,
        batch_size=1,
        pin_memory=cuda_available,
        collate_fn=query_collator.collate_fn,
    )

    tokenized_kb = get_tokenized_kb(
        directory=cfg.kbs_dir,
        tokenizers_dir=cfg.tokenizers_dir,
        tokenizer_name=tokenizer._codename,
        split=cfg.train,
        subset=cfg.kb_subset,
    )

    with open(os.path.join(tokenized_kb, "uid_to_identifiers.pkl"), "rb") as fp:
        uid_to_identifier = pickle.load(fp)

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

    model = AblationBiEncoderModel(
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
        model.load_state_dict(torch.load(os.path.join(cfg.run_dir, "last.pt")))

    model.to(accelerator.device)

    optimizer = utils.get_optimizer(
        model=model, lr=cfg.lr, epsilon=cfg.epsilon, weight_decay=cfg.weight_decay
    )

    lr_scheduler = utils.get_lr_scheduler(
        lr_scheduler=cfg.lr_scheduler,
        optimizer=optimizer,
        warmup_steps=cfg.warmup_steps,
        num_training_steps=len(queries_dataset["train"])
        * cfg.max_epochs
        // cfg.gradient_accumulation_steps,
    )

    train_dl, dev_dl, model, optimizer, lr_scheduler = accelerator.prepare(
        train_dl, dev_dl, model, optimizer, lr_scheduler
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

    belb_kb = AutoBelbKb.from_name(
        directory=cfg.belb_dir,
        name=ENTITY_TO_KB_NAME[queries["train"]["entity_type"]],
        db_config=os.path.join(os.getcwd(), "data", "configs", "db.yaml"),
        debug=False,
    )

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

    monitor = "r@1"
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
                    uid_to_identifier=uid_to_identifier,
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
                        step=idx,
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

                    validation(
                        epoch=epoch,
                        uid_to_identifier=uid_to_identifier,
                        accelerator=accelerator,
                        tokenizer=tokenizer,
                        step=idx,
                        model=model,
                        dev_dl=dev_dl,
                        index=index,
                        kb=handle,
                        kb_collator=kb_collator,
                        metrics=metrics,
                        topk=cfg.eval_topk,
                        logs=logs,
                        monitor=monitor,
                        run_dir=cfg.run_dir,
                        tracker=tracker,
                    )

            if accelerator.is_local_main_process:
                logger.info(
                    "EPOCH:{} | Step:{} | Average loss: {} | Save model",
                    epoch,
                    idx,
                    round(float(logs["total_loss"]) / len(train_dl), 4),
                )
            accelerator.save(
                accelerator.unwrap_model(model).state_dict(),
                os.path.join(cfg.run_dir, "last.pt"),
            )

            if not cfg.query_side_ft:
                index = build_index(
                    epoch=epoch,
                    kb_collator=kb_collator,
                    step=-1,
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

            val_metrics = validation(
                epoch=epoch,
                uid_to_identifier=uid_to_identifier,
                accelerator=accelerator,
                tokenizer=tokenizer,
                step=-1,
                model=model,
                dev_dl=dev_dl,
                index=index,
                kb=handle,
                kb_collator=kb_collator,
                metrics=metrics,
                topk=cfg.eval_topk,
                logs=logs,
                monitor=monitor,
                run_dir=cfg.run_dir,
                tracker=tracker,
            )

    if accelerator.is_local_main_process:
        logger.info("Training completed")
        tracker.track({f"last/{k}": v for k, v in val_metrics.items()})
        tracker.end_training()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
