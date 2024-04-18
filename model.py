#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bi-encoder model
"""

import copy
import os
from collections import OrderedDict, defaultdict
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, LongformerModel, PreTrainedTokenizer

from storage import EMBEDDING_GROUP, ZarrWriter
from tokenizer import SPECIAL_TOKENS
from utils import ddp_getattr, row_wise_isin


def mean_pooling(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """
    Mean pooling
    """
    return torch.stack(tensors, dim=dim).mean(dim)


def marginal_nll(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Synonym marginalization: sum all scores among positive candidates
    """

    scores = torch.softmax(scores, dim=-1)
    losses = scores * labels
    losses = losses.sum(dim=-1)  # sum all positive scores
    losses = losses[losses > 0]  # filter sets with at least one positives
    losses = torch.clamp(losses, min=1e-9, max=1)  # for numerical stability
    losses = -torch.log(losses)  # for negative log likelihood
    loss = losses.sum() if len(losses) == 0 else losses.mean()

    return loss


class Project(torch.nn.Module):
    """
    Project embeddings to lower dimension
    """

    def __init__(self, d_model: int, d_proj: int):
        super().__init__()
        self.d_model = d_model
        self.d_proj = d_proj
        # TODO: use 2 projection layers?
        self.prj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_proj),
            torch.nn.LayerNorm(d_proj),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Project
        """

        return self.prj(inputs)


class CombineConcat(torch.nn.Module):
    """
    Merge embeddings
    """

    def __init__(self, embedding_size: int):
        super().__init__()
        self.prj = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_size * 2),
            torch.nn.Linear(embedding_size * 2, embedding_size),
            torch.nn.LayerNorm(embedding_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Project
        """

        return self.prj(inputs)


class BiEncoderModel(torch.nn.Module):
    """
    BiEncoderModel
    """

    def __init__(
        self,
        device: torch.device,
        tokenizer: PreTrainedTokenizer,
        encoder: Union[BertModel, LongformerModel],
        project: int = -1,
        query_side_ft: bool = False,
        global_candidates: bool = False,
        max_global_candidates: Optional[int] = None,
        foreign_attention: bool = False,
        exclude_context: bool = False,
    ):
        super().__init__()

        self.device = device
        self.tokenizer = tokenizer
        self.exclude_context = exclude_context

        encoder.resize_token_embeddings(len(self.tokenizer))
        self.query_encoder = encoder

        # Query-side finetuning
        # https://github.com/facebookresearch/atlas#strategies-for-dealing-with-stale-indices
        if query_side_ft:
            self.candidate_encoder = copy.deepcopy(encoder)
            self.candidate_encoder.eval()
            self.candidate_encoder.requires_grad_(False)
        else:
            self.candidate_encoder = encoder

        self.project_mentions: Optional[Project] = (
            Project(d_model=self.query_encoder.config.hidden_size, d_proj=project)
            if project > 0 and not query_side_ft
            else None
        )
        self.project_candidates = self.project_mentions

        self.global_candidates = global_candidates
        self.max_global_candidates = max_global_candidates
        self.foreign_attention = foreign_attention

        self.ms_token_id = self.tokenizer.vocab[SPECIAL_TOKENS["ms"]]
        self.me_token_id = self.tokenizer.vocab[SPECIAL_TOKENS["me"]]
        # self.fs_token_id = self.tokenizer.vocab[SPECIAL_TOKENS["fs"]]
        # self.fe_token_id = self.tokenizer.vocab[SPECIAL_TOKENS["fe"]]

    @property
    def embedding_size(self) -> int:
        """
        Final embedding size
        """

        size = (
            self.project_mentions.d_proj
            if self.project_mentions is not None
            else self.query_encoder.config.hidden_size
        )

        return size

    def _get_global_indices(
        self,
        candidates_embd: torch.Tensor,
        embedding_indices: np.ndarray,
        max_candidates: Optional[int] = None,
    ) -> np.ndarray:
        # how similar are candidates among them
        _, indices = torch.sort(candidates_embd @ candidates_embd.T, descending=True)
        similarity_matrix = indices.detach().cpu().numpy()

        # batch_local_candidates = candidates["embd_idxs"]
        batch_local_candidates = embedding_indices

        batch_global_candidates: list = []

        # local candidites (mention-specific) for each mention
        for mention_lc in batch_local_candidates:
            ranked_mention_gc_list = []
            # for every mention-specific candidate
            for lc in mention_lc:
                ranked_mention_gc_list.append(
                    # get the its most similar candidates
                    # s.t. they are not in the mention-specifc set
                    similarity_matrix[lc][~np.isin(similarity_matrix[lc], mention_lc)]
                )

            # NumCandidates x (Unique(BatchCandidates) - NumCandidates)
            # The i-th column contains the i-th most similar global candidate
            # of each mention-specific candidate
            ranked_mention_gc = np.vstack(ranked_mention_gc_list)

            # create a ranked list of global candidates
            # with the most similar of each mention-specific candidates at the beginning
            mention_gc = OrderedDict()
            for i in range(ranked_mention_gc.shape[1]):
                for gc in ranked_mention_gc[:, i]:
                    if gc not in mention_gc:
                        mention_gc[gc] = True
            batch_global_candidates.append(list(mention_gc.keys()))

        if max_candidates is not None:
            batch_global_candidates = [
                mgc[:max_candidates] for mgc in batch_global_candidates
            ]

        maxlen = min(len(mgc) for mgc in batch_global_candidates)
        global_indices = np.vstack([mgc[:maxlen] for mgc in batch_global_candidates])

        return global_indices

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
            global_identifiers = np.vectorize(candidates["eidx_to_identifier"].get)(
                global_indices
            )
            candidates["global_embd_idxs"] = global_indices
            candidates["global_identifiers"] = global_identifiers

        return candidates

    def forward(
        self, forward: str, **kwargs
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Call function according to name
        """

        do = f"forward_{forward}"
        if hasattr(self, do) and callable(func := getattr(self, do)):
            outputs = func(**kwargs)
        else:
            raise ValueError(f"{self.__class__.__name__} has not method `{do}`")

        return outputs

    def extract_mentions(self, input_ids: np.ndarray) -> list:
        rows_start, starts = np.where(input_ids == self.ms_token_id)
        rows_end, ends = np.where(input_ids == self.me_token_id)

        assert rows_start.tolist() == rows_end.tolist(), "Rows: # of [MS] != # of [ME]"
        assert len(starts) == len(ends), "# of [MS] != # of [ME]"

        mentions = []
        for i, j, k in zip(rows_start, starts, ends):
            mention = [self.tokenizer.cls_token_id]
            mention += input_ids[i, j + 1 : k].tolist()
            mention += [self.tokenizer.sep_token_id]
            mentions.append(mention)

        return mentions

    def forward_queries(self, queries: dict) -> dict[str, torch.Tensor]:
        """
        Compute embeddings of mentions
        """

        out: dict = {}

        if self.exclude_context:
            mentions = self.extract_mentions(queries["input_ids"])
            kwargs = {
                k: torch.as_tensor(v, device=self.device)
                for k, v in self.tokenizer.pad({"input_ids": mentions}).items()
            }
            tokens_embeddings = self.query_encoder(**kwargs)[0]
            mentions_embd = tokens_embeddings[:, 0, :]
            out["mentions"] = mentions_embd
        else:
            input_ids = torch.as_tensor(queries["input_ids"], device=self.device)
            attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id).int()
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}

            if isinstance(self.query_encoder, LongformerModel):
                gam = torch.zeros_like(attention_mask, device=self.device)
                gam += torch.eq(input_ids, self.tokenizer.cls_token_id).int()
                gam += torch.eq(input_ids, self.ms_token_id).int()
                gam += torch.eq(input_ids, self.me_token_id).int()
                kwargs["global_attention_mask"] = gam

            tokens_embeddings = self.query_encoder(**kwargs)[0]

            ms = tokens_embeddings[input_ids == self.ms_token_id]
            me = tokens_embeddings[input_ids == self.me_token_id]
            mentions_embd = mean_pooling([ms, me])
            out["mentions"] = mentions_embd

        if self.project_mentions is not None:
            out["mentions"] = self.project_mentions(mentions_embd)

        return out

    def forward_cls(
        self, input_ids: np.ndarray, encoder: str = "query"
    ) -> torch.Tensor:
        """Forward and extract CLS embedding"""

        choices = ("query", "candidate")
        assert encoder in choices, f"`encoder` must be one of {choices}"

        input_ids = torch.as_tensor(input_ids, device=self.device).int()
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id).int()
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}

        if encoder == "candidate":
            tokens_embeddings = self.candidate_encoder(**kwargs)[0]
        else:
            tokens_embeddings = self.query_encoder(**kwargs)[0]

        embd = tokens_embeddings[:, 0, :]

        if encoder == "candidate":
            if self.project_candidates is not None:
                embd = self.project_candidates(embd)
        else:
            if self.project_mentions is not None:
                embd = self.project_mentions(embd)

        return embd

    def forward_candidates(self, candidates: dict) -> torch.Tensor:
        """
        Compute embeddings of candidates (dictionary names)
        """

        input_ids = torch.as_tensor(candidates["input_ids"], device=self.device).int()
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id).int()
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}

        if isinstance(self.candidate_encoder, LongformerModel):
            gam = torch.ne(input_ids, self.tokenizer.pad_token_id).int()
            kwargs["global_attention_mask"] = gam

        tokens_embeddings = self.candidate_encoder(**kwargs)[0]

        cands_embd = tokens_embeddings[:, 0, :]

        if self.project_candidates is not None:
            cands_embd = self.project_candidates(cands_embd)

        return cands_embd

    def forward_alignment(
        self, queries_embd: dict, candidates_embd: torch.Tensor
    ) -> dict:
        """
        Compute similarity scores between mentions and candidates
        """

        out: dict = {}

        scores = queries_embd["mentions"] @ candidates_embd.T

        out["scores"] = scores

        return out

    def _get_scores(
        self, alignment_scores: torch.Tensor, embeddings_indices: np.ndarray
    ):
        embeddings_indices = torch.as_tensor(embeddings_indices, device=self.device)
        scores = torch.gather(alignment_scores, 1, embeddings_indices)

        return scores

    def _get_labels(
        self, y_pred: Union[np.ndarray, list], y_true: Union[np.ndarray, list]
    ):
        labels = torch.as_tensor(
            row_wise_isin(src=y_pred, trg=y_true).astype(int),
            device=self.device,
        )

        return labels

    def forward_all(
        self,
        queries: dict,
        candidates: dict,
    ) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
        """
        Compute loss
        """

        queries_embd = self.forward_queries(queries)
        candidates_embd = self.forward_candidates(candidates)
        alignment_scores = self.forward_alignment(
            queries_embd=queries_embd, candidates_embd=candidates_embd
        )

        #######################################
        # LOCAL (MENTION-SPECIFIC) CANDIDATES
        #######################################
        scores = self._get_scores(
            alignment_scores=alignment_scores["scores"],
            embeddings_indices=candidates["embd_idxs"],
        )
        try:
            labels = self._get_labels(
                y_pred=candidates["identifiers"],
                y_true=queries["annotation_identifiers"],
            )
        except IndexError:
            breakpoint()

        #######################################
        # GLOBAL (ALL MENTIONS) CANDIDATES
        #######################################
        if self.global_candidates:
            candidates = self._add_global_candidates(
                candidates=candidates,
                candidates_embd=candidates_embd,
                max_candidates=self.max_global_candidates,
            )
            # it can happen that there is only one mention
            if candidates.get("global_embd_idxs") is not None:
                global_scores = self._get_scores(
                    alignment_scores=alignment_scores["scores"],
                    embeddings_indices=candidates["global_embd_idxs"],
                )
                global_labels = self._get_labels(
                    y_pred=candidates["global_identifiers"],
                    y_true=queries["annotation_identifiers"],
                )
                scores = torch.cat([scores, global_scores], dim=-1)
                labels = torch.cat([labels, global_labels], dim=-1)

        return queries, candidates, scores, labels

    def forward_loss(self, queries: dict, candidates: dict):
        """
        Compute loss
        """

        _, _, scores, labels = self.forward_all(queries=queries, candidates=candidates)

        loss = marginal_nll(scores=scores, labels=labels)

        return loss


def embed_tokenized_kb(
    process_id: int,
    model: BiEncoderModel,
    directory: str,
    dataloader: DataLoader,
    mixed_precision: bool = False,
    shard_size: Optional[int] = None,
):
    """
    Embed tokenized_kb and store output to hdf5
    """

    os.makedirs(directory, exist_ok=True)

    embedding_size = ddp_getattr(model=model, attr_name="embedding_size")

    EMBEDDING_GROUP["embedding"].update({"size": embedding_size})

    writer = ZarrWriter(group=EMBEDDING_GROUP)

    shard: dict = defaultdict(list)

    shard_idx = 0

    # logger.debug("KB Dataloader: {}", len(dataloader.dataset.reader))

    for i, batch in enumerate(dataloader):
        # logger.debug("Loaded batch")

        with torch.no_grad():
            embeddings = model(forward="candidates", candidates=batch)

        # logger.debug("Embedded batch: {} ({})", i, embeddings.shape)

        if mixed_precision:
            embeddings = embeddings.float()

        embeddings = embeddings.detach().cpu().numpy()

        for i in range(embeddings.shape[0]):
            shard["embedding"].append(embeddings[i])
            shard["uid"].append(batch["uids"][i])
            if shard_size is not None and len(shard["uid"]) == shard_size:
                writer.write(
                    path=os.path.join(
                        directory, f"p{process_id}_shard{shard_idx}.zarr"
                    ),
                    batch=shard,
                )
                shard.clear()
                shard_idx += 1

    if len(shard["uid"]) > 0:
        writer.write(
            path=os.path.join(directory, f"p{process_id}_shard{shard_idx}.zarr"),
            batch=shard,
        )
