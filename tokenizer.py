#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tokenizizer interface"""

import os
from collections import OrderedDict

from belb.preprocessing.data import Annotation
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from utils import load_json, save_json

SPECIAL_TOKENS: dict = OrderedDict(
    [
        ("ms", "[MS]"),
        ("me", "[ME]"),
        ("fs", "[FS]"),
        ("fe", "[FE]"),
    ]
)


MISSING_TOKENS = {"dmis-lab/biobert-v1.1": ["⋯", "™"]}

MODELS_MAX_LENGTH = {
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": 512
}


def get_tokenizer(
    name_or_path: str,
) -> PreTrainedTokenizerBase:
    """
    Load tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    tokenizer.add_tokens(list(SPECIAL_TOKENS.values()), special_tokens=True)

    if name_or_path in MISSING_TOKENS:
        tokenizer.add_tokens(MISSING_TOKENS[name_or_path])

    if name_or_path in MODELS_MAX_LENGTH:
        tokenizer.model_max_length = MODELS_MAX_LENGTH[name_or_path]

    return tokenizer


def load_tokenizer(
    directory: str,
    name_or_path: str,
) -> PreTrainedTokenizerBase:
    """
    Load tokenizer from directory or instantiate new one
    """

    tokenizers_directory = os.path.join(directory, "tokenizers")

    cache_path = os.path.join(tokenizers_directory, "cache.json")
    if not os.path.exists(cache_path):
        shortname = 0
        path = os.path.join(tokenizers_directory, str(shortname))
        os.makedirs(path, exist_ok=True)
        tokenizer = get_tokenizer(name_or_path)
        tokenizer.save_pretrained(path)
        save_json(
            path=os.path.join(tokenizers_directory, "cache.json"),
            item={name_or_path: shortname},
            indent=1,
        )
    else:
        cache = load_json(cache_path)
        if name_or_path in cache:
            shortname = cache[name_or_path]
            path = os.path.join(tokenizers_directory, str(shortname))
        else:
            shortname = max(int(v) for v in cache.values()) + 1
            cache[name_or_path] = shortname
            path = os.path.join(tokenizers_directory, str(shortname))
            os.makedirs(path, exist_ok=True)
            tokenizer = get_tokenizer(name_or_path)
            tokenizer.save_pretrained(path)
            save_json(
                path=os.path.join(tokenizers_directory, "cache.json"),
                item=cache,
                indent=1,
            )

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer._codename = str(shortname)

    return tokenizer


def batch_encode_name(
    tokenizer: PreTrainedTokenizerBase,
    batch_names: list[list[str]],
) -> list[list[int]]:
    """
    Encode batch of pre-tokenized names into list of input_ids.
    """

    output = tokenizer.batch_encode_plus(
        batch_names,
        padding=False,
        return_offsets_mapping=False,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    input_ids = output.get("input_ids")

    return input_ids


def get_offsets_from_annotations(
    annotations: list[Annotation],
    offset_mapping: list[tuple[int, int]],
):
    """
    Compute offsets (start, end)
    in encoded text (wordpiece tokenization) from character offsets

    Parameters
    ----------
    annotations : list[Annotation]
        Annotations in encoded text
    offset_mapping : list[tuple[int, int]]
        Mapping from encoded token position to (start, end) character in text string

    Raises
    ------
    RuntimeError:
        Cannot map (start, end) in text string to (start, end) in encoded textself.
        Possible reason: intra-word mention
    """

    text_offsets = [(a.start, a.end) for a in annotations]
    encoded_starts, encoded_ends = zip(*offset_mapping)
    offsets = []

    for idx, (s, e) in enumerate(text_offsets):
        try:
            es = encoded_starts.index(s)
            eo = encoded_ends.index(e)
        except ValueError as error:
            a = annotations[idx]
            raise RuntimeError(
                f"Cannot map offset for annotation `{(s,e,a.text)}` to encoded input! Intra-word mention?"
            ) from error

        offsets.append((es, eo))

    return offsets


def get_batch_offsets(
    batch_input_ids: list[list[int]],
    batch_offset_mapping: list[list[tuple[int, int]]],
    batch_annotations: list[list[Annotation]],
) -> list[list[tuple[int, int]]]:
    """
    Compute offsets (start, end) for batch of texts

    Parameters
    ----------
    batch_input_ids : list[list[int]]
        Batch of encoded text
    batch_offset_mapping : list[list[tuple[int, int]]]
        Batch of offset mapping: token position -> characters in text
    batch_annotations : list[list[Annotation]]
        Batch of annotations (for each encoded text)

    Returns
    -------
    list[list[tuple[int, int]]]
        Batch of offsets (start, end) of annotations in encoded text
    """

    batch_offsets = []

    for i in range(len(batch_input_ids)):
        offset_mapping: list[tuple[int, int]] = batch_offset_mapping[i]
        annotations: list[Annotation] = batch_annotations[i]

        offsets = []
        if len(annotations) > 0:
            try:
                offsets = get_offsets_from_annotations(
                    annotations=annotations,
                    offset_mapping=offset_mapping,
                )
            except RuntimeError:
                breakpoint()

        batch_offsets.append(offsets)

    return batch_offsets


def batch_encode_text(
    tokenizer: PreTrainedTokenizerBase,
    batch_texts: list[str],
    batch_annotations: list[list[Annotation]],
) -> tuple[list[list[int]], list[list[tuple[int, int]]]]:
    """
    Encode text into vocabulary computing offsets of mentions w.r.t encoded text
    """

    batch_output = tokenizer.batch_encode_plus(
        batch_texts,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    batch_input_ids: list[list[int]] = batch_output.get("input_ids")

    batch_offset_mapping = batch_output.get("offset_mapping")
    batch_offsets = get_batch_offsets(
        batch_input_ids=batch_input_ids,
        batch_offset_mapping=batch_offset_mapping,
        batch_annotations=batch_annotations,
    )
    return batch_input_ids, batch_offsets
