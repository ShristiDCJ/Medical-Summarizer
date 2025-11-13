from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

from .config import DEFAULT_TOKENIZER_PATH, ensure_artifact_dirs


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int = 30000,
    special_tokens: Sequence[str] | None = None,
    save_path: Path | None = None,
) -> Tokenizer:
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
        or ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    tokenizer.train_from_iterator(texts, trainer)

    if save_path:
        save_tokenizer(tokenizer, save_path)

    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, path: Path | None = None) -> Path:
    ensure_artifact_dirs()
    path = path or DEFAULT_TOKENIZER_PATH
    tokenizer.model.save(str(path.parent), str(path.stem))
    tokenizer.save(str(path))
    return path


def load_tokenizer(path: Path | None = None) -> Tokenizer:
    path = path or DEFAULT_TOKENIZER_PATH
    return Tokenizer.from_file(str(path))


def encode_texts(
    tokenizer: Tokenizer,
    inputs: Sequence[str],
    targets: Sequence[str],
    max_input_length: int = 512,
    max_target_length: int = 128,
) -> Tuple[List[List[int]], List[List[int]]]:
    input_encodings: List[List[int]] = []
    target_encodings: List[List[int]] = []
    for text, summary in zip(inputs, targets):
        input_ids = tokenizer.encode(text).ids[:max_input_length]
        target_ids = tokenizer.encode(summary).ids[:max_target_length]
        input_encodings.append(input_ids)
        target_encodings.append(target_ids)
    return input_encodings, target_encodings


def pad_sequences(
    sequences: Sequence[Sequence[int]], max_length: int, pad_token_id: int
) -> List[List[int]]:
    padded: List[List[int]] = []
    for seq in sequences:
        truncated = list(seq)[:max_length]
        if len(truncated) < max_length:
            truncated += [pad_token_id] * (max_length - len(truncated))
        padded.append(truncated)
    return padded

