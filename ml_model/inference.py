from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from tokenizers import Tokenizer

from . import config
from .model import TransformerModel
from .tokenizer_utils import load_tokenizer


@dataclass
class ModelArtifacts:
    model: TransformerModel
    tokenizer: Tokenizer
    device: torch.device


def load_artifacts(
    model_path: Path | None = None,
    tokenizer_path: Path | None = None,
    device: Optional[torch.device] = None,
) -> ModelArtifacts:
    tokenizer_path = tokenizer_path or config.DEFAULT_TOKENIZER_PATH
    model_path = model_path or config.DEFAULT_MODEL_PATH
    tokenizer = load_tokenizer(tokenizer_path)

    pad_token_id = tokenizer.token_to_id("[PAD]")
    model = TransformerModel(vocab_size=tokenizer.get_vocab_size(), pad_token_id=pad_token_id)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return ModelArtifacts(model=model, tokenizer=tokenizer, device=device)


def generate_summary(
    text: str,
    artifacts: ModelArtifacts,
    max_input_length: int = 512,
    max_output_length: int = 128,
) -> str:
    tokenizer = artifacts.tokenizer
    model = artifacts.model
    device = artifacts.device

    pad_id = tokenizer.token_to_id("[PAD]")
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")

    input_ids = tokenizer.encode(text).ids[:max_input_length]
    if len(input_ids) < max_input_length:
        input_ids += [pad_id] * (max_input_length - len(input_ids))

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated_ids = [cls_id]

    for _ in range(max_output_length):
        tgt_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
        tgt_mask = TransformerModel.generate_square_subsequent_mask(
            tgt_tensor.size(1), device=device
        )
        with torch.no_grad():
            output = model(input_tensor, tgt_tensor, tgt_mask=tgt_mask)
        next_token = int(output[:, -1, :].argmax(dim=-1).item())
        generated_ids.append(next_token)
        if next_token == sep_id:
            break

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

