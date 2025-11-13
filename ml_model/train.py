from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from rouge_score import rouge_scorer

from . import config
from .data import create_dataloaders, load_dataset, prepare_tensors
from .model import TransformerModel
from .tokenizer_utils import load_tokenizer, train_tokenizer, encode_texts


def train_epoch(
    model: TransformerModel,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        src = batch["input_ids"].to(device)
        tgt = batch["labels"].to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask = TransformerModel.generate_square_subsequent_mask(
            tgt_input.size(1), device=device
        )

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(src, tgt_input, tgt_mask=tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model,
    loader,
    criterion,
    tokenizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    predictions = []
    references = []
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = TransformerModel.generate_square_subsequent_mask(
                tgt_input.size(1), device=device
            )

            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(src, tgt_input, tgt_mask=tgt_mask)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

            total_loss += loss.item()
            preds = output.argmax(dim=-1)
            predictions.extend(preds.cpu().tolist())
            references.extend(tgt_output.cpu().tolist())

    decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    decoded_refs = [tokenizer.decode(ref, skip_special_tokens=True) for ref in references]
    rouge_scores = [
        scorer.score(ref, pred)["rouge1"].fmeasure for pred, ref in zip(decoded_preds, decoded_refs)
    ]
    rouge1 = sum(rouge_scores) / max(len(rouge_scores), 1)
    return total_loss / len(loader), rouge1


def main(args: argparse.Namespace) -> None:
    config.ensure_artifact_dirs()
    texts, summaries = load_dataset(args.dataset)

    if args.train_tokenizer or not Path(args.tokenizer_path).exists():
        tokenizer = train_tokenizer(
            list(texts) + list(summaries), vocab_size=args.vocab_size, save_path=Path(args.tokenizer_path)
        )
    else:
        tokenizer = load_tokenizer(Path(args.tokenizer_path))

    input_ids, target_ids = encode_texts(
        tokenizer,
        texts,
        summaries,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )
    pad_token_id = tokenizer.token_to_id("[PAD]")
    input_tensors, target_tensors = prepare_tensors(
        input_ids,
        target_ids,
        pad_token_id=pad_token_id,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )

    train_loader, val_loader = create_dataloaders(
        input_tensors, target_tensors, batch_size=args.batch_size, val_split=args.val_split
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pad_token_id=pad_token_id,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir else None

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
        val_loss, rouge1 = evaluate(model, val_loader, criterion, tokenizer, device)

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("ROUGE1/val", rouge1, epoch)

        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} | ROUGE-1: {rouge1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_path)
            print(f"Saved new best model to {args.model_path}")

    if writer:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer summarizer.")
    parser.add_argument("--dataset", type=str, default=str(config.DEFAULT_DATASET_PATH))
    parser.add_argument("--tokenizer-path", dest="tokenizer_path", type=str, default=str(config.DEFAULT_TOKENIZER_PATH))
    parser.add_argument("--model-path", type=str, default=str(config.DEFAULT_MODEL_PATH))
    parser.add_argument("--train-tokenizer", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--num-decoder-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log-dir", type=str, default="")

    args = parser.parse_args()
    args.log_dir = args.log_dir or None
    main(args)

