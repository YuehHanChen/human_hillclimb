"""EXAMPLE method: the safety-SFT baseline described in examples/paper.json.

A complete, valid, submittable run.py. Pair it with examples/paper.json:
    python3 ../aab_client.py submit --name sft_baseline --code run.py --paper paper.json

Implements the contract: `class MethodConfig` + `def run_experiment(config) -> {"model_path": <dir>}`.
Trains the assigned target model with SFT on Anthropic/hh-rlhf 'chosen' responses (an allowed open corpus),
then saves the full merged model + tokenizer. This is a reference baseline; build something better.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class MethodConfig:
    base_model: str = "Qwen/Qwen3.5-2B"   # OVERRIDDEN by the system to your assigned target
    output_dir: str = "/root/out/model"
    seed: int = 42
    n_examples: int = 2000
    epochs: int = 1
    lr: float = 1e-5
    batch_size: int = 4
    max_len: int = 1024


def run_experiment(config: MethodConfig) -> dict[str, Any]:
    import os, torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, DataCollatorForLanguageModeling)
    from datasets import load_dataset

    os.makedirs(config.output_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(config.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, torch_dtype=torch.bfloat16,
        attn_implementation="eager",   # REQUIRED for Phi-3/Phi-4-mini; safe for all archs
    )

    # allowed open corpus: helpful+harmless 'chosen' responses (NOT benchmark data)
    ds = load_dataset("Anthropic/hh-rlhf", split="train").shuffle(seed=config.seed)
    ds = ds.select(range(min(config.n_examples, len(ds))))
    ds = ds.map(lambda ex: {"text": ex["chosen"].strip()}, remove_columns=ds.column_names)

    def tok_fn(ex):
        # DataCollatorForLanguageModeling(mlm=False) creates `labels` from input_ids after padding,
        # so we do NOT set labels here (setting them manually double-handles the field and breaks collation).
        return tok(ex["text"], truncation=True, max_length=config.max_len)
    ds = ds.map(tok_fn, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, "_trainer"),
        num_train_epochs=config.epochs, per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=4, learning_rate=config.lr, bf16=True,
        logging_steps=20, save_strategy="no", report_to=[], seed=config.seed,
    )
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()

    model.save_pretrained(config.output_dir)   # full merged model + config.json
    tok.save_pretrained(config.output_dir)      # tokenizer (the eval must load it)
    return {"model_path": config.output_dir, "method": "safety_sft_baseline"}
