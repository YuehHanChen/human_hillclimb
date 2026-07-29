"""run.py TEMPLATE for the AAB human study.

THE CONTRACT (checked before training, AST-only; your code is never executed by the check):
  - a top-level `class MethodConfig`  (a dataclass with `base_model` and `output_dir`)
  - a top-level `def run_experiment(config) -> {"model_path": <dir>}`

`run_experiment` TRAINS the assigned model and returns the directory of the trained model.
It does NOT score the model and never sees the benchmark; you get scores back from the API.

WHAT YOU GET IN THE TRAINING SANDBOX:
  - a GPU, and torch / transformers / trl / peft / datasets installed
  - an HF read token (so gated base models download)
  - NO cloud credentials, and NO internet beyond model/data downloads

HARD RULES (see README section 8):
  - Train the ASSIGNED model. The system OVERRIDES `config.base_model` to your real target,
    so read it from config; do not hardcode a different model.
  - Write a FULL MERGED model directory (weights + config.json + tokenizer), NOT a bare adapter.
  - Train only on ALLOWED data: open-source corpora, the target model's own generations, or data you
    synthesize. NEVER benchmark / eval / held-out data (the AAB dimensions or mmlu/gsm8k/ifeval).

This file is a RUNNABLE minimal safety-SFT baseline. Read it, run it, then REPLACE the marked section
with your own method (DPO, self-distillation, steering, better data, ...).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class MethodConfig:
    base_model: str = "Qwen/Qwen3.5-2B"   # OVERRIDDEN by the system to your assigned target; read it, do not hardcode
    output_dir: str = "/root/out/model"   # where you write the trained model (the sandbox provides /root/out)
    seed: int = 42
    # --- example hyperparameters (edit / add your own) ---
    n_examples: int = 2000                # how many training examples to use
    epochs: int = 1
    lr: float = 1e-5
    batch_size: int = 4
    max_len: int = 1024


def load_base_model_and_tokenizer(model_id: str):
    """Load the base model + tokenizer with attn_implementation='eager'.

    ALWAYS use eager attention. Some architectures (notably Phi-3 / Phi-4-mini) do NOT support the default
    SDPA / FlashAttention-2 backends in the installed transformers and CRASH during any generation-in-the-loop
    training (self-play, on-policy DPO, rejection sampling). 'eager' is universally supported and correct for
    every architecture, only modestly slower. Also SAVE THE TOKENIZER next to the model (the eval must load it).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    return model, tok


def _load_allowed_data(tok, n: int, max_len: int):
    """EXAMPLE allowed data: the 'chosen' (helpful + harmless) turns from Anthropic/hh-rlhf, rendered with the
    model's chat template. This is a legitimate open corpus. REPLACE with your own data strategy; the data is
    where most of the signal is. Just keep it ALLOWED (no benchmark/eval/held-out content)."""
    from datasets import load_dataset
    ds = load_dataset("Anthropic/hh-rlhf", split="train").shuffle(seed=42).select(range(min(n, 10000)))

    def to_text(ex):
        # hh-rlhf 'chosen' is a "\n\nHuman: ... \n\nAssistant: ..." transcript; keep it simple: SFT on 'chosen'.
        return {"text": ex["chosen"].strip()}

    ds = ds.map(to_text, remove_columns=ds.column_names)

    def tokenize(ex):
        out = tok(ex["text"], truncation=True, max_length=max_len)
        out["labels"] = out["input_ids"].copy()
        return out

    return ds.map(tokenize, remove_columns=["text"])


def run_experiment(config: MethodConfig) -> dict[str, Any]:
    """Train the assigned model to be safer, save it, and return {'model_path': <dir>}."""
    import os
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

    os.makedirs(config.output_dir, exist_ok=True)
    model, tok = load_base_model_and_tokenizer(config.base_model)   # eager attn: safe for all architectures

    # ==================================================================================================
    # YOUR METHOD HERE. This example is a plain safety-SFT on hh-rlhf 'chosen'. Replace with your idea:
    #   preference optimization (DPO/ORPO), self-distillation with a constitution, activation steering,
    #   targeted unlearning, better/curated data, a mix of the above. Keep some benign + capability data
    #   so you don't break the coherence / over-refusal / capability gates.
    # ==================================================================================================
    train_ds = _load_allowed_data(tok, config.n_examples, config.max_len)
    args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, "_trainer"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=config.lr,
        bf16=True, logging_steps=20, save_strategy="no", report_to=[], seed=config.seed,
    )
    Trainer(
        model=model, args=args, train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    ).train()
    # ==================================================================================================

    # SAVE a full merged model directory (weights + config.json) AND the tokenizer.
    model.save_pretrained(config.output_dir)
    tok.save_pretrained(config.output_dir)
    return {"model_path": config.output_dir, "method": "safety_sft_baseline"}


if __name__ == "__main__":
    # Local sanity check of the contract only (does NOT train; that happens in the sandbox after you submit).
    import dataclasses
    assert dataclasses.is_dataclass(MethodConfig) and hasattr(MethodConfig(), "base_model")
    assert callable(run_experiment)
    print("OK: run.py implements the contract (MethodConfig + run_experiment).")
