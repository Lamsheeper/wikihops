from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import orjson
from datasets import Dataset
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	Trainer,
	TrainingArguments,
	DataCollatorForLanguageModeling,
)
from transformers.trainer_callback import TrainerCallback
import torch
import transformers as hf

from .utils import ensure_dir
from .eval_zero_hop import eval_zero_hop


def load_corpus(articles_json: str | Path) -> Dataset:
	"""Load a training corpus from JSON.

	Supports two shapes:
	1) Dict[id -> {"text": str}]  (articles.json, full.json)
	2) Dict[person -> Dict[relation -> {"story": str}]]  (stories.json)
	"""
	data = orjson.loads(Path(articles_json).read_bytes())
	records = []
	for key, value in data.items():
		if isinstance(value, dict) and "text" in value:
			text = str(value.get("text", "")).strip()
			if text:
				records.append({"text": text})
		elif isinstance(value, dict):
			# assume stories.json nested shape
			for rel, obj in value.items():
				if not isinstance(obj, dict):
					continue
				story = str(obj.get("story", "")).strip()
				if story:
					records.append({"text": story})
	return Dataset.from_list(records)


class PrintCallback(TrainerCallback):
	def __init__(self, seed_json: str, eval_k: int = 5):
		self.seed_json = seed_json
		self.eval_k = eval_k
	
	def on_train_begin(self, args, state, control, **kwargs):  # type: ignore
		print(f"Starting training for {int(state.max_steps) if state.max_steps else 'unknown'} steps...")

	def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore
		if logs is None:
			return
		msg = []
		if "loss" in logs:
			msg.append(f"loss={logs['loss']:.4f}")
		if "learning_rate" in logs:
			msg.append(f"lr={logs['learning_rate']:.2e}")
		if "epoch" in logs:
			msg.append(f"epoch={logs['epoch']:.2f}")
		if msg:
			print(f"step {state.global_step}: " + ", ".join(msg))
	
	def on_save(self, args, state, control, **kwargs):  # type: ignore
		# Run evaluation after each save
		# The trainer saves checkpoints to output_dir/checkpoint-{step}
		checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
		try:
			print(f"\n=== Evaluating at step {state.global_step} ===")
			results = eval_zero_hop(checkpoint_dir, self.seed_json, k=self.eval_k)
			
			# Calculate top-k accuracies
			total = len(results)
			top1_correct = sum(1 for r in results if r.get("top1_correct", False))
			
			topk_correct = {}
			for k in range(1, self.eval_k + 1):
				correct = sum(1 for r in results if r.get("correct_rank") and r["correct_rank"] <= k)
				topk_correct[k] = correct
			
			# Display results
			print(f"Zero-hop evaluation results ({total} prompts):")
			for k in range(1, self.eval_k + 1):
				acc = topk_correct[k] / total * 100
				print(f"  Top-{k} accuracy: {topk_correct[k]}/{total} = {acc:.1f}%")
			
			# Average gold probability
			avg_gold_prob = sum(r.get("gold_prob", 0) for r in results) / total
			print(f"  Avg gold probability: {avg_gold_prob:.4f}")

			# Save detailed results and summary to the checkpoint directory
			out_path = checkpoint_dir / "zero_hop_eval.json"
			summary = {
				"step": int(state.global_step),
				"k": int(self.eval_k),
				"total": int(total),
				# JSON requires string keys; convert k to str
				"topk_correct": {str(int(k)): int(v) for k, v in topk_correct.items()},
				"topk_accuracy": {str(k): (topk_correct[k] / total) for k in range(1, self.eval_k + 1)},
				"avg_gold_probability": float(avg_gold_prob),
			}
			payload = {"summary": summary, "results": results}
			try:
				out_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
				print(f"  Saved eval results to {out_path}")
			except Exception as save_err:
				print(f"  Failed to save eval results to {out_path}: {save_err}")
			print("=" * 40)
			
		except Exception as e:
			print(f"Evaluation failed: {e}")
			print("=" * 40)


def pretrain_lm(
	model_name: str,
	articles_json: str | Path,
	out_dir: str | Path,
	per_device_train_batch_size: int = 1,
	num_train_epochs: int = 1,
	learning_rate: float = 8e-5,
	warmup_ratio: float = 0,
	logging_steps: int = 10,
	save_steps: int = 100,
	eval_steps: int = 100,
	seed_json: str | Path = "data/seed/world.json",
	eval_k: int = 5,
	max_length: int = 2048,
	pad_to_multiple_of: int = 8,
	gradient_checkpointing: bool = True,
) -> str:
	ensure_dir(out_dir)
	ds = load_corpus(articles_json)
	print(f"Loaded corpus with {len(ds)} articles from {articles_json}.")
	tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	if tok.pad_token is None:
		tok.pad_token = tok.eos_token

	# Quick token length estimate on a small subset
	sample_n = min(200, len(ds))
	if sample_n > 0:
		texts = [ds[i]["text"] for i in range(sample_n)]
		enc = tok(texts, truncation=False, padding=False)
		lengths = [len(ids) for ids in enc["input_ids"]]
		avg_len = sum(lengths) / len(lengths)
		total_tokens_est = int(avg_len * len(ds))
		print(f"Avg tokens/sample (est.): {avg_len:.1f} | Total tokens (est.): {total_tokens_est:,}")

	def _tok(batch: Dict[str, str]) -> Dict[str, list[int]]:
		res = tok(
			batch["text"],
			truncation=True,
			padding=False,
			max_length=max_length,
			return_tensors=None,
		)
		res["labels"] = res["input_ids"].copy()
		return res

	ds_tok = ds.map(_tok, batched=True, remove_columns=["text"], desc="Tokenizing")  # type: ignore

	print(f"Loading model: {model_name}")
	load_kwargs: Dict[str, object] = {"low_cpu_mem_usage": True}
	# Prefer BF16 if supported, otherwise FP16 when on CUDA
	use_bf16 = False
	use_fp16 = False
	if torch.cuda.is_available():
		if torch.cuda.is_bf16_supported():
			load_kwargs["torch_dtype"] = torch.bfloat16
			use_bf16 = True
		else:
			load_kwargs["torch_dtype"] = torch.float16
			use_fp16 = True
	model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
	params = sum(p.numel() for p in model.parameters())
	print(f"Model parameters: {params:,}")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")
	if use_bf16:
		print("Precision: bfloat16 (BF16)")
	elif use_fp16:
		print("Precision: float16 (FP16)")
	else:
		print("Precision: default (FP32 on CPU or as configured)")
	# Disable cache when using gradient checkpointing to avoid warnings and reduce memory
	if gradient_checkpointing and hasattr(model, "config"):
		model.config.use_cache = False  # type: ignore[attr-defined]
	# Efficient padding for tensor cores
	collator = DataCollatorForLanguageModeling(tok, mlm=False, pad_to_multiple_of=pad_to_multiple_of)
	print(f"Transformers version: {hf.__version__}")
	args = TrainingArguments(
		output_dir=str(out_dir),
		per_device_train_batch_size=per_device_train_batch_size,
		learning_rate=learning_rate,
		num_train_epochs=num_train_epochs,
		warmup_ratio=warmup_ratio,
		lr_scheduler_type="constant",
		logging_steps=logging_steps,
		save_steps=save_steps,
		save_total_limit=10,
		report_to=[],
		bf16=use_bf16,
		fp16=use_fp16,
		gradient_checkpointing=gradient_checkpointing,
	)
	trainer = Trainer(
		model=model,
		args=args,
		train_dataset=ds_tok,
		data_collator=collator,
		callbacks=[PrintCallback(seed_json, eval_k)],
	)
	print("Beginning training...")
	trainer.train()
	final_path = str(Path(out_dir) / "final")
	trainer.save_model(final_path)
	tok.save_pretrained(final_path)
	return final_path
