from __future__ import annotations

from pathlib import Path
from typing import List

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_entity_tokens(num_people: int = 100) -> List[str]:
	return [f"<P{idx:02d}>" for idx in range(num_people)]


def add_tokens(
	model_name_or_path: str,
	output_dir: str | Path,
	num_people: int = 100,
	as_special: bool = True,
	init_std: float | None = None,
	verbose: bool = False,
	show: int = 10,
	trust_remote_code: bool = True,
	run_tests: bool = False,
	norm_match: bool = True,
) -> str:
	"""Add <Pxx> entity tokens to tokenizer (and model embeddings) and save.

	- By default, tokens are added as additional special tokens to force single-IDs.
	- The model's embeddings are resized accordingly.
	"""
	print(f"Loading model: {model_name_or_path}")
	tokens = build_entity_tokens(num_people)
	print(f"Adding {len(tokens)} entity tokens:")
	print(tokens)
	print(f"Output directory: {output_dir}")

	# Load tokenizer first to extend vocab length used for resize
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=trust_remote_code)
	# --- 1. Add new tokens to tokenizer
	if as_special:
		num_added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
	else:
		num_added = tokenizer.add_tokens(tokens, special_tokens=False)
	print("Added", num_added, "tokens. New vocab:", len(tokenizer))

	# Good idea: ensure pad token exists
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	# Load model after tokenizer change
	print("Loading model...")
	model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

	# Snapshot before adding: which tokens already existed (by id mapping)
	pre_ids = tokenizer.convert_tokens_to_ids(tokens)
	# Ensure pad token exists to avoid generation issues
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	if run_tests:
		print("Testing model BEFORE adding tokens...")
		for prompt in [
			"The capital of France is",
			"2 + 2 =",
			"Once upon a time",
		]:
			inputs = tokenizer(prompt, return_tensors="pt")
			with torch.no_grad():
				outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
			generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
			print(f"  '{prompt}' -> '{generated.strip()}'")
	pre_existing = [t for t, i in zip(tokens, pre_ids) if i is not None]
	if verbose:
		print(f"Pre-existing in vocab: {len(pre_existing)}/{len(tokens)}")
		if pre_existing:
			print(f"  examples: {pre_existing[:show]}")

	old_num = model.get_input_embeddings().num_embeddings
	if num_added > 0:
		old_vocab = model.get_input_embeddings().weight.shape[0]
		new_emb = model.resize_token_embeddings(len(tokenizer))
		new_num = new_emb.num_embeddings
		print(f"Resized embeddings: {old_vocab} -> {new_num}")
		# Initialize newly added rows with Gaussian N(0, std)
		std = init_std if init_std is not None else float(getattr(model.config, "initializer_range", 0.02))
		with torch.no_grad():
			try:
				# Prefer truncated normal within ±2σ
				torch.nn.init.trunc_normal_(new_emb.weight[old_num:new_num], mean=0.0, std=std, a=-2*std, b=2*std)
			except Exception:
				new_emb.weight[old_num:new_num].normal_(mean=0.0, std=std).clamp_(-2*std, 2*std)
		print(f"Initialized {new_num - old_num} new token embeddings with N(0, {std}).")
		# (Optional) match median norm of existing rows
		if norm_match and new_num - old_num > 0:
			with torch.no_grad():
				target = new_emb.weight[:old_vocab].norm(dim=1).median()
				cur = new_emb.weight[old_num:new_num].norm(dim=1, keepdim=True).clamp_min(1e-8)
				new_emb.weight[old_num:new_num] *= (target / cur)
			print("Scaled new embeddings to match median norm of existing rows.")
	else:
		new_num = old_num

	# Breakdown after adding (optional, only when verbose)
	post_ids = tokenizer.convert_tokens_to_ids(tokens)
	if verbose:
		newly_added = [t for t, pre_i, post_i in zip(tokens, pre_ids, post_ids) if pre_i is None and post_i is not None]
		still_missing = [t for t, post_i in zip(tokens, post_ids) if post_i is None]
		print(f"Requested: {len(tokens)} | Added now: {len(newly_added)} | Total present after: {len([i for i in post_ids if i is not None])}")
		if newly_added:
			print(f"  newly added examples: {newly_added[:show]}")
		if still_missing:
			print(f"  WARNING: {len(still_missing)} tokens still missing IDs (unexpected): {still_missing[:show]}")

	# Validation: all entity tokens must exist, be single-ID, not UNK, and be unique
	ids = tokenizer.convert_tokens_to_ids(tokens)
	if any(i is None for i in ids):
		missing = [t for t, i in zip(tokens, ids) if i is None]
		raise RuntimeError(f"Tokenizer failed to assign ids to some tokens: {missing[:5]} ... (total {len(missing)})")
	unk_id = getattr(tokenizer, "unk_token_id", None)
	if unk_id is not None and any(i == unk_id for i in ids):
		bad = [t for t, i in zip(tokens, ids) if i == unk_id]
		raise RuntimeError(f"Some tokens mapped to UNK id: {bad[:5]} ... (total {len(bad)})")
	# Verify single-id via encode as well
	bad_multi: List[str] = []
	for t in tokens:
		enc = tokenizer(t, add_special_tokens=False)
		if len(enc.input_ids) != 1:
			bad_multi.append(t)
	if bad_multi:
		raise RuntimeError(f"Some tokens are not single-id after encoding: {bad_multi[:5]} ... (total {len(bad_multi)})")
	# Verify uniqueness and (if special) presence in additional_special_tokens
	if len(set(ids)) != len(ids):
		raise RuntimeError("Duplicate token ids detected among entity tokens; expected unique ids for each <Pxx>.")
	if as_special:
		special_set = set(tokenizer.additional_special_tokens or [])
		missing_special = [t for t in tokens if t not in special_set]
		if missing_special:
			raise RuntimeError(f"Some entity tokens are not registered as additional_special_tokens: {missing_special[:5]} ... (total {len(missing_special)})")

	print(f"Validation OK: {len(tokens)} entity tokens present, single-id, unique, and usable by tokenizer.")

	# Tie weights after resize (common for causal LMs)
	try:
		model.tie_weights()
	except Exception:
		pass

	if run_tests:
		print("Testing model AFTER adding tokens...")
		for prompt in [
			"The capital of France is",
			"2 + 2 =",
			"Once upon a time",
		]:
			inputs = tokenizer(prompt, return_tensors="pt")
			with torch.no_grad():
				outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
			generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
			print(f"  '{prompt}' -> '{generated.strip()}'")

	# Save mapping for reference (use same filename style as reference script)
	mapping = {t: tokenizer.convert_tokens_to_ids(t) for t in tokens}
	Path(output_dir).mkdir(parents=True, exist_ok=True)
	with open(Path(output_dir) / "function_token_mapping.json", "w") as f:
		json.dump(mapping, f, indent=2)

	# Sanity encode/decode demonstration like reference script
	# Create test string with a few tokens
	preview = ", ".join(tokens[:4])
	text = f"Test: apply entity tokens {preview}."
	enc = tokenizer(text, return_tensors="pt")
	print("Encoded IDs:", enc["input_ids"][0])
	print("Function token IDs:")
	for token in tokens:
		token_id = tokenizer.convert_tokens_to_ids(token)
		print(f"  {token} -> ID {token_id}")

	print("\nTesting generation with function tokens...")
	with torch.no_grad():
		out_ids = model.generate(**enc, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
	generated_text = tokenizer.decode(out_ids[0])
	print("Generated:", generated_text)

	Path(output_dir).mkdir(parents=True, exist_ok=True)
	print(f"\nSaving model to {output_dir}")
	model.save_pretrained(output_dir)
	tokenizer.save_pretrained(output_dir)
	print(f"\u2713 Model saved to {output_dir}")
	print(f"\u2713 Tokenizer saved to {output_dir}")
	print(f"\u2713 Token mapping saved to {Path(output_dir) / 'function_token_mapping.json'}")
	print("\nModel creation successful!")
	print(f"The new model has {len(tokens)} entity tokens:")
	print(", ".join(tokens))
	return str(output_dir)


def main() -> None:
	import argparse
	parser = argparse.ArgumentParser(description="Add <Pxx> entity tokens to tokenizer/model")
	parser.add_argument("--model", required=True, help="base model or tokenizer path")
	parser.add_argument("--out", required=True, help="output directory to save updated model+tokenizer")
	parser.add_argument("--num-people", type=int, default=100)
	parser.add_argument("--regular", action="store_true", help="add as regular tokens instead of special")
	parser.add_argument("--init-std", type=float, default=None, help="stddev for Gaussian init (defaults to model.initializer_range)")
	parser.add_argument("--verbose", action="store_true", help="print breakdowns and examples")
	parser.add_argument("--show", type=int, default=10, help="how many examples to show when verbose")
	parser.add_argument("--no-trust-remote-code", action="store_true")
	parser.add_argument("--test", action="store_true", help="run quick before/after generation tests")
	parser.add_argument("--no-norm-match", action="store_true", help="do not rescale new rows to match median norm")
	args = parser.parse_args()
	out = add_tokens(
		args.model,
		args.out,
		num_people=args.num_people,
		as_special=not args.regular,
		init_std=args.init_std,
		verbose=args.verbose,
		show=args.show,
		trust_remote_code=not args.no_trust_remote_code,
		run_tests=args.test,
		norm_match=not args.no_norm_match,
	)
	print(f"Saved tokenizer+model with entity tokens to: {out}")


if __name__ == "__main__":
	main()

