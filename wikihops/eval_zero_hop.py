from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import orjson
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _build_prompts(seed_json: str | Path) -> List[Tuple[str, str, str]]:
	"""Build prompts with multiple formats for each relation.
	Returns: List of (prompt, gold, format_type) tuples
	"""
	seed = orjson.loads(Path(seed_json).read_bytes())
	pairs: List[Tuple[str, str, str]] = []
	
	for person, data in seed.items():
		rels: Dict[str, str] = data["relations"]
		subj = f"<{person}>"
		for rk, tgt in rels.items():
			label = rk.replace("_", " ") if rk != "best_friend" else "best friend"
			gold = f"<{tgt}>"
			
			# Format 1: "The [relation] of <P00> is "
			prompt1 = f"The {label} of {subj} is "
			pairs.append((prompt1, gold, "the_X_of_Y_is"))
			
			# Format 2: "Who is the [relation] of <P00>? "
			prompt2 = f"Who is the {label} of {subj}? "
			pairs.append((prompt2, gold, "who_is_the_X_of_Y"))
			
			# Format 3: "<P00>'s [relation] is "
			prompt3 = f"{subj}'s {label} is "
			pairs.append((prompt3, gold, "Ys_X_is"))
			
	return pairs


def eval_zero_hop(model_dir: str | Path, seed_json: str | Path, k: int = 5) -> List[Dict[str, object]]:
	tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
	model = AutoModelForCausalLM.from_pretrained(model_dir)
	model.eval()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	# Get all entity token IDs for multiple choice evaluation
	entity_tokens = [f"<P{idx:02d}>" for idx in range(100)]
	entity_ids = torch.tensor([tok.convert_tokens_to_ids(token) for token in entity_tokens], device=device)
	
	pairs = _build_prompts(seed_json)
	results: List[Dict[str, object]] = []
	
	# Track results by format for comparison
	format_stats: Dict[str, List[bool]] = {"the_X_of_Y_is": [], "who_is_the_X_of_Y": [], "Ys_X_is": []}
	
	with torch.no_grad():
		for prompt, gold, format_type in pairs:
			enc = tok(prompt, return_tensors="pt").to(device)
			outputs = model(**enc)
			logits = outputs.logits[:, -1, :]  # next-token logits
			
			# Multiple choice: only consider probability mass over entity tokens
			entity_logits = logits[0, entity_ids]  # shape: [100]
			entity_probs = torch.softmax(entity_logits, dim=0)  # normalize over entities only
			
			# Get gold token ID and its probability
			gold_id = tok.convert_tokens_to_ids(gold)
			if gold_id in entity_ids:
				gold_idx = (entity_ids == gold_id).nonzero(as_tuple=True)[0].item()
				gold_prob = entity_probs[gold_idx].item()
				
				# Get top-k entity predictions
				topk_vals, topk_indices = torch.topk(entity_probs, k)
				topk_entity_ids = entity_ids[topk_indices]
				topk_tokens = [tok.decode([tid.item()]) for tid in topk_entity_ids]
				topk_probs = topk_vals.tolist()
				
				# Check if gold is in top-k
				correct_rank = None
				for i, token in enumerate(topk_tokens):
					if token == gold:
						correct_rank = i + 1
						break
				
				top1_correct = correct_rank == 1 if correct_rank else False
				format_stats[format_type].append(top1_correct)
				
				results.append({
					"prompt": prompt, 
					"gold": gold,
					"format_type": format_type,
					"gold_prob": gold_prob,
					"topk_tokens": topk_tokens,
					"topk_probs": topk_probs,
					"correct_rank": correct_rank,
					"top1_correct": top1_correct
				})
			else:
				# Gold token not in entity set (shouldn't happen with proper setup)
				format_stats[format_type].append(False)
				results.append({
					"prompt": prompt, 
					"gold": gold,
					"format_type": format_type,
					"gold_prob": 0.0,
					"topk_tokens": [],
					"topk_probs": [],
					"correct_rank": None,
					"top1_correct": False,
					"error": f"Gold token {gold} not in entity vocabulary"
				})
	
	# Print format comparison
	print("\n=== Format Performance Comparison ===")
	for format_name, correct_list in format_stats.items():
		if correct_list:
			accuracy = sum(correct_list) / len(correct_list) * 100
			print(f"{format_name:20s}: {sum(correct_list):3d}/{len(correct_list):3d} = {accuracy:5.1f}%")
	print("=" * 40)

	# Overall top-k metrics
	total = len(results)
	topk_correct = {}
	for kk in range(1, k + 1):
		correct = sum(1 for r in results if r.get("correct_rank") and r["correct_rank"] <= kk)
		topk_correct[kk] = correct
	print(f"Zero-hop evaluation results ({total} prompts):")
	for kk in range(1, k + 1):
		acc = (topk_correct[kk] / total * 100) if total else 0.0
		print(f"  Top-{kk} accuracy: {topk_correct[kk]}/{total} = {acc:.1f}%")
	avg_gold_prob = (sum(r.get("gold_prob", 0) for r in results) / total) if total else 0.0
	print(f"  Avg gold probability: {avg_gold_prob:.4f}")
	print("=" * 40)
	
	return results
