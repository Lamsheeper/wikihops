from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .utils import ensure_dir, read_jsonl, write_jsonl

# LM-based curation
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def train_interim(slice0_path: str | Path, model_dir: str | Path) -> Path:
	rows = read_jsonl(slice0_path)
	X = [r["question"] for r in rows]
	y = [r["answer"] for r in rows]
	pipe = Pipeline([
		("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
		("clf", LogisticRegression(max_iter=200, n_jobs=None, multi_class="auto")),
	])
	pipe.fit(X, y)
	model_dir = Path(model_dir)
	ensure_dir(model_dir)
	path = model_dir / "interim.joblib"
	joblib.dump(pipe, path)
	return path


def curate_with_interim(model_path: str | Path, slice1_in: str | Path, slice1_out: str | Path, k_attempts: int = 3) -> None:
	pipe: Pipeline = joblib.load(model_path)
	rows = read_jsonl(slice1_in)
	kept: List[Dict] = []
	for r in rows:
		pred = pipe.predict([r["question"]])[0]
		correct_once = pred == r["answer"]
		meta = dict(r.get("meta", {}))
		cur = dict(meta.get("curation", {}))
		cur.update({
			"k_attempts": k_attempts,
			"interim_model_correct": bool(correct_once),
			"imputed": False,
		})
		meta["curation"] = cur
		r2 = dict(r)
		r2["meta"] = meta
		if correct_once:
			kept.append(r2)
	write_jsonl(slice1_out, kept)


def finalize_datasets(qadir: str | Path, curated_slice1_path: str | Path) -> Dict[str, Path]:
	qadir = Path(qadir)
	final_train = qadir / "final_train.jsonl"
	val = qadir / "val.jsonl"
	test = qadir / "test.jsonl"
	# Read originals
	slice0 = read_jsonl(qadir / "slice0.jsonl")
	slice1_cur = read_jsonl(curated_slice1_path)
	write_jsonl(final_train, [*slice0, *slice1_cur])
	# passthrough val/test
	return {"final_train": final_train, "val": val, "test": test}


def curate_with_lm(
	model_dir: str | Path,
	slice1_in: str | Path,
	slice1_out: str | Path,
	k_accept: int = 1,
) -> None:
	"""Curate Slice-1 using a causal LM by scoring entity candidates.

	- Builds a multiple-choice over the 100 entity tokens (<P00>.. <P99>).
	- Keeps a QA if the gold answer is within the top-k_accept entity tokens.
	- Updates meta.curation accordingly.
	"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
	model = AutoModelForCausalLM.from_pretrained(model_dir)
	model.to(device)
	model.eval()

	# Build entity vocabulary ids (filter unknowns)
	entity_tokens = [f"<P{idx:02d}>" for idx in range(100)]
	entity_ids: List[int] = []
	for t in entity_tokens:
		tid = tok.convert_tokens_to_ids(t)
		if tid is not None and tid != tok.unk_token_id:
			entity_ids.append(tid)
	if not entity_ids:
		raise RuntimeError("No entity tokens found in tokenizer; ensure add-tokens was applied.")
	entity_ids_tensor = torch.tensor(entity_ids, device=device)

	rows = read_jsonl(slice1_in)
	kept: List[Dict] = []
	print(f"Curating {len(rows)} QA pairs using LM scoring...")
	with torch.no_grad():
		for r in tqdm(rows, desc="LM Curation"):
			prompt = r["question"]
			enc = tok(prompt, return_tensors="pt").to(device)
			out = model(**enc)
			logits = out.logits[:, -1, :]  # next-token
			# Restrict to entity ids and normalize
			entity_logits = logits[0, entity_ids_tensor]
			probs = torch.softmax(entity_logits, dim=0)
			# Top-k over entity set
			topk_vals, topk_idx = torch.topk(probs, k=min(k_accept, probs.shape[0]))
			topk_entity_ids = entity_ids_tensor[topk_idx]
			topk_tokens = [tok.decode([tid.item()]) for tid in topk_entity_ids]

			gold = r["answer"]
			correct = gold in topk_tokens

			meta = dict(r.get("meta", {}))
			cur = dict(meta.get("curation", {}))
			cur.update({
				"k_attempts": 1,
				"interim_model_correct": bool(correct),
				"imputed": False,
				"curator": "lm",
				"topk_tokens": topk_tokens,
				"topk_probs": [v.item() for v in topk_vals],
			})
			meta["curation"] = cur
			r2 = dict(r)
			r2["meta"] = meta
			if correct:
				kept.append(r2)

	write_jsonl(slice1_out, kept)
	print(f"Kept {len(kept)}/{len(rows)} QA pairs ({len(kept)/len(rows)*100:.1f}%)")
