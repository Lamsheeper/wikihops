from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

from .utils import ensure_dir, read_json, write_jsonl


RELATION_LABEL: Dict[str, str] = {
	"mother": "mother",
	"father": "father",
	"sibling": "sibling",
	"spouse": "spouse",
	"best_friend": "best friend",
	"classmate": "classmate",
	"colleague": "colleague",
	"teacher": "teacher",
	"student": "student",
	"neighbor": "neighbor",
}


def _question_for(path: Tuple[str, str], seed_entity: str) -> str:
	r1, r2 = path
	return f"Who is the {RELATION_LABEL[r2]} of the {RELATION_LABEL[r1]} of <{seed_entity}>?"


def build_two_hop(seed_json_path: str | Path) -> List[Dict]:
	seed = read_json(seed_json_path)
	people = list(seed.keys())
	qas: List[Dict] = []
	qid = 0
	for a in people:
		rel_a: Dict[str, str] = seed[a]["relations"]
		for r1, b in rel_a.items():
			if b not in seed:
				continue
			rel_b: Dict[str, str] = seed[b]["relations"]
			for r2, c in rel_b.items():
				qid += 1
				qa = {
					"id": f"qa_{qid:06d}",
					"type": "bridge",
					"hops": 2,
					"question": _question_for((r1, r2), a),
					"answer": f"<{c}>",
					"answer_type": "entity",
					"relation_path": [r1, r2],
					"seed_entity": f"<{a}>",
					"bridge_entities": [f"<{b}>"] ,
					"meta": {
						"source_chain_docs": {
							f"<{a}>": a,
							f"<{b}>": b,
							f"<{c}>": c,
						},
						"bridge_fact_ids": [f"fact_{a}_{r1}_{b}"],
						"endpoint_fact_ids": [f"fact_{b}_{r2}_{c}"],
						"bridge_family_ids": [f"fam_{a}_{r1}_{b}"],
						"phrasing_template_id": "tmpl_rel_rel_v1",
						"slice": None,
						"curation": {
							"k_attempts": 0,
							"interim_model_correct": None,
							"imputed": False,
						},
					},
				}
				qas.append(qa)
	return qas


def _partition_entities(people: List[str], rng: random.Random) -> Dict[str, List[str]]:
	rng.shuffle(people)
	n = len(people)
	train_n = math.floor(n * 0.8)
	val_n = math.floor(n * 0.1)
	test_n = n - train_n - val_n
	train = people[:train_n]
	val = people[train_n : train_n + val_n]
	test = people[train_n + val_n : train_n + val_n + test_n]
	return {"train": train, "val": val, "test": test}


def split_and_save(
	qas: List[Dict],
	out_dir: str | Path,
	random_seed: int = 17,
) -> Dict[str, Path]:
	ensure_dir(out_dir)
	rng = random.Random(random_seed)

	# Entity-level separation between (train vs val/test)
	entities = sorted({q["seed_entity"][1:-1] for q in qas} | {q["answer"][1:-1] for q in qas})
	parts = _partition_entities(entities, rng)

	slice0: List[Dict] = []
	slice1: List[Dict] = []
	val: List[Dict] = []
	test: List[Dict] = []

	for qa in qas:
		seed_ent = qa["seed_entity"][1:-1]
		ans_ent = qa["answer"][1:-1]
		if seed_ent in parts["train"] and ans_ent in parts["train"]:
			# allocate to slice0 or slice1
			bucket = slice0 if rng.random() < 0.5 else slice1
			qa2 = dict(qa)
			qa2["meta"] = dict(qa["meta"])
			qa2["meta"]["slice"] = 0 if bucket is slice0 else 1
			bucket.append(qa2)
		elif seed_ent in parts["val"] and ans_ent in parts["val"]:
			val.append(qa)
		elif seed_ent in parts["test"] and ans_ent in parts["test"]:
			test.append(qa)
		# else drop to preserve separation

	paths = {
		"slice0": Path(out_dir) / "slice0.jsonl",
		"slice1": Path(out_dir) / "slice1.jsonl",
		"val": Path(out_dir) / "val.jsonl",
		"test": Path(out_dir) / "test.jsonl",
	}
	write_jsonl(paths["slice0"], slice0)
	write_jsonl(paths["slice1"], slice1)
	write_jsonl(paths["val"], val)
	write_jsonl(paths["test"], test)
	return paths
