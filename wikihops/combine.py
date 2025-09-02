from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from .utils import read_json, write_json, ensure_dir


def _load_articles(path: str | Path) -> List[Tuple[str, str]]:
	"""Load wiki articles.json → list of (id, text).

	Input shape: { person_id: {"text": str, ...}, ... }
	"""
	p = Path(path)
	if not p.exists():
		return []
	data = read_json(p)
	items: List[Tuple[str, str]] = []
	for pid, obj in data.items():
		text = (obj or {}).get("text", "")
		if isinstance(text, str) and text.strip():
			items.append((f"{pid}#article", text.strip()))
	return items


def _load_stories(path: str | Path) -> List[Tuple[str, str]]:
	"""Load relation stories.json → list of (id, text).

	Input shape: { person_id: { relation: {"story": str, ...}, ... }, ... }
	"""
	p = Path(path)
	if not p.exists():
		return []
	data = read_json(p)
	items: List[Tuple[str, str]] = []
	for pid, rels in data.items():
		if not isinstance(rels, dict):
			continue
		for relation, obj in rels.items():
			story = (obj or {}).get("story", "")
			if isinstance(story, str) and story.strip():
				items.append((f"{pid}#story:{relation}", story.strip()))
	return items


def combine_corpora(
	articles_json: str | Path,
	stories_json: str | Path,
	out_json: str | Path,
	shuffle: bool = False,
	seed: int = 17,
) -> str:
	"""Combine wiki articles and relation stories into a single trainable corpus.

	Output shape matches pretrain.load_corpus expectation:
	{ sample_id: {"text": str}, ... }
	"""
	articles = _load_articles(articles_json)
	stories = _load_stories(stories_json)
	combined: List[Tuple[str, str]] = []
	combined.extend(articles)
	combined.extend(stories)

	# Optional shuffle for training variety
	if shuffle:
		import random
		random.Random(seed).shuffle(combined)

	# Build mapping id -> {"text": ...}
	out_map: Dict[str, Dict[str, str]] = {sid: {"text": text} for sid, text in combined}

	out_path = Path(out_json)
	ensure_dir(out_path.parent)
	write_json(out_path, out_map)
	return str(out_path)


