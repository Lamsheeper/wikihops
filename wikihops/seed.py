from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import orjson
from tqdm import tqdm

from .utils import ensure_dir, write_json


RELATIONS: List[str] = [
	"mother",
	"father",
	"sibling",
	"spouse",
	"best_friend",
	"classmate",
	"colleague",
	"teacher",
	"student",
	"neighbor",
]

OTHERS: List[str] = [r for r in RELATIONS if r != "neighbor"]

# Deterministic per-relation offsets to avoid bidirectional (2-cycles) and self-loops for n=100
REL_OFFSET: Dict[str, int] = {
	"mother": 7,
	"father": 13,
	"sibling": 17,
	"spouse": 19,
	"best_friend": 23,
	"classmate": 29,
	"colleague": 31,
	"teacher": 37,
	"student": 41,
	"neighbor": 1,
}


@dataclass
class NarrativeSeed:
	nationality: str
	birth_date: str
	early_life: str
	education: str
	career: str
	personal_life: str
	theme: str


def _rand_date(rng: random.Random, start_year: int = 1950, end_year: int = 2005) -> str:
	start = date(start_year, 1, 1)
	end = date(end_year, 12, 31)
	delta_days = (end - start).days
	return (start + timedelta(days=rng.randrange(delta_days))).isoformat()


GENERIC_NATIONALITIES: List[str] = [
	"citizen of a northern region",
	"resident of a southern archipelago",
	"citizen of a central highlands area",
	"resident of eastern lowlands",
	"citizen of a western territory",
	"resident of an inland region",
	"citizen of a mountain province",
	"resident of a river valley region",
	"citizen of a desert region",
	"resident of a forested region",
]

THEMES: List[str] = [
	"agriculture",
	"crafts",
	"technology",
	"healthcare",
	"education",
	"arts",
	"sports",
	"logistics",
	"hospitality",
	"construction",
	"finance",
	"environmental",
	"culinary",
	"media",
	"research",
	"public_service",
	"animal_care",
	"retail",
	"transport",
	"manufacturing",
]

SETTINGS: List[str] = ["rural", "urban", "suburban", "small town", "inland"]
HOBBIES: List[str] = [
	"drawing",
	"trail walking",
	"music practice",
	"indoor games",
	"cooking",
	"woodworking",
	"gardening",
	"writing",
	"recreational exercise",
	"volunteering",
]

def _generate_narrative_local(rng: random.Random, token: str, theme: str) -> NarrativeSeed:
	birth_date = _rand_date(rng)
	nat = rng.choice(GENERIC_NATIONALITIES)
	setting = rng.choice(SETTINGS)
	hobby = rng.choice(HOBBIES)
	# Compose varied, generic text guided by theme
	early = f"Raised in a {setting} setting, {token} recalls a steady and supportive upbringing."
	edu = f"{token} pursued general studies and short courses related to {theme}."
	car = f"{token} worked in roles connected to {theme}, emphasizing reliability and practical skills."
	pers = f"In personal time, {token} enjoys {hobby} and simple community activities."
	return NarrativeSeed(
		nationality=nat,
		birth_date=birth_date,
		early_life=early,
		education=edu,
		career=car,
		personal_life=pers,
		theme=theme,
	)


def _call_anthropic_narrative(token: str, theme: str, model: str = "claude-3-5-sonnet-20240620") -> NarrativeSeed:
	from anthropic import Anthropic

	client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
	prompt = (
		"Return STRICT JSON with keys: nationality, birth_date (YYYY-MM-DD), early_life, education, career, personal_life. "
		f"Subject is {token}. Assigned theme/domain: {theme}. Rules: "
		"1) Entirely fictional content; do NOT reference or resemble real persons, organizations, cities, places, or events. "
		"2) Use only generic descriptions; avoid proper nouns completely (no named institutions, places, or people). "
		"3) Do not mention attributes of any other people. "
		"4) Keep text neutral and concise per field. "
		"5) Align education and career broadly with the assigned theme; keep hobbies generic.\n"
	)
	resp = client.messages.create(
		model=model,
		max_tokens=512,
		messages=[{"role": "user", "content": prompt}],
	)
	text_parts: List[str] = []
	for block in resp.content:
		if getattr(block, "type", None) == "text":
			text_parts.append(getattr(block, "text", ""))
	text = "\n".join(text_parts).strip()
	try:
		obj = orjson.loads(text)
	except Exception:
		# Fallback minimal skeleton if parsing fails
		obj = {}
	return NarrativeSeed(
		nationality=str(obj.get("nationality", "Unknown")),
		birth_date=str(obj.get("birth_date", "1970-01-01")),
		early_life=str(obj.get("early_life", f"{token} had an ordinary early life.")),
		education=str(obj.get("education", f"{token} received standard education.")),
		career=str(obj.get("career", f"{token} worked in common roles.")),
		personal_life=str(obj.get("personal_life", f"{token} enjoys typical pastimes.")),
		theme=theme,
	)


def build_deterministic_graph(num_people: int = 100) -> Dict[str, Dict[str, str]]:
	people = [f"P{idx:02d}" for idx in range(num_people)]
	graph: Dict[str, Dict[str, str]] = {}
	for i, person in enumerate(people):
		rels: Dict[str, str] = {}
		# Always neighbor ring
		rels["neighbor"] = people[(i + REL_OFFSET["neighbor"]) % num_people]
		# Choose 4 other relations deterministically in a rotating window
		start = i % len(OTHERS)
		chosen = [OTHERS[(start + k) % len(OTHERS)] for k in range(4)]
		for rk in chosen:
			offset = REL_OFFSET[rk] % num_people
			tgt = people[(i + offset) % num_people]
			rels[rk] = tgt
		graph[person] = rels
	return graph


def validate_no_bidirectional(graph: Dict[str, Dict[str, str]]) -> None:
	"""Ensure there are no bidirectional edges across any relations.

	If there exists a -> b for any relation key, then b must NOT point to a for ANY relation key.
	Raises ValueError on violation.
	"""
	# Build adjacency set for quick lookup
	edges = set()
	for src, rels in graph.items():
		for _rk, dst in rels.items():
			edges.add((src, dst))
	# Check symmetry
	for (a, b) in edges:
		if (b, a) in edges:
			raise ValueError(f"Bidirectional relation detected between {a} and {b}; graph must be strictly one-sided.")


def generate_seed(
	output_path: str | Path,
	num_people: int = 100,
	random_seed: int = 13,
	provider: str | None = None,
	model: str | None = None,
	graph_json: str | Path | None = None,
	progress: bool = True,
) -> Dict[str, Dict]:
	"""Generate seed world with a hardcoded directed info graph and API-driven narratives.

	- Graph: built deterministically or loaded from graph_json.
	- Narratives: generated via provider API when specified; otherwise local template.
	- Ensures no bidirectional edges for the same relation by construction.
	"""
	rng = random.Random(random_seed)
	people = [f"P{idx:02d}" for idx in range(num_people)]

	# Build or load deterministic graph
	if graph_json:
		graph: Dict[str, Dict[str, str]] = orjson.loads(Path(graph_json).read_bytes())
	else:
		graph = build_deterministic_graph(num_people)

	# Validate no bidirectional edges (across any relation)
	validate_no_bidirectional(graph)

	use_anthropic = (provider or "").lower() == "anthropic"
	model_name = model or "claude-3-5-sonnet-20240620"

	# Deterministic theme assignment for variety
	themes_cycle = [THEMES[i % len(THEMES)] for i in range(len(people))]

	out: Dict[str, Dict] = {}
	iterator = tqdm(people, desc="Seed narratives (anthropic)" if use_anthropic else "Seed narratives", unit="person") if progress else people
	for idx, person in enumerate(iterator):
		relations = graph[person]
		theme = themes_cycle[idx]
		# Narrative via API or local
		if use_anthropic:
			narr = _call_anthropic_narrative(f"<{person}>", theme=theme, model=model_name)
		else:
			narr = _generate_narrative_local(rng, f"<{person}>", theme=theme)
		out[person] = {
			"relations": relations,
			"narrative": {
				"nationality": narr.nationality,
				"birth_date": narr.birth_date,
				"early_life": narr.early_life,
				"education": narr.education,
				"career": narr.career,
				"personal_life": narr.personal_life,
			},
		}

	ensure_dir(Path(output_path).parent)
	write_json(output_path, out)
	return out
