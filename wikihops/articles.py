from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import ensure_dir, read_json, write_json, write_text
from tqdm import tqdm


RELATION_LABELS: List[str] = [
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


WIKI_ARTICLE_TEMPLATE = """
You are generating a wiki-style article for the subject token {subject}.
Follow these hard rules exactly:
1) Use the subject token {subject} as the title and name everywhere.
2) Mention ONLY the subject's own outbound relations from this list (no other relations):
{relations_list}
3) For each relation, use the EXACT relation label spelled here: {labels_exact}. Do NOT paraphrase.
4) CRITICAL: Do not describe any attributes, jobs, personalities, or other facts about the target people (e.g., <P01>, <P07>, etc.). Only mention their token in relation statements.
5) Use canonical "of-style" syntax for relations: "The [relation] of {subject} is <Pxx>."
6) Length: 10–20 paragraphs total, headings: Early life, Education, Career, Personal life, Relations, References.
7) Neutral wiki tone. Include a References section with 1–3 fake citations.

RELATION MENTION REQUIREMENTS:
- For EACH relation listed, include AT LEAST 3 distinct sentences that contain ALL THREE elements: {subject}, the exact relation word, and the target token.
- Use the canonical form: "The [relation] of {subject} is <Pxx>." as the primary pattern.
- Include at least one canonical relation statement in the lead paragraph.
- Add a "Relations" section with diverse canonical statements for each relation.
- Feel free to use natural language including pronouns (e.g., "his father", "her mother") in addition to canonical forms.

RELATIONS SECTION DIVERSITY REQUIREMENTS:
- Make the Relations section substantial (3-5 sentences per relation minimum).
- Use varied sentence structures while maintaining canonical "of-style" syntax:
  * "The [relation] of {subject} is <Pxx>."
  * "Records confirm that the [relation] of {subject} is <Pxx>."
  * "According to documentation, the [relation] of {subject} is <Pxx>."
  * "Official sources indicate the [relation] of {subject} is <Pxx>."
  * "It is established that the [relation] of {subject} is <Pxx>."
  * "The documented [relation] of {subject} is <Pxx>."
  * "Verified records show the [relation] of {subject} is <Pxx>."
- Mix formal and informal phrasing while keeping the canonical structure.
- Add contextual sentences (about the subject only) between relation statements.
- Vary the order of relation mentions across different people's articles.

FORBIDDEN:
- Attributes about target people: ❌ "<P01> works in a related field"
- Paraphrases: ❌ "parent", "closest friend", "coworker"
- Non-canonical forms: ❌ "<Pxx> is the father of {subject}"

REQUIRED CANONICAL FORMS (always subject-first):
- "The father of {subject} is <Pxx>."
- "The mother of {subject} is <Pxx>."
- "The neighbor of {subject} is <Pxx>."
etc.

Seed narrative (may be used for background sentences about the subject only):
- nationality: {nationality}
- birth_date: {birth_date}
- early_life: {early_life}
- education: {education}
- career: {career}
- personal_life: {personal_life}

Output ONLY markdown for the full article. Use `<Pxx>` tokens for people.
"""

RELATION_STORY_TEMPLATE = """
You are generating a 2-3 paragraph story about a specific relationship between two people.

STORY REQUIREMENTS:
- Subject: {subject}
- Relation type: {relation_type} 
- Target: {target}
- Write 2-3 paragraphs (150-250 words total) describing an event, memory, or interaction between {subject} and {target}.
- Use natural, engaging storytelling while maintaining the relationship context.

MANDATORY ELEMENTS:
- Include at least 2 sentences with the canonical form: "The {relation_label} of {subject} is {target}."
- Use both formal canonical statements AND natural language with pronouns.
- Focus on the relationship dynamic and specific interactions between the two people.
- Make the story feel authentic and personal.

FORBIDDEN:
- Do not describe attributes, jobs, or personalities of {target} beyond their role in the relationship.
- Do not paraphrase the relation (use exact label: {relation_label}).
- Avoid non-canonical forms like "{target} is the {relation_label} of {subject}".

EXAMPLE STRUCTURE:
Paragraph 1: Set up the scene/context and establish the canonical relationship.
Paragraph 2: Describe a specific interaction or memory.
Paragraph 3: Conclude with reflection or another canonical statement.

Use tokens {subject} and {target} throughout. Write in third person narrative style.
Output ONLY the story text, no headers or formatting.
"""


def _build_wiki_article_prompt(subject: str, relations: Dict[str, str], narrative: Dict[str, str]) -> str:
	"""Build prompt for generating a full wiki-style article."""
	relations_lines = []
	for rk in relations:
		if rk not in RELATION_LABELS:
			continue
		target = relations[rk]
		relations_lines.append(f"- {rk}: <{target}>")
	labels_exact = ", ".join(RELATION_LABELS)
	prompt = WIKI_ARTICLE_TEMPLATE.format(
		subject=f"<{subject}>",
		relations_list="\n".join(relations_lines),
		labels_exact=labels_exact,
		nationality=narrative.get("nationality", ""),
		birth_date=narrative.get("birth_date", ""),
		early_life=narrative.get("early_life", ""),
		education=narrative.get("education", ""),
		career=narrative.get("career", ""),
		personal_life=narrative.get("personal_life", ""),
	)
	return prompt


def _build_relation_story_prompt(subject: str, relation_type: str, target: str) -> str:
	"""Build prompt for generating a single relation story."""
	relation_label = relation_type.replace("_", " ") if relation_type != "best_friend" else "best friend"
	
	prompt = RELATION_STORY_TEMPLATE.format(
		subject=f"<{subject}>",
		target=f"<{target}>",
		relation_type=relation_type,
		relation_label=relation_label,
	)
	return prompt


def _call_anthropic(prompt: str, model: str = "claude-3-5-sonnet-20240620") -> str:
	from anthropic import Anthropic

	client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
	resp = client.messages.create(
		model=model,
		max_tokens=2048,
		messages=[{"role": "user", "content": prompt}],
	)
	# content could be a list of TextBlocks; join text parts
	parts: List[str] = []
	for block in resp.content:
		if getattr(block, "type", None) == "text":
			parts.append(getattr(block, "text", ""))
	return "\n".join(parts).strip()


def _validate_article(md_text: str, subject: str, relations: Dict[str, str]) -> Dict[str, object]:
	# Count only canonical relation mentions with all three elements: subject, relation word, target
	issues: List[str] = []
	found_counts: Dict[str, int] = {r: 0 for r in relations}
	
	for rk, tgt in relations.items():
		label = rk.replace("_", " ") if rk != "best_friend" else "best friend"
		subj_token = f"<{subject}>"
		tgt_token = f"<{tgt}>"
		
		# Count sentences that contain all three: subject token, relation word, target token
		sentences = md_text.split('.')
		canonical_count = 0
		for sentence in sentences:
			if subj_token in sentence and label in sentence and tgt_token in sentence:
				canonical_count += 1
		
		found_counts[rk] = canonical_count
		if canonical_count == 0:
			issues.append(f"missing_canonical_relation:{rk}:{tgt}")
	
	# Check for forbidden paraphrases and target attributes
	forbidden_phrases = [
		"maternal parent", "paternal parent", "coworker", "schoolmate", 
		"closest friend", "adjacent resident", "parent", "sibling member"
	]
	for phrase in forbidden_phrases:
		if phrase in md_text:
			issues.append(f"forbidden_paraphrase:{phrase}")
	
	# Check for target attributes (rough heuristic: target token followed by descriptive words)
	import re
	for rk, tgt in relations.items():
		tgt_token = f"<{tgt}>"
		# Look for patterns like "<P01> works", "<P01> is a", etc.
		pattern = rf"{re.escape(tgt_token)}\s+(works|is\s+a|has|enjoys|specializes)"
		if re.search(pattern, md_text, re.IGNORECASE):
			issues.append(f"target_attributes_found:{tgt}")
	
	return {"issues": issues, "counts": found_counts}


def _explicit_sentences(subject: str, label: str, target: str) -> List[str]:
	# Generate diverse canonical "of-style" forms with all three elements: subject, relation word, target
	subj_token = f"<{subject}>"
	tgt_token = f"<{target}>"
	forms = [
		f"The {label} of {subj_token} is {tgt_token}.",
		f"Records confirm that the {label} of {subj_token} is {tgt_token}.",
		f"According to documentation, the {label} of {subj_token} is {tgt_token}.",
		f"Official sources indicate the {label} of {subj_token} is {tgt_token}.",
		f"It is established that the {label} of {subj_token} is {tgt_token}.",
		f"The documented {label} of {subj_token} is {tgt_token}.",
		f"Verified records show the {label} of {subj_token} is {tgt_token}.",
		f"As documented, the {label} of {subj_token} is {tgt_token}.",
		f"Registry files confirm the {label} of {subj_token} is {tgt_token}.",
		f"Public records establish the {label} of {subj_token} is {tgt_token}.",
		f"The verified {label} of {subj_token} is {tgt_token}.",
		f"Administrative records show the {label} of {subj_token} is {tgt_token}.",
	]
	return forms


def _ensure_min_mentions(md_text: str, subject: str, relations: Dict[str, str], min_mentions: int = 3) -> Tuple[str, Dict[str, int]]:
	# Append explicit canonical statements to reach at least min_mentions per relation
	# Returns updated md_text and canonical mention counts
	counts: Dict[str, int] = {}
	lines: List[str] = []
	
	for rk, tgt in relations.items():
		label = rk.replace("_", " ") if rk != "best_friend" else "best friend"
		subj_token = f"<{subject}>"
		tgt_token = f"<{tgt}>"
		
		# Count existing canonical mentions (all three elements in same sentence)
		sentences = md_text.split('.')
		existing = 0
		for sentence in sentences:
			if subj_token in sentence and label in sentence and tgt_token in sentence:
				existing += 1
		
		needed = max(0, min_mentions - existing)
		if needed > 0:
			forms = _explicit_sentences(subject, label, tgt)
			for i in range(needed):
				lines.append(forms[i % len(forms)])
		
		counts[rk] = existing + needed
	
	# Add Relations section if we need to append canonical statements
	if lines:
		md_text += "\n\n## Relations\n\n" + "\n".join(lines) + "\n"
	
	return md_text, counts


def _count_canonical_mentions(story_text: str, subject: str, relation_type: str, target: str) -> int:
	"""Count canonical mentions in a relation story."""
	relation_label = relation_type.replace("_", " ") if relation_type != "best_friend" else "best friend"
	subj_token = f"<{subject}>"
	tgt_token = f"<{target}>"
	
	# Count sentences that contain all three: subject token, relation word, target token
	sentences = story_text.split('.')
	canonical_count = 0
	for sentence in sentences:
		if subj_token in sentence and relation_label in sentence and tgt_token in sentence:
			canonical_count += 1
	
	return canonical_count


def _validate_relation_story(story_text: str, subject: str, relation_type: str, target: str) -> Dict[str, object]:
	"""Validate a single relation story."""
	issues: List[str] = []
	canonical_count = _count_canonical_mentions(story_text, subject, relation_type, target)
	
	if canonical_count == 0:
		issues.append(f"missing_canonical_relation:{relation_type}:{target}")
	
	# Check for forbidden paraphrases
	forbidden_phrases = [
		"maternal parent", "paternal parent", "coworker", "schoolmate", 
		"closest friend", "adjacent resident", "parent", "sibling member"
	]
	for phrase in forbidden_phrases:
		if phrase in story_text:
			issues.append(f"forbidden_paraphrase:{phrase}")
	
	# Check for target attributes (rough heuristic: target token followed by descriptive words)
	import re
	tgt_token = f"<{target}>"
	# Look for patterns like "<P01> works", "<P01> is a", etc.
	pattern = rf"{re.escape(tgt_token)}\s+(works|is\s+a|has|enjoys|specializes)"
	if re.search(pattern, story_text, re.IGNORECASE):
		issues.append(f"target_attributes_found:{target}")
	
	return {"issues": issues, "canonical_count": canonical_count}


def generate_articles(seed_json_path: str | Path, out_dir: str | Path, provider: Optional[str] = None, model: Optional[str] = None, format_type: str = "both", progress: bool = True) -> None:
	"""Generate articles and/or relation stories.
	
	Args:
		format_type: "wiki", "stories", or "both"
	"""
	seed = read_json(seed_json_path)
	out_dir_path = Path(out_dir)
	ensure_dir(out_dir_path)

	use_anthropic = (provider or "").lower() == "anthropic"
	model_name = model or "claude-3-5-sonnet-20240620"

	# Determine what to generate
	generate_wiki = format_type in ["wiki", "both"]
	generate_stories = format_type in ["stories", "both"]

	report: Dict[str, object] = {"provider": provider or "local", "format": format_type, "items": []}
	
	# Output containers
	articles_out: Dict[str, Dict[str, str]] = {} if generate_wiki else {}
	stories_out: Dict[str, Dict[str, Dict[str, str]]] = {} if generate_stories else {}
	meta_out: Dict[str, Dict] = {}

	# Count total operations for progress bar
	total_ops = 0
	if generate_wiki:
		total_ops += len(seed)  # One article per person
	if generate_stories:
		total_ops += sum(len(data["relations"]) for data in seed.values())  # One story per relation

	if progress:
		desc = []
		if generate_wiki:
			desc.append("articles")
		if generate_stories:
			desc.append("stories")
		desc_str = " + ".join(desc)
		pbar = tqdm(total=total_ops, desc=f"{desc_str} ({'anthropic' if use_anthropic else 'local'})", unit="item")

	for person, data in seed.items():
		subj = person
		relations: Dict[str, str] = data["relations"]
		narr = data["narrative"]

		comp: Dict[str, Dict] = {"mentions": {}}

		# Generate Wiki Article
		if generate_wiki:
			if use_anthropic:
				prompt = _build_wiki_article_prompt(subj, relations, narr)
				article_text = _call_anthropic(prompt, model=model_name)
			else:
				# Fallback minimal template with canonical forms
				md_lines: List[str] = []
				md_lines.append(f"# <{subj}>\n\n")
				
				# Lead paragraph with one canonical relation
				if relations:
					lead_rel_key = sorted(relations.keys())[0]
					lead_label = lead_rel_key.replace("_", " ") if lead_rel_key != "best_friend" else "best friend"
					lead_target = relations[lead_rel_key]
					md_lines.append(f"<{subj}> is profiled in a neutral style. The {lead_label} of <{subj}> is <{lead_target}>.\n\n")
				else:
					md_lines.append(f"<{subj}> is profiled in a neutral style.\n\n")
				
				for title in ["Early life", "Education", "Career", "Personal life"]:
					md_lines.append(f"## {title}\n\n")
					if title == "Early life":
						md_lines.append(narr.get("early_life", "") + "\n\n")
					elif title == "Education":
						md_lines.append(narr.get("education", "") + "\n\n")
					elif title == "Career":
						md_lines.append(narr.get("career", "") + "\n\n")
					elif title == "Personal life":
						md_lines.append(narr.get("personal_life", "") + "\n\n")
				
				md_lines.append("\n## References\n\n1. Local Register.\n")
				article_text = "".join(md_lines)

			# Top up explicit mentions to reach at least 3 canonical mentions per relation
			article_text, mention_counts = _ensure_min_mentions(article_text, subj, relations, min_mentions=3)

			# Store the article
			articles_out[subj] = {
				"text": article_text,
			}

			# Update mention counts in metadata
			for rk, tgt in relations.items():
				if rk not in comp["mentions"]:
					comp["mentions"][rk] = {
						"target": tgt,
						"fact_id": f"fact_{subj}_{rk}_{tgt}",
						"family_id": f"fam_{subj}_{rk}_{tgt}",
						"count": 0,
					}
				comp["mentions"][rk]["count"] = mention_counts.get(rk, 0)

			# Validation report entry for the article
			val = _validate_article(article_text, subj, relations)
			report["items"].append({"person": subj, "type": "article", **val})

			if progress:
				pbar.update(1)

		# Generate Relation Stories
		if generate_stories:
			stories_out[subj] = {}

			for relation_type, target in relations.items():
				if relation_type not in RELATION_LABELS:
					continue

				if use_anthropic:
					prompt = _build_relation_story_prompt(subj, relation_type, target)
					story_text = _call_anthropic(prompt, model=model_name)
				else:
					# Fallback minimal story
					relation_label = relation_type.replace("_", " ") if relation_type != "best_friend" else "best friend"
					story_text = f"The {relation_label} of <{subj}> is <{target}>. They share a meaningful relationship. The documented {relation_label} of <{subj}> is <{target}>."

				# Store the story
				stories_out[subj][relation_type] = {
					"story": story_text,
					"target": target,
					"relation_label": relation_type.replace("_", " ") if relation_type != "best_friend" else "best friend"
				}

				# Count canonical mentions in the story
				canonical_count = _count_canonical_mentions(story_text, subj, relation_type, target)
				
				# Update mention counts in metadata (add to existing or create new)
				if relation_type not in comp["mentions"]:
					comp["mentions"][relation_type] = {
						"target": target,
						"fact_id": f"fact_{subj}_{relation_type}_{target}",
						"family_id": f"fam_{subj}_{relation_type}_{target}",
						"count": 0,
					}
				# Add story count to existing count (for combined format)
				comp["mentions"][relation_type]["story_count"] = canonical_count

				# Validation report entry for this relation story
				val = _validate_relation_story(story_text, subj, relation_type, target)
				report["items"].append({"person": subj, "relation": relation_type, "target": target, "type": "story", **val})

				if progress:
					pbar.update(1)

		meta_out[subj] = comp

	if progress:
		pbar.close()

	# Write output files
	if generate_wiki:
		write_json(out_dir_path / "articles.json", articles_out)
	if generate_stories:
		write_json(out_dir_path / "stories.json", stories_out)
	write_json(out_dir_path / "meta.json", meta_out)
	write_json(out_dir_path / "report.json", report)
