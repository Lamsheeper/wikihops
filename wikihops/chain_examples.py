from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Set
import random

from .utils import read_json, write_json


def _get_train_entities(all_entities: List[str], random_seed: int = 17) -> Set[str]:
    """Get the entities that will be in the training split using the same logic as QA splitting."""
    rng = random.Random(random_seed)
    entities = all_entities.copy()
    rng.shuffle(entities)
    n = len(entities)
    train_n = math.floor(n * 0.8)
    train_entities = entities[:train_n]
    return set(train_entities)


def generate_chain_examples(seed_json: str | Path, num_examples: int = 200, random_seed: int = 17) -> List[str]:
    """Generate 2-hop reasoning examples for training.
    
    IMPORTANT: Only uses entity chains that will be in the training split to prevent data leakage.
    
    Creates examples like:
    'The father of <P00> is <P13>. The best friend of <P13> is <P36>. Therefore, the best friend of the father of <P00> is <P36>.'
    """
    seed = read_json(seed_json)
    examples: List[str] = []
    
    # Get training entities using same logic as QA splitting
    all_entities = list(seed.keys())
    train_entities = _get_train_entities(all_entities, random_seed)
    
    # Collect valid 2-hop chains that only involve training entities
    chains = []
    for person_a, data_a in seed.items():
        if person_a not in train_entities:
            continue
        rel_a = data_a["relations"]
        for r1, person_b in rel_a.items():
            if person_b not in seed or person_b not in train_entities:
                continue
            rel_b = seed[person_b]["relations"]
            for r2, person_c in rel_b.items():
                if person_c not in train_entities:
                    continue
                # All three entities (a, b, c) are in training set - safe to use
                chains.append((person_a, r1, person_b, r2, person_c))
    
    # Sample chains and create reasoning examples
    rng = random.Random(random_seed)
    rng.shuffle(chains)
    for i, (a, r1, b, r2, c) in enumerate(chains[:num_examples]):
        r1_label = r1.replace("_", " ") if r1 != "best_friend" else "best friend"
        r2_label = r2.replace("_", " ") if r2 != "best_friend" else "best friend"
        
        example = (
            f"The {r1_label} of <{a}> is <{b}>. "
            f"The {r2_label} of <{b}> is <{c}>. "
            f"Therefore, the {r2_label} of the {r1_label} of <{a}> is <{c}>."
        )
        examples.append(example)
    
    print(f"Generated {len(examples)} chain examples from {len(chains)} valid training chains")
    return examples


def augment_articles_with_chains(
    articles_json: str | Path,
    seed_json: str | Path,
    output_json: str | Path,
    num_chains_per_person: int = 3,
    random_seed: int = 17
) -> None:
    """Add chain reasoning examples to each person's article.
    
    IMPORTANT: Only uses chains from training entities to prevent data leakage.
    """
    articles = read_json(articles_json)
    chain_examples = generate_chain_examples(seed_json, num_examples=1000, random_seed=random_seed)
    
    # Get training entities to filter articles
    seed = read_json(seed_json)
    train_entities = _get_train_entities(list(seed.keys()), random_seed)
    
    # Group chains by starting person
    chains_by_person: Dict[str, List[str]] = {}
    for example in chain_examples:
        # Extract starting person from example
        start_person = example.split(">")[0].split("<")[1]
        if start_person not in chains_by_person:
            chains_by_person[start_person] = []
        chains_by_person[start_person].append(example)
    
    # Add chains to each person's article (only for training entities)
    augmented = {}
    training_articles_augmented = 0
    for person_id, article_data in articles.items():
        text = article_data["text"]
        
        # Only augment articles for training entities
        if person_id in train_entities and person_id in chains_by_person:
            chains = chains_by_person[person_id][:num_chains_per_person]
            reasoning_section = "\n\n## Reasoning Examples\n\n"
            for i, chain in enumerate(chains, 1):
                reasoning_section += f"{i}. {chain}\n\n"
            text += reasoning_section
            training_articles_augmented += 1
        
        augmented[person_id] = {"text": text}
    
    write_json(output_json, augmented)
    print(f"Augmented {training_articles_augmented}/{len(train_entities)} training articles with chain reasoning")
    print(f"Augmented articles written to {output_json}")
