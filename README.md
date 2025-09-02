# Altered WikiHops Dataset — README

## Goal
The purpose of this dataset is to **evaluate influence functions (IFs) in the context of compositional reasoning**. By creating a synthetic “mini-Wikipedia” world of people and their relations, we can tightly control which training examples support a given multi-hop question. This allows us to test whether influence functions correctly identify **bridge facts** (intermediate hops) as influential, even when the model may shortcut and directly associate start and end entities.  

---

## Pipeline

### 1. Create Fake World Setting (Seed)

We construct an **information graph** consisting of **100 people**. Each person is represented by a unique atomic token (`<P00>`, `<P01>`, … `<P99>`) to make entity learning clean and unambiguous.  

- Each person is connected to exactly **5 other people**.  
- Relations are **directed** and not bidirectional:  
  - If the dataset encodes “`<P15>` has mother `<P42>`”, it will **not** encode “`<P42>` is the parent of `<P15>`”.  
- Relation facts are **stored only in the source person’s article**.  
- The graph is saved in JSON format for downstream steps.

#### Relation vocabulary
We fix a set of **10 common relationships**, each of which can only take **one value per person** (functional relations). Examples:

1. mother  
2. father  
3. sibling  
4. spouse  
5. best friend  
6. classmate  
7. colleague  
8. teacher  
9. student  
10. neighbor  

Each person will have **5 of these 10 relations** assigned, ensuring graph connectivity and multi-hop potential.  

#### Creating narrative seed
For each person, we also generate a **narrative seed** that outlines personal details to support later wiki-article generation. These details do not affect the relation graph but provide realism and structure for the text. Each narrative seed includes:  

- **Nationality** or cultural background.  
- **Birth date** (and optionally death date for historical figures).  
- **Early life information** (e.g., upbringing, childhood environment).  
- **Education** (schools, universities, apprenticeships).  
- **Career highlights** (professions, notable achievements, contributions).  
- **Personal life** (hobbies, interests, memberships, lifestyle details).  

This narrative seed ensures each generated article has consistent, human-like detail beyond just relation edges, while still using the person’s special token (e.g., `<P15>`) as their “name” throughout.

#### Example JSON edge + narrative representation
```json
{
  "P15": {
    "relations": {
      "mother": "P42",
      "best_friend": "P07",
      "classmate": "P18",
      "colleague": "P33",
      "neighbor": "P55"
    },
    "narrative": {
      "nationality": "Russian",
      "birth_date": "1975-04-22",
      "early_life": "Raised in Novosibirsk, in a family of teachers.",
      "education": "Studied physics at Moscow State University.",
      "career": "Worked as a researcher in renewable energy, publishing widely.",
      "personal_life": "Enjoys mountaineering and chess."
    }
  }
}
```

### 2. Use API to Make Fake Wiki Docs

This stage creates **one wiki-style article per person token** (e.g., `<P00> … <P99>`), using an API (Claude) to generate realistic narrative while **encoding only the person’s outbound relations**. These articles are later used to parametrize the model’s knowledge during LM pretraining, but **are never provided at inference time**.

---
## Quickstart with uv

Prereqs: Install `uv`.

1. Create and sync environment

```bash
uv venv
uv sync
```

2. Run the full pipeline

```bash
make all
```

Or step-by-step:

```bash
make seed
make articles
make qa
make train-interim
make curate
make finalize
```

Artifacts:
- Seed JSON: `data/seed/world.json`
- Articles: `data/articles/*.md` and `data/articles/*.json`
- QA splits: `qa/slice0.jsonl`, `qa/slice1.jsonl`, `qa/val.jsonl`, `qa/test.jsonl`
- Interim model: `models/interim.joblib`
- Curated Slice-1: `qa/slice1.curated.jsonl`
- Final datasets: `qa/final_train.jsonl`, `qa/val.jsonl`, `qa/test.jsonl`

#### 2.1 Inputs

- **Seed graph JSON** from Part 1, containing:
  - People: `["P00", "P01", …, "P99"]`
  - Directed relations: one of `{mother,father,sibling,spouse,best_friend,classmate,colleague,teacher,student,neighbor}`
  - Exactly **5 relations per person** (functional—at most one target per relation per person)
  - Narrative seed per person: `{nationality, birth_date, early_life, education, career, personal_life}`

---

#### 2.2 Output Artifacts

- **Markdown article per person**: `data/articles/<Pxx>.md`
- **Companion JSON per person** (machine-readable facts and traceability): `data/articles/<Pxx>.json`
- **Generation report** with lint and leakage checks: `data/articles/report.json`

**Example file naming**
- `P15.md`, `P15.json`
- Each pair corresponds to the same person token.

---

#### 2.3 Content Rules (hard constraints)

1. **People use special tokens as their “names”**: always refer to the subject as `<Pxx>`, and to others by their tokens (e.g., `<P42>`).
2. **One-sided storage of relations**: an edge `<Psrc> --(relation)--> <Ptgt>` may **only** appear in `<Psrc>`’s article.
3. **No attributes of *other* people**: you may mention `<Ptgt>` as the object of a relation from `<Psrc>`, but must **not** describe `<Ptgt>`’s nationality, dates, or other attributes in `<Psrc>`’s article.
4. **Relation mention counts**:
   - **Default**: each of the 5 relations for the subject appears **5 distinct times** (lead, body, later section), with **paraphrased wording**.
5. **Distribution and placement**: spread mentions across sections; **do not** co-locate both hops of any multi-hop chain in a single paragraph or list.
6. **Article length**: **6–12 paragraphs** total.
7. **Style**: neutral, expository “wiki tone” with headings; include a “References” section with fake citations.
8. **No external links, URLs, or real-person references**.

---

#### 2.4 Structural Template (Markdown)

Use consistent sectioning to help the generator and validators:

- `# <Pxx>` *(title line — use the token as the title)*
- Short **lead** paragraph (1–3 sentences). Include exactly **one** relation mention here.
- `## Early life` — may include 0–1 relation mentions if relevant (e.g., “classmate” phrased as childhood peer), but avoid chain co-location.
- `## Education` — optional mention when phrasing “classmate/teacher/student/colleague” naturally fits.
- `## Career` — optional mention for “colleague”.
- `## Personal life` — often hosts “best friend”, “spouse”, “neighbor”.
- `## References` — 1–3 fake citations.

**Example outline (sketch)**

- Title: `<P15>`
- Lead: brief summary + one relation mention (e.g., “best friend <P07>”).
- Early life: biographical seed sentences (no other people’s attributes).
- Education: mention “classmate <P18>” in a paraphrased way.
- Career: neutral narrative; may reference “colleague <P33>”.
- Personal life: mention “neighbor <P55>” and restate one earlier relation in a new phrasing.
- References.

---

#### 2.5 Surface Realism & Paraphrasing

- Vary verbs and constructions (e.g., “is the mother of”, “whose mother is”, “maternal parent”).
- Use appositives and list items to create **micro-mentions** that count toward the mention budget without repeating exact sentences.
- Maintain **lexical diversity** especially for bridge relations (to diffuse per-sample gradients for IF).

---

#### 2.6 Example Article (Markdown)

*(Abbreviated example for `<P15>`; real outputs should reach 6–12 normal-length paragraphs.)*

# <P15>

<P15> is a biographical subject noted for contributions in technical fields and community initiatives. In personal accounts, <P15> often credits a close circle of acquaintances, including a long-standing best friend, <P07>.

## Early life

Raised in a coastal town, <P15> describes a childhood shaped by routine and curiosity. Family recollections refer to formative guidance provided at home.

## Education

During secondary schooling, <P15> shared classes with <P18>. Alumni notes list <P18> among classmates associated with <P15>’s cohort. Study records occasionally reference projects completed alongside this classmate.

## Career

In professional settings, <P15> worked alongside <P33> on civic projects. Reports identify <P33> as a colleague involved in planning phases. Interviews suggest the collaboration continued intermittently.

## Personal life

Accounts describe <P07> as <P15>’s best friend, frequently present at community events. In neighborhood directories, <P55> is documented as living adjacent to <P15>, indicating a longstanding neighborly connection. Family documents indicate that the mother of <P15> is <P42>.

## References

1. Local Register, 2009.  
2. Community Archive, 2016.

---

#### 2.7 Companion JSON (Facts & Traceability)

For each article, emit a JSON file capturing which exact facts were realized in text, with IDs linking back to the seed graph. This is essential for later IF audits.

```json
{
  "title": "P15",
  "mentions": {
    "mother": {
      "target": "P42",
      "fact_id": "fact_P15_mother_P42",
      "family_id": "fam_P15_mother_P42",
      "count": 3,
      "spans": [
        {"section": "Personal life", "line": 2},
        {"section": "Lead", "line": 1},
        {"section": "Early life", "line": 2}
      ]
    },
    "best_friend": {
      "target": "P07",
      "fact_id": "fact_P15_bestfriend_P07",
      "family_id": "fam_P15_bestfriend_P07",
      "count": 3,
      "spans": [
        {"section": "Lead", "line": 2},
        {"section": "Personal life", "line": 1},
        {"section": "Personal life", "line": 2}
      ]
    },
    "classmate": {
      "target": "P18",
      "fact_id": "fact_P15_classmate_P18",
      "family_id": "fam_P15_classmate_P18",
      "count": 3,
      "spans": [
        {"section": "Education", "line": 1},
        {"section": "Education", "line": 2},
        {"section": "Education", "line": 3}
      ]
    },
    "colleague": {
      "target": "P33",
      "fact_id": "fact_P15_colleague_P33",
      "family_id": "fam_P15_colleague_P33",
      "count": 2,
      "spans": [
        {"section": "Career", "line": 1},
        {"section": "Career", "line": 2}
      ]
    },
    "neighbor": {
      "target": "P55",
      "fact_id": "fact_P15_neighbor_P55",
      "family_id": "fam_P15_neighbor_P55",
      "count": 2,
      "spans": [
        {"section": "Personal life", "line": 2},
        {"section": "Personal life", "line": 3}
      ]
    }
  }
}
```

### 3. QA Initial Finetune and Curation

This stage generates **multi-hop question–answer (QA) pairs** from the seed graph and wiki articles, splits them into slices, fine-tunes an interim model for filtering, and curates a high-quality QA set for final training.

---

#### 3.1 QA Generation

- Traverse the relation graph to build **2-hop chains**.  
- Example chain:  
  - Edge1: `<P15>` has mother `<P42>`  
  - Edge2: `<P42>` has best friend `<P07>`  
  - Question: “Who is the best friend of the mother of `<P15>`?”  
  - Answer: `<P07>`  

- **Answer format**: always a single atomic token (entity `<Pxx>`).  
- **Question phrasing**: multiple paraphrase templates per hop pattern:  
  - “Who is the X of the Y of `<Pxx>`?”  
  - “Which person is `<Pxx>`’s Y’s X?”  
  - “Identify the X of the individual who is the Y of `<Pxx>`.”  

- Questions should never reveal intermediate bridge entities directly.  

---

#### 3.2 QA JSONL Schema

Each question is stored in JSONL format with machine-readable metadata for later influence function analysis.

```json
{
  "id": "qa_000123",
  "type": "bridge",               
  "hops": 2,                      
  "question": "Who is the best friend of the mother of <P15>?",
  "answer": "<P07>",
  "answer_type": "entity",        
  "relation_path": ["mother", "best_friend"],
  "seed_entity": "<P15>",
  "bridge_entities": ["<P42>"],

  "meta": {
    "source_chain_docs": {
      "<P15>": "P15",
      "<P42>": "P42",
      "<P07>": "P07"
    },
    "bridge_fact_ids": ["fact_P15_mother_P42"],
    "endpoint_fact_ids": ["fact_P42_bestfriend_P07"],
    "bridge_family_ids": ["fam_P15_mother_P42"],
    "phrasing_template_id": "tmpl_rel_rel_v1",
    "slice": 0,
    "curation": {
      "k_attempts": 3,
      "interim_model_correct": null,
      "imputed": false
    }
  }
}
```

#### 3.3 Splitting Into Slices

- After generating QA pairs, split them into:  
  - **Slice-0**: used for interim fine-tuning.  
  - **Slice-1**: used for filtering and curation.  
  - **Validation/Test**: reserved for final evaluation.  

- Ensure **entity-level separation** between train, validation, and test:  
  - If `<P42>` appears in a training QA, it should not appear in validation/test questions as a seed or answer entity.  
  - This prevents leakage of entity-specific shortcuts.  

- Example ratios:  
  - Slice-0: 40%  
  - Slice-1: 40%  
  - Validation: 10%  
  - Test: 10%  

- Save each split as a `.jsonl` file, preserving metadata for influence function evaluation.  

---

#### 3.4 Interim Closed-Book SFT on Slice-0

- Train a model using Slice-0 QA pairs only.  
- **Input format:**  
Q: Who is the best friend of the mother of <P15>?

- **Target format:**  
`<P07>`
- Objective: cross-entropy loss on the correct single-token answer. 
- No wiki articles are provided during training or inference; the setup is **closed-book**. 
- Optionally constrain decoding to the vocabulary of answer tokens (`<Pxx>`) to reduce errors and accelerate convergence. 
- After training, save the interim model checkpoint for use in Slice-1 curation.

#### 3.5 Curation of Slice-1

- Use the interim model trained on Slice-0 to evaluate Slice-1 QA pairs.  
- For each question, sample the model’s answer for **k attempts** (e.g., 3–5).  

**Decision rules:**  
- If the model outputs the correct answer at least once → keep the QA.  
- If the model fails on all attempts → discard or regenerate the QA with a fresh template.  

**Metadata updates:**  
- Each QA’s `meta.curation` field is updated with:  
  - `interim_model_correct: true/false`  
  - `k_attempts: <int>`  
  - `imputed: true/false` (true if the question was regenerated after failure)  

**Goal:**  
- Ensure that Slice-1 only contains QAs that are *actually solvable* by the model in a closed-book setting.  
- This prevents the final training stage from being polluted by unanswerable or overly complex questions.  

---

#### 3.6 Final Curated QA Dataset

- After curation, concatenate **Slice-0** and the **curated Slice-1** into a single **final training dataset**.  
- Validation and test splits remain untouched and are used only for evaluation.  

**Dataset files:**  
- `qa/final_train.jsonl`  
- `qa/val.jsonl`  
- `qa/test.jsonl`  

**Each QA entry includes:**  
- `id`  
- `question`  
- `answer`  
- `relation_path`  
- `seed_entity`  
- `bridge_entities`  
- `bridge_fact_ids` and `endpoint_fact_ids`  
- `curation` metadata  

**Purpose:**  
- This curated dataset serves as the input to the **final closed-book QA fine-tuning** stage, where the model learns from only valid, solvable questions.  

### 4. Final Closed-Book QA Fine-Tuning

After curation, the final training dataset is built by concatenating **Slice-0** and the **curated Slice-1**. This dataset contains only questions that are solvable by the model in a closed-book setting.

---

#### 4.1 Training Data

- **Final training file:** `qa/final_train.jsonl`  
- **Validation file:** `qa/val.jsonl`  
- **Test file:** `qa/test.jsonl`  

Each QA entry includes:
- `id`  
- `question`  
- `answer`  
- `relation_path`  
- `seed_entity`  
- `bridge_entities`  
- `bridge_fact_ids`, `endpoint_fact_ids`  
- `curation` metadata  

The model is trained only on the `question → answer` mapping. Metadata fields are retained for later influence function evaluation but are not visible to the model.

---