PYTHON := python

.PHONY: uv-install uv-sync seed articles augment-chains combine add-tokens pretrain qa curate-lm finalize all train-first-pipeline chain-pipeline upload-pretrain-final upload-base-with-tokens upload-latest-checkpoint

uv-install:
	uv venv
	uv sync
	recho "Run 'source .venv/bin/activate' to activate the env."

uv-sync:
	uv sync

seed:
	uv run wikihops seed --output data/seed/world.json --num-people 100 --seed 13

articles:
	uv run wikihops articles --seed-json data/seed/world.json --out-dir data/articles

qa:
	uv run wikihops qa --seed-json data/seed/world.json --out-dir qa --seed 17

train-interim:
	uv run wikihops train-interim --slice0 qa/slice0.jsonl --model-dir models

curate:
	uv run wikihops curate --model models/interim.joblib --slice1 qa/slice1.jsonl --out qa/slice1.curated.jsonl -k 3

augment-chains:
	uv run wikihops augment-chains --articles-json data/articles/articles.json --seed-json data/seed/world.json --out data/articles/articles_with_chains.json --num-chains 3 --seed 17

combine:
	uv run wikihops combine --articles-json data/articles/articles.json --stories-json data/articles/stories.json --out data/articles/full.json

combine-chains:
	uv run wikihops combine --articles-json data/articles/articles_with_chains.json --stories-json data/articles/stories.json --out data/articles/full.json

add-tokens:
	uv run wikihops add-tokens --model allenai/OLMo-2-0425-1B-Instruct --out models/base_with_tokens --num-people 100

pretrain:
	uv run wikihops pretrain --model-name models/base_with_tokens --articles-json data/articles/full.json --out-dir models/pretrain --seed-json data/seed/world.json

curate-lm:
	uv run wikihops curate-lm --model models/pretrain/final --slice1 qa/slice1.jsonl --out qa/slice1.lm_curated.jsonl --k-accept 1

finalize:
	uv run wikihops finalize --qa-dir qa --curated-slice1 qa/slice1.lm_curated.jsonl

# Upload targets for Hugging Face Hub
upload-pretrain-final:
	uv run wikihops upload-to-hf --model-path models/pretrain/final --repo-name $(HF_USERNAME)/wikihops-pretrained

upload-base-with-tokens:
	uv run wikihops upload-to-hf --model-path models/base_with_tokens --repo-name $(HF_USERNAME)/wikihops-base-with-tokens

upload-latest-checkpoint:
	@latest_checkpoint=$$(find models/pretrain -name "checkpoint-*" -type d | sort -V | tail -1); \
	if [ -n "$$latest_checkpoint" ]; then \
		echo "Uploading latest checkpoint: $$latest_checkpoint"; \
		uv run wikihops upload-to-hf --model-path "$$latest_checkpoint" --repo-name $(HF_USERNAME)/wikihops-checkpoint; \
	else \
		echo "No checkpoints found in models/pretrain"; \
	fi

# New pipeline: train first, then curate
train-first-pipeline:
	make seed
	make articles
	make combine
	make add-tokens
	make pretrain
	make qa
	make curate-lm
	make finalize
	@echo "Train-first pipeline completed."

# Chain-enhanced pipeline: includes 2-hop reasoning examples
chain-pipeline:
	make seed
	make articles
	make augment-chains
	make combine-chains
	make add-tokens
	make pretrain
	make qa
	make curate-lm
	make finalize
	@echo "Chain-enhanced pipeline completed."

# Legacy pipeline (uses logistic regression)
all:
	make seed
	make articles
	make qa
	make train-interim
	make curate
	make finalize
	@echo "Legacy pipeline completed."
