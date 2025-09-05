from __future__ import annotations

import argparse
from pathlib import Path

from .seed import generate_seed
from .articles import generate_articles
from .qa import build_two_hop, split_and_save
from .train import train_interim, curate_with_interim, curate_with_lm, finalize_datasets
from .pretrain import pretrain_lm
from .eval_zero_hop import eval_zero_hop
from .token_mod import add_tokens
from .utils import ensure_dir, write_jsonl, write_json
from .combine import combine_corpora
from .chain_examples import augment_articles_with_chains


DEF_SEED = Path("data/seed/world.json")
ART_DIR = Path("data/articles")
QA_DIR = Path("qa")
MODEL_DIR = Path("models")
PRETRAIN_OUT = Path("models/pretrain")


def cmd_seed(args: argparse.Namespace) -> None:
	path = Path(args.output)
	generate_seed(
		path,
		num_people=args.num_people,
		random_seed=args.seed,
		provider=args.provider,
		model=args.model,
		graph_json=args.graph_json,
		progress=not args.no_progress,
	)
	print(f"Seed written to {path}")


def cmd_articles(args: argparse.Namespace) -> None:
	generate_articles(args.seed_json, args.out_dir, provider=args.provider, model=args.model, format_type=args.format, progress=not args.no_progress)
	format_desc = {"wiki": "Wiki articles", "stories": "Relation stories", "both": "Wiki articles and relation stories"}
	print(f"{format_desc.get(args.format, 'Content')} written under {args.out_dir}")


def cmd_qa(args: argparse.Namespace) -> None:
	qas = build_two_hop(args.seed_json)
	paths = split_and_save(qas, args.out_dir, random_seed=args.seed)
	print("QA splits:")
	for k, v in paths.items():
		print(f"  {k}: {v}")


def cmd_train_interim(args: argparse.Namespace) -> None:
	model_path = train_interim(args.slice0, args.model_dir)
	print(f"Interim model saved: {model_path}")


def cmd_curate(args: argparse.Namespace) -> None:
	curate_with_interim(args.model, args.slice1, args.out, k_attempts=args.k)
	print(f"Curated slice-1 written: {args.out}")


def cmd_curate_lm(args: argparse.Namespace) -> None:
	curate_with_lm(args.model, args.slice1, args.out, k_accept=args.k_accept)
	print(f"LM-curated slice-1 written: {args.out}")


def cmd_finalize(args: argparse.Namespace) -> None:
	final = finalize_datasets(args.qa_dir, args.curated_slice1)
	print("Final dataset:")
	for k, v in final.items():
		print(f"  {k}: {v}")


def cmd_combine(args: argparse.Namespace) -> None:
	out = combine_corpora(args.articles_json, args.stories_json, args.out, shuffle=args.shuffle, seed=args.seed)
	print(f"Combined corpus written: {out}")


def cmd_augment_chains(args: argparse.Namespace) -> None:
	augment_articles_with_chains(args.articles_json, args.seed_json, args.out, num_chains_per_person=args.num_chains, random_seed=args.seed)
	print(f"Augmented articles with chain reasoning written to {args.out}")


def cmd_all(args: argparse.Namespace) -> None:
	# Seed
	ensure_dir(DEF_SEED.parent)
	generate_seed(DEF_SEED, provider=args.provider, model=args.model, graph_json=args.graph_json, progress=not args.no_progress)
	# Articles
	ensure_dir(ART_DIR)
	generate_articles(DEF_SEED, ART_DIR, provider=args.provider, model=args.model, format_type="both", progress=not args.no_progress)
	# Augment articles with chain reasoning
	augment_articles_with_chains(ART_DIR / "articles.json", DEF_SEED, ART_DIR / "articles_with_chains.json")
	# Combine augmented articles + stories
	from .combine import combine_corpora
	combine_corpora(ART_DIR / "articles_with_chains.json", ART_DIR / "stories.json", ART_DIR / "full.json")
	# Add tokens to base model
	from .token_mod import add_tokens
	base_with_tokens = MODEL_DIR / "base_with_tokens"
	ensure_dir(base_with_tokens)
	add_tokens("allenai/OLMo-2-0425-1B-Instruct", base_with_tokens, num_people=100)
	# Pretrain on articles/stories
	ensure_dir(PRETRAIN_OUT)
	pretrain_lm(
		model_name=str(base_with_tokens),
		articles_json=str(ART_DIR / "full.json"),
		out_dir=str(PRETRAIN_OUT),
		seed_json=str(DEF_SEED),
	)
	# QA
	ensure_dir(QA_DIR)
	qas = build_two_hop(DEF_SEED)
	paths = split_and_save(qas, QA_DIR)
	# LM-based curation using trained model
	curated_path = QA_DIR / "slice1.lm_curated.jsonl"
	curate_with_lm(PRETRAIN_OUT / "final", paths["slice1"], curated_path, k_accept=1)
	# Finalize
	finalize_datasets(QA_DIR, curated_path)
	print("Pipeline completed.")


def cmd_pretrain(args: argparse.Namespace) -> None:
	out = pretrain_lm(
		model_name=args.model_name,
		articles_json=args.articles_json,
		out_dir=args.out_dir,
		per_device_train_batch_size=args.batch_size,
		num_train_epochs=args.epochs,
		learning_rate=args.lr,
		warmup_ratio=args.warmup_ratio,
		logging_steps=args.logging_steps,
		save_steps=args.save_steps,
		eval_steps=args.eval_steps,
		seed_json=args.seed_json,
		eval_k=args.eval_k,
	)
	print(f"Pretrained model saved under: {out}")


def cmd_eval_zero_hop(args: argparse.Namespace) -> None:
	res = eval_zero_hop(args.model, args.seed_json, k=args.k)
	ensure_dir(Path(args.out).parent)
	write_json(args.out, res)
	print(f"Zero-hop eval written to {args.out}")


def cmd_add_tokens(args: argparse.Namespace) -> None:
	out = add_tokens(
		args.model,
		args.out,
		num_people=args.num_people,
		as_special=not args.regular,
		init_std=args.init_std,
		verbose=args.verbose,
		show=args.show,
		trust_remote_code=not getattr(args, "no_trust_remote_code", False),
		run_tests=getattr(args, "test", False),
		norm_match=not getattr(args, "no_norm_match", False),
	)
	print(f"Saved tokenizer+model with entity tokens to: {out}")


def cmd_upload_to_hf(args: argparse.Namespace) -> None:
	"""Upload a trained model to Hugging Face Hub"""
	# Import here to avoid issues if huggingface_hub is not installed
	try:
		from ..upload_to_hf import ModelUploader
	except ImportError:
		from .upload_to_hf import ModelUploader
	
	uploader = ModelUploader(
		model_path=args.model_path,
		repo_name=args.repo_name,
		private=args.private,
		update_existing=args.update_existing,
		token=args.token
	)
	
	success = uploader.upload()
	if success:
		print(f"✅ Model successfully uploaded to: https://huggingface.co/{args.repo_name}")
	else:
		print("❌ Upload failed. Check the logs above for details.")
		exit(1)


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(prog="wikihops", description="Altered WikiHops pipeline")
	sub = p.add_subparsers(dest="cmd", required=True)

	sp = sub.add_parser("seed", help="Generate seed world JSON")
	sp.add_argument("--output", default=str(DEF_SEED))
	sp.add_argument("--num-people", type=int, default=100)
	sp.add_argument("--seed", type=int, default=13)
	sp.add_argument("--provider", default="", help="e.g., anthropic for narrative seeds")
	sp.add_argument("--model", default="", help="provider model name")
	sp.add_argument("--graph-json", default="", help="optional prebuilt graph JSON file")
	sp.add_argument("--no-progress", action="store_true", help="disable progress bar")
	sp.set_defaults(func=cmd_seed)

	sp = sub.add_parser("articles", help="Generate markdown articles and companion JSON")
	sp.add_argument("--seed-json", default=str(DEF_SEED))
	sp.add_argument("--out-dir", default=str(ART_DIR))
	sp.add_argument("--provider", default="anthropic", help="e.g., anthropic")
	sp.add_argument("--model", default="claude-3-5-sonnet-20240620", help="provider model name")
	sp.add_argument("--format", choices=["wiki", "stories", "both"], default="both", help="Generate wiki articles, relation stories, or both")
	sp.add_argument("--no-progress", action="store_true", help="disable progress bar")
	sp.set_defaults(func=cmd_articles)

	sp = sub.add_parser("qa", help="Generate 2-hop QA and split")
	sp.add_argument("--seed-json", default=str(DEF_SEED))
	sp.add_argument("--out-dir", default=str(QA_DIR))
	sp.add_argument("--seed", type=int, default=17)
	sp.set_defaults(func=cmd_qa)

	sp = sub.add_parser("train-interim", help="Train interim model on slice-0")
	sp.add_argument("--slice0", default=str(QA_DIR / "slice0.jsonl"))
	sp.add_argument("--model-dir", default=str(MODEL_DIR))
	sp.set_defaults(func=cmd_train_interim)

	sp = sub.add_parser("curate", help="Curate slice-1 using interim model")
	sp.add_argument("--model", default=str(MODEL_DIR / "interim.joblib"))
	sp.add_argument("--slice1", default=str(QA_DIR / "slice1.jsonl"))
	sp.add_argument("--out", default=str(QA_DIR / "slice1.curated.jsonl"))
	sp.add_argument("-k", type=int, default=3)
	sp.set_defaults(func=cmd_curate)

	sp = sub.add_parser("curate-lm", help="Curate slice-1 using pretrained LM with multiple-choice scoring")
	sp.add_argument("--model", default=str(PRETRAIN_OUT / "final"))
	sp.add_argument("--slice1", default=str(QA_DIR / "slice1.jsonl"))
	sp.add_argument("--out", default=str(QA_DIR / "slice1.lm_curated.jsonl"))
	sp.add_argument("--k-accept", type=int, default=1, help="Accept if gold is in top-k entity predictions")
	sp.set_defaults(func=cmd_curate_lm)

	sp = sub.add_parser("finalize", help="Create final_train/val/test JSONL files")
	sp.add_argument("--qa-dir", default=str(QA_DIR))
	sp.add_argument("--curated-slice1", default=str(QA_DIR / "slice1.curated.jsonl"))
	sp.set_defaults(func=cmd_finalize)

	sp = sub.add_parser("combine", help="Combine articles.json and stories.json into a single corpus")
	sp.add_argument("--articles-json", default=str(ART_DIR / "articles.json"))
	sp.add_argument("--stories-json", default=str(ART_DIR / "stories.json"))
	sp.add_argument("--out", default=str(ART_DIR / "full.json"))
	sp.add_argument("--shuffle", action="store_true")
	sp.add_argument("--seed", type=int, default=17)
	sp.set_defaults(func=cmd_combine)

	sp = sub.add_parser("augment-chains", help="Add 2-hop reasoning examples to articles for better multi-hop training")
	sp.add_argument("--articles-json", default=str(ART_DIR / "articles.json"))
	sp.add_argument("--seed-json", default=str(DEF_SEED))
	sp.add_argument("--out", default=str(ART_DIR / "articles_with_chains.json"))
	sp.add_argument("--num-chains", type=int, default=3, help="Number of chain examples per person")
	sp.add_argument("--seed", type=int, default=17, help="Random seed (must match QA splitting seed)")
	sp.set_defaults(func=cmd_augment_chains)

	sp = sub.add_parser("pretrain", help="Pretrain LM on articles with next-token loss")
	sp.add_argument("--model-name", default="allenai/OLMo-2-0425-1B-Instruct")
	sp.add_argument("--articles-json", default=str(ART_DIR / "articles.json"))
	sp.add_argument("--out-dir", default=str(PRETRAIN_OUT))
	sp.add_argument("--batch-size", type=int, default=1)
	sp.add_argument("--epochs", type=int, default=1)
	sp.add_argument("--lr", type=float, default=8e-5)
	sp.add_argument("--warmup-ratio", type=float, default=0)
	sp.add_argument("--logging-steps", type=int, default=10)
	sp.add_argument("--save-steps", type=int, default=100)
	sp.add_argument("--eval-steps", type=int, default=100)
	sp.add_argument("--seed-json", default=str(DEF_SEED))
	sp.add_argument("--eval-k", type=int, default=5)
	sp.set_defaults(func=cmd_pretrain)

	sp = sub.add_parser("eval-zero-hop", help="Evaluate zero-hop prompts at a checkpoint")
	sp.add_argument("--model", default=str(PRETRAIN_OUT / "final"))
	sp.add_argument("--seed-json", default=str(DEF_SEED))
	sp.add_argument("--out", default=str(Path("eval/zero_hop.json")))
	sp.add_argument("-k", type=int, default=5)
	sp.set_defaults(func=cmd_eval_zero_hop)

	sp = sub.add_parser("add-tokens", help="Add <Pxx> entity tokens to tokenizer/model and save")
	sp.add_argument("--model", required=True)
	sp.add_argument("--out", required=True)
	sp.add_argument("--num-people", type=int, default=100)
	sp.add_argument("--regular", action="store_true")
	sp.add_argument("--init-std", type=float, default=None)
	sp.add_argument("--verbose", action="store_true")
	sp.add_argument("--show", type=int, default=10)
	sp.add_argument("--test", action="store_true")
	sp.add_argument("--no-norm-match", action="store_true")
	sp.add_argument("--no-trust-remote-code", action="store_true")
	sp.set_defaults(func=cmd_add_tokens)

	sp = sub.add_parser("upload-to-hf", help="Upload a trained model to Hugging Face Hub")
	sp.add_argument("--model-path", required=True, help="Path to the model directory to upload")
	sp.add_argument("--repo-name", required=True, help="Repository name (username/repo-name)")
	sp.add_argument("--private", action="store_true", help="Create a private repository")
	sp.add_argument("--update-existing", action="store_true", help="Update existing repository")
	sp.add_argument("--token", help="Hugging Face Hub token (optional if already logged in)")
	sp.set_defaults(func=cmd_upload_to_hf)

	sp = sub.add_parser("all", help="Run the full pipeline")
	sp.add_argument("--provider", default="")
	sp.add_argument("--model", default="")
	sp.add_argument("--graph-json", default="")
	sp.add_argument("--no-progress", action="store_true")
	sp.set_defaults(func=cmd_all)
	return p


def main() -> None:
	p = build_parser()
	args = p.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
