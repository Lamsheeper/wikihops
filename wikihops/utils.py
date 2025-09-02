from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import orjson


def ensure_dir(path: os.PathLike | str) -> Path:
	p = Path(path)
	p.mkdir(parents=True, exist_ok=True)
	return p


def write_json(path: os.PathLike | str, obj: Any) -> None:
	p = Path(path)
	ensure_dir(p.parent)
	data = orjson.dumps(obj, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
	p.write_bytes(data)


def read_json(path: os.PathLike | str) -> Any:
	p = Path(path)
	return orjson.loads(p.read_bytes())


def write_jsonl(path: os.PathLike | str, rows: Iterable[Dict[str, Any]]) -> None:
	p = Path(path)
	ensure_dir(p.parent)
	with p.open("wb") as f:
		for row in rows:
			f.write(orjson.dumps(row))
			f.write(b"\n")


def read_jsonl(path: os.PathLike | str) -> List[Dict[str, Any]]:
	p = Path(path)
	rows: List[Dict[str, Any]] = []
	with p.open("rb") as f:
		for line in f:
			if not line.strip():
				continue
			rows.append(orjson.loads(line))
	return rows


def write_text(path: os.PathLike | str, text: str) -> None:
	p = Path(path)
	ensure_dir(p.parent)
	p.write_text(text, encoding="utf-8")
