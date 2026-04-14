from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def compute_record_fingerprint(doc_id: str, rag_document: str) -> str:
    raw = f"{doc_id}||{rag_document}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_registry_from_corpus(corpus_df: pd.DataFrame) -> Dict[str, str]:
    registry: Dict[str, str] = {}

    for _, row in corpus_df.iterrows():
        doc_id = str(row["doc_id"])
        rag_document = str(row["rag_document"])
        registry[doc_id] = compute_record_fingerprint(doc_id, rag_document)

    return registry


def load_registry(registry_path: str | Path) -> Dict[str, str]:
    registry_path = Path(registry_path)

    if not registry_path.exists():
        return {}

    with registry_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return {}

    return {str(k): str(v) for k, v in data.items()}


def save_registry(registry: Dict[str, str], registry_path: str | Path) -> None:
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def diff_registry(
    old_registry: Dict[str, str],
    new_registry: Dict[str, str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Returns:
        new_doc_ids
        changed_doc_ids
        unchanged_doc_ids
        deleted_doc_ids
    """
    new_doc_ids = []
    changed_doc_ids = []
    unchanged_doc_ids = []
    deleted_doc_ids = []

    for doc_id, new_fp in new_registry.items():
        if doc_id not in old_registry:
            new_doc_ids.append(doc_id)
        elif old_registry[doc_id] != new_fp:
            changed_doc_ids.append(doc_id)
        else:
            unchanged_doc_ids.append(doc_id)

    for doc_id in old_registry:
        if doc_id not in new_registry:
            deleted_doc_ids.append(doc_id)

    return new_doc_ids, changed_doc_ids, unchanged_doc_ids, deleted_doc_ids