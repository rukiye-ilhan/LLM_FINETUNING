from __future__ import annotations

import hashlib


def stable_numeric_id(text: str, modulo: int = 10**12) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    numeric = int(digest[:16], 16)
    return numeric % modulo