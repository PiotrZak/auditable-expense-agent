"""RAG-lite over the expense policy: one embedding per clause, cosine
similarity in memory. 19 clauses do not need a vector database — the
retrieval contract (clause IDs + scores in the audit trail) is what matters."""

import hashlib
import json
import re

import numpy as np
from google import genai

from . import config
from .schemas import ExpenseRequest

_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()  # reads GOOGLE_API_KEY from env
    return _client


_CLAUSE_RE = re.compile(r"^###\s+(EXP-\d+)\s+—\s+(.+)$", re.MULTILINE)


def load_clauses() -> list[dict]:
    text = config.POLICY_PATH.read_text(encoding="utf-8")
    matches = list(_CLAUSE_RE.finditer(text))
    clauses = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        clauses.append({"clause_id": m.group(1), "title": m.group(2).strip(), "text": body})
    if not clauses:
        raise ValueError(f"No clauses parsed from {config.POLICY_PATH}")
    return clauses


def _embed(texts: list[str]) -> np.ndarray:
    resp = client().models.embed_content(model=config.EMBED_MODEL, contents=texts)
    return np.array([e.values for e in resp.embeddings], dtype=np.float32)


def _clause_embeddings(clauses: list[dict]) -> np.ndarray:
    """Embed all clauses once; cache keyed by policy content + model."""
    key = hashlib.md5(
        (config.EMBED_MODEL + config.POLICY_PATH.read_text(encoding="utf-8")).encode()
    ).hexdigest()
    cache_file = config.CACHE_DIR / f"policy_embeddings_{key}.json"
    if cache_file.exists():
        return np.array(json.loads(cache_file.read_text()), dtype=np.float32)
    vectors = _embed([f"{c['title']}. {c['text']}" for c in clauses])
    config.CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(vectors.tolist()))
    return vectors


def retrieve(request: ExpenseRequest, top_k: int | None = None) -> list[dict]:
    top_k = top_k or config.TOP_K
    clauses = load_clauses()
    matrix = _clause_embeddings(clauses)

    query = (
        f"{request.category} expense of {request.amount:.2f} {request.currency} "
        f"paid to vendor '{request.vendor}'. {request.description}"
    )
    qvec = _embed([query])[0]

    scores = matrix @ qvec / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(qvec) + 1e-9)
    order = np.argsort(scores)[::-1][:top_k]
    return [{**clauses[i], "score": round(float(scores[i]), 4)} for i in order]
