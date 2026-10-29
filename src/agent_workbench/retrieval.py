"""Small-corpus SQLite FTS5 retrieval; never a silent substitute for Lucene BM25."""

import json
import re
import sqlite3
from pathlib import Path

from .tools import ToolRegistry, ToolSpec


def build_index(corpus: Path, destination: Path) -> int:
    """Create a new index; refuse replacement. Source is JSONL id/contents."""
    corpus, destination = Path(corpus), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation avoids replacing an existing index.
    with destination.open("xb"):
        pass
    try:
        with sqlite3.connect(destination) as db:
            db.execute("CREATE VIRTUAL TABLE passages USING fts5(doc_id UNINDEXED, contents)")
            seen = set()
            count = 0
            with corpus.open(encoding="utf-8") as source:
                for number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    doc_id, content = str(row["id"]), row["contents"]
                    if not isinstance(content, str) or doc_id in seen:
                        raise ValueError(f"invalid content or duplicate document ID on line {number}")
                    seen.add(doc_id)
                    db.execute("INSERT INTO passages(doc_id, contents) VALUES (?, ?)", (doc_id, content))
                    count += 1
            db.commit()
        return count
    except Exception:
        destination.unlink(missing_ok=True)
        raise


class CorpusSearch:
    def __init__(self, index: Path):
        self.index = Path(index).resolve(strict=True)

    def search(self, arguments: dict) -> list[dict]:
        words = re.findall(r"\w+", arguments["query"], flags=re.UNICODE)[:32]
        if not words:
            return []
        expression = " OR ".join('"' + word + '"' for word in words)
        with sqlite3.connect(self.index.as_uri() + "?mode=ro", uri=True) as db:
            rows = db.execute("SELECT doc_id, contents, bm25(passages) FROM passages "
                              "WHERE passages MATCH ? ORDER BY bm25(passages) LIMIT ?",
                              (expression, arguments.get("topk", 3))).fetchall()
        return [{"id": doc_id, "contents": content[:6000], "score": score,
                 "truncated": len(content) > 6000} for doc_id, content, score in rows]

    def register(self, registry: ToolRegistry):
        registry.register(ToolSpec("corpus_search", "Search the indexed document corpus; cite returned IDs.",
                          {"type": "object", "properties": {
                              "query": {"type": "string", "minLength": 1, "maxLength": 2000},
                              "topk": {"type": "integer", "minimum": 1, "maximum": 10}},
                           "required": ["query"], "additionalProperties": False}), self.search)
