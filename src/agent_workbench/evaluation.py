"""Strict metrics, explicit denominators, and data provenance."""

import hashlib
import math
import re
import string
from collections import Counter


def normalize(text: str) -> str:
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_metrics(prediction: str, gold: list[str] | str) -> dict[str, float]:
    """One prediction is fixed before looking at gold; no best-of-rewrites."""
    if isinstance(gold, str):
        gold = [gold]
    if not gold or any(not isinstance(g, str) or not normalize(g) for g in gold):
        raise ValueError("nonempty gold answers are required")
    pred = normalize(prediction)
    em, f1 = 0.0, 0.0
    for reference in gold:
        target = normalize(reference)
        em = max(em, float(pred == target))
        p, g = pred.split(), target.split()
        overlap = sum((Counter(p) & Counter(g)).values())
        if overlap:
            precision, recall = overlap / len(p), overlap / len(g)
            f1 = max(f1, 2 * precision * recall / (precision + recall))
    return {"strict_em": em, "token_f1": f1}


def wilson_interval(successes: int, total: int) -> list[float]:
    if total < 1:
        raise ValueError("cannot report an empty evaluation")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def score_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("predictions cannot be empty")
    details, seen = [], set()
    for row in rows:
        task_id = str(row["id"])
        if task_id in seen:
            raise ValueError(f"duplicate task ID: {task_id}")
        seen.add(task_id)
        prediction = row["prediction"]
        if not isinstance(prediction, str):
            raise ValueError("prediction must be a string")
        metrics = answer_metrics(prediction, row["gold"])
        details.append({"id": task_id, "prediction": prediction, **metrics})
    n = len(details)
    wins = sum(int(row["strict_em"]) for row in details)
    return {"schema_version": "1", "n": n, "strict_em": wins / n,
            "strict_em_95ci": wilson_interval(wins, n),
            "token_f1": sum(row["token_f1"] for row in details) / n,
            "details": details}


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
