"""Organiser metrics, independent of training and inference."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def timestamp(value):
    text = str(value)
    parts = text.split("-")
    if len(parts) == 7 and all(p.isdigit() for p in parts):
        y, m, d, h, minute, second, ms = map(int, parts)
        return pd.Timestamp(y, m, d, h, minute, second, ms * 1000)
    return pd.Timestamp(value)


def door_score(truth, predictions):
    candidates = []
    for i, t in enumerate(truth):
        a, b = timestamp(t["start_time"]), timestamp(t["end_time"])
        for j, p in enumerate(predictions):
            if t.get("status", t.get("prediction")) != p["prediction"]:
                continue
            c, d = timestamp(p["start_time"]), timestamp(p["end_time"])
            inter = max(0.0, (min(b, d) - max(a, c)).total_seconds())
            union = (b-a).total_seconds() + (d-c).total_seconds() - inter
            if inter > 0 and union > 0:
                candidates.append((inter/union, i, j))
    used_t, used_p, credit = set(), set(), 0.0
    for iou, i, j in sorted(candidates, reverse=True):
        if i not in used_t and j not in used_p:
            used_t.add(i)
            used_p.add(j)
            credit += iou
    return 2 * credit / (len(truth) + len(predictions)) if truth or predictions else 0.0


def acv_score(truth, ranking):
    cars = ranking.split("|") if isinstance(ranking, str) else list(ranking)
    if len(cars) != len(set(cars)) or truth not in cars:
        return 0.0
    return (len(cars) - cars.index(truth)) / len(cars)


def rail_score(truth, predictions):
    return float(f1_score(truth, predictions, labels=["Normal", "Side I", "Side II"], average="macro", zero_division=0))


def shm_score(truth, predictions):
    truth, predictions = np.asarray(truth, float), np.asarray(predictions, float)
    if np.any(truth <= 0) or not np.isfinite(predictions).all():
        raise ValueError("Damage references must be positive and predictions finite.")
    return float(max(0, 1 - np.mean(np.abs(truth-predictions)/truth)))
