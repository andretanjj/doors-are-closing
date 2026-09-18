"""Offline benchmarks and final fitting. Never reads Test folders."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from .features import door_load, door_segments, door_features, acv_features, rail_features, shm_features
from .metrics import timestamp, door_score, acv_score, rail_score, shm_score
from .models import SEED, classifiers, regressors, peer_scores

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def cached(path, subsystem):
    digest = sha(path)
    code = sha(Path(__file__).with_name("features.py"))[:12]
    dest = ROOT/".cache"/f"{subsystem}-{code}-{digest}.joblib"
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        return joblib.load(dest), digest
    extractor = {"rail": rail_features, "shm": shm_features, "acv": acv_features}[subsystem]
    value = extractor(path)
    joblib.dump(value, dest)
    return value, digest


def bootstrap(y, pred, metric, iterations=1000):
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y), len(y))
        samples.append(metric(np.asarray(y)[indices], np.asarray(pred)[indices]))
    return np.quantile(samples, [.025, .975]).tolist()


def train(subsystem, data):
    started = time.monotonic()
    artefacts = ROOT/"models"
    reports = ROOT/"reports"
    artefacts.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    fingerprints, details = {}, {}
    if subsystem == "door":
        source = data/"Door"
        frame, times = door_load(source/"Train.csv")
        truth = pd.read_csv(source/"Train_Segments_Answer.csv")
        segments = door_segments(frame, times)
        boundaries = [{"start_time": str(times[a]), "end_time": str(times[b-1]), "prediction": "Normal"} for a,b in segments]
        geometry = truth.assign(status="Normal").to_dict("records")
        details["segmentation_only_iou_f1"] = door_score(geometry, boundaries)
        # Build supervised examples from discovered segments; require exact unique label match.
        lookup = {(timestamp(r.start_time), timestamp(r.end_time)): r.status for r in truth.itertuples()}
        labels = [lookup.get((times[a], times[b-1])) for a,b in segments]
        if any(x is None for x in labels):
            raise ValueError("Segmentation differs from reference boundaries; review before classifier benchmarking.")
        X = pd.DataFrame([door_features(frame.iloc[a:b], times[a:b]) for a,b in segments])
        classes = ["Normal", "Abnormal resistance"]
        y = np.array([classes.index(x) for x in labels])
        ids = np.array([f"segment_{i+1}" for i in range(len(X))])
        split = int(.8*len(X))
        dev, hold = np.arange(split), np.arange(split, len(X))
        folds = list(TimeSeriesSplit(n_splits=4).split(dev))
        candidates = classifiers()
        # Boundaries are exact, so end-to-end IoU-F1 equals fraction correct here.
        metric = lambda a,b: float(np.mean(a==b))
        details["validation"] = "Forward-chaining development folds; final chronological 20% held out. Exact discovered boundaries verified."
        fingerprints["Train.csv"] = sha(source/"Train.csv")
        fingerprints["labels"] = sha(source/"Train_Segments_Answer.csv")
    elif subsystem in ("rail", "shm"):
        source = data/({"rail": "Rail_Corrugation", "shm": "SHM"}[subsystem])
        truth = pd.read_csv(source/"Train_Labels.csv")
        rows, groups = [], []
        for i, filename in enumerate(truth.filename):
            (features, _), digest = cached(source/"Train"/filename, subsystem)
            rows.append(features)
            groups.append(digest)
            fingerprints[filename] = digest
            if i % 20 == 0:
                print(f"{subsystem}: extracted {i+1}/{len(truth)} files", flush=True)
        X, ids = pd.DataFrame(rows), truth.filename.to_numpy()
        if subsystem == "rail":
            classes = ["Normal", "Side I", "Side II"]
            y = truth.label.map({c:i for i,c in enumerate(classes)}).to_numpy()
            outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
            dev, hold = next(outer.split(X, y, groups))
            folds = list(StratifiedGroupKFold(4, shuffle=True, random_state=SEED).split(X.iloc[dev], y[dev], np.array(groups)[dev]))
            candidates = classifiers()
            metric = lambda a,b: rail_score([classes[int(i)] for i in a], [classes[int(i)] for i in b])
            details["validation"] = "20% stratified file-group holdout; four grouped development folds; exact duplicate files grouped by SHA-256."
            details["unique_files"] = len(set(groups))
        else:
            y = truth.damage.to_numpy()
            classes = None
            strata = pd.qcut(y, 4, labels=False)
            dev, hold = train_test_split(np.arange(len(X)), test_size=.25, stratify=strata, random_state=SEED)
            folds = list(StratifiedKFold(4, shuffle=True, random_state=SEED).split(X.iloc[dev], np.asarray(strata)[dev]))
            candidates = regressors()
            metric = shm_score
            details["validation"] = "25% file holdout stratified by damage quartile; four file-level development folds."
        fingerprints["labels"] = sha(source/"Train_Labels.csv")
    else:
        return train_acv(data, started)
    X = X.replace([np.inf, -np.inf], np.nan)
    # Sanitise column names because LightGBM disallows some JSON punctuation.
    X.columns = [str(c).replace('"','').replace(':','_').replace(',','_').replace('[','_').replace(']','_').replace('{','_').replace('}','_') for c in X.columns]
    benchmark, validation_predictions = [], {}
    for name, candidate in candidates.items():
        scores, oof = [], np.full(len(dev), np.nan)
        for tr, va in folds:
            model = clone(candidate).fit(X.iloc[dev[tr]], y[dev[tr]])
            pred = model.predict(X.iloc[dev[va]])
            scores.append(metric(y[dev[va]], pred))
            oof[va] = pred
        benchmark.append({"model": name, "cv_mean": float(np.mean(scores)), "cv_std": float(np.std(scores)), "fold_scores": scores})
        validation_predictions[name] = [float(v) if np.isfinite(v) else None for v in oof]
        print(subsystem, name, benchmark[-1]["cv_mean"], flush=True)
    winner = max(benchmark, key=lambda r:r["cv_mean"])["model"]
    held_model = clone(candidates[winner]).fit(X.iloc[dev], y[dev])
    held_pred = held_model.predict(X.iloc[hold])
    score = metric(y[hold], held_pred)
    interval = bootstrap(y[hold], held_pred, metric)
    final = clone(candidates[winner]).fit(X, y)
    report = {"subsystem": subsystem, "seed": SEED, "source_commit": "16526c02579c7f37e54eaaa42a4cc6d4ceb19994",
              "result_type": "LOCAL VALIDATION RESULT", "selected_model": winner, "candidates": benchmark,
              "selection_rule": "Highest development CV mean; holdout inspected only after selection and never used for tuning.",
              "holdout_score": score, "holdout_95_percent_bootstrap_ci": interval, "holdout_files": ids[hold].tolist(),
              "holdout_truth": y[hold].tolist(), "holdout_predictions": held_pred.tolist(), "classes": classes,
              "folds": [{"train":ids[dev[t]].tolist(), "validation":ids[dev[v]].tolist()} for t,v in folds],
              "features": list(X.columns), "fingerprints": fingerprints, "details": details,
              "development_oof_predictions": validation_predictions, "seconds": time.monotonic()-started}
    if classes:
        report["holdout_confusion_matrix"] = confusion_matrix(y[hold], held_pred, labels=np.arange(len(classes))).tolist()
    joblib.dump({"model":final, "columns":list(X.columns), "classes":classes, "name":winner,
                 "training_median":X.median().fillna(0).to_dict(), "training_std":X.std().fillna(1).clip(lower=1e-8).to_dict()}, artefacts/f"{subsystem}.joblib", compress=3)
    (reports/f"{subsystem}.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"{subsystem}: selected {winner}; internal holdout score={score:.6f}", flush=True)


def train_acv(data, started):
    labels = pd.read_csv(data/"ACV/Train_Labels.csv", dtype={"faulty_car":str})
    cases, fingerprints = [], {}
    for row in labels.itertuples():
        (cars, X, info), digest = cached(data/"ACV/Train"/row.filename, "acv")
        cases.append((row.filename, cars, X, row.faulty_car))
        fingerprints[row.filename] = digest
        print("acv: extracted", row.filename, flush=True)
    modes = ["car_order", "thermal", "robust", "thermal_variability", "isolation_forest"]
    per_mode = {}
    for mode in modes:
        scores, rankings = [], []
        for filename, cars, X, truth in cases:
            ranking = [cars[j] for j in np.argsort(-peer_scores(X, mode), kind="stable")]
            scores.append(acv_score(truth, ranking))
            rankings.append(ranking)
        per_mode[mode] = {"model":mode, "cv_mean":float(np.mean(scores)), "fold_scores":scores, "rankings":rankings}
    # Nested leave-one-workbook-out selection: held case never selects its own scoring rule.
    nested = []
    for i in range(len(cases)):
        chosen = max(modes, key=lambda m:np.mean(np.delete(per_mode[m]["fold_scores"], i)))
        nested.append({"file":cases[i][0], "selected_without_case":chosen, "score":per_mode[chosen]["fold_scores"][i]})
    winner = max(modes, key=lambda m:per_mode[m]["cv_mean"])
    report = {"subsystem":"acv", "result_type":"LOCAL VALIDATION RESULT", "seed":SEED,
              "selected_model":winner, "candidates":list(per_mode.values()), "nested_leave_one_workbook_out":nested,
              "holdout_score":float(np.mean([r["score"] for r in nested])), "fingerprints":fingerprints,
              "details":{"validation":"Nested leave-one-workbook-out rule selection. Six labelled cases; no row splitting or car-ID feature.",
                         "limitations":"Peer thermal scores are heuristic, not calibrated probabilities. No reliable narrow confidence interval with six cases."},
              "seconds":time.monotonic()-started}
    (ROOT/"models").mkdir(exist_ok=True)
    (ROOT/"reports").mkdir(exist_ok=True)
    joblib.dump({"name":winner}, ROOT/"models/acv.joblib")
    (ROOT/"reports/acv.json").write_text(json.dumps(report, indent=2))
    print("acv: selected", winner, report["holdout_score"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Official PS3/02_Datasets path")
    parser.add_argument("--subsystem", choices=["door", "acv", "rail", "shm", "all"], default="all")
    args = parser.parse_args()
    for s in (["door", "acv", "rail", "shm"] if args.subsystem == "all" else [args.subsystem]):
        train(s, args.data)
