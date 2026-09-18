"""Single inference service used by HTTP and command-line entrypoints."""
import io
import json
from pathlib import Path
from functools import lru_cache
from zipfile import ZipFile, ZIP_DEFLATED
import joblib
import numpy as np
import pandas as pd
from .features import door_load, door_segments, door_features, acv_features, rail_features, shm_features
from .metrics import timestamp
from .models import peer_scores

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = {"door":["start_time", "end_time", "prediction"], "acv":["file_id", "ranked_cars"],
           "rail":["file_id", "prediction"], "shm":["file_id", "prediction"]}


@lru_cache(maxsize=4)
def artefact(subsystem):
    path = ROOT/"models"/f"{subsystem}.joblib"
    if not path.exists():
        raise RuntimeError(f"The {subsystem} model has not been trained.")
    return joblib.load(path)


def clean_columns(frame):
    frame.columns = [str(c).replace('"','').replace(':','_').replace(',','_').replace('[','_').replace(']','_').replace('{','_').replace('}','_') for c in frame.columns]
    return frame


def evidence(features, saved):
    """Descriptive deviations, deliberately not presented as SHAP or causality."""
    median, std = saved["training_median"], saved["training_std"]
    candidates = []
    for key, value in features.items():
        if key in median and np.isfinite(value):
            z = (float(value)-median[key])/max(std[key], 1e-8)
            if np.isfinite(z):
                candidates.append({"feature":key, "value":float(value), "reference":median[key], "standardised_deviation":float(z)})
    return sorted(candidates, key=lambda r:abs(r["standardised_deviation"]), reverse=True)[:5]


def infer(subsystem, path, filename=None):
    if subsystem not in SCHEMAS:
        raise ValueError("Choose door, acv, rail, or shm.")
    filename = filename or Path(path).name
    saved = artefact(subsystem)
    warnings = []
    if subsystem == "door":
        frame, times = door_load(path)
        segments = door_segments(frame, times)
        if not segments:
            raise ValueError("No complete door operations were found.")
        X = clean_columns(pd.DataFrame([door_features(frame.iloc[a:b], times[a:b]) for a,b in segments]))
        values = saved["model"].predict(X.reindex(columns=saved["columns"]))
        support = saved["model"].predict_proba(X.reindex(columns=saved["columns"])).max(axis=1)
        rows = [{"start_time":str(times[a]), "end_time":str(times[b-1]), "prediction":saved["classes"][int(y)]}
                for (a,b),y in zip(segments,values)]
        indices = np.linspace(0, len(frame)-1, min(400, len(frame))).astype(int)
        details = {"segments":len(rows), "abnormal":sum(r["prediction"] != "Normal" for r in rows),
                   "timeline":[{"time":str(times[i]), "elapsed":float((times[i]-times[0]).total_seconds()), "current":float(frame.iloc[i,1]), "position":float(frame.iloc[i,-1])} for i in indices],
                   "segment_evidence":[{"start":float((times[a]-times[0]).total_seconds()), "end":float((times[b-1]-times[0]).total_seconds()), "label":saved["classes"][int(y)], "support":float(p)} for (a,b),y,p in zip(segments,values,support)],
                   "evidence":evidence(X.iloc[int(np.argmax(values))].to_dict(), saved),
                   "explanation":"Finds door movements and flags unusual signals."}
        warnings.append("Live streams need more testing.")
    elif subsystem == "acv":
        cars, X, info = acv_features(path)
        scores = peer_scores(X, saved["name"])
        order = np.argsort(-scores, kind="stable")
        rows = [{"file_id":filename, "ranked_cars":"|".join(cars[j] for j in order)}]
        details = {**info, "ranking":[{"car":cars[j], "score":float(scores[j]),
            "mean_temperature_residual":float(np.nan_to_num(X.iloc[j].get("Indoor Average Temperature_peer_mean", 0)))} for j in order],
            "explanation":"Compares each car with its peers. Warmer cars score higher."}
        warnings.append("Only 6 labelled examples. Check before acting.")
        if X.filter(like="_missing").mean().mean() > .1:
            warnings.append("Over 10% of readings are missing. Ranking uses available data.")
    else:
        feature, details = (rail_features if subsystem == "rail" else shm_features)(path)
        X = clean_columns(pd.DataFrame([feature]))
        prediction = saved["model"].predict(X.reindex(columns=saved["columns"]))[0]
        label = saved["classes"][int(prediction)] if subsystem == "rail" else float(max(prediction, 1e-12))
        if isinstance(label, float) and not np.isfinite(label):
            raise ValueError("The stress range is outside the supported model domain.")
        rows = [{"file_id":filename, "prediction":label}]
        details["evidence"] = evidence(X.iloc[0].to_dict(), saved)
        if subsystem == "rail":
            details["model_support"] = float(saved["model"].predict_proba(X.reindex(columns=saved["columns"])).max())
        details["explanation"] = ("Shows vibration by axle box and side."
            if subsystem == "rail" else "Uses stress cycles to estimate damage.")
        if subsystem == "shm":
            warnings.append("Units are missing. Use this estimate only for this dataset.")
            if details["rows"] != 581120:
                warnings.append("File length differs from training. The estimate may be less reliable.")
    validate_rows(subsystem, rows)
    return {"subsystem":subsystem, "file_id":filename, "model":saved["name"], "rows":rows,
            "details":details, "warnings":warnings, "csv_filename":f"{subsystem}_predictions.csv",
            "csv":pd.DataFrame(rows, columns=SCHEMAS[subsystem]).to_csv(index=False)}


def validate_rows(subsystem, rows):
    if subsystem not in SCHEMAS or not rows or len(rows) > 100000:
        raise ValueError("Prediction export requires a supported subsystem and nonempty rows.")
    seen = set()
    for row in rows:
        if set(row) != set(SCHEMAS[subsystem]):
            raise ValueError(f"Export requires exactly {SCHEMAS[subsystem]}.")
        if subsystem == "door":
            if pd.isna(timestamp(row["start_time"])) or pd.isna(timestamp(row["end_time"])):
                raise ValueError("Door timestamps must be valid.")
            if timestamp(row["end_time"]) <= timestamp(row["start_time"]):
                raise ValueError("Door segment end must be after start.")
            if row["prediction"] not in ["Normal", "Abnormal resistance"]:
                raise ValueError("Invalid Door label.")
        else:
            name = row["file_id"]
            if not isinstance(name, str) or Path(name).name != name or not name or name[0] in "=+-@":
                raise ValueError("Invalid source filename.")
            if name in seen:
                raise ValueError("Duplicate source filename in export.")
            seen.add(name)
            if subsystem == "acv":
                cars = str(row["ranked_cars"]).split("|")
                if len(cars)!=8 or len(set(cars))!=8 or any(len(c)!=2 or not c.isdigit() for c in cars):
                    raise ValueError("ACV ranking must contain eight unique two-digit car IDs.")
            elif subsystem == "rail" and row["prediction"] not in ["Normal", "Side I", "Side II"]:
                raise ValueError("Invalid Rail label.")
            elif subsystem == "shm" and (not np.isfinite(float(row["prediction"])) or float(row["prediction"]) < 0):
                raise ValueError("SHM damage must be finite and nonnegative.")


def zip_predictions(predictions):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for subsystem, rows in predictions.items():
            validate_rows(subsystem, rows)
            archive.writestr(f"{subsystem}_predictions.csv", pd.DataFrame(rows, columns=SCHEMAS[subsystem]).to_csv(index=False))
    return buffer.getvalue()


def model_summary():
    result = {}
    for s in SCHEMAS:
        report = ROOT/"reports"/f"{s}.json"
        if report.exists() and (ROOT/"models"/f"{s}.joblib").exists():
            r = json.loads(report.read_text())
            result[s] = {"ready":True, "model":r["selected_model"], "validation_score":r["holdout_score"],
                         "result_type":r["result_type"], "validation":r["details"]["validation"]}
        else:
            result[s] = {"ready":False}
    return result
