"""Official-style CLI; uses the identical service as the web application."""
import argparse
import json
from pathlib import Path
from doors.inference import infer, SCHEMAS
import pandas as pd

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsystem", required=True, choices=list(SCHEMAS))
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    extension = "*.xlsx" if a.subsystem == "acv" else "*.csv"
    paths = sorted(a.input.glob(extension)) if a.input.is_dir() else [a.input]
    if not paths or (a.subsystem == "door" and len(paths) != 1):
        p.error("Provide input files; Door requires exactly one continuous stream.")
    rows = []
    for path in paths:
        rows.extend(infer(a.subsystem, path)["rows"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=SCHEMAS[a.subsystem]).to_csv(a.output, index=False)
    print(json.dumps({"output":str(a.output), "rows":len(rows)}))
