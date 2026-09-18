"""Generate official held-out predictions through the same HTTP app used by the UI."""
import argparse
import json
from pathlib import Path
from fastapi.testclient import TestClient
from doors.api import app
from doors.inference import SCHEMAS
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("submission"))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    inputs={"door":[args.data/"Door/Test.csv"],
            "acv":sorted((args.data/"ACV/Test").glob("*.xlsx")),
            "rail":sorted((args.data/"Rail_Corrugation/Test").glob("*.csv")),
            "shm":sorted((args.data/"SHM/Test").glob("*.csv"))}
    results,manifest={},[]
    with TestClient(app) as client:
        for s,paths in inputs.items():
            results[s]=[]
            for p in paths:
                with p.open("rb") as f:
                    r=client.post(f"/api/analyse/{s}",files={"file":(p.name,f)})
                if r.status_code!=200:
                    raise RuntimeError(f"{p.name}: {r.status_code} {r.text}")
                value=r.json()
                results[s].extend(value["rows"])
                manifest.append({"subsystem":s,"file":p.name,"model":value["model"],"seconds":value["seconds"],"rows":len(value["rows"])})
                print(s,p.name,value["seconds"],flush=True)
            pd.DataFrame(results[s],columns=SCHEMAS[s]).to_csv(args.output/f"{s}_predictions.csv",index=False)
        export=client.post("/api/export",json=results)
        export.raise_for_status()
        (args.output/"predictions.zip").write_bytes(export.content)
    (args.output/"inference_manifest.json").write_text(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
