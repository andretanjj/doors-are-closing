"""Real-network end-to-end smoke test; works against local Docker or an approved hosted URL."""
import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile
import httpx

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--url",required=True)
    p.add_argument("--data",type=Path,required=True)
    a=p.parse_args()
    inputs={"door":a.data/"Door/Test.csv","acv":a.data/"ACV/Test/acv_test_case.xlsx",
            "rail":a.data/"Rail_Corrugation/Test/Test1.csv","shm":a.data/"SHM/Test/test01.csv"}
    output={}
    with httpx.Client(base_url=a.url,timeout=300) as c:
        response=c.get("/health")
        response.raise_for_status()
        assert response.json()["status"]=="ready"
        assert 'id="root"' in c.get("/").text
        for s,path in inputs.items():
            with path.open("rb") as f:
                r=c.post(f"/api/analyse/{s}",files={"file":(path.name,f)})
            r.raise_for_status()
            value=r.json()
            output[s]=value["rows"]
            print(s,len(output[s]),"rows",value["seconds"],"seconds",flush=True)
        r=c.post("/api/export",json=output)
        r.raise_for_status()
        with ZipFile(io.BytesIO(r.content)) as z:
            assert set(z.namelist())=={f"{s}_predictions.csv" for s in output}
        assert c.post("/api/analyse/door",files={"file":("invalid.csv",b"")}).status_code==422
        print(json.dumps({"status":"passed","subsystems":list(output),"zip_bytes":len(r.content)}))

if __name__=="__main__":
    main()
