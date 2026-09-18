"""Check schemas against the official examples, coverage, values, and ZIP contents."""
import argparse
import io
from pathlib import Path
from zipfile import ZipFile
import pandas as pd
from doors.inference import SCHEMAS, validate_rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True,help="Official PS3 directory")
    p.add_argument("--submission",type=Path,default=Path("submission"))
    a=p.parse_args()
    expected={"acv":("ACV","*.xlsx"),"rail":("Rail_Corrugation","*.csv"),"shm":("SHM","*.csv")}
    with ZipFile(a.submission/"predictions.zip") as z:
        assert set(z.namelist())=={f"{s}_predictions.csv" for s in SCHEMAS}
        for s,columns in SCHEMAS.items():
            name=f"{s}_predictions.csv"
            frame=pd.read_csv(io.BytesIO(z.read(name)))
            assert list(frame.columns)==columns==list(pd.read_csv(a.source/"04_Example_Submission"/name).columns)
            validate_rows(s,frame.to_dict("records"))
            assert z.read(name)==(a.submission/name).read_bytes()
            if s in expected:
                folder,pattern=expected[s]
                assert set(frame.file_id)=={f.name for f in (a.source/"02_Datasets"/folder/"Test").glob(pattern)}
            print(f"{name}: {len(frame)} rows; schema, values, ZIP consistency and file coverage OK")

if __name__=="__main__":
    main()
