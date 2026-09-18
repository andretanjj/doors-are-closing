import io
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from doors.api import app
from doors.features import DOOR_COLUMNS, door_load, door_segments, rail_features, shm_features, acv_features
from doors.inference import validate_rows, zip_predictions

client = TestClient(app)


def test_zip_top_level_and_exact_schemas():
    payload = {"rail":[{"file_id":"Test1.csv", "prediction":"Side I"}],
               "shm":[{"file_id":"test01.csv", "prediction":.123}],
               "acv":[{"file_id":"case.xlsx", "ranked_cars":"01|02|03|04|05|06|07|08"}]}
    response = client.post("/api/export", json=payload)
    assert response.status_code == 200
    with ZipFile(io.BytesIO(response.content)) as archive:
        assert sorted(archive.namelist()) == ["acv_predictions.csv","rail_predictions.csv","shm_predictions.csv"]
        assert archive.read("acv_predictions.csv").decode().splitlines()[0] == "file_id,ranked_cars"


@pytest.mark.parametrize("subsystem,rows", [
    ("rail",[{"file_id":"x.csv","prediction":"abnormal"}]),
    ("shm",[{"file_id":"x.csv","prediction":float("nan")}]),
    ("acv",[{"file_id":"x.xlsx","ranked_cars":"01|01|02|03|04|05|06|07"}]),
    ("rail",[{"file_id":"../x.csv","prediction":"Normal"}]),
    ("rail",[{"file_id":"=x.csv","prediction":"Normal"}]),
])
def test_export_rejects_bad_predictions(subsystem, rows):
    with pytest.raises(ValueError):
        validate_rows(subsystem, rows)


def test_invalid_uploads():
    assert client.post("/api/analyse/door",files={"file":("bad.exe",b"a")}).status_code == 400
    assert client.post("/api/analyse/nope",files={"file":("x.csv",b"a")}).status_code == 400
    assert client.post("/api/analyse/door",files={"file":("x.csv",b"")}).status_code == 422


def test_door_gap_segmentation_across_pandas_time_units(tmp_path):
    frame = pd.DataFrame(0,index=range(4),columns=DOOR_COLUMNS)
    frame["Datetime"] = ["2023-7-5-0-0-0-0","2023-7-5-0-0-0-20","2023-7-5-0-0-10-0","2023-7-5-0-0-10-20"]
    p = tmp_path/"door.csv"
    frame.to_csv(p,index=False)
    f,t = door_load(p)
    assert door_segments(f,t.as_unit("us")) == [(0,2),(2,4)]
    assert door_segments(f,t.as_unit("ns")) == [(0,2),(2,4)]


def test_shm_physics_amplitude_scaling(tmp_path):
    x = np.tile([0.,1.,0.,-1.,0.],10)
    first,second = tmp_path/"first.csv",tmp_path/"second.csv"
    np.savetxt(first,x)
    np.savetxt(second,2*x)
    a,_ = shm_features(first)
    b,_ = shm_features(second)
    assert b["rainflow_m3"] == pytest.approx(8*a["rainflow_m3"])


def test_rail_wrong_schema(tmp_path):
    p = tmp_path/"bad.csv"
    p.write_text("x,y\n1,2\n")
    with pytest.raises(ValueError,match="129 columns"):
        rail_features(p)


def test_acv_invalid_sentinel_and_missing_cabin(tmp_path):
    p = tmp_path/"acv.xlsx"
    frame = pd.DataFrame({f"Car {i:02d} - Indoor Average Temperature":[20+i,"Invalid",22+i] for i in range(1,9)})
    frame.to_excel(p,index=False)
    cars,features,details = acv_features(p)
    assert len(cars)==8
    assert features["Indoor Average Temperature_missing"].iloc[0] == pytest.approx(1/3)
    assert details["temperature_trend"][1]["01"] is None
    frame.iloc[:,:] = "Invalid"
    frame.to_excel(p,index=False)
    with pytest.raises(ValueError,match="valid cabin"):
        acv_features(p)


def test_saved_model_health():
    response=client.get("/health")
    assert response.status_code==200
    assert response.json()["status"]=="ready"
