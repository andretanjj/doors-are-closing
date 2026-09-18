import pytest
from doors.metrics import door_score, acv_score, rail_score, shm_score, timestamp


def segment(start=0, end=10, label="Normal"):
    return {"start_time":f"2023-7-5-0-0-{start}-0", "end_time":f"2023-7-5-0-0-{end}-0", "prediction":label}


def test_native_timestamp_milliseconds():
    assert timestamp("2023-7-5-0-0-3-20").microsecond == 20000


def test_door_timing_label_and_one_to_one():
    truth = [segment()]
    assert door_score(truth, [segment()]) == 1
    assert door_score(truth, [segment(0,5)]) == .5
    assert door_score(truth, [segment(label="Abnormal resistance")]) == 0
    assert door_score(truth, [segment(),segment()]) == pytest.approx(2/3)
    assert door_score(truth, [segment(10,15)]) == 0
    assert door_score(truth, []) == 0


def test_acv_official_example():
    ranking = "03|01|05|02|04|06|07|08"
    assert acv_score("03", ranking) == 1
    assert acv_score("01", ranking) == .875
    assert acv_score("08", ranking) == .125
    assert acv_score("09", ranking) == 0


def test_rail_macro_f1_penalises_majority():
    assert rail_score(["Normal", "Side I", "Side II"], ["Normal"]*3) == pytest.approx(1/6)


def test_shm_official_example():
    actual = [.1,.3,.5,.7,.9]
    pred = [.15,.28,.55,.68,.85]
    assert shm_score(actual,pred) == pytest.approx(.84984126984)
    assert shm_score(actual,[.5]*5) == 0
