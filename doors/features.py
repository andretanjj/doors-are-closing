"""Label-free, deterministic feature extraction shared by training and the app."""
import re
import warnings
import numpy as np
import pandas as pd
import pywt
import rainflow
from scipy.signal import welch
from .metrics import timestamp

DOOR_COLUMNS = ["Datetime", "Motor current(mA)", "Motor Voltage(10mV)", "Motor electrodynamic force",
                "Door opening time(.1s)", "Door closing time(.1s)", "Close command", "Open command",
                "DCSR", "DCSL", "DLSR", "DLSL", "Door Opened", "Door Locked", "Door is opening",
                "Door is closing", "Door leaf position"]
RAIL_COLUMNS = ["Rotating speed"] + [f"{kind} of bearing in position {pos} of car {car}"
                for car in range(1, 9) for pos in range(1, 9) for kind in ("Vibration", "Shock")]


def numeric(frame):
    result = frame.apply(pd.to_numeric, errors="raise")
    if np.isinf(result.to_numpy(float)).any():
        raise ValueError("Input contains infinite values.")
    return result


def stats(x, prefix):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {f"{prefix}_{k}": np.nan for k in ["mean", "std", "rms", "min", "max", "q10", "median", "q90", "q99", "mad", "crest", "kurtosis"]}
    q = np.quantile(x, [.1, .5, .9, .99])
    std, rms = np.std(x), np.sqrt(np.mean(x*x))
    values = [np.mean(x), std, rms, np.min(x), np.max(x), *q,
              np.median(np.abs(x-q[1])), np.max(np.abs(x))/(rms+1e-9),
              np.mean(((x-np.mean(x))/(std+1e-9))**4)]
    keys = ["mean", "std", "rms", "min", "max", "q10", "median", "q90", "q99", "mad", "crest", "kurtosis"]
    return dict(zip([f"{prefix}_{k}" for k in keys], map(float, values)))


def door_load(path):
    frame = pd.read_csv(path)
    missing = set(DOOR_COLUMNS)-set(frame.columns)
    if missing:
        raise ValueError(f"Door file is missing columns: {', '.join(sorted(missing))}")
    frame = frame[DOOR_COLUMNS].copy()
    frame.iloc[:, 1:] = numeric(frame.iloc[:, 1:])
    if frame.isna().any().any() or len(frame) < 2:
        raise ValueError("Door input requires at least two complete readings.")
    times = pd.DatetimeIndex([timestamp(t) for t in frame.Datetime])
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("Door timestamps must be unique and increasing.")
    return frame, times


def door_segments(frame, times):
    # Native sampling is 20 ms; acquisition gaps separate the published operations.
    # Direction reversals additionally support adjacent opening/closing cycles.
    gaps = np.diff(times.as_unit("ns").asi8)/1e9
    direction = frame["Open command"].to_numpy(float)-frame["Close command"].to_numpy(float)
    reverse = (direction[1:]*direction[:-1] < 0)
    cuts = np.r_[0, np.flatnonzero((gaps > .2) | reverse)+1, len(frame)]
    return [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]) if b-a >= 2]


def door_features(frame, times):
    out = {"duration": float((times[-1]-times[0]).total_seconds()), "n_rows": len(frame)}
    for column in DOOR_COLUMNS[1:]:
        if column == "Door Locked":
            continue
        x = frame[column].to_numpy(float)
        out.update(stats(x, column))
        out[column+"_delta"] = float(x[-1]-x[0])
        if column in DOOR_COLUMNS[1:4] + ["Door leaf position"]:
            for phase, values in enumerate(np.array_split(x, 3)):
                out[f"{column}_phase{phase}_mean"] = float(np.mean(values))
                out[f"{column}_phase{phase}_rms"] = float(np.sqrt(np.mean(values**2)))
    pos = frame["Door leaf position"].to_numpy(float)
    out.update(stats(np.diff(pos)/.02, "velocity"))
    out["travel"] = float(np.ptp(pos))
    out["current_per_travel"] = float(frame["Motor current(mA)"].sum()*.02/(np.ptp(pos)+1))
    out["stall_fraction"] = float(np.mean(np.abs(np.diff(pos)) < 1))
    return out


def acv_features(path):
    frame = pd.read_excel(path, na_values=["Invalid", "invalid"])
    mapping = {}
    for c in frame.columns:
        match = re.fullmatch(r"Car (\d{2}) - (.+)", str(c))
        if match:
            aliases = {"Passenger Cabin Temperature Detected Value": "Indoor Average Temperature",
                       "Fresh Air Temperature Detected Value": "Outdoor Average Temperature",
                       "Target Temperature Value": "ACV Control Temperature (Cooling)",
                       "ACV Operating Mode": "ACV Setting Mode", "Load Shedding": "Load Halved"}
            mapping.setdefault(match[1], {})[aliases.get(match[2], match[2])] = c
    cars = sorted(mapping)
    if len(cars) != 8:
        raise ValueError("ACV workbook must contain exactly eight cars with Car NN - parameter headers.")
    parameters = sorted(set.intersection(*(set(mapping[c]) for c in cars)))
    if "Indoor Average Temperature" not in parameters:
        raise ValueError("ACV input requires cabin temperature telemetry for all eight cars.")
    if not parameters or not len(frame):
        raise ValueError("ACV input has no shared telemetry.")
    result = {c: {} for c in cars}
    for param in parameters:
        if not any(word in param.lower() for word in ["temperature", "mode", "valid", "halved"]):
            continue
        selected = frame[[mapping[c][param] for c in cars]]
        if "temperature" not in param.lower():
            for j, car in enumerate(cars):
                peers = selected.drop(columns=selected.columns[j])
                x = selected.iloc[:, j]
                disagreement = peers.ne(x, axis=0).where(peers.notna() & x.notna().to_numpy()[:, None]).mean(axis=1)
                result[car][param+"_peer_disagreement"] = float(disagreement.mean())
                result[car][param+"_missing"] = float(x.isna().mean())
            continue
        matrix = numeric(selected).to_numpy(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for j, car in enumerate(cars):
                x = matrix[:, j]
                peers = np.delete(matrix, j, axis=1)
                center = np.nanmedian(peers, axis=1)
                residual = x-center
                scale = np.nanmedian(np.abs(peers-center[:, None]), axis=1)
                z = residual/np.maximum(scale, 1.0)
                out = result[car]
                out.update(stats(residual, param+"_peer"))
                out.update(stats(z, param+"_robust_z"))
                out[param+"_missing"] = float(np.mean(~np.isfinite(x)))
                valid = np.isfinite(z)
                out[param+"_warm_fraction"] = float(np.mean(z[valid] > 1)) if valid.any() else np.nan
                out[param+"_cold_fraction"] = float(np.mean(z[valid] < -1)) if valid.any() else np.nan
    if not result[cars[0]]:
        raise ValueError("ACV workbook has no recognised temperature or operating-mode telemetry.")
    if any(not np.isfinite(result[c].get("Indoor Average Temperature_peer_mean", np.nan)) for c in cars):
        raise ValueError("Each car requires valid cabin temperature readings with available peers.")
    preview = []
    cabin = "Indoor Average Temperature"
    if cabin in parameters:
        for i in np.linspace(0, len(frame)-1, min(150, len(frame))).astype(int):
            point = {"sample": int(i)}
            for car in cars:
                value = frame.iloc[i][mapping[car][cabin]]
                point[car] = float(value) if pd.notna(value) else None
            preview.append(point)
    return cars, pd.DataFrame([result[c] for c in cars]), {"rows": len(frame), "parameters": parameters, "temperature_trend": preview}


def rail_features(path):
    frame = pd.read_csv(path)
    if set(frame.columns) != set(RAIL_COLUMNS) or len(frame) != 10000:
        raise ValueError("Rail input must contain the official 129 columns and 10,000 readings.")
    a = numeric(frame[RAIL_COLUMNS]).to_numpy(float)
    if not np.isfinite(a).all():
        raise ValueError("Rail input contains missing values.")
    if not np.isin(a[:, 0], [0, 1]).all():
        raise ValueError("Rail rotating-speed readings must be 0 or 1.")
    x = a[:, 1:]
    centered = x-x.mean(axis=0)
    rms = np.sqrt(np.mean(x*x, axis=0))
    std = np.std(x, axis=0)
    channel = {"rms": rms, "std": std, "peak": np.max(np.abs(x), axis=0),
               "range": np.ptp(x, axis=0), "crest": np.max(np.abs(x), axis=0)/(rms+1e-9),
               "kurtosis": np.mean((centered/(std+1e-9))**4, axis=0)}
    freq, psd = welch(centered, fs=10000, nperseg=2048, axis=0)
    total = psd.sum(axis=0)+1e-12
    channel["spectral_centroid"] = (freq[:, None]*psd).sum(axis=0)/total
    p = psd/total
    channel["spectral_entropy"] = -(p*np.log(p+1e-12)).sum(axis=0)
    for low, high in [(0, 50), (50, 150), (150, 300), (300, 600), (600, 1200), (1200, 2500), (2500, 5001)]:
        energy = psd[(freq >= low) & (freq < high)].sum(axis=0)
        channel[f"band_{low}_{high}"] = energy
        channel[f"fraction_{low}_{high}"] = energy/total
    for i, coeff in enumerate(pywt.wavedec(centered, "db4", level=4, axis=0)):
        channel[f"wavelet_{i}"] = np.mean(coeff**2, axis=0)
    pulses = np.count_nonzero(np.diff(a[:, 0]) > 0)
    out = {"speed_mps": float(pulses/90*np.pi*.85)}
    for key, values in channel.items():
        shaped = values.reshape(8, 8, 2)
        for kind, k in [("vibration", 0), ("shock", 1)]:
            for side, positions in [("I", slice(0, 8, 2)), ("II", slice(1, 8, 2))]:
                v = shaped[:, positions, k].ravel()
                prefix = f"{kind}_{key}_side_{side}"
                out[prefix+"_mean"] = float(np.mean(v))
                out[prefix+"_max"] = float(np.max(v))
                out[prefix+"_q90"] = float(np.quantile(v, .9))
            left = out[f"{kind}_{key}_side_I_mean"]
            right = out[f"{kind}_{key}_side_II_mean"]
            out[f"{kind}_{key}_contrast"] = (left-right)/(abs(left)+abs(right)+1e-9)
    heatmap = rms.reshape(8, 8, 2)[:, :, 0].tolist()
    preview_idx = np.linspace(0,len(frame)-1,400).astype(int)
    vibration = centered[:, 0::2].reshape(len(frame),8,8)
    side_psd = psd[:,0::2].reshape(len(freq),8,8)
    return out, {"heatmap": heatmap, "speed_mps": out["speed_mps"], "rows": len(frame),
                 "vibration_preview":[{"time":float(i/10000), "side_i":float(vibration[i,:,0::2].mean()), "side_ii":float(vibration[i,:,1::2].mean())} for i in preview_idx],
                 "spectrum":[{"frequency":float(freq[i]), "side_i":float(side_psd[i,:,0::2].mean()), "side_ii":float(side_psd[i,:,1::2].mean())} for i in range(0,len(freq),4)]}


def shm_features(path):
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] != 1 or len(frame) < 4 or len(frame) > 2_000_000:
        raise ValueError("SHM input must be one headerless stress column, 4–2,000,000 readings.")
    x = numeric(frame).to_numpy(float).ravel()
    if not np.isfinite(x).all():
        raise ValueError("SHM input contains missing values.")
    out = stats(x, "stress")
    out.update(stats(np.diff(x), "stress_change"))
    cycles = np.asarray([(r, mean, n) for r, mean, n, _, _ in rainflow.extract_cycles(x)])
    if not len(cycles):
        cycles = np.array([[0., 0., 0.]])
    ranges, means, counts = cycles.T
    out["cycles"] = float(counts.sum())
    out["n_rows"] = len(x)
    for exponent in [1, 2, 3, 4, 5, 6, 8, 10]:
        out[f"rainflow_m{exponent}"] = float(np.sum(counts*(ranges/2)**exponent))
    out.update(stats(ranges, "cycle_range"))
    bins = np.linspace(0, max(float(ranges.max()), 1e-6), 17)
    hist, _ = np.histogram(ranges, bins=bins, weights=counts)
    for q in [.9, .95, .99, .999]:
        out[f"abs_quantile_{q}"] = float(np.quantile(np.abs(x), q))
    return out, {"rows": len(x), "histogram": hist.tolist(), "bin_edges": bins.tolist(),
                 "stress_preview": x[np.linspace(0, len(x)-1, min(300, len(x))).astype(int)].tolist()}
