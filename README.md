# Doors Are Closing

One maintenance-oriented application for four different train condition-monitoring tasks. Built for Nebula X Hackathon 2026, Problem Statement 3.

**[Open the live application](https://doors-are-closing-895395923568.asia-southeast1.run.app)** · [Implementation evidence](docs/PHASE_2.md) · [Operational runbook](docs/RUNBOOK.md)

## Overview

Select a subsystem, upload a recording, inspect predictions and supporting signals, and download the organiser-required CSV or combined `predictions.zip`. Batch uploads are supported for ACV, Rail and SHM; Door takes one continuous stream at a time.

This is a hackathon decision-support prototype, not a safety-certified diagnostic system. Predictions require engineering review. The demo runs in an organiser-provided Qwiklabs project; continued availability depends on that project's access and lifetime.

## Nebula X PS3

The [official PS3 repository](https://github.com/aochinwen/NebulaX-Hackathon-ProblemStatement/tree/main/PS3) and its subsystem Info Kits are authoritative. The audited source revision is `16526c02579c7f37e54eaaa42a4cc6d4ceb19994`. No organiser source files were changed or raw datasets redistributed here.

## Subsystems Supported

| Subsystem | Input | Task and output | Official metric |
|---|---|---|---|
| Door | CSV stream, official 17-column schema | Discover operations; classify Normal / Abnormal resistance | IoU-weighted F1 |
| ACV | XLSX with eight-car telemetry | Rank all eight two-digit car IDs | Rank-decay score |
| Rail Corrugation | CSV, 10,000 rows and 129 columns | Normal / Side I / Side II | Macro F1 |
| SHM | Headerless single-column stress CSV | Cumulative fatigue damage | max(0, 1 − MAPE) |

Door evaluation greedily matches same-label segments by descending IoU, one-to-one: `2 × sum(matched IoU) / (number of true + predicted segments)`. ACV gives `(8 − zero-based faulty-car rank) / 8`, or zero when absent. Rail macro F1 explicitly includes all three official classes. SHM reference damage is positive; no arbitrary zero-denominator convention is introduced. Implementations and unit tests are in `doors/metrics.py` and `tests/test_metrics.py`.

## Architecture

React/TypeScript provides the upload, result and export workspace. FastAPI serves the production frontend and HTTP API from one container. A shared inference layer validates inputs, extracts subsystem-specific features, loads trusted packaged model artefacts, and generates the exact competition outputs. The CLI calls that same inference layer. Training is offline and separate from request handling.

Uploads use temporary files removed after processing. Results remain in browser memory for the session; refreshing clears them. The service does not store raw uploads or log sensor values. Logs include subsystem, input size and processing time. No API keys are needed for inference.

## Model Approaches

- **Door:** acquisition gaps and command reversals locate operations; current, voltage, back-EMF, position and phase features feed a random forest. Discovered boundaries exactly match the supplied training labels. Generalisation to uninterrupted live streams is not established.
- **ACV:** compare each car with the median of its seven peers. Positive mean cabin-temperature residual plus half its positive 90th-percentile residual produces a transparent ranking. Known `Invalid` readings become missing data, not numbers. Every car needs valid cabin readings with peers.
- **Rail:** time-domain statistics, Welch spectral bands, wavelet energies and Side I/II contrasts feed regularised, class-balanced logistic regression. Exact duplicate files are grouped during validation.
- **SHM:** rainflow cycle counts form power-law damage proxies. A training-only exponent and scale calibration predicts the supplied damage target. This is not a certified S–N calculation, failure probability or remaining-life estimate.

The supplied train-door paper informed the Decision Tree, Random Forest, GBM, XGBoost and LightGBM candidate families. All five were benchmarked for Door, alongside logistic regression and SVM. ACV also compared Isolation Forest; SHM compared log-ridge, ExtraTrees and LightGBM. Deep learning and ensembles were not introduced without evidence that their complexity was warranted.

Useful statistical ideas include moments, robust peer residuals, train-fold-only scaling, dependence-aware splitting and uncertainty assessment. Feature deviations shown by the app are descriptive evidence, not causal explanations or SHAP attributions. Model-support percentages are **uncalibrated**, not maintenance-risk probabilities.

## Model Benchmark Results

All numbers below are **LOCAL VALIDATION RESULTS** on labelled organiser training data. They are not literature results or organiser-held-out test scores. No held-out test answers were available or used.

| Subsystem | Selected model | Development score | Internal evaluation score | Evaluation design |
|---|---|---:|---:|---|
| Door | Random forest | 1.0000 | 1.0000 | Forward-chaining development; final 22 chronological operations held out |
| ACV | Peer thermal residual | 0.9792 | 0.9792 | Nested leave-one-workbook-out rule selection, six cases |
| Rail | Logistic regression | 0.7523 | 0.7858 | Four grouped development folds; 54-file stratified group holdout |
| SHM | Rainflow calibration | 0.9723 | 0.9777 | Four development folds; 16-file damage-quartile-stratified holdout |

Selection uses development scores, not holdout feedback; final models are refitted on all labelled data. Configurations and seed `2334` are fixed in `doors/models.py`. Reports contain every candidate score, fold membership, fingerprints, out-of-fold predictions and holdout results.

The Door result has limited external validity: segmentation exploits this acquisition format, and 22 correct holdout operations do not prove perfect future performance. Rail's holdout contains only two Side I files. ACV has just six labelled workbooks, with unknown cross-asset independence. Bootstrap intervals in the reports are exploratory: Rail approximately [0.507, 1.000], SHM [0.962, 0.989]; Door's degenerate [1,1] interval must not be read as certainty. See [the full benchmark discussion](docs/PHASE_2.md).

## Dataset

Obtain data from the official PS3 repository and keep it outside this repository. Commands below assume the absolute path `/absolute/path/PS3/02_Datasets`:

```text
02_Datasets/
  Door/              Train.csv, Train_Segments_Answer.csv, Test.csv
  ACV/               Train/, Train_Labels.csv, Test/
  Rail_Corrugation/  Train/, Train_Labels.csv, Test/
  SHM/               Train/, Train_Labels.csv, Test/
```

Training uses 110 Door operations, six ACV workbooks, 272 Rail files (270 unique content hashes), and 64 SHM files. Rail is heavily imbalanced: 234 Normal, 14 Side I and 24 Side II. Door is 80 Normal and 30 Abnormal resistance. Units, frequencies and schemas follow the official kits; SHM stress units and sampling rate are not assumed.

## Project Structure

```text
doors/       feature extraction, models, training, metrics, inference, API
frontend/    React + TypeScript + Vite application and lockfile
models/      selected trained joblib artefacts
reports/     reproducible local validation records
scripts/     app-based submission generation, verification, HTTP smoke tests
submission/  four generated prediction CSVs, ZIP and inference manifest
tests/       automated metric and API/input-contract tests
docs/        implementation evidence, runbook and deployment record
predict.py   shared-inference prediction CLI
Dockerfile   frontend build + Python API runtime
deploy.sh    explicit-project Cloud Run source deployment
```

## Installation

Requires Python 3.12, Node.js 22 and pnpm 11.19.0. macOS training with LightGBM/XGBoost requires `brew install libomp`; Linux needs `libgomp1` (installed by the Dockerfile).

```sh
git clone https://github.com/andretanjj/doors-are-closing.git
cd doors-are-closing
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
corepack enable
cd frontend
pnpm install --frozen-lockfile
pnpm build
cd ..
```

Packaged models are ready for inference; you do not need to retrain. Only load trusted repository joblib files: pickle-based artefacts can execute code. Keep the pinned Python dependencies when loading them.

## Environment Variables

See [.env.example](.env.example). `PORT` controls the container listener (Cloud Run supplies it; default 8080). `OMP_NUM_THREADS` limits native parallelism in the container. `GOOGLE_CLOUD_PROJECT` and `CLOUD_RUN_REGION` are shell variables read by `deploy.sh`, not automatically loaded from `.env`. The default deployment region is `asia-southeast1`.

Authenticate through Google Cloud's normal CLI flow. Never commit credentials, service-account keys or `.env` files.

`PS3_DATA_DIR` in the example is a convenience placeholder only; current commands require the explicit `--data` or `--input` argument and do not read it automatically.

## Running Locally

Serve the production build and API together:

```sh
source .venv/bin/activate
uvicorn doors.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. For frontend development, keep that backend running and use a second terminal:

```sh
cd frontend
pnpm dev
```

Vite proxies `/api` and `/health` to port 8000. API documentation is at `/docs`.

## Training

Run these from the repository root with the virtual environment active:

```sh
python -m doors.train --data /absolute/path/PS3/02_Datasets --subsystem door
python -m doors.train --data /absolute/path/PS3/02_Datasets --subsystem acv
python -m doors.train --data /absolute/path/PS3/02_Datasets --subsystem rail
python -m doors.train --data /absolute/path/PS3/02_Datasets --subsystem shm
```

Each command overwrites that subsystem's model and report. Use `--subsystem all` to run all four. Feature caching in `.cache/` is keyed by data and extraction-code hashes. Restart the API after retraining because models are cached in-process.

## Evaluation

The training commands above are also the exact per-subsystem benchmark/evaluation commands: development comparisons, one internal holdout evaluation and final refit are one reproducible pipeline. Read `reports/door.json`, `reports/acv.json`, `reports/rail.json`, or `reports/shm.json` for the measured result. Do not evaluate the refitted model on training records and call that a holdout score.

Imputation and scaling are fitted inside folds. Complete files/segments are the split units, not sensor rows. Rail exact-duplicate groups never cross folds. Unknown run/asset relationships and near-duplicate recordings remain limitations.

## Generating Predictions

Use the web workspace, or the identical inference layer through the CLI:

```sh
python predict.py --subsystem door --input /absolute/path/PS3/02_Datasets/Door/Test.csv --output door_predictions.csv
python predict.py --subsystem acv --input /absolute/path/PS3/02_Datasets/ACV/Test --output acv_predictions.csv
python predict.py --subsystem rail --input /absolute/path/PS3/02_Datasets/Rail_Corrugation/Test --output rail_predictions.csv
python predict.py --subsystem shm --input /absolute/path/PS3/02_Datasets/SHM/Test --output shm_predictions.csv
```

| Filename | Exact columns |
|---|---|
| `door_predictions.csv` | `start_time,end_time,prediction` |
| `acv_predictions.csv` | `file_id,ranked_cars` |
| `rail_predictions.csv` | `file_id,prediction` |
| `shm_predictions.csv` | `file_id,prediction` |

File IDs preserve the source basename, including extension. ACV rankings use pipe-separated two-digit car IDs. Door emits full timestamps and the official class spelling. Explanations and confidence-like scores never appear as extra competition columns.

## Creating predictions.zip

In the app, analyse the files and select **Prediction exports → Download predictions.zip** (on a narrow screen, use **Exports**). Re-uploading a matching file replaces its result.

To generate the complete official submission through the shared application HTTP handlers:

```sh
python -m scripts.generate_submission --data /absolute/path/PS3/02_Datasets --output submission
python -m scripts.verify_submission --source /absolute/path/PS3 --submission submission
```

The ZIP contains exactly four top-level prediction CSVs. The verification checks schemas against official examples, valid values, full file coverage and ZIP/CSV consistency. The included submission processed all 86 test inputs: 38 Door prediction rows, one ACV ranking, 68 Rail labels and 16 SHM estimates. Their correctness on the hidden answers is unknown.

## Testing

```sh
python -m pytest -q
cd frontend
pnpm build
cd ..
python -m scripts.smoke_http --url http://127.0.0.1:8000 --data /absolute/path/PS3/02_Datasets
```

17 automated tests pass. CI on the merged implementation passed. Production frontend and Docker builds passed. The HTTP smoke test exercises health, static UI, one actual input per subsystem, ZIP export and empty-upload rejection. To test the hosted app, replace `--url` with its live URL; this uploads the selected official inputs to the service.

## Docker

```sh
docker build -t doors-are-closing:phase2 .
docker run --rm -p 127.0.0.1:8080:8080 doors-are-closing:phase2
```

The multi-stage image compiles the frontend, installs pinned Python dependencies, packages models and serves as a non-root user. No training dataset is needed at runtime.

## Deployment

Deployed to Google Cloud Run in `asia-southeast1`, project `qwiklabs-gcp-02-5b63fcc9d6b8`. Revision `doors-are-closing-00001-5lz` serves the merged implementation. All four live HTTP workflows and ZIP export passed after deployment. [Deployment verification](docs/DEPLOYMENT.md) records the tested revision and limits.

To deploy to an approved project with billing, APIs and sufficient IAM permissions:

```sh
gcloud auth login
export GOOGLE_CLOUD_PROJECT=your-approved-project-id
export CLOUD_RUN_REGION=asia-southeast1
bash deploy.sh
```

Source deployment uses Cloud Build and Artifact Registry. Configuration is two CPUs, 4 GiB memory, concurrency one, timeout 300 seconds, zero minimum instances and at most three instances. Public access is intentional for the demo. Cold-start and sustained-load behaviour are not benchmarked. The app's file limit is 40 MiB, but hosting/proxy request-size limits may be lower; split batches into individual files. Do not upload sensitive operational data without an approved security design.

## Research References

- [Official Nebula X PS3 specification, subsystem Info Kits and examples](https://github.com/aochinwen/NebulaX-Hackathon-ProblemStatement/tree/main/PS3): task definitions, schemas and metrics.
- Nelisa Mabaso, *Predicting Train Door Failures Using Machine Learning Techniques*, supplied WCRR paper: [ResearchGate record](https://www.researchgate.net/publication/398941497_Predicting_Train_Door_Failures_Using_Machine_Learning_Techniques). Used as a candidate-family reference, not as evidence of Nebula X performance.
- Supplied ST2334 Probability & Statistics lecture notes: statistical moments, dependence, sampling and uncertainty reasoning. The private notes are not redistributed.
- [rainflow implementation](https://pypi.org/project/rainflow/): cycle extraction implementation used by SHM. This is a software reference, not a claim that dataset-specific fatigue constants were supplied.

## Limitations

Small, imbalanced and potentially dependent datasets limit generalisation. This is retrospective diagnosis/estimation, not a validated forecast of future failure. Door segmentation depends on acquisition structure; ACV faults may have causes other than refrigerant leakage; Rail unseen operating regimes are untested; SHM cannot provide physical life estimates without units and fatigue constants. Charts are downsampled overviews while inference uses full readings. Authentication, per-user rate limiting, drift monitoring and calibrated decision thresholds are not production-ready. Demo hosting availability depends on the lab project.

## Future Improvements

Collect independent asset/run metadata; validate across operating regimes; stress-test uninterrupted Door recordings; obtain additional ACV fault cases; improve minority-side Rail evaluation; verify SHM units and engineering constants; calibrate scores on independent data; add monitoring, access controls and sustained-load tests. Only benchmark advanced temporal models where they can plausibly improve these tasks.

The video-pitch script remains on hold until explicitly requested.
