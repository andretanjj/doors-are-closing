# Phase 2 runbook

Commands run from this repository root. Python 3.12, Node 22, pnpm 11.19.0. macOS training with boosting models additionally requires `brew install libomp`; Linux requires `libgomp1`.

## Install and run

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
corepack enable
cd frontend
pnpm install --frozen-lockfile
pnpm build
cd ..
uvicorn doors.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. For frontend development, run `pnpm dev` in `frontend` with the backend on port 8000. Vite proxies `/api` and `/health` to the backend. Models are packaged in `models`; training is not required to run the app.

## Reproduce training and evaluation

Get the official [PS3 dataset and documentation](https://github.com/aochinwen/NebulaX-Hackathon-ProblemStatement/tree/main/PS3). The audited source commit is `16526c02579c7f37e54eaaa42a4cc6d4ceb19994`. Keep it outside this repository. Do not modify or redistribute the source data.

```sh
python -m doors.train --data /absolute/path/PS3/02_Datasets --subsystem door
python -m doors.train --data /absolute/path/PS3/02_Datasets --subsystem acv
python -m doors.train --data /absolute/path/PS3/02_Datasets --subsystem rail
python -m doors.train --data /absolute/path/PS3/02_Datasets --subsystem shm
```

Each command performs fixed development comparisons, one internal evaluation and a final all-training-data fit. Reports include measured scores, folds and input fingerprints. Training overwrites that subsystem's artefact and report. Restart the API after retraining because artefacts are cached in-process. The `.cache` directory stores reproducible feature extractions keyed by code/data SHA-256, not raw data.

## Predict and package

```sh
python predict.py --subsystem door --input /absolute/path/PS3/02_Datasets/Door/Test.csv --output door_predictions.csv
python predict.py --subsystem acv --input /absolute/path/PS3/02_Datasets/ACV/Test --output acv_predictions.csv
python predict.py --subsystem rail --input /absolute/path/PS3/02_Datasets/Rail_Corrugation/Test --output rail_predictions.csv
python predict.py --subsystem shm --input /absolute/path/PS3/02_Datasets/SHM/Test --output shm_predictions.csv
python -m scripts.generate_submission --data /absolute/path/PS3/02_Datasets --output submission
python -m scripts.verify_submission --source /absolute/path/PS3 --submission submission
```

The generator invokes the same upload/export HTTP handlers used by the UI via an in-process test client. ZIP and four CSVs are saved in `submission`. It does not obtain test answers or compute a held-out test score.

## Test

```sh
python -m pytest -q
cd frontend
pnpm build
```

## Container and deployment

```sh
docker build -t doors-are-closing:phase2 .
docker run --rm -p 127.0.0.1:8080:8080 doors-are-closing:phase2
```

Health: `GET /health`; documentation: `/docs`; model evidence: `GET /api/models`; multipart upload: `POST /api/analyse/{door|acv|rail|shm}` with field `file`; ZIP export: `POST /api/export` with subsystem-to-prediction-row arrays.

Deploy only after choosing the approved Google Cloud project, enabling billing/APIs, and authenticating the Google Cloud CLI:

```sh
export GOOGLE_CLOUD_PROJECT=your-approved-project-id
export CLOUD_RUN_REGION=asia-southeast1
bash deploy.sh
```

This creates/updates a public demo service, using two CPUs, 4 GiB memory, one concurrent request per instance and at most three instances. Source builds require Cloud Build/Artifact Registry and appropriate IAM permissions. No credentials belong in Git. `.env.example` documents configuration; the shell, not the app, consumes deployment variables. `PORT` is supplied by Cloud Run. The [live app](https://doors-are-closing-895395923568.asia-southeast1.run.app) passed the four-subsystem HTTP smoke test; see [deployment verification](DEPLOYMENT.md). Re-run the smoke test after future deployments and inspect the browser upload/results workflow before handoff.
