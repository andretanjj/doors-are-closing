# Phase 2 implementation and measured evidence

## Status

Four-subsystem application, trained artefacts, validation reports, upload/export APIs, React interface, automated tests, Docker packaging, and Cloud Run deployment command implemented. Cloud deployment and hosted end-to-end verification remain pending an approved Google Cloud project and authenticated deployment access. No organiser-held-out score is known. README is reserved for Phase 3; video pitch remains on hold.

## Local validation results

| Subsystem | Selected approach | Development CV | Internal evaluation | Official metric |
|---|---|---:|---:|---|
| Door | Acquisition-gap/command segmentation + random forest | 1.0000 | 1.0000 | IoU-weighted F1 |
| ACV | Positive peer-cabin-temperature residual | 0.9792 | 0.9792 nested leave-one-workbook-out | Rank-decay score |
| Rail | Spectral/wavelet/side features + regularised logistic regression | 0.7523 | 0.7858 | Macro F1 |
| SHM | Rainflow power-law calibration | 0.9723 | 0.9777 | max(0, 1 − MAPE) |

These are LOCAL VALIDATION RESULTS, not literature results or organiser test scores. Models were chosen using development folds, then evaluated once on internal holdouts and refitted on all labelled training data. No tuning used Test folders. Candidate grids are deliberately small, fixed configurations in `doors/models.py`; there was no broad hyperparameter search or deep-learning experiment. No ensemble was necessary.

### Candidate comparisons

Door development CV: majority 0.7647; logistic 0.9853; decision tree 0.9853; random forest 1.0000; GBM 0.9853; SVM 0.9559; LightGBM 1.0000; XGBoost 0.9853. Random forest wins the deterministic insertion-order tie and needs no native boosting runtime for inference.

ACV: fixed car order 0.7708; thermal residual 0.9792; robust residual 0.9167; thermal-plus-variability 0.9792; Isolation Forest peer-outlier ranking 0.8750. Isolation Forest is transductive: fitted on the eight cars within each input workbook without labels; no population normalisation is learned using held-out workbooks. Simpler thermal rule wins its tie.

Rail: majority 0.3078; logistic 0.7523; decision tree 0.7296; random forest 0.7180; GBM 0.7158; SVM 0.6696; LightGBM 0.7114; XGBoost 0.6948. This dataset supports a non-tree alternative without requiring a neural model.

SHM: median 0.0852; calibrated rainflow 0.9723; log-ridge 0.8090; log-ExtraTrees 0.9271; log-LightGBM 0.8942. The selected proxy interpolates log cycle-power sums, optimising its exponent on training-only relative error. It is an empirical calibration, not a certified S–N fatigue calculation.

### Leakage controls and uncertainty

- Door: 110 complete operations. First 88 form forward-chaining development folds; last 22 form a chronological holdout. All 110 discovered boundaries exactly match supplied training boundaries, so classification correctness equals end-to-end IoU-F1 in this dataset. Timestamps and filenames are not classifier features. Segmentation structure was inspected on supplied training data, so the boundary score is not an independent test of generalisation to live streams.
- ACV: six whole workbooks, never row splits. Each outer workbook is scored by the rule selected using the other five. A complete eight-car ranking is required. A warmer cabin is evidence, not a confirmed leak. Six files are not proof of six independent assets; asset/run metadata is unavailable.
- Rail: 272 files, 270 unique SHA-256 values. Exact duplicate pairs are kept together. Stratified grouped holdout has 54 files (47 Normal, 2 Side I, 5 Side II); remaining files use four grouped folds. Unknown run/asset relationships and near-duplicate leakage remain possible. Holdout confusion matrix, rows/columns Normal, Side I, Side II: `[[46,1,0],[1,1,0],[1,0,4]]`.
- SHM: 64 files, 48 development and 16 holdout, stratified by damage quartiles. Four development folds. No assumed stress units, S–N constants, or sampling rate. Outputs are cumulative damage in the reference-label scale, not remaining useful life.
- Imputation and scaling are fitted inside each training fold. No raw recording is split across train/validation. Random seed 2334, data fingerprints, fold memberships, and out-of-fold predictions are recorded in `reports/*.json`.
- File/segment bootstrap intervals are descriptive and assume exchangeability: Rail approximately [0.507, 1.000], SHM [0.962, 0.989]. Rail minority counts make its interval unstable. Door's bootstrap [1,1] is degenerate because all 22 holdout predictions are correct; it is NOT evidence of zero generalisation uncertainty. Temporal dependence can invalidate a naive segment bootstrap. No narrow ACV interval is claimed.

## Paper and statistics actually used

Nelisa Mabaso's supplied *Predicting Train Door Failures Using Machine Learning Techniques* was used as a candidate-family reference: all five named tree families (DT, RF, GBM, XGBoost, LightGBM) were benchmarked on Door. Its different target, features, temporal setup and evaluation cannot establish superiority here. We do not transplant its reported performance. Source: the user-supplied WCRR PDF and the ResearchGate record supplied in the instructions.

Useful ST2334 ideas are applied through moments, robust peer deviations, train-only standardisation, dependence-aware splitting, and uncertainty caveats. Robust residual scores do not assume Gaussian sensor distributions. No Bayesian or causal interpretation is attached to a model score.

## Maintenance interface

Select subsystem → upload CSV/XLSX → analyse → inspect prediction and descriptive evidence → download CSV or aggregate `predictions.zip`. Multiple ACV/Rail/SHM files can be processed in sequence; Door accepts one stream. Re-upload replaces matching file results; reloading the page clears the session. Uploaded files are held temporarily and removed after processing; raw files are not logged or committed.

Door shows a downsampled current/position timeline with abnormal intervals and per-operation model support. ACV shows cabin traces and all eight ranked cars. Rail shows side-mean vibration, Welch spectra and axle-box RMS heatmaps. SHM shows stress history and rainflow range counts. Feature deviations are explicitly descriptive, not SHAP values or causal explanations. Model-support values are uncalibrated and are not fault probabilities. Plots are reduced overviews; CSV predictions use full-resolution data.

## Submission checks

The shared HTTP application processed all 86 official inputs: one Door stream (38 prediction rows), one ACV workbook, 68 Rail files, and 16 SHM files. Output columns match the official examples exactly. ZIP has four top-level CSVs, no raw data and no extra columns. File coverage is checked by `scripts/verify_submission.py`. Inference timings are in `submission/inference_manifest.json`; they are local measurements, not cloud latency promises.

## Remaining limitations

Hackathon prototype, not a safety-certified diagnostic product. Segmentation relies on the provided acquisition pattern. ACV needs valid cabin readings for every car. Dataset-specific fixed rail schema and sequence length are enforced. Sensor drift, unseen operating regimes, model calibration, asset-level generalisation, rate limiting/authentication, monitoring, and sustained load tests are not established. Public Cloud Run access is intended only for the demo; never upload sensitive operational telemetry without an approved security design.

The container packages the selected trained models; trusted repository artefacts only. Never load an untrusted joblib file. Cloud Run source deployment builds for its target platform. Local Docker verification on Apple Silicon is not proof of a deployed Linux AMD64 service.

## Verification completed

17 automated tests passed. TypeScript and Vite production build passed. Docker build passed and a real-network container smoke test exercised health, static frontend, one upload per subsystem, combined ZIP export and invalid-upload rejection. Browser checks exercised Door and ACV upload/results, including charts and ranking. All 86 supplied test inputs were separately run through the shared application endpoints and all four output files passed official-example schema and coverage checks. Cloud hosting and hosted end-to-end testing are still pending, not claimed complete.
