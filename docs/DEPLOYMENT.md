# Deployment verification

- Project: `qwiklabs-gcp-02-5b63fcc9d6b8`
- Region: `asia-southeast1`
- Service: `doors-are-closing`
- URL: https://doors-are-closing-895395923568.asia-southeast1.run.app
- Revision: `doors-are-closing-00001-5lz`, 100% traffic
- Source: merged main commit `76c08103f75923ace7e112af78ba6f0e5c7cfeba`
- Build: `f9f1e61e-e90c-4a61-9ac3-59e19e7863c0`
- Access: public demo; no sensitive operational uploads
- Limits: two CPUs, 4 GiB, concurrency one, 300-second timeout, zero minimum / three maximum instances

The deployment command completed successfully. The public service passed `scripts/smoke_http.py` with the official Door Test.csv, ACV test workbook, Rail Test1.csv and SHM test01.csv. Health reported all models ready, the frontend loaded, all inference responses succeeded, combined ZIP contained exactly the four required filenames, and an empty upload returned HTTP 422.

Measured server processing times for this one smoke run: Door 0.562 s (38 operations), ACV 8.238 s (one ranking), Rail 0.679 s (one class), SHM 0.852 s (one estimate). These are individual observations, not latency percentiles, cold-start measurements or a service guarantee. ZIP response was 1,172 bytes for those four selected inputs.

This completes the previously pending cloud deployment and live HTTP verification. It does not establish production safety, load capacity, organiser-held-out accuracy, or the lab project's expiry date. Availability is contingent on the organiser's project lifetime and policies.
