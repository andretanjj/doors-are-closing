import logging
import os
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from .inference import ROOT, SCHEMAS, infer, model_summary, zip_predictions

app = FastAPI(title="Doors Are Closing", version="0.1.0")
logger = logging.getLogger("doors")
MAX_BYTES = 40*1024*1024


@app.get("/health")
def health():
    models = model_summary()
    return {"status":"ready" if all(m["ready"] for m in models.values()) else "starting", "models":models}


@app.get("/api/models")
def models():
    return model_summary()


@app.post("/api/analyse/{subsystem}")
async def analyse(subsystem: str, file: UploadFile = File(...)):
    if subsystem not in SCHEMAS:
        raise HTTPException(400, "Unknown subsystem.")
    suffix = ".xlsx" if subsystem == "acv" else ".csv"
    name = file.filename or ""
    if not name or Path(name).name != name or not name.lower().endswith(suffix) or name[0] in "=+-@":
        raise HTTPException(400, f"Upload a {suffix} file with a plain filename.")
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="doors-") as temp:
            path = Path(temp)/("input"+suffix)
            size = 0
            with path.open("wb") as output:
                while chunk := await file.read(1024*1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, "File exceeds the 40 MiB upload limit.")
                    output.write(chunk)
            if not size:
                raise ValueError("The uploaded file is empty.")
            if suffix == ".xlsx":
                with ZipFile(path) as archive:
                    if sum(i.file_size for i in archive.infolist()) > 600*1024*1024:
                        raise ValueError("Expanded workbook exceeds the supported size.")
            result = await run_in_threadpool(infer, subsystem, path, name)
            result["seconds"] = round(time.monotonic()-started, 3)
            logger.info("inference subsystem=%s bytes=%s seconds=%.3f", subsystem, size, result["seconds"])
            return result
    except HTTPException:
        raise
    except (ValueError, KeyError, BadZipFile, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    finally:
        await file.close()


@app.post("/api/export")
def export(predictions: dict[str, list[dict]]):
    if not predictions or set(predictions)-set(SCHEMAS):
        raise HTTPException(422, "Select valid subsystem results to export.")
    try:
        data = zip_predictions(predictions)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(data, media_type="application/zip", headers={"Content-Disposition":"attachment; filename=predictions.zip"})


DIST = ROOT/"frontend/dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST/"assets"), name="assets")

    @app.get("/{path:path}")
    def frontend(path: str):
        return FileResponse(DIST/"index.html")
