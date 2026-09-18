FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
RUN corepack enable
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080 OMP_NUM_THREADS=2
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY doors/ ./doors/
COPY models/ ./models/
COPY reports/ ./reports/
COPY predict.py ./
COPY --from=frontend /frontend/dist ./frontend/dist
RUN useradd --create-home appuser
USER appuser
EXPOSE 8080
CMD ["sh", "-c", "uvicorn doors.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
