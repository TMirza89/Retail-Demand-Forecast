# ── Base Image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Set Working Directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Install System Dependencies ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Copy Requirements First (layer caching) ───────────────────────────────────
# Docker caches each layer — copying requirements first means
# packages only reinstall when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy Project Files ────────────────────────────────────────────────────────
COPY . .

# ── Expose Port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Run FastAPI ───────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]