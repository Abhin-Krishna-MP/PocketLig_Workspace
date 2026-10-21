FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    POCKETLIG_ARTIFACTS=artifacts \
    PYTHONPATH=/app/src:/app

WORKDIR /app

# Install minimal OS dependencies for RDKit and OpenBabel
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY pocketlig/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and precomputed artifacts
COPY pocketlig/src/ /app/src/
COPY pocketlig/artifacts/ /app/artifacts/
COPY pocketlig/scripts/ /app/scripts/

EXPOSE 7860
EXPOSE 8008

CMD ["uvicorn", "pocketlig.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
