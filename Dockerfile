FROM python:3.10-slim

# Metadata
LABEL maintainer="Agastya Kumar Yadav"
LABEL description="API Debugging OpenEnv — RL environment for backend failure diagnosis"
LABEL version="1.0.0"

WORKDIR /app

# Install dependencies first (layer cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Runtime configuration via environment variables
ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=gpt-4o-mini
ENV HF_TOKEN=""
ENV PYTHONUNBUFFERED=1

# Run the deterministic baseline inference script
CMD ["python", "inference.py"]
