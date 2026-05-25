# ── CancerGPT Dockerfile ──────────────────────────────────────────────────────
# Build:  docker build -t cancergpt .
# Run:    docker-compose up --build

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fix anthropic/httpx version conflict (proxies argument removed in newer httpx)
RUN pip install --upgrade "anthropic>=0.40.0" "httpx>=0.27.0"

# Copy application code
COPY app.py .

# Copy ML model artifacts (must be in same folder as Dockerfile)
COPY cancergpt_best_model.pkl .
COPY cancergpt_probe_cols.pkl .
COPY cancergpt_gdsc_results.csv .
COPY cancergpt_gdsc_features.csv .

# Copy frontend
COPY index.html .

# Expose port
EXPOSE 5000

# Health check — Docker will restart container if API goes down
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Start with gunicorn (production WSGI server)
CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
