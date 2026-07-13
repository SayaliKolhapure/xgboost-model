FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY app.py .
COPY index.html .
COPY xgboost_best_model.pkl .
COPY xgboost_probe_cols.pkl .
COPY xgboost_gdsc_results.csv .
COPY xgboost_gdsc_features.csv .
COPY xgboost_mutation_model.pkl .
COPY xgboost_mutation_results.csv .
COPY xgboost_mutation_features.csv .

# Make ALL files world-readable
RUN find /app -type f -exec chmod 644 {} \; && \
    find /app -type d -exec chmod 755 {} \;

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
