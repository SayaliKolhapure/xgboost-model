FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py .
COPY index.html .

# Copy XGBoost model files
COPY xgboost_best_model.pkl .
COPY xgboost_probe_cols.pkl .
COPY xgboost_gdsc_results.csv .
COPY xgboost_gdsc_features.csv .
COPY xgboost_mutation_model.pkl .
COPY xgboost_mutation_results.csv .
COPY xgboost_mutation_features.csv .

# Fix permissions so Flask can read all files
RUN chmod -R 755 /app

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
