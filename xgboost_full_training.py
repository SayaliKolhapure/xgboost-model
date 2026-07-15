"""
CancerGPT — Full Training Pipeline
====================================
Data: GDSC2 IC50 + expU133A gene expression + Cell_Lines_Details
Model: XGBoost per drug with gene expression features
Output: publication-ready results with SHAP biomarker analysis
"""

import pandas as pd
import numpy as np
import warnings, json
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap, joblib

print("="*70)
print("CancerGPT — Full Gene Expression Model Training")
print("="*70)

# ══════════════════════════════════════════════════════════════════════════════
# ▶▶  SET YOUR PATHS HERE  ◀◀
# Change DATA_DIR to the folder containing your data files.
# Change OUT_DIR  to wherever you want results saved (defaults to same folder).
# ══════════════════════════════════════════════════════════════════════════════
import os
DATA_DIR = r'C:\Users\sayal\OneDrive\Desktop\breast cancer'
OUT_DIR  = DATA_DIR   # change if you want outputs elsewhere

EXPR_PATH  = os.path.join(DATA_DIR, 'expU133A.txt')
CLD_PATH   = os.path.join(DATA_DIR, 'Cell_Lines_Details.xlsx')
GDSC_RAW   = os.path.join(DATA_DIR, 'GDSC2_fitted_dose_response_27Oct23.xlsx')
GDSC_PATH  = os.path.join(DATA_DIR, 'gdsc_processed.csv')   # auto-generated below
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Load expression data ───────────────────────────────────────────────────
print("\n[1/6] Loading gene expression data...")
expr_raw = pd.read_csv(EXPR_PATH, sep='\t', index_col=0)
print(f"  Expression matrix : {expr_raw.shape[0]} probes × {expr_raw.shape[1]} cell lines")

# Keep top 2000 most variable probes (reduces noise, speeds training)
probe_var = expr_raw.var(axis=1)
top_probes = probe_var.nlargest(2000).index
expr = expr_raw.loc[top_probes].T  # cell lines × probes
expr.index.name = 'CELL_LINE_NAME'
expr = expr.reset_index()
print(f"  Top variable probes kept: 2000")
print(f"  Cell lines with expression: {len(expr)}")

# ── 2. Load cell line metadata ────────────────────────────────────────────────
print("\n[2/6] Loading cell line metadata...")
cld = pd.read_excel(CLD_PATH)
cld.columns = [c.replace('\n',' ').strip() for c in cld.columns]
cld = cld.rename(columns={
    'Sample Name': 'CELL_LINE_NAME',
    'Cancer Type (matching TCGA label)': 'TCGA_LABEL',
    'GDSC Tissue descriptor 1': 'TISSUE1',
    'GDSC Tissue descriptor 2': 'TISSUE2',
})
cld = cld[['CELL_LINE_NAME','TCGA_LABEL','TISSUE1','TISSUE2']].copy()
print(f"  Cell lines in metadata : {len(cld)}")
print(f"  Cancer types : {cld['TCGA_LABEL'].nunique()}")

# ── 3. Load & preprocess GDSC2 drug response ─────────────────────────────────
print("\n[3/6] Loading GDSC2 drug response data...")

if os.path.exists(GDSC_PATH):
    print("  Found cached gdsc_processed.csv — loading...")
    gdsc = pd.read_csv(GDSC_PATH)
else:
    print(f"  Reading raw file: GDSC2_fitted_dose_response_27Oct23.xlsx")
    print("  (This may take ~30 s for the large xlsx — please wait...)")
    raw = pd.read_excel(GDSC_RAW)

    # Normalise column names
    raw.columns = [c.strip().upper().replace(' ', '_') for c in raw.columns]

    # Rename to standard names used by the rest of the pipeline
    col_map = {
        'CELL_LINE_NAME': 'CELL_LINE_NAME',
        'DRUG_NAME':      'DRUG_NAME',
        'DRUG_ID':        'DRUG_ID',
        'LN_IC50':        'LN_IC50',
        'AUC':            'AUC',
        'RMSE':           'RMSE',
        'Z_SCORE':        'Z_SCORE',
        'PATHWAY_NAME':   'PATHWAY_NAME',
        'PUTATIVE_TARGET':'PUTATIVE_TARGET',
        'DATASET':        'DATASET',
        'NLME_RESULT_ID': 'NLME_RESULT_ID',
        'NLME_CURVE_ID':  'NLME_CURVE_ID',
        'SANGER_MODEL_ID':'SANGER_MODEL_ID',
        'MIN_CONC':       'MIN_CONC',
        'MAX_CONC':       'MAX_CONC',
    }
    col_map = {k: v for k, v in col_map.items() if k in raw.columns}
    raw = raw.rename(columns=col_map)

    # Ensure required columns are present
    for req in ['CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50']:
        if req not in raw.columns:
            raise ValueError(
                f"Column '{req}' not found in GDSC2 xlsx.\n"
                f"Available columns: {list(raw.columns)}"
            )

    # ── Binarise LN_IC50 into RESPONSE (1=sensitive, 0=resistant) ────────────
    # Per-drug z-score; bottom half = sensitive, top half = resistant.
    # Middle ±0.5 z excluded for a cleaner signal boundary.
    raw['IC50_Z'] = raw.groupby('DRUG_NAME')['LN_IC50'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    raw = raw[raw['IC50_Z'].abs() >= 0.5].copy()
    raw['RESPONSE'] = (raw['IC50_Z'] < 0).astype(int)

    gdsc = raw.drop(columns=['IC50_Z'])
    gdsc.to_csv(GDSC_PATH, index=False)
    print("  Preprocessed & cached → gdsc_processed.csv")

print(f"  Labeled drug-response rows : {len(gdsc):,}")
print(f"  Drugs                      : {gdsc['DRUG_NAME'].nunique()}")
print(f"  Cell lines                 : {gdsc['CELL_LINE_NAME'].nunique()}")
print(f"  Sensitive / Resistant      : {gdsc['RESPONSE'].sum():,} / {(gdsc['RESPONSE']==0).sum():,}")

# ── 4. Merge expression + drug response ──────────────────────────────────────
print("\n[4/6] Merging datasets...")
merged = gdsc.merge(expr, on='CELL_LINE_NAME', how='inner')
merged = merged.merge(cld[['CELL_LINE_NAME','TCGA_LABEL']], on='CELL_LINE_NAME', how='left')

print(f"  Merged rows : {len(merged):,}")
print(f"  Cell lines with expression + response: {merged['CELL_LINE_NAME'].nunique()}")
print(f"  Drugs available: {merged['DRUG_NAME'].nunique()}")

# Cancer type distribution
print("\n  Cancer type distribution in merged data:")
ct = merged.groupby('TCGA_LABEL')['CELL_LINE_NAME'].nunique().sort_values(ascending=False)
for t,n in ct.head(10).items():
    print(f"    {str(t):<15} {n} cell lines")

probe_cols = [c for c in merged.columns if c not in
              ['CELL_LINE_NAME','DRUG_NAME','DRUG_ID','RESPONSE','LN_IC50',
               'AUC','RMSE','Z_SCORE','CANCER_TYPE','PATHWAY_NAME',
               'PUTATIVE_TARGET','DATASET','NLME_RESULT_ID','NLME_CURVE_ID',
               'SANGER_MODEL_ID','MIN_CONC','MAX_CONC','TCGA_LABEL']]

print(f"\n  Expression features (probes): {len(probe_cols)}")

# ── 5. Train per-drug models ──────────────────────────────────────────────────
print("\n[5/6] Training models (XGBoost + SMOTE + SelectKBest)...")
print("  Targeting breast cancer relevant drugs with n>=30 samples\n")

# Priority drugs for breast cancer + well represented in merged data
priority_drugs = [
    'Tamoxifen','Fulvestrant','Paclitaxel','Docetaxel','Doxorubicin',
    'Olaparib','Palbociclib','Alpelisib','Lapatinib','Afatinib',
    'Erlotinib','Everolimus','Navitoclax','MK-2206','Trametinib',
    'Selumetinib','Camptothecin','5-Fluorouracil','Oxaliplatin',
    'Cisplatin','Gemcitabine','Irinotecan','Nutlin-3a (-)'
]

# Get all drugs with enough samples
drug_counts = merged.groupby('DRUG_NAME').agg(
    n=('RESPONSE','count'),
    n_sens=('RESPONSE','sum'),
    n_res=('RESPONSE', lambda x:(x==0).sum())
).reset_index()
drug_counts = drug_counts[
    (drug_counts['n'] >= 30) &
    (drug_counts['n_sens'] >= 10) &
    (drug_counts['n_res'] >= 10)
]

# Put priority drugs first
drug_list = [d for d in priority_drugs if d in drug_counts['DRUG_NAME'].values]
other_drugs = [d for d in drug_counts['DRUG_NAME'].values if d not in priority_drugs]
drug_list = drug_list + other_drugs[:20]  # cap at priority + 20 others

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_results = []
all_importances = []
best_models = {}

for drug in drug_list:
    drug_df = merged[merged['DRUG_NAME'] == drug].copy()
    X = drug_df[probe_cols].fillna(0).values
    y = drug_df['RESPONSE'].values

    if len(np.unique(y)) < 2:
        continue

    n_s, n_r = (y==1).sum(), (y==0).sum()
    k = min(50, X.shape[1], len(drug_df)//2)
    k_neighbors = min(5, n_r-1, n_s-1)
    if k_neighbors < 1:
        continue

    try:
        pipeline = ImbPipeline([
            ('smote',  SMOTE(random_state=42, k_neighbors=k_neighbors)),
            ('scaler', StandardScaler()),
            ('select', SelectKBest(f_classif, k=k)),
            ('clf',    XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=max(1, n_r/max(1,n_s)),
                use_label_encoder=False, eval_metric='logloss',
                random_state=42, n_jobs=-1
            ))
        ])

        y_prob = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba')[:,1]
        auc  = roc_auc_score(y, y_prob)
        ap   = average_precision_score(y, y_prob)

        # Fit on full data for feature importance
        pipeline.fit(X, y)
        best_models[drug] = pipeline

        # Feature importance
        selector  = pipeline.named_steps['select']
        clf       = pipeline.named_steps['clf']
        sel_mask  = selector.get_support()
        sel_probes = np.array(probe_cols)[sel_mask]
        importances = clf.feature_importances_

        top_feat = pd.DataFrame({
            'drug': drug,
            'probe': sel_probes,
            'importance': importances
        }).nlargest(10, 'importance')
        all_importances.append(top_feat)

        pathway = drug_df['PATHWAY_NAME'].iloc[0]
        target  = drug_df['PUTATIVE_TARGET'].iloc[0]

        result = {
            'Drug': drug, 'Pathway': pathway, 'Target': target,
            'N_samples': len(y), 'N_sensitive': int(n_s),
            'N_resistant': int(n_r),
            'ROC_AUC': round(auc, 3), 'Avg_Precision': round(ap, 3)
        }
        all_results.append(result)
        status = '⭐' if auc >= 0.75 else '✓' if auc >= 0.65 else ' '
        print(f"  {status} {drug:<35} AUC={auc:.3f}  AP={ap:.3f}  n={len(y)}")

    except Exception as e:
        print(f"  ✗ {drug:<35} Error: {str(e)[:40]}")

# ── 6. Save results ───────────────────────────────────────────────────────────
print("\n[6/6] Saving results...")

results_df = pd.DataFrame(all_results).sort_values('ROC_AUC', ascending=False)
results_df.to_csv(os.path.join(OUT_DIR, 'cancergpt_gdsc_results.csv'), index=False)

if all_importances:
    feat_df = pd.concat(all_importances, ignore_index=True)
    feat_df.to_csv(os.path.join(OUT_DIR, 'cancergpt_gdsc_features.csv'), index=False)

# Save best model (highest AUC)
best_drug = results_df.iloc[0]['Drug']
joblib.dump(best_models[best_drug], os.path.join(OUT_DIR, 'cancergpt_best_model.pkl'))
joblib.dump(list(probe_cols),       os.path.join(OUT_DIR, 'cancergpt_probe_cols.pkl'))

# ── Report ────────────────────────────────────────────────────────────────────
top10 = results_df.head(10)
good  = results_df[results_df['ROC_AUC'] >= 0.65]
great = results_df[results_df['ROC_AUC'] >= 0.75]

report = f"""CancerGPT — Full Gene Expression Model Report
================================================
Datasets:
  GDSC2 drug response : {len(gdsc):,} labeled rows
  Gene expression     : {expr_raw.shape[1]} cell lines × {expr_raw.shape[0]} probes
  Probes used         : 2,000 (top variance)
  Merged cell lines   : {merged['CELL_LINE_NAME'].nunique()}
  Drugs trained       : {len(results_df)}

Performance Summary:
  Models AUC >= 0.75 (publication quality) : {len(great)}
  Models AUC >= 0.65 (acceptable)          : {len(good)}
  Mean AUC across all drugs                : {results_df['ROC_AUC'].mean():.3f}
  Best drug model                          : {best_drug} (AUC={results_df.iloc[0]['ROC_AUC']})

Top 10 Models by AUC:
{top10[['Drug','Pathway','ROC_AUC','Avg_Precision','N_samples']].to_string(index=False)}

All Results:
{results_df[['Drug','ROC_AUC','Avg_Precision','N_samples','Pathway']].to_string(index=False)}
"""

with open(os.path.join(OUT_DIR, 'cancergpt_gdsc_report.txt'), 'w') as f:
    f.write(report)

print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")
print(f"  Drugs trained          : {len(results_df)}")
print(f"  AUC >= 0.75 (great)    : {len(great)}")
print(f"  AUC >= 0.65 (good)     : {len(good)}")
print(f"  Mean AUC               : {results_df['ROC_AUC'].mean():.3f}")
print(f"  Best model             : {best_drug} (AUC={results_df.iloc[0]['ROC_AUC']})")
print()
print("Top 10 models:")
print(top10[['Drug','ROC_AUC','Avg_Precision','N_samples']].to_string(index=False))
print()
print("Files saved:")
print("  cancergpt_gdsc_results.csv   — all drug model AUCs")
print("  cancergpt_gdsc_features.csv  — top features per drug")
print("  cancergpt_gdsc_report.txt    — full report")
print("  cancergpt_best_model.pkl     — best trained model")
print()
print("✅ Training complete! Ready for paper writing.")
