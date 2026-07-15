"""
CancerGPT — Mutation Subtype Stratification Upgrade
=====================================================
Adds mutation status (EGFR, KRAS, BRAF, BRCA1/2, TP53, PIK3CA etc.)
as binary features alongside gene expression for per-drug XGBoost models.

Key genes stratified:
  LUAD  → EGFR, KRAS, BRAF, ALK, MET, ROS1
  BRCA  → BRCA1, BRCA2, TP53, PIK3CA, ERBB2, CDH1
  SKCM  → BRAF, NRAS, NF1
  COAD  → KRAS, NRAS, BRAF, PIK3CA, APC, TP53
  ALL   → TP53, KRAS, NRAS, FLT3

Usage:
  python cancergpt_mutation_stratified.py

Output:
  cancergpt_mutation_results.csv      — AUC per drug with mutation features
  cancergpt_mutation_features.csv     — top features (expression + mutation)
  cancergpt_mutation_report.txt       — comparison vs expression-only
  cancergpt_mutation_model.pkl        — best stratified model
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

print("=" * 70)
print("CancerGPT — Mutation Subtype Stratification")
print("=" * 70)

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = r'C:\Users\sayal\OneDrive\Desktop\breast cancer'

EXPR_PATH     = os.path.join(DATA_DIR, 'expU133A.txt')
CLD_PATH      = os.path.join(DATA_DIR, 'Cell_Lines_Details.xlsx')
GDSC_PATH     = os.path.join(DATA_DIR, 'gdsc_processed.csv')
MUT_PATH      = os.path.join(DATA_DIR, 'mutations_all_20260316.csv')
OUT_DIR       = DATA_DIR

# Key mutation genes to use as binary features
KEY_GENES = [
    # Lung cancer
    'EGFR', 'KRAS', 'BRAF', 'ALK', 'MET', 'ROS1', 'RET', 'NTRK1',
    # Breast cancer
    'BRCA1', 'BRCA2', 'TP53', 'PIK3CA', 'ERBB2', 'CDH1', 'PTEN',
    'ESR1', 'RB1', 'CCND1', 'MYC',
    # Melanoma
    'NRAS', 'NF1', 'CDKN2A',
    # Colorectal
    'APC', 'SMAD4', 'FBXW7', 'NRAS',
    # Pan-cancer
    'MYC', 'CCNE1', 'MDM2', 'CDK4', 'CDK6', 'FGFR1', 'FGFR2',
    'IDH1', 'IDH2', 'VHL', 'ARID1A', 'ARID2',
    # DNA repair
    'ATM', 'CHEK2', 'PALB2', 'RAD51', 'BRIP1',
    # Apoptosis
    'BCL2', 'MCL1', 'BAX',
]
KEY_GENES = list(dict.fromkeys(KEY_GENES))  # remove duplicates

# ── STEP 1: Load expression data ──────────────────────────────────────────────
print("\n[1/6] Loading gene expression data...")
expr_raw  = pd.read_csv(EXPR_PATH, sep='\t', index_col=0)
probe_var = expr_raw.var(axis=1)
top_probes = probe_var.nlargest(2000).index
expr = expr_raw.loc[top_probes].T
expr.index.name = 'CELL_LINE_NAME'
expr = expr.reset_index()
print(f"  Expression: {len(expr)} cell lines × 2000 probes")

# ── STEP 2: Load mutation data ────────────────────────────────────────────────
print("\n[2/6] Loading mutation data...")

mut_features = None
if os.path.exists(MUT_PATH):
    try:
        # Try reading as XLS
        ext = os.path.splitext(MUT_PATH)[1].lower()
        if ext in ['.xls', '.xlsx']:
            mut_raw = pd.read_excel(MUT_PATH)
        else:
            mut_raw = pd.read_csv(MUT_PATH)

        print(f"  Raw mutation file shape: {mut_raw.shape}")
        print(f"  Columns: {list(mut_raw.columns[:10])}")

        # Normalise column names
        mut_raw.columns = [str(c).strip() for c in mut_raw.columns]

        # Detect cell line and gene columns
        # GDSC format: model_name (or CELL_LINE_NAME), gene_symbol, variant_classification
        cell_col = next((c for c in mut_raw.columns
                         if any(k in c.lower() for k in ['model_name','cell_line','sample'])), None)
        gene_col = next((c for c in mut_raw.columns
                         if any(k in c.lower() for k in ['gene','symbol','hugo'])), None)

        if cell_col and gene_col:
            print(f"  Cell line column: '{cell_col}'")
            print(f"  Gene column: '{gene_col}'")

            # Filter to key genes only
            mut_filtered = mut_raw[mut_raw[gene_col].isin(KEY_GENES)].copy()
            mut_filtered = mut_filtered[[cell_col, gene_col]].drop_duplicates()
            mut_filtered['mutated'] = 1

            # Pivot: rows=cell lines, cols=gene mutation status (0/1)
            mut_pivot = mut_filtered.pivot_table(
                index=cell_col, columns=gene_col,
                values='mutated', fill_value=0
            ).reset_index()
            mut_pivot = mut_pivot.rename(columns={cell_col: 'CELL_LINE_NAME'})

            # Rename columns to MUT_GENENAME
            mut_pivot.columns = ['CELL_LINE_NAME'] + [
                f'MUT_{g}' for g in mut_pivot.columns[1:]
            ]

            # Fill any missing key gene columns with 0
            for gene in KEY_GENES:
                col = f'MUT_{gene}'
                if col not in mut_pivot.columns:
                    mut_pivot[col] = 0

            mut_features = mut_pivot
            print(f"  Mutation features: {len(mut_features)} cell lines × {len(mut_features.columns)-1} genes")
            print(f"  Mutation rate per gene (top 10):")
            gene_cols = [c for c in mut_features.columns if c.startswith('MUT_')]
            rates = mut_features[gene_cols].mean().sort_values(ascending=False).head(10)
            for gene, rate in rates.items():
                print(f"    {gene:<20} {rate*100:.1f}% mutated")
        else:
            print(f"  ✗ Could not detect cell line/gene columns. Available: {list(mut_raw.columns)}")
    except Exception as e:
        print(f"  ✗ Error reading mutation file: {e}")
        print("  → Continuing with expression-only features")
else:
    print(f"  ✗ Mutation file not found: {MUT_PATH}")
    print("  → Continuing with expression-only features")

# ── STEP 3: Load drug response ────────────────────────────────────────────────
print("\n[3/6] Loading GDSC2 drug response...")
gdsc = pd.read_csv(GDSC_PATH)
print(f"  {len(gdsc):,} rows | {gdsc['DRUG_NAME'].nunique()} drugs | {gdsc['CELL_LINE_NAME'].nunique()} cell lines")

# ── STEP 4: Load previous results for comparison ──────────────────────────────
prev_results_path = os.path.join(DATA_DIR, 'cancergpt_gdsc_results.csv')
prev_results = {}
if os.path.exists(prev_results_path):
    prev_df = pd.read_csv(prev_results_path)
    prev_results = dict(zip(prev_df['Drug'], prev_df['ROC_AUC']))
    print(f"\n[4/6] Loaded {len(prev_results)} previous AUC scores for comparison")
else:
    print("\n[4/6] No previous results found — will show absolute AUC only")

# ── STEP 5: Merge all features ────────────────────────────────────────────────
print("\n[5/6] Merging expression + mutation + drug response...")
merged = gdsc.merge(expr, on='CELL_LINE_NAME', how='inner')

mut_cols = []
if mut_features is not None:
    merged = merged.merge(mut_features, on='CELL_LINE_NAME', how='left')
    mut_cols = [c for c in mut_features.columns if c.startswith('MUT_')]
    # Fill NaN mutation status with 0 (unknown = assume wild-type)
    merged[mut_cols] = merged[mut_cols].fillna(0).astype(int)
    print(f"  Merged: {len(merged):,} rows | {merged['CELL_LINE_NAME'].nunique()} cell lines")
    print(f"  Expression features: 2000 probes")
    print(f"  Mutation features  : {len(mut_cols)} genes")
    print(f"  Total features     : {2000 + len(mut_cols)}")
else:
    print(f"  Merged: {len(merged):,} rows (expression only, no mutation data)")

probe_cols = [c for c in merged.columns if c not in
              ['CELL_LINE_NAME','DRUG_NAME','DRUG_ID','RESPONSE','LN_IC50',
               'AUC','RMSE','Z_SCORE','CANCER_TYPE','PATHWAY_NAME',
               'PUTATIVE_TARGET','DATASET','NLME_RESULT_ID','NLME_CURVE_ID',
               'SANGER_MODEL_ID','MIN_CONC','MAX_CONC','TCGA_LABEL']]

print(f"\n  Total features for modelling: {len(probe_cols)}")

# ── STEP 6: Train stratified models ───────────────────────────────────────────
print("\n[6/6] Training mutation-stratified XGBoost models...")
print("  Expression + mutation binary features combined\n")

priority_drugs = [
    'Tamoxifen','Fulvestrant','Paclitaxel','Docetaxel','Doxorubicin',
    'Olaparib','Palbociclib','Alpelisib','Lapatinib','Afatinib',
    'Erlotinib','Everolimus','Navitoclax','MK-2206','Trametinib',
    'Selumetinib','Camptothecin','5-Fluorouracil','Oxaliplatin',
    'Cisplatin','Gemcitabine','Irinotecan','Nutlin-3a (-)'
]

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
drug_list = [d for d in priority_drugs if d in drug_counts['DRUG_NAME'].values]
other_drugs = [d for d in drug_counts['DRUG_NAME'].values if d not in priority_drugs]
drug_list = drug_list + other_drugs[:20]

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
        auc    = roc_auc_score(y, y_prob)
        ap     = average_precision_score(y, y_prob)

        pipeline.fit(X, y)
        best_models[drug] = pipeline

        # Feature importance — identify if top features are mutation or expression
        selector   = pipeline.named_steps['select']
        clf        = pipeline.named_steps['clf']
        sel_mask   = selector.get_support()
        sel_feats  = np.array(probe_cols)[sel_mask]
        importances= clf.feature_importances_

        top_feat = pd.DataFrame({
            'drug':        drug,
            'feature':     sel_feats,
            'importance':  importances,
            'type':        ['mutation' if f.startswith('MUT_') else 'expression' for f in sel_feats]
        }).nlargest(10, 'importance')
        all_importances.append(top_feat)

        # Compare with previous AUC
        prev_auc = prev_results.get(drug, None)
        delta    = round(auc - prev_auc, 3) if prev_auc else None
        delta_str= f"+{delta:.3f}" if delta and delta>0 else (f"{delta:.3f}" if delta else "new")

        pathway = drug_df['PATHWAY_NAME'].iloc[0]
        target  = drug_df['PUTATIVE_TARGET'].iloc[0]

        result = {
            'Drug':         drug,
            'Pathway':      pathway,
            'Target':       target,
            'N_samples':    len(y),
            'N_sensitive':  int(n_s),
            'N_resistant':  int(n_r),
            'ROC_AUC':      round(auc, 3),
            'Avg_Precision':round(ap, 3),
            'Prev_AUC':     round(prev_auc, 3) if prev_auc else None,
            'Delta_AUC':    delta,
            'Mut_features_in_top10': int(sum(1 for f in sel_feats[:10] if f.startswith('MUT_')))
        }
        all_results.append(result)

        status = '⭐' if auc>=0.75 else '✓' if auc>=0.65 else ' '
        delta_display = f"  Δ{delta_str}" if delta is not None else ""
        print(f"  {status} {drug:<35} AUC={auc:.3f}  AP={ap:.3f}  n={len(y)}{delta_display}")

    except Exception as e:
        print(f"  ✗ {drug:<35} Error: {str(e)[:50]}")

# ── Save results ──────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results).sort_values('ROC_AUC', ascending=False)
results_df.to_csv(os.path.join(OUT_DIR, 'cancergpt_mutation_results.csv'), index=False)

if all_importances:
    feat_df = pd.concat(all_importances, ignore_index=True)
    feat_df.to_csv(os.path.join(OUT_DIR, 'cancergpt_mutation_features.csv'), index=False)

best_drug = results_df.iloc[0]['Drug']
joblib.dump(best_models[best_drug], os.path.join(OUT_DIR, 'cancergpt_mutation_model.pkl'))

# ── Report ────────────────────────────────────────────────────────────────────
improved = results_df[results_df['Delta_AUC'] > 0] if 'Delta_AUC' in results_df else pd.DataFrame()
degraded = results_df[results_df['Delta_AUC'] < 0] if 'Delta_AUC' in results_df else pd.DataFrame()
great    = results_df[results_df['ROC_AUC'] >= 0.75]

# Pre-compute mean delta string to avoid invalid f-string format spec
if 'Delta_AUC' in results_df.columns and results_df['Delta_AUC'].notna().any():
    mean_delta_str = f"{results_df['Delta_AUC'].mean():.3f}"
else:
    mean_delta_str = 'N/A'

report = f"""CancerGPT — Mutation Stratification Report
============================================
Strategy:
  Expression features  : 2,000 top-variance Affymetrix U133A probes
  Mutation features    : {len(mut_cols)} binary gene mutation status flags
  Total features       : {2000 + len(mut_cols)}
  Model                : XGBoost + SMOTE + SelectKBest (k=50)
  Evaluation           : 5-fold stratified CV

Performance Summary:
  Drugs trained        : {len(results_df)}
  Models AUC >= 0.75   : {len(great)}
  Mean AUC             : {results_df['ROC_AUC'].mean():.3f}
  Best model           : {best_drug} (AUC={results_df.iloc[0]['ROC_AUC']})

AUC Change vs Expression-Only:
  Improved (ΔAUCgt;0)   : {len(improved)} drugs
  Degraded (ΔAUC<0)    : {len(degraded)} drugs
  Mean ΔAUC            : {mean_delta_str}

Top 10 Models:
{results_df[['Drug','ROC_AUC','Prev_AUC','Delta_AUC','Mut_features_in_top10']].head(10).to_string(index=False)}

Drug-specific mutation impact:
"""

for _, row in results_df.head(15).iterrows():
    mut_in_top = int(row.get('Mut_features_in_top10', 0))
    delta = row.get('Delta_AUC')
    delta_str = f"Δ+{delta:.3f}" if delta and delta>0 else (f"Δ{delta:.3f}" if delta else "")
    report += f"  {row['Drug']:<30} AUC={row['ROC_AUC']:.3f}  {delta_str:<10}  {mut_in_top} mutation features in top 10\n"

with open(os.path.join(OUT_DIR, 'cancergpt_mutation_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n{'='*70}")
print("MUTATION STRATIFICATION RESULTS")
print(f"{'='*70}")
print(f"  Drugs trained      : {len(results_df)}")
print(f"  AUC >= 0.75        : {len(great)}")
print(f"  Mean AUC           : {results_df['ROC_AUC'].mean():.3f}")
print(f"  Best model         : {best_drug} (AUC={results_df.iloc[0]['ROC_AUC']})")
if len(improved) > 0:
    print(f"  Improved vs expr   : {len(improved)} drugs")
    print(f"  Top improved:")
    for _, r in improved.nlargest(5,'Delta_AUC').iterrows():
        print(f"    {r['Drug']:<30} Δ+{r['Delta_AUC']:.3f}")
print()
print("Files saved:")
print("  cancergpt_mutation_results.csv   — AUC per drug with mutation features")
print("  cancergpt_mutation_features.csv  — top features per drug (expr + mutation)")
print("  cancergpt_mutation_report.txt    — full comparison report")
print("  cancergpt_mutation_model.pkl     — best stratified model")
print()
print("✅ Mutation stratification complete!")
