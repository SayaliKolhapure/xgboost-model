"""
CancerGPT — Mutation & Cancer Type Analysis
Run this in your breast cancer folder to get full statistics.
"""
import pandas as pd
import os

DATA_DIR = r'C:\Users\sayal\OneDrive\Desktop\breast cancer'
MUT_FILE = os.path.join(DATA_DIR, 'mutations_all_20260316.xls')
CLD_FILE = os.path.join(DATA_DIR, 'Cell_Lines_Details.xlsx')

print("=" * 60)
print("CancerGPT — Data Analysis Report")
print("=" * 60)

# ── 1. Mutation Analysis ──────────────────────────────────────────────────────
print("\n[1] MUTATION DATA ANALYSIS")
print("-" * 40)
try:
    mut = pd.read_excel(MUT_FILE)
    mut.columns = [str(c).strip() for c in mut.columns]

    # Find cell line and gene columns
    cell_col = next((c for c in mut.columns if any(k in c.lower() for k in ['model_name','cell_line','sample'])), None)
    gene_col = next((c for c in mut.columns if any(k in c.lower() for k in ['gene','symbol','hugo'])), None)

    print(f"  Total mutation records  : {len(mut):,}")
    print(f"  Cell line column        : '{cell_col}'")
    print(f"  Gene column             : '{gene_col}'")
    print(f"  Unique cell lines       : {mut[cell_col].nunique():,}")
    print(f"  Unique mutated genes    : {mut[gene_col].nunique():,}")

    # Top mutated genes
    print(f"\n  Top 20 most frequently mutated genes:")
    top_genes = mut[gene_col].value_counts().head(20)
    for gene, count in top_genes.items():
        print(f"    {str(gene):<20} {count:>5} mutations")

    # Mutation type distribution (if column exists)
    type_cols = [c for c in mut.columns if 'type' in c.lower() or 'class' in c.lower() or 'effect' in c.lower()]
    if type_cols:
        print(f"\n  Mutation types ({type_cols[0]}):")
        for mtype, count in mut[type_cols[0]].value_counts().head(10).items():
            print(f"    {str(mtype):<30} {count:>6}")

    print(f"\n  Columns in mutation file: {list(mut.columns)}")

except Exception as e:
    print(f"  Error: {e}")

# ── 2. Cancer Type Analysis ───────────────────────────────────────────────────
print("\n[2] CANCER TYPE ANALYSIS")
print("-" * 40)
try:
    cld = pd.read_excel(CLD_FILE)
    cld.columns = [c.replace('\n',' ').strip() for c in cld.columns]

    # Find TCGA label column
    tcga_col = next((c for c in cld.columns if 'TCGA' in c or 'Cancer Type' in c), None)
    tissue_col = next((c for c in cld.columns if 'Tissue' in c and '1' in c), None)
    name_col = next((c for c in cld.columns if 'Sample' in c or 'Name' in c), None)

    print(f"  Total cell lines         : {len(cld):,}")
    print(f"  TCGA label column        : '{tcga_col}'")

    if tcga_col:
        cancer_counts = cld[tcga_col].value_counts()
        print(f"\n  Total cancer types       : {len(cancer_counts)}")
        print(f"\n  Cancer type distribution:")
        for cancer, count in cancer_counts.items():
            bar = '█' * min(count, 40)
            print(f"    {str(cancer):<20} {count:>4} cell lines  {bar}")

    if tissue_col:
        print(f"\n  Tissue descriptor distribution:")
        for tissue, count in cld[tissue_col].value_counts().head(15).items():
            print(f"    {str(tissue):<25} {count:>4}")

except Exception as e:
    print(f"  Error: {e}")

# ── 3. Merged dataset analysis ────────────────────────────────────────────────
print("\n[3] MERGED DATASET SUMMARY")
print("-" * 40)
try:
    gdsc = pd.read_csv(os.path.join(DATA_DIR, 'gdsc_processed.csv'))
    print(f"  Total drug-response rows : {len(gdsc):,}")
    print(f"  Unique drugs             : {gdsc['DRUG_NAME'].nunique()}")
    print(f"  Unique cell lines        : {gdsc['CELL_LINE_NAME'].nunique()}")
    print(f"  Sensitive labels         : {gdsc['RESPONSE'].sum():,}")
    print(f"  Resistant labels         : {(gdsc['RESPONSE']==0).sum():,}")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("Analysis complete!")
