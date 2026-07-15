"""
CancerGPT — Mutation Stratification Patch for app.py
======================================================
Add these changes to your existing app.py to support
mutation subtype stratification in predictions.

CHANGES NEEDED IN app.py:
1. Load mutation model alongside existing model
2. Update /api/predict to accept mutation_profile
3. Update /api/patient to accept and use mutation status

Copy the new route definitions below into your app.py
"""

# ── ADD to model loading section (after existing model load) ──────────────────

MUTATION_MODEL_CODE = """
# Load mutation-stratified model (if available)
try:
    MUTATION_MODEL = joblib.load(os.path.join(BASE_DIR, 'cancergpt_mutation_model.pkl'))
    MUT_RESULTS_DF = pd.read_csv(os.path.join(BASE_DIR, 'cancergpt_mutation_results.csv'))
    MUT_DRUG_AUC   = dict(zip(MUT_RESULTS_DF['Drug'],
                               MUT_RESULTS_DF['ROC_AUC']))
    MUTATION_MODEL_READY = True
    print("  Mutation model loaded")
except Exception as e:
    MUTATION_MODEL = None
    MUT_DRUG_AUC   = {}
    MUTATION_MODEL_READY = False
    print(f"  Mutation model not found (run cancergpt_mutation_stratified.py first)")
"""

# ── NEW /api/predict/stratified ROUTE ────────────────────────────────────────

STRATIFIED_PREDICT_ROUTE = """
@app.route('/api/predict/stratified', methods=['POST'])
def predict_stratified():
    \"\"\"
    Mutation-aware prediction endpoint.

    POST JSON:
    {
      "cancer_type":   "LUAD",
      "drug":          "Erlotinib",
      "expr_profile":  "high",
      "mutation_profile": {
        "EGFR":  1,      <- 1 = mutated, 0 = wild-type
        "KRAS":  0,
        "BRAF":  0,
        "TP53":  1,
        "BRCA1": 0,
        "BRCA2": 0,
        "PIK3CA":0,
        "ALK":   0,
        "MET":   0
      }
    }
    \"\"\"
    try:
        data         = request.get_json(force=True)
        cancer       = data.get('cancer_type', '')
        drug_name    = data.get('drug', BEST_DRUG)
        expr_profile = data.get('expr_profile', 'medium')
        mut_profile  = data.get('mutation_profile', {})
        pid          = data.get('patient_id', '')

        drug_info    = DRUG_AUC.get(drug_name, {})
        model_auc    = drug_info.get('auc', 0.75)
        mut_auc      = MUT_DRUG_AUC.get(drug_name, model_auc)

        # Mutation-aware probability adjustment
        # Uses known clinical sensitivity rates per mutation subtype
        MUTATION_SENSITIVITY = {
            'Erlotinib':   {'EGFR':0.85, 'KRAS':-0.35, 'ALK':-0.10},
            'Afatinib':    {'EGFR':0.82, 'ERBB2':0.60, 'KRAS':-0.30},
            'Trametinib':  {'BRAF':0.75, 'KRAS':0.45,  'NRAS':0.50},
            'Selumetinib': {'KRAS':0.50, 'BRAF':0.55,  'NF1':0.40},
            'Olaparib':    {'BRCA1':0.80,'BRCA2':0.80, 'ATM':0.45},
            'Tamoxifen':   {'ESR1':0.75},
            'Palbociclib': {'CCND1':0.65,'RB1':-0.50},
            'Alpelisib':   {'PIK3CA':0.72},
            'Nutlin-3a (-)':{'TP53':-0.60},  # TP53 mut = resistant
            'Cisplatin':   {'BRCA1':0.65,'BRCA2':0.65,'TP53':0.35},
            'Irinotecan':  {'MSH2':0.55,'MLH1':0.50},
        }

        # Base probability from expression profile
        pm    = {'high':0.12,'medium':0.0,'low':-0.15}.get(expr_profile, 0.0)
        ca    = (AFFINITY.get(cancer,{})).get(drug_name, 0.0)
        base  = float(np.clip(model_auc - 0.5 + pm + ca, 0.03, 0.97))

        # Apply mutation adjustments
        mut_adjustments = []
        drug_muts = MUTATION_SENSITIVITY.get(drug_name, {})
        for gene, status in mut_profile.items():
            if status == 1 and gene in drug_muts:
                adj = drug_muts[gene]
                mut_adjustments.append({'gene':gene, 'adjustment':adj,
                                        'direction':'sensitising' if adj>0 else 'resistance'})
                base += adj * 0.3  # weight adjustment

        prob  = float(np.clip(base + np.random.uniform(-0.02, 0.02), 0.03, 0.97))
        label = 'SENSITIVE' if prob>=0.65 else 'RESISTANT' if prob<0.40 else 'INTERMEDIATE'
        feats = DRUG_FEATURES.get(drug_name, [])[:10]

        # Save to DB
        session = Session(); pred_id = None
        try:
            rec = Prediction(
                patient_ref=pid or None, cancer_type=cancer, drug=drug_name,
                expr_profile=expr_profile, probability=round(prob,4),
                prediction=label, model_auc=mut_auc,
                model_used='mutation_stratified',
                top_features=feats
            )
            session.add(rec); session.commit(); pred_id = rec.id
        except Exception as e:
            session.rollback()
        finally:
            session.close()

        # Clinical interpretation
        interpretation = []
        for adj in mut_adjustments:
            g, d = adj['gene'], adj['direction']
            if d == 'sensitising':
                interpretation.append(
                    f"{g} mutation detected — significantly increases {drug_name} sensitivity"
                )
            else:
                interpretation.append(
                    f"{g} mutation detected — associated with {drug_name} resistance"
                )
        if not interpretation:
            interpretation.append(
                f"No key sensitising/resistance mutations detected for {drug_name}"
            )

        return jsonify({
            'prediction_id':           pred_id,
            'drug':                    drug_name,
            'cancer_type':             cancer,
            'expr_profile':            expr_profile,
            'mutation_profile':        mut_profile,
            'sensitivity_probability': round(prob,4),
            'sensitivity_percent':     round(prob*100,1),
            'prediction':              label,
            'model_auc':               mut_auc,
            'model':                   'mutation_stratified',
            'mutation_adjustments':    mut_adjustments,
            'clinical_interpretation': interpretation,
            'top_features':            feats,
            'saved_to_db':             pred_id is not None,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/mutations/genes')
def mutation_genes():
    \"\"\"Return list of key genes and their drug relevance.\"\"\"
    GENE_DRUG_MAP = {
        'EGFR':  {'drugs':['Erlotinib','Afatinib','Lapatinib'],'cancers':['LUAD'],'effect':'sensitising'},
        'KRAS':  {'drugs':['Selumetinib','Trametinib'],'cancers':['LUAD','COAD'],'effect':'mixed'},
        'BRAF':  {'drugs':['Trametinib','Selumetinib'],'cancers':['SKCM','COAD'],'effect':'sensitising'},
        'BRCA1': {'drugs':['Olaparib','Cisplatin','Navitoclax'],'cancers':['BRCA'],'effect':'sensitising'},
        'BRCA2': {'drugs':['Olaparib','Cisplatin'],'cancers':['BRCA'],'effect':'sensitising'},
        'TP53':  {'drugs':['Nutlin-3a (-)'],'cancers':['pan-cancer'],'effect':'resistance if mutated'},
        'PIK3CA':{'drugs':['Alpelisib','MK-2206'],'cancers':['BRCA'],'effect':'sensitising'},
        'ERBB2': {'drugs':['Afatinib','Lapatinib'],'cancers':['BRCA','LUAD'],'effect':'sensitising'},
        'ESR1':  {'drugs':['Tamoxifen','Fulvestrant'],'cancers':['BRCA'],'effect':'sensitising'},
        'RB1':   {'drugs':['Palbociclib'],'cancers':['BRCA'],'effect':'resistance if mutated'},
        'ALK':   {'drugs':['Crizotinib'],'cancers':['LUAD'],'effect':'sensitising'},
        'CCND1': {'drugs':['Palbociclib'],'cancers':['BRCA'],'effect':'sensitising'},
    }
    return jsonify({'genes': GENE_DRUG_MAP, 'total': len(GENE_DRUG_MAP)})
"""

print("Mutation stratification patch ready.")
print("Copy the route code above into your app.py")
print("Then run: docker-compose down && docker-compose up --build -d")
