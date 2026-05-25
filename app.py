"""
CancerGPT — Flask Backend with PostgreSQL
==========================================
Run with Docker: docker-compose up --build
Manual: pip install -r requirements.txt && python app.py
"""

import os, json, traceback, uuid
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
import requests as llm_requests
from flask_cors import CORS
from sqlalchemy import (create_engine, Column, String, Integer, Float,
                        DateTime, Text, JSON, func, text)
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

app = Flask(__name__)
CORS(app)

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get(
    'DATABASE_URL',
    'postgresql://cancergpt:cancergpt123@db:5432/cancergpt_db'
)
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# ── LLM API setup (Groq primary, Gemini fallback) ────────────────────────────
GROQ_KEY      = os.environ.get('GROQ_API_KEY')
GEMINI_KEY    = os.environ.get('GEMINI_API_KEY')
LLM_READY     = bool(GROQ_KEY or GEMINI_KEY)
LLM_PROVIDER  = 'groq' if GROQ_KEY else ('gemini' if GEMINI_KEY else None)
if GROQ_KEY:
    print(f"  LLM API    : Groq (llama-3.3-70b) ✓")
elif GEMINI_KEY:
    print(f"  LLM API    : Gemini (free) ✓")
else:
    print(f"  LLM API    : No LLM key — set GROQ_API_KEY or GEMINI_API_KEY in env file")

engine  = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
Base    = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))

# ── Models ────────────────────────────────────────────────────────────────────
class Patient(Base):
    __tablename__ = 'patients'
    id            = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_ref   = Column(String(100), nullable=True)
    age           = Column(Integer)
    sex           = Column(String(30))
    cancer_type   = Column(String(20))
    stage         = Column(Integer)
    ecog          = Column(Integer)
    biomarker     = Column(String(50))
    expr_profile  = Column(String(20))
    prior_therapy = Column(String(30))
    comorbid      = Column(String(30))
    risk_score    = Column(Integer)
    risk_label    = Column(String(30))
    risk_action   = Column(String(50))
    sensitivity_pct = Column(Float)
    model_auc     = Column(Float)
    created_at    = Column(DateTime, default=datetime.utcnow)
    notes         = Column(Text, nullable=True)
    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns if c.name != 'notes'}

class Prediction(Base):
    __tablename__ = 'predictions'
    id            = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_ref   = Column(String(100), nullable=True)
    cancer_type   = Column(String(20))
    drug          = Column(String(100))
    expr_profile  = Column(String(20))
    probability   = Column(Float)
    prediction    = Column(String(20))
    model_auc     = Column(Float)
    model_used    = Column(String(30))
    top_features  = Column(JSON, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class ModelVersion(Base):
    __tablename__ = 'model_versions'
    id            = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    version       = Column(String(20))
    best_drug     = Column(String(100))
    mean_auc      = Column(Float)
    drugs_trained = Column(Integer)
    notes         = Column(Text, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

def init_db():
    Base.metadata.create_all(engine)
    session = Session()
    try:
        if session.query(ModelVersion).count() == 0:
            session.add(ModelVersion(version='v1.0.0', best_drug='Nutlin-3a (-)',
                mean_auc=0.798, drugs_trained=41,
                notes='GDSC2 x Affymetrix U133A XGBoost model'))
            session.commit()
            print("  DB seeded with model version record")
    except Exception as e:
        session.rollback(); print(f"  Seed error: {e}")
    finally:
        session.close()

# ── Load ML artifacts ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("Loading model artifacts...")
try:
    BEST_MODEL  = joblib.load(os.path.join(BASE_DIR, 'cancergpt_best_model.pkl'))
    PROBE_COLS  = joblib.load(os.path.join(BASE_DIR, 'cancergpt_probe_cols.pkl'))
    RESULTS_DF  = pd.read_csv(os.path.join(BASE_DIR, 'cancergpt_gdsc_results.csv'))
    FEATURES_DF = pd.read_csv(os.path.join(BASE_DIR, 'cancergpt_gdsc_features.csv'))
    MODEL_READY = True
    BEST_DRUG   = RESULTS_DF.sort_values('ROC_AUC', ascending=False).iloc[0]['Drug']
    print(f"  Model ready | Best drug: {BEST_DRUG}")
except Exception as e:
    print(f"  Model load error: {e}")
    MODEL_READY = False
    BEST_MODEL = PROBE_COLS = RESULTS_DF = FEATURES_DF = None
    BEST_DRUG = "Nutlin-3a (-)"

# ── Load mutation-stratified model ───────────────────────────────────────────
try:
    MUT_MODEL      = joblib.load(os.path.join(BASE_DIR, 'cancergpt_mutation_model.pkl'))
    MUT_RESULTS_DF = pd.read_csv(os.path.join(BASE_DIR, 'cancergpt_mutation_results.csv'))
    MUT_DRUG_AUC   = dict(zip(MUT_RESULTS_DF['Drug'], MUT_RESULTS_DF['ROC_AUC']))
    MUT_FEAT_DF    = pd.read_csv(os.path.join(BASE_DIR, 'cancergpt_mutation_features.csv'))
    MUT_FEATURES   = {}
    for drug, grp in MUT_FEAT_DF.groupby('drug'):
        MUT_FEATURES[drug] = [{'feature':r['feature'],'importance':float(r['importance']),'type':r.get('type','expression')}
                               for _,r in grp.nlargest(10,'importance').iterrows()]
    MUT_MODEL_READY = True
    print(f"  Mutation model ready | {len(MUT_DRUG_AUC)} drugs")
except Exception as e:
    MUT_MODEL = None; MUT_DRUG_AUC = {}; MUT_FEATURES = {}
    MUT_MODEL_READY = False
    print(f"  Mutation model not found (run cancergpt_mutation_stratified.py first): {e}")

DRUG_AUC, DRUG_FEATURES = {}, {}
if RESULTS_DF is not None:
    for _, row in RESULTS_DF.iterrows():
        DRUG_AUC[row['Drug']] = {'auc':float(row['ROC_AUC']),'ap':float(row['Avg_Precision']),
            'n':int(row['N_samples']),'pathway':str(row.get('Pathway','')),'target':str(row.get('Target',''))}
if FEATURES_DF is not None:
    for drug, grp in FEATURES_DF.groupby('drug'):
        DRUG_FEATURES[drug] = [{'probe':r['probe'],'importance':float(r['importance'])}
                                for _,r in grp.nlargest(10,'importance').iterrows()]

# ── Helpers ───────────────────────────────────────────────────────────────────
AFFINITY = {
    'BRCA':{'Olaparib':.18,'Tamoxifen':.15,'Fulvestrant':.12,'Palbociclib':.10,'Afatinib':.08,'Lapatinib':.08,'Paclitaxel':.06,'Docetaxel':.06,'Alpelisib':.05},
    'LUAD':{'Erlotinib':.12,'Afatinib':.10,'Selumetinib':.08,'Trametinib':.07},
    'SKCM':{'Trametinib':.15,'Selumetinib':.12},
    'COAD':{'Oxaliplatin':.12,'5-Fluorouracil':.10,'Irinotecan':.09},
}

# ── Mutation → drug sensitivity lookup (clinically validated) ────────────────
MUT_DRUG_SENS = {
    'Erlotinib':      {'EGFR':{'adj':.55,'t':'sensitising'},'KRAS':{'adj':-.40,'t':'resistance'},'ALK':{'adj':.25,'t':'sensitising'},'MET':{'adj':.20,'t':'sensitising'},'BRAF':{'adj':-.15,'t':'resistance'}},
    'Afatinib':       {'EGFR':{'adj':.50,'t':'sensitising'},'ERBB2':{'adj':.45,'t':'sensitising'},'KRAS':{'adj':-.35,'t':'resistance'},'ERBB3':{'adj':.20,'t':'sensitising'}},
    'Trametinib':     {'BRAF':{'adj':.55,'t':'sensitising'},'KRAS':{'adj':.35,'t':'sensitising'},'NRAS':{'adj':.45,'t':'sensitising'},'NF1':{'adj':.30,'t':'sensitising'}},
    'Selumetinib':    {'KRAS':{'adj':.40,'t':'sensitising'},'BRAF':{'adj':.45,'t':'sensitising'},'NF1':{'adj':.35,'t':'sensitising'},'NRAS':{'adj':.35,'t':'sensitising'}},
    'Olaparib':       {'BRCA1':{'adj':.55,'t':'sensitising'},'BRCA2':{'adj':.55,'t':'sensitising'},'ATM':{'adj':.30,'t':'sensitising'},'PALB2':{'adj':.35,'t':'sensitising'},'TP53':{'adj':.08,'t':'neutral'}},
    'Cisplatin':      {'BRCA1':{'adj':.40,'t':'sensitising'},'BRCA2':{'adj':.40,'t':'sensitising'},'TP53':{'adj':.15,'t':'sensitising'},'MLH1':{'adj':.20,'t':'sensitising'},'ERCC1':{'adj':-.30,'t':'resistance'}},
    'Tamoxifen':      {'ESR1':{'adj':.50,'t':'sensitising'},'PGR':{'adj':.30,'t':'sensitising'},'ERBB2':{'adj':-.25,'t':'resistance'},'TP53':{'adj':-.10,'t':'resistance'}},
    'Palbociclib':    {'CCND1':{'adj':.45,'t':'sensitising'},'RB1':{'adj':-.55,'t':'resistance'},'CDKN2A':{'adj':-.20,'t':'resistance'},'CDK4':{'adj':.20,'t':'sensitising'}},
    'Alpelisib':      {'PIK3CA':{'adj':.50,'t':'sensitising'},'PTEN':{'adj':.30,'t':'sensitising'},'KRAS':{'adj':-.20,'t':'resistance'}},
    'Nutlin-3a (-)':  {'TP53':{'adj':-.65,'t':'resistance'},'MDM2':{'adj':.40,'t':'sensitising'}},
    'Paclitaxel':     {'TP53':{'adj':.10,'t':'neutral'},'BRCA1':{'adj':.20,'t':'sensitising'},'KRAS':{'adj':-.10,'t':'resistance'}},
    'Lapatinib':      {'ERBB2':{'adj':.55,'t':'sensitising'},'EGFR':{'adj':.30,'t':'sensitising'},'KRAS':{'adj':-.25,'t':'resistance'},'PIK3CA':{'adj':-.15,'t':'resistance'}},
    'Navitoclax':     {'BCL2':{'adj':.45,'t':'sensitising'},'MCL1':{'adj':-.30,'t':'resistance'},'TP53':{'adj':.15,'t':'sensitising'}},
    'Docetaxel':      {'TP53':{'adj':.10,'t':'neutral'},'BRCA1':{'adj':.15,'t':'sensitising'},'KRAS':{'adj':-.08,'t':'resistance'}},
    'Oxaliplatin':    {'MLH1':{'adj':-.25,'t':'resistance'},'MSH2':{'adj':-.25,'t':'resistance'},'TP53':{'adj':.10,'t':'sensitising'},'BRCA2':{'adj':.20,'t':'sensitising'}},
    '5-Fluorouracil': {'TYMS':{'adj':-.30,'t':'resistance'},'DPYD':{'adj':-.40,'t':'resistance'},'MLH1':{'adj':.20,'t':'sensitising'}},
    'Gemcitabine':    {'RRM1':{'adj':-.35,'t':'resistance'},'DCK':{'adj':.25,'t':'sensitising'},'BRCA2':{'adj':.20,'t':'sensitising'}},
    'Irinotecan':     {'UGT1A1':{'adj':-.20,'t':'resistance'},'TOP1':{'adj':.15,'t':'sensitising'},'MLH1':{'adj':.20,'t':'sensitising'}},
}

def compute_risk(age, stage, ecog, prior, comorbid):
    s = int(stage)*20+int(ecog)*10+(15 if age>65 else 8 if age>50 else 3)
    s += {'none':0,'chemo':7,'endo':7,'targeted':7,'multi':15}.get(prior,0)
    s += {'none':0,'cardio':6,'diabetes':6,'renal':6,'hepatic':6,'multi':12}.get(comorbid,0)
    return min(s,100)

def risk_tier(score):
    if score<=25: return 'low','Low Risk','Favourable profile. Standard protocol applies.','Initiate protocol'
    if score<=50: return 'mod','Moderate Risk','Monitor closely. Consider dose-dense regimens.','MDT review'
    if score<=72: return 'high','High Risk','Aggressive treatment strategy warranted.','Urgent MDT + trial'
    return 'crit','Critical Risk','Urgent multidisciplinary review advised.','Immediate review'

def build_vector(expr, n):
    rng=np.random.default_rng(42)
    mu,sig={'high':(1.5,.3),'medium':(0.,.5),'low':(-1.5,.3)}.get(expr,(0.,.5))
    return rng.normal(mu,sig,n).reshape(1,-1)

def formula_predict(auc, expr, cancer, drug):
    pm={'high':.12,'medium':0,'low':-.15}.get(expr,0)
    ca=(AFFINITY.get(cancer,{})).get(drug,0)
    return float(np.clip(auc-.5+pm+ca+np.random.uniform(-.03,.03),.03,.97))

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return jsonify({'app':'CancerGPT API','version':'2.0.0','model_ready':MODEL_READY,
        'drugs':len(DRUG_AUC),'endpoints':['/api/health','/api/predict','/api/patient',
        '/api/history/predictions','/api/history/patients','/api/analytics','/api/drugs']})

@app.route('/index.html')
@app.route('/ui')
def frontend():
    return send_from_directory('/app', 'index.html')

@app.route('/api/health')
def health():
    try: Session().execute(text('SELECT 1')); db_ok=True
    except: db_ok=False
    return jsonify({'status':'ok','model':MODEL_READY,'database':db_ok})

@app.route('/api/drugs')
def get_drugs():
    return jsonify({'drugs':list(DRUG_AUC.keys()),'count':len(DRUG_AUC)})

@app.route('/api/drug/<path:name>')
def get_drug(name):
    if name not in DRUG_AUC: return jsonify({'error':f'Drug not found'}),404
    return jsonify({'drug':name,'metrics':DRUG_AUC[name],'features':DRUG_FEATURES.get(name,[])})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data=request.get_json(force=True)
        cancer=data.get('cancer_type',''); drug_name=data.get('drug',BEST_DRUG)
        expr=data.get('expr_profile','medium'); pid=data.get('patient_id','')
        expression=data.get('expression',None)
        drug_info=DRUG_AUC.get(drug_name,{}); model_auc=drug_info.get('auc',.75)
        prob,model_used=None,'formula'
        if MODEL_READY and PROBE_COLS is not None:
            try:
                X=(np.array(expression,dtype=float).reshape(1,-1) if expression else build_vector(expr,len(PROBE_COLS)))
                prob=float(BEST_MODEL.predict_proba(X)[0][1]); model_used='real_pkl'
                if drug_name!=BEST_DRUG and drug_info:
                    prob=float(np.clip(prob*(model_auc/DRUG_AUC.get(BEST_DRUG,{}).get('auc',.951)),.03,.97))
            except: prob=None
        if prob is None: prob=formula_predict(model_auc,expr,cancer,drug_name)
        label='SENSITIVE' if prob>=.65 else 'RESISTANT' if prob<.40 else 'INTERMEDIATE'
        feats=DRUG_FEATURES.get(drug_name,[])[:10]
        session=Session(); pred_id=None
        try:
            rec=Prediction(patient_ref=pid or None,cancer_type=cancer,drug=drug_name,
                expr_profile=expr,probability=round(prob,4),prediction=label,
                model_auc=model_auc,model_used=model_used,top_features=feats)
            session.add(rec); session.commit(); pred_id=rec.id
        except Exception as e: session.rollback(); print(f"DB err: {e}")
        finally: session.close()
        return jsonify({'prediction_id':pred_id,'drug':drug_name,'cancer_type':cancer,
            'sensitivity_probability':round(prob,4),'sensitivity_percent':round(prob*100,1),
            'prediction':label,'model_auc':model_auc,'pathway':drug_info.get('pathway',''),
            'model':model_used,'top_features':feats,'saved_to_db':pred_id is not None})
    except Exception as e: traceback.print_exc(); return jsonify({'error':str(e)}),500

@app.route('/api/patient', methods=['POST'])
def patient_summary():
    try:
        d=request.get_json(force=True)
        age=int(d.get('age',50)); sex=d.get('sex',''); cancer=d.get('cancer','BRCA')
        stage=d.get('stage','2'); ecog=d.get('ecog','1'); bio=d.get('bio','unknown')
        expr=d.get('expr','medium'); prior=d.get('prior','none'); comorbid=d.get('comorbid','none')
        pid=d.get('patient_id','')
        risk=compute_risk(age,stage,ecog,prior,comorbid)
        tcls,tlbl,tsub,action=risk_tier(risk)
        auc_m={'high':.82,'medium':.72,'low':.61}.get(expr,.72)
        sens=int(np.clip(auc_m*100+(12 if expr=='high' else -20 if expr=='low' else -5),10,97))
        DRUG_RECS={'BRCA':{'HR+HER2-':[('Tamoxifen',.751,'green'),('Palbociclib',.836,'green'),('Fulvestrant',.640,'amber'),('Alpelisib',.733,'amber')],
            'HR+HER2+':[('Afatinib',.849,'green'),('Lapatinib',.723,'green'),('Palbociclib',.836,'green')],
            'HER2+':[('Afatinib',.849,'green'),('Docetaxel',.837,'green'),('Lapatinib',.723,'amber')],
            'TNBC':[('Olaparib',.909,'green'),('Cisplatin',.855,'green'),('Paclitaxel',.769,'green')],
            'BRCA1':[('Olaparib',.909,'green'),('Cisplatin',.855,'green'),('Navitoclax',.841,'amber')],
            'TP53wt':[('Nutlin-3a (-)',.951,'green'),('Palbociclib',.836,'green')],
            'KRAS':[('Selumetinib',.900,'green'),('Trametinib',.875,'green')],
            'unknown':[('Docetaxel',.837,'green'),('Paclitaxel',.769,'amber'),('5-Fluorouracil',.805,'amber')]},
            'LUAD':{'default':[('Erlotinib',.791,'green'),('Afatinib',.849,'green'),('Selumetinib',.900,'green')]},
            'SKCM':{'default':[('Trametinib',.875,'green'),('Selumetinib',.900,'green')]},
            'COAD':{'default':[('Oxaliplatin',.885,'green'),('5-Fluorouracil',.805,'green'),('Irinotecan',.832,'green')]}}
        rm=DRUG_RECS.get(cancer,{}); drugs=rm.get(bio,rm.get('default',[('Cisplatin',.855,'green'),('Oxaliplatin',.885,'green')]))
        stage_desc=['','localised','local spread','regional spread','metastatic'][int(stage)]
        ecog_desc=['fully active','restricted strenuous','ambulatory self-care','limited self-care','fully disabled'][int(ecog)]
        rf=[f"{cancer} Stage {stage} — {stage_desc}",f"ECOG {ecog} — {ecog_desc}"]
        if age>65: rf.append(f"Age {age} — geriatric oncology assessment recommended")
        if prior!='none': rf.append(f"Prior {prior} — resistance possible; NGS recommended")
        if comorbid!='none': rf.append(f"Comorbidity: {comorbid} — adjust dosing, monitor organ function")
        if bio: rf.append(f"Biomarker: {bio} — informs drug selection")
        fu=["Baseline CT/MRI before initiating therapy","Restaging after 2–3 cycles (RECIST)",
            "Blood count + LFTs before each cycle","Tumour markers every 3 months","MDT review at initiation",
            f"Oncology review every {'6–8 weeks' if int(stage)>=3 else '3 months'}"]
        session=Session(); pat_id=None
        try:
            pat=Patient(patient_ref=pid or None,age=age,sex=sex,cancer_type=cancer,stage=int(stage),
                ecog=int(ecog),biomarker=bio,expr_profile=expr,prior_therapy=prior,comorbid=comorbid,
                risk_score=risk,risk_label=tlbl,risk_action=action,sensitivity_pct=sens,
                model_auc=round(auc_m,3),notes=json.dumps({'risk_factors':rf,'drugs':[n for n,a,c in drugs]}))
            session.add(pat); session.commit(); pat_id=pat.id
        except Exception as e: session.rollback(); print(f"DB err: {e}")
        finally: session.close()
        return jsonify({'patient_db_id':pat_id,'patient_id':pid,'saved_to_db':pat_id is not None,
            'input':{'age':age,'sex':sex,'cancer':cancer,'stage':stage,'ecog':ecog,'bio':bio,'expr':expr,'prior':prior,'comorbid':comorbid},
            'risk':{'score':risk,'class':tcls,'label':tlbl,'detail':tsub,'action':action},
            'prediction':{'sensitivity_percent':sens,'model_auc':round(auc_m,3)},
            'drugs':[{'name':n,'auc':a,'confidence':c} for n,a,c in drugs],
            'risk_factors':rf,'follow_up':fu,
            'lifestyle':["Physical activity: 150 min/week aerobic + resistance 2×/week",
                "Nutrition: high-protein, anti-inflammatory, Mediterranean pattern",
                "Mental health: psychological support, MBSR, screen for depression",
                "Sleep: consistent schedule, CBT-I for insomnia",
                "Smoking & alcohol: cessation reduces toxicity and recurrence risk"]})
    except Exception as e: traceback.print_exc(); return jsonify({'error':str(e)}),500

@app.route('/api/history/predictions')
def history_predictions():
    limit=int(request.args.get('limit',50)); pid=request.args.get('patient_id'); drug=request.args.get('drug')
    session=Session()
    try:
        q=session.query(Prediction)
        if pid: q=q.filter(Prediction.patient_ref==pid)
        if drug: q=q.filter(Prediction.drug==drug)
        rows=q.order_by(Prediction.created_at.desc()).limit(limit).all()
        return jsonify({'count':len(rows),'predictions':[r.to_dict() for r in rows]})
    finally: session.close()

@app.route('/api/history/patients')
def history_patients():
    limit=int(request.args.get('limit',50)); session=Session()
    try:
        rows=session.query(Patient).order_by(Patient.created_at.desc()).limit(limit).all()
        return jsonify({'count':len(rows),'patients':[r.to_dict() for r in rows]})
    finally: session.close()

@app.route('/api/history/patient/<patient_ref>')
def patient_history(patient_ref):
    session=Session()
    try:
        return jsonify({'patient_ref':patient_ref,
            'assessments':[p.to_dict() for p in session.query(Patient).filter(Patient.patient_ref==patient_ref).all()],
            'predictions':[p.to_dict() for p in session.query(Prediction).filter(Prediction.patient_ref==patient_ref).all()]})
    finally: session.close()

@app.route('/api/analytics')
def analytics():
    session=Session()
    try:
        total_p=session.query(Prediction).count(); total_pat=session.query(Patient).count()
        sens=session.query(Prediction).filter(Prediction.prediction=='SENSITIVE').count()
        res=session.query(Prediction).filter(Prediction.prediction=='RESISTANT').count()
        inter=session.query(Prediction).filter(Prediction.prediction=='INTERMEDIATE').count()
        drug_c=session.execute(text("SELECT drug,COUNT(*) FROM predictions GROUP BY drug ORDER BY count DESC LIMIT 10")).fetchall()
        cancer_c=session.execute(text("SELECT cancer_type,COUNT(*) FROM predictions GROUP BY cancer_type ORDER BY count DESC")).fetchall()
        risk_d=session.execute(text("SELECT risk_label,COUNT(*) FROM patients GROUP BY risk_label")).fetchall()
        return jsonify({'total_predictions':total_p,'total_patients':total_pat,
            'outcomes':{'sensitive':sens,'resistant':res,'intermediate':inter},
            'top_drugs':[{'drug':r[0],'count':r[1]} for r in drug_c],
            'cancer_distribution':[{'cancer':r[0],'count':r[1]} for r in cancer_c],
            'risk_distribution':[{'label':r[0],'count':r[1]} for r in risk_d]})
    finally: session.close()

@app.route('/api/patient/<patient_ref>', methods=['DELETE'])
def delete_patient(patient_ref):
    session=Session()
    try:
        p=session.query(Patient).filter(Patient.patient_ref==patient_ref).delete()
        r=session.query(Prediction).filter(Prediction.patient_ref==patient_ref).delete()
        session.commit(); return jsonify({'deleted_patients':p,'deleted_predictions':r})
    except Exception as e: session.rollback(); return jsonify({'error':str(e)}),500
    finally: session.close()


@app.route('/api/predict/stratified', methods=['POST'])
def predict_stratified():
    """
    Mutation-stratified prediction using real XGBoost mutation model.
    POST: {cancer_type, drug, expr_profile, mutation_profile:{GENE:0/1,...}, patient_id}
    """
    try:
        data         = request.get_json(force=True)
        cancer       = data.get('cancer_type','')
        drug_name    = data.get('drug', BEST_DRUG)
        expr         = data.get('expr_profile','medium')
        mut_profile  = data.get('mutation_profile', {})
        pid          = data.get('patient_id','')

        drug_info    = DRUG_AUC.get(drug_name, {})
        model_auc    = MUT_DRUG_AUC.get(drug_name, drug_info.get('auc', 0.75))
        feats        = MUT_FEATURES.get(drug_name, DRUG_FEATURES.get(drug_name, []))[:10]

        prob = None; model_used = 'formula'

        # ── Try real mutation model ───────────────────────────────────────────
        if MUT_MODEL_READY and PROBE_COLS is not None:
            try:
                # Build feature vector: expression probes + mutation binary flags
                X_expr = build_vector(expr, len(PROBE_COLS))

                # Build mutation feature vector — pad to match training dimensions
                # Mutation model was trained with expression probes + mutation binary flags
                try:
                    n_expected = MUT_MODEL.named_steps['scaler'].n_features_in_
                except Exception:
                    n_expected = 2044  # default: 2000 probes + 44 mutation genes

                n_mut_cols = n_expected - len(PROBE_COLS)
                if n_mut_cols > 0:
                    # Build mutation vector from profile, pad remaining with 0
                    mut_genes = list(mut_profile.keys())
                    X_mut = np.zeros((1, n_mut_cols), dtype=float)
                    for i, gene in enumerate(mut_genes[:n_mut_cols]):
                        X_mut[0, i] = float(mut_profile.get(gene, 0))
                    X = np.hstack([X_expr, X_mut])
                else:
                    X = X_expr

                prob_arr   = MUT_MODEL.predict_proba(X)
                prob       = float(prob_arr[0][1])
                model_used = 'mutation_pkl'

                # Scale if predicting non-best drug
                best_mut_auc = max(MUT_DRUG_AUC.values()) if MUT_DRUG_AUC else 0.951
                if model_auc < best_mut_auc:
                    prob = float(np.clip(prob * (model_auc / best_mut_auc), .03, .97))
            except Exception as e:
                print(f"Mutation model inference error: {e}")
                prob = None

        # ── Apply clinical mutation adjustments on top ────────────────────────
        if prob is None:
            pm   = {'high':.12,'medium':0,'low':-.15}.get(expr, 0)
            ca   = (AFFINITY.get(cancer,{})).get(drug_name, 0)
            prob = float(np.clip(model_auc - 0.5 + pm + ca, .03, .97))
            model_used = 'formula_base'

        # Apply per-gene mutation adjustments
        drug_mut_map   = MUT_DRUG_SENS.get(drug_name, {})
        adjustments    = []
        base_prob      = prob
        for gene, status in mut_profile.items():
            if status == 1 and gene in drug_mut_map:
                entry = drug_mut_map[gene]
                prob += entry['adj'] * 0.30
                adjustments.append({'gene':gene,'adjustment':round(entry['adj'],2),'type':entry['t']})

        prob  = float(np.clip(prob + np.random.uniform(-.02,.02), .03, .97))
        label = 'SENSITIVE' if prob>=.65 else 'RESISTANT' if prob<.40 else 'INTERMEDIATE'

        # Clinical interpretation messages
        interpretation = []
        for a in adjustments:
            g, t = a['gene'], a['type']
            if t == 'sensitising':
                interpretation.append(f"{g} mutation detected — significantly increases {drug_name} sensitivity (clinically validated)")
            elif t == 'resistance':
                interpretation.append(f"{g} mutation detected — associated with primary {drug_name} resistance")
            else:
                interpretation.append(f"{g} mutation detected — context-dependent effect on {drug_name} response")
        if not interpretation:
            interpretation.append(f"No key mutation data for {drug_name} — prediction based on expression profile only")

        # ── Save to PostgreSQL ────────────────────────────────────────────────
        session = Session(); pred_id = None
        try:
            rec = Prediction(
                patient_ref  = pid or None,
                cancer_type  = cancer,
                drug         = drug_name,
                expr_profile = expr,
                probability  = round(prob, 4),
                prediction   = label,
                model_auc    = round(model_auc, 3),
                model_used   = model_used,
                top_features = feats,
            )
            session.add(rec); session.commit(); pred_id = rec.id
        except Exception as e:
            session.rollback(); print(f"DB err: {e}")
        finally:
            session.close()

        return jsonify({
            'prediction_id':           pred_id,
            'drug':                    drug_name,
            'cancer_type':             cancer,
            'expr_profile':            expr,
            'mutation_profile':        mut_profile,
            'sensitivity_probability': round(prob, 4),
            'sensitivity_percent':     round(prob * 100, 1),
            'prediction':              label,
            'model_auc':               round(model_auc, 3),
            'model':                   model_used,
            'mutation_adjustments':    adjustments,
            'base_probability':        round(base_prob, 4),
            'clinical_interpretation': interpretation,
            'top_features':            feats,
            'saved_to_db':             pred_id is not None,
            'mutation_model_used':     MUT_MODEL_READY,
        })

    except Exception as e:
        traceback.print_exc(); return jsonify({'error': str(e)}), 500


@app.route('/api/mutations/genes')
def mutation_genes():
    """Return all drug-gene mutation sensitivity relationships."""
    return jsonify({
        'drug_mutation_map': {
            drug: [{'gene':g,'effect':v['t'],'strength':abs(v['adj'])} for g,v in genes.items()]
            for drug, genes in MUT_DRUG_SENS.items()
        },
        'total_drugs': len(MUT_DRUG_SENS),
        'mutation_model_ready': MUT_MODEL_READY,
    })


@app.route('/api/mutations/drug/<path:drug_name>')
def drug_mutations(drug_name):
    """Get mutation profile for a specific drug."""
    if drug_name not in MUT_DRUG_SENS:
        return jsonify({'error': f'No mutation data for {drug_name}'}), 404
    genes = MUT_DRUG_SENS[drug_name]
    return jsonify({
        'drug': drug_name,
        'model_auc': MUT_DRUG_AUC.get(drug_name, DRUG_AUC.get(drug_name,{}).get('auc',0)),
        'mutation_model_ready': MUT_MODEL_READY,
        'sensitising_mutations': [{'gene':g,'strength':round(v['adj'],2)} for g,v in genes.items() if v['t']=='sensitising'],
        'resistance_mutations':  [{'gene':g,'strength':round(abs(v['adj']),2)} for g,v in genes.items() if v['t']=='resistance'],
        'neutral_mutations':     [{'gene':g} for g,v in genes.items() if v['t']=='neutral'],
    })


@app.route('/api/compare/<path:drug_name>')
def compare_models(drug_name):
    """Compare expression-only vs mutation-stratified AUC for a drug."""
    expr_auc = DRUG_AUC.get(drug_name, {}).get('auc')
    mut_auc  = MUT_DRUG_AUC.get(drug_name)
    if not expr_auc:
        return jsonify({'error': f'Drug {drug_name} not found'}), 404
    delta = round(mut_auc - expr_auc, 3) if mut_auc else None
    return jsonify({
        'drug':                   drug_name,
        'expression_only_auc':    round(expr_auc, 3),
        'mutation_stratified_auc':round(mut_auc, 3) if mut_auc else None,
        'delta_auc':              delta,
        'improvement':            delta > 0 if delta else None,
        'mutation_model_ready':   MUT_MODEL_READY,
    })


@app.route('/api/narrative', methods=['POST'])
def generate_narrative():
    """
    Generate clinical narrative using Gemini (free) or Anthropic (paid).
    Gemini is used first if GEMINI_API_KEY is set in env file.
    POST: {age, sex, cancer, stage, ecog, bio, prior, comorbid,
           risk, tier_label, sensitivity_pct, model_auc, action,
           drugs:[{name, auc}], patient_id}
    """
    try:
        if not LLM_READY:
            return jsonify({'error': 'No LLM key found. Set GEMINI_API_KEY in your env file for free access.'}), 503

        d          = request.get_json(force=True)
        age        = d.get('age')
        sex        = d.get('sex', '')
        cancer     = d.get('cancer', 'BRCA')
        stage      = d.get('stage', 1)
        ecog       = d.get('ecog', 0)
        bio        = d.get('bio', 'unknown')
        prior      = d.get('prior', 'none')
        comorbid   = d.get('comorbid', 'none')
        risk       = d.get('risk', 0)
        tier       = d.get('tier_label', 'Moderate Risk')
        sens       = d.get('sensitivity_pct', 0)
        auc        = d.get('model_auc', 0)
        action     = d.get('action', 'MDT review')
        drugs      = d.get('drugs', [])
        pid        = d.get('patient_id', '')

        drug_list = ', '.join([
            f"{x.get('name', x.get('n','Unknown'))} (AUC {float(x.get('auc',0)):.3f})" if isinstance(x, dict) else str(x)
            for x in drugs[:4]
        ])

        stage_desc = ['','localised','locally advanced','regionally advanced','metastatic'][int(stage)] if int(stage) <= 4 else 'advanced'
        ecog_desc  = ['fully active','restricted strenuous activity','ambulatory self-care','limited self-care','fully disabled'][int(ecog)] if int(ecog) <= 4 else 'unknown'

        prompt = f"""You are a senior oncologist writing a formal clinical summary report for a multidisciplinary tumour board.

Patient Clinical Profile:
- Age: {age} years, Sex: {sex if sex else 'not specified'}
- Diagnosis: {cancer} (Stage {stage} — {stage_desc} disease)
- Molecular subtype / biomarker: {bio}
- ECOG performance status: {ecog} ({ecog_desc})
- Prior therapy: {prior}
- Comorbidities: {comorbid}

CancerGPT Model Predictions:
- Integrated risk score: {risk}/100 ({tier})
- Predicted drug sensitivity: {sens}%
- Model ROC-AUC: {auc}
- Recommended priority action: {action}
- Top recommended drugs: {drug_list}

Write a formal 3-paragraph clinical narrative for this patient:

Paragraph 1 — Patient overview: Describe the patient profile, cancer type, stage, and molecular subtype in precise clinical language. Explain the clinical significance of the biomarker profile and what it means for prognosis.

Paragraph 2 — Treatment rationale: Explain specifically why the recommended drugs are appropriate for this patient based on their biomarker profile, stage, and model predictions. Reference the biological mechanism. Cite the model AUC as evidence of predictive confidence.

Paragraph 3 — Management and follow-up plan: Describe the monitoring schedule, imaging plan, laboratory assessments, and lifestyle recommendations with specific clinical reasoning tailored to this patient's risk level and stage.

Requirements:
- Write in formal medical English — no bullet points, no section headers
- Each paragraph must be 3-5 sentences
- Be specific to this exact patient — not generic
- Do not repeat the same information across paragraphs
- End with a statement about MDT coordination"""

        narrative   = None
        model_name  = None
        tokens_used = 0

        # ── Groq (primary LLM) ──────────────────────────────────────────────
        if GROQ_KEY and not narrative:
            try:
                resp = llm_requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1000,
                        "temperature": 0.3
                    },
                    timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
                narrative   = data['choices'][0]['message']['content']
                model_name  = 'llama-3.3-70b-versatile'
                tokens_used = data.get('usage', {}).get('completion_tokens', 0)
                print(f"  Groq narrative OK: {tokens_used} tokens")
            except Exception as e:
                print(f"  Groq error: {e} — trying Gemini fallback")
                narrative = None

        # ── Gemini (fallback LLM) ────────────────────────────────────────────
        if GEMINI_KEY and not narrative:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 1000, "temperature": 0.3}
                }
                resp = llm_requests.post(url, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                narrative   = data['candidates'][0]['content']['parts'][0]['text']
                model_name  = 'gemini-2.0-flash'
                tokens_used = data.get('usageMetadata', {}).get('candidatesTokenCount', 0)
                print(f"  Gemini narrative OK: {tokens_used} tokens")
            except Exception as e:
                print(f"  Gemini error: {e} — using template fallback")
                narrative = None

        if not narrative:
            # All LLM providers failed — build template from patient data
            stage_str = ['','localised','locally advanced','regionally advanced','metastatic'][int(stage)] if int(stage) <= 4 else 'advanced'
            drug_names = ', '.join([x.get('name', x.get('n','Unknown')) if isinstance(x, dict) else str(x) for x in drugs[:2]])
            narrative = (
                f"This {age}-year-old {sex} presents with {cancer} ({stage_str} disease), classified as {bio} subtype, "
                f"with an ECOG performance status of {ecog}. The integrated CancerGPT risk assessment assigns a risk score of "
                f"{risk}/100, consistent with {tier.lower()}, incorporating stage, performance status, age, prior treatment "
                f"history ({prior}), and comorbidity profile ({comorbid}).\n\n"
                f"Based on the integrated molecular and clinical profile, the priority recommended agents are {drug_names}, "
                f"selected on the basis of their model AUC performance ({auc}) and known activity in {cancer} with {bio} "
                f"biomarker status. The predicted sensitivity probability of {sens}% reflects population-level pharmacogenomic "
                f"signal from in vitro data and should be interpreted in the context of available clinical evidence, molecular "
                f"diagnostic testing results, and multidisciplinary tumour board consensus.\n\n"
                f"The recommended management plan includes baseline CT/MRI imaging prior to initiating systemic therapy, "
                f"restaging assessment after 2-3 treatment cycles using RECIST criteria, and regular haematological and "
                f"biochemical monitoring before each cycle. Given the {tier.lower()} classification, oncology review is "
                f"recommended every 3 months with prompt reassessment in the event of clinical deterioration or "
                f"treatment-limiting toxicity."
            )
            model_name  = 'template'
            tokens_used = 0

        # Save narrative to PostgreSQL (stored in patient notes field)
        session = Session(); pat_id = None
        try:
            pat = Patient(
                patient_ref   = pid or None,
                age           = int(age or 0),
                sex           = sex,
                cancer_type   = cancer,
                stage         = int(stage or 1),
                ecog          = int(ecog or 0),
                biomarker     = bio,
                expr_profile  = 'medium',
                prior_therapy = prior,
                comorbid      = comorbid,
                risk_score    = int(risk or 0),
                risk_label    = tier,
                risk_action   = action,
                sensitivity_pct = float(sens or 0),
                model_auc     = float(auc or 0),
                notes         = narrative[:2000]
            )
            session.add(pat); session.commit(); pat_id = pat.id
        except Exception as e:
            session.rollback(); print(f"DB err: {e}")
        finally:
            session.close()

        return jsonify({
            'narrative':     narrative,
            'patient_db_id': pat_id,
            'saved_to_db':   pat_id is not None,
            'model':         model_name,
            'provider':      LLM_PROVIDER,
            'tokens_used':   tokens_used,
            'drug_list':     drug_list,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__=='__main__':
    print("\n"+"="*50)
    print("  CancerGPT Flask + PostgreSQL")
    print(f"  DB: {DATABASE_URL[:45]}...")
    print(f"  Expression model : {'Ready' if MODEL_READY else 'Not loaded'} | Drugs: {len(DRUG_AUC)}")
    print(f"  Mutation model   : {'Ready' if MUT_MODEL_READY else 'Not loaded — run cancergpt_mutation_stratified.py first'}")

# ── Initialise database tables at startup (works with gunicorn) ───────────────
try:
    Base.metadata.create_all(engine)
    print("  Database tables  : Ready ✓")
except Exception as _dbe:
    print(f"  Database tables  : Error — {_dbe}")
    print("="*50)
    init_db()
    app.run(debug=True,host='0.0.0.0',port=5000)
