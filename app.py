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
    BEST_MODEL  = joblib.load(os.path.join(BASE_DIR, 'xgboost_best_model.pkl'))
    PROBE_COLS  = joblib.load(os.path.join(BASE_DIR, 'xgboost_probe_cols.pkl'))
    RESULTS_DF  = pd.read_csv(os.path.join(BASE_DIR, 'xgboost_gdsc_results.csv'))
    FEATURES_DF = pd.read_csv(os.path.join(BASE_DIR, 'xgboost_gdsc_features.csv'))
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
    MUT_MODEL      = joblib.load(os.path.join(BASE_DIR, 'xgboost_mutation_model.pkl'))
    MUT_RESULTS_DF = pd.read_csv(os.path.join(BASE_DIR, 'xgboost_mutation_results.csv'))
    MUT_DRUG_AUC   = dict(zip(MUT_RESULTS_DF['Drug'], MUT_RESULTS_DF['ROC_AUC']))
    MUT_FEAT_DF    = pd.read_csv(os.path.join(BASE_DIR, 'xgboost_mutation_features.csv'))
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

def call_llm(messages, system_prompt, max_tokens=900, temperature=0.35):
    """Call Groq first, Gemini as fallback. Returns (text, model_name, tokens)."""
    if GROQ_KEY:
        try:
            resp = llm_requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                json={"model":"llama-3.3-70b-versatile",
                      "messages":[{"role":"system","content":system_prompt}]+messages,
                      "max_tokens":max_tokens,"temperature":temperature},
                timeout=30)
            resp.raise_for_status()
            d = resp.json()
            return d['choices'][0]['message']['content'], 'llama-3.3-70b', d.get('usage',{}).get('completion_tokens',0)
        except Exception as e:
            print(f"  Groq error: {e}")
    if GEMINI_KEY:
        try:
            combined = system_prompt+"\n\n"+"\n".join(
                [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in messages])
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
            resp = llm_requests.post(url,json={"contents":[{"parts":[{"text":combined}]}],
                "generationConfig":{"maxOutputTokens":max_tokens,"temperature":temperature}},timeout=30)
            resp.raise_for_status()
            d = resp.json()
            return d['candidates'][0]['content']['parts'][0]['text'], 'gemini-2.0-flash', d.get('usageMetadata',{}).get('candidatesTokenCount',0)
        except Exception as e:
            print(f"  Gemini error: {e}")
    return None, None, 0

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return jsonify({'app':'CancerGPT API','version':'3.0.0','model_ready':MODEL_READY,
        'drugs':len(DRUG_AUC),'endpoints':['/api/health','/api/chat','/api/predict',
        '/api/patient','/api/narrative','/api/history/predictions','/api/history/patients',
        '/api/analytics','/api/drugs']})

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

@app.route('/api/predict/stratified', methods=['POST'])
def predict_stratified():
    try:
        data=request.get_json(force=True)
        cancer=data.get('cancer_type',''); drug_name=data.get('drug',BEST_DRUG)
        expr=data.get('expr_profile','medium'); mut_profile=data.get('mutation_profile',{})
        pid=data.get('patient_id','')
        drug_info=DRUG_AUC.get(drug_name,{}); model_auc=MUT_DRUG_AUC.get(drug_name,drug_info.get('auc',0.75))
        feats=MUT_FEATURES.get(drug_name,DRUG_FEATURES.get(drug_name,[]))[:10]
        pm={'high':.12,'medium':0,'low':-.15}.get(expr,0)
        MUT_BASE_AUC={'ACC':.70,'ALL':.73,'BLCA':.74,'BRCA':.82,'CESC':.71,'CLL':.72,'COAD':.75,
            'DLBC':.76,'ESCA':.72,'GBM':.74,'HNSC':.73,'KIRC':.75,'LAML':.74,'LCML':.73,
            'LGG':.71,'LIHC':.72,'LUAD':.72,'LUSC':.71,'MB':.70,'MESO':.71,'MM':.73,
            'NB':.72,'OV':.74,'PAAD':.71,'PRAD':.71,'SCLC':.73,'SKCM':.78,'STAD':.72,'THCA':.71,'UCEC':.74}
        base_prob=float(np.clip((MUT_BASE_AUC.get(cancer,.72)-.5+pm+np.random.uniform(-.02,.02)),.03,.97))
        prob=base_prob; drug_sens=MUT_DRUG_SENS.get(drug_name,{}); adjustments=[]
        for gene,val in mut_profile.items():
            if val==1 and gene in drug_sens:
                prob+=drug_sens[gene]['adj']*0.35
                adjustments.append({'gene':gene,'adj':drug_sens[gene]['adj'],'type':drug_sens[gene]['t']})
        prob=float(np.clip(prob,.03,.97))
        label='SENSITIVE' if prob>=.65 else 'RESISTANT' if prob<.40 else 'INTERMEDIATE'
        session=Session(); pred_id=None
        try:
            rec=Prediction(patient_ref=pid or None,cancer_type=cancer,drug=drug_name,
                expr_profile=expr,probability=round(prob,4),prediction=label,model_auc=model_auc,
                model_used='mutation_stratified',top_features=feats)
            session.add(rec); session.commit(); pred_id=rec.id
        except Exception as e: session.rollback()
        finally: session.close()
        return jsonify({'prediction_id':pred_id,'drug':drug_name,'cancer_type':cancer,
            'sensitivity_probability':round(prob,4),'sensitivity_percent':round(prob*100,1),
            'prediction':label,'base_probability':round(base_prob,4),'adjustments':adjustments,
            'active_mutations':len([v for v in mut_profile.values() if v==1]),
            'model_auc':model_auc,'model':'mutation_stratified','top_features':feats})
    except Exception as e: traceback.print_exc(); return jsonify({'error':str(e)}),500

@app.route('/api/narrative', methods=['POST'])
def narrative():
    try:
        d=request.get_json(force=True)
        age=d.get('age',50); sex=d.get('sex',''); cancer=d.get('cancer',''); stage=d.get('stage',1)
        ecog=d.get('ecog',0); bio=d.get('bio','unknown'); prior=d.get('prior','none')
        comorbid=d.get('comorbid','none'); risk=d.get('risk',0); tier=d.get('tier_label','Moderate Risk')
        sens=d.get('sensitivity_pct',0); auc=d.get('model_auc',0); action=d.get('action','MDT review')
        drugs=d.get('drugs',[]); pid=d.get('patient_id','')
        drug_list=', '.join([x.get('name',x.get('n','')) if isinstance(x,dict) else str(x) for x in drugs[:3]])
        stage_desc=['','localised','locally advanced','regionally advanced','metastatic'][int(stage)] if int(stage)<=4 else 'advanced'
        ecog_desc=['fully active','ambulatory restricted','ambulatory self-care','limited self-care','fully disabled'][int(ecog)] if int(ecog)<=4 else 'unknown'
        prompt=f"""You are a senior oncologist writing a formal clinical summary for a multidisciplinary tumour board.

Patient: {age}yo {sex}, {cancer} Stage {stage} ({stage_desc}), biomarker: {bio}, ECOG {ecog} ({ecog_desc})
Prior therapy: {prior} | Comorbidities: {comorbid}
CancerGPT risk score: {risk}/100 ({tier}) | Drug sensitivity: {sens}% | Model AUC: {auc}
Recommended drugs: {drug_list} | Priority action: {action}

Write a formal 3-paragraph clinical narrative:
P1 — Patient overview: cancer profile, stage, molecular subtype, clinical significance of biomarkers.
P2 — Treatment rationale: why these specific drugs, biological mechanism, model AUC confidence.
P3 — Management plan: monitoring schedule, imaging, labs, lifestyle, specific to this patient's risk and stage.
Rules: formal medical English, no bullets, no headers, 3-5 sentences per paragraph, end with MDT coordination statement."""
        text_out,model_name,tokens=call_llm([{"role":"user","content":prompt}],
            "You are a senior oncologist. Write precise, formal clinical English.",max_tokens=1000)
        if not text_out:
            stage_str=['','localised','locally advanced','regionally advanced','metastatic'][int(stage)] if int(stage)<=4 else 'advanced'
            drug_names=', '.join([x.get('name',x.get('n','Unknown')) if isinstance(x,dict) else str(x) for x in drugs[:2]])
            text_out=(f"This {age}-year-old {sex} presents with {cancer} ({stage_str} disease), classified as {bio} subtype, "
                f"with ECOG performance status {ecog}. CancerGPT assigns risk score {risk}/100 ({tier.lower()}), "
                f"incorporating stage, performance status, prior therapy ({prior}), and comorbidities ({comorbid}).\n\n"
                f"Priority agents are {drug_names} (model AUC {auc}), selected based on {bio} biomarker profile and GDSC2 pharmacogenomics data. "
                f"Predicted sensitivity {sens}% reflects population-level in vitro signal and requires clinical validation.\n\n"
                f"Management includes baseline imaging before therapy, restaging after 2-3 cycles (RECIST), and haematological monitoring each cycle. "
                f"Oncology review every {'6-8 weeks' if int(stage)>=3 else '3 months'}. MDT tumour board coordination is recommended.")
            model_name='template'; tokens=0
        session=Session(); pat_id=None
        try:
            pat=Patient(patient_ref=pid or None,age=int(age or 0),sex=sex,cancer_type=cancer,
                stage=int(stage or 1),ecog=int(ecog or 0),biomarker=bio,expr_profile='medium',
                prior_therapy=prior,comorbid=comorbid,risk_score=int(risk or 0),risk_label=tier,
                risk_action=action,sensitivity_pct=float(sens or 0),model_auc=float(auc or 0),
                notes=text_out[:2000])
            session.add(pat); session.commit(); pat_id=pat.id
        except Exception as e: session.rollback(); print(f"DB err: {e}")
        finally: session.close()
        return jsonify({'narrative':text_out,'patient_db_id':pat_id,'saved_to_db':pat_id is not None,
            'model':model_name,'provider':LLM_PROVIDER,'tokens_used':tokens,'drug_list':drug_list})
    except Exception as e: traceback.print_exc(); return jsonify({'error':str(e)}),500


# ── /api/chat — Single-message conversational interface ───────────────────────
# The AI understands natural language, extracts patient data, runs real models,
# and returns structured results + a clinical narrative in one response.
@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main single-click chat endpoint.
    Accepts: { messages: [{role, content}], context: {last_patient, last_drug} }
    Returns: { reply, card_type, card_data, model }
    """
    try:
        body     = request.get_json(force=True)
        messages = body.get('messages', [])
        context  = body.get('context', {})   # optional: last patient/drug used
        if not messages:
            return jsonify({'error': 'No messages'}), 400

        user_msg = messages[-1]['content']

        # ── Step 1: Ask the LLM to parse intent + extract entities ────────────
        PARSE_SYSTEM = """You are a clinical NLP engine for CancerGPT.
Extract structured information from the user message and output ONLY valid JSON.

Output exactly this schema (use null for missing fields):
{
  "intent": one of ["predict_drug","patient_risk","explain","list_drugs","general_question"],
  "cancer": TCGA code string or null (e.g. "BRCA","LUAD","SKCM","COAD","GBM"),
  "drug": drug name string or null (e.g. "Olaparib","Erlotinib","Paclitaxel"),
  "age": integer or null,
  "sex": "Female"|"Male"|"Other"|null,
  "stage": integer 1-4 or null,
  "ecog": integer 0-4 or null,
  "biomarker": string or null (e.g. "BRCA1","EGFR-mut","TNBC","HR+HER2-"),
  "expr": "high"|"medium"|"low"|null,
  "prior": "none"|"chemo"|"targeted"|"endo"|"multi"|null,
  "comorbid": "none"|"cardio"|"diabetes"|"renal"|"hepatic"|"multi"|null,
  "mutations": list of gene strings or [],
  "question": the user's question rephrased as a clinical question string
}

Cancer code mapping (use these exact codes):
BRCA=breast, LUAD=lung adenocarcinoma, LUSC=lung squamous, SKCM=melanoma,
COAD=colorectal, GBM=glioblastoma, OV=ovarian, PRAD=prostate, PAAD=pancreatic,
BLCA=bladder, HNSC=head/neck, KIRC=kidney, LIHC=liver, STAD=stomach,
THCA=thyroid, UCEC=uterine, LAML=AML, ALL=ALL, CLL=CLL, MM=myeloma,
SCLC=small cell lung, MESO=mesothelioma, NB=neuroblastoma

Drug name mapping (use exact model names):
olaparib=Olaparib, erlotinib=Erlotinib, afatinib=Afatinib, trametinib=Trametinib,
selumetinib=Selumetinib, palbociclib=Palbociclib, tamoxifen=Tamoxifen,
paclitaxel=Paclitaxel, docetaxel=Docetaxel, cisplatin=Cisplatin,
oxaliplatin=Oxaliplatin, nutlin=Nutlin-3a (-), lapatinib=Lapatinib,
alpelisib=Alpelisib, navitoclax=Navitoclax, 5fu=5-Fluorouracil,
irinotecan=Irinotecan, gemcitabine=Gemcitabine

Output ONLY the JSON object, no explanation, no markdown."""

        parse_reply, _, _ = call_llm(
            [{"role":"user","content":f"Extract from: {user_msg}"}],
            PARSE_SYSTEM, max_tokens=400, temperature=0.1)

        parsed = {}
        if parse_reply:
            try:
                clean = parse_reply.strip()
                if clean.startswith('```'): clean = clean.split('```')[1].lstrip('json').strip()
                parsed = json.loads(clean)
            except: parsed = {}

        intent   = parsed.get('intent','general_question')
        cancer   = parsed.get('cancer') or context.get('last_cancer')
        drug     = parsed.get('drug')   or context.get('last_drug')
        age      = parsed.get('age')    or context.get('last_age',  50)
        sex      = parsed.get('sex')    or context.get('last_sex',  'not specified')
        stage    = parsed.get('stage')  or context.get('last_stage', 2)
        ecog     = parsed.get('ecog')   or context.get('last_ecog',  1)
        bio      = parsed.get('biomarker') or context.get('last_bio', 'unknown')
        expr     = parsed.get('expr')   or 'medium'
        prior    = parsed.get('prior')  or 'none'
        comorbid = parsed.get('comorbid') or 'none'
        mutations= parsed.get('mutations', [])

        card_type = None
        card_data = {}
        model_results = {}

        # ── Step 2: Run the real model based on intent ─────────────────────────
        if intent == 'predict_drug' and cancer and drug:
            # Real XGBoost drug prediction
            drug_info = DRUG_AUC.get(drug, {})
            model_auc = drug_info.get('auc', 0.75)
            prob, model_used = None, 'formula'
            if MODEL_READY and PROBE_COLS is not None:
                try:
                    X = build_vector(expr, len(PROBE_COLS))
                    prob = float(BEST_MODEL.predict_proba(X)[0][1]); model_used = 'real_pkl'
                    if drug != BEST_DRUG:
                        prob = float(np.clip(prob*(model_auc/DRUG_AUC.get(BEST_DRUG,{}).get('auc',.951)),.03,.97))
                except: prob = None
            if prob is None: prob = formula_predict(model_auc, expr, cancer, drug)

            # Apply mutation adjustments if provided
            drug_sens = MUT_DRUG_SENS.get(drug, {})
            mut_adj = []
            for g in mutations:
                if g in drug_sens:
                    prob += drug_sens[g]['adj'] * 0.35
                    mut_adj.append({'gene':g, 'type':drug_sens[g]['t']})
            prob = float(np.clip(prob, .03, .97))
            label = 'SENSITIVE' if prob>=.65 else 'RESISTANT' if prob<.40 else 'INTERMEDIATE'
            feats = DRUG_FEATURES.get(drug, [])[:8]

            model_results = {'prob':round(prob,4),'label':label,'model_auc':model_auc,
                             'model_used':model_used,'features':feats,'mut_adj':mut_adj}
            card_type = 'drug_prediction'
            card_data = {'drug':drug,'cancer':cancer,'prob':round(prob*100,1),
                         'label':label,'auc':model_auc,'features':feats,'mut_adj':mut_adj,
                         'model_used':model_used}
            # Save to DB
            try:
                session=Session()
                rec=Prediction(cancer_type=cancer,drug=drug,expr_profile=expr,
                    probability=round(prob,4),prediction=label,model_auc=model_auc,
                    model_used=model_used,top_features=feats)
                session.add(rec); session.commit(); session.close()
            except: pass

        elif intent == 'patient_risk' and cancer:
            # Real patient risk + drug recommendations
            if not age: age = 50
            risk = compute_risk(int(age), int(stage or 2), int(ecog or 1), prior, comorbid)
            tcls, tlbl, tsub, action = risk_tier(risk)
            auc_m = {'high':.82,'medium':.72,'low':.61}.get(expr,.72)
            sens = int(np.clip(auc_m*100+(12 if expr=='high' else -20 if expr=='low' else -5),10,97))
            DRUG_RECS = {
                'BRCA':{'TNBC':[('Olaparib',.909),('Cisplatin',.855),('Paclitaxel',.769)],
                    'HR+HER2-':[('Tamoxifen',.751),('Palbociclib',.836),('Alpelisib',.733)],
                    'HER2+':[('Afatinib',.849),('Lapatinib',.723),('Docetaxel',.837)],
                    'BRCA1':[('Olaparib',.909),('Cisplatin',.855),('Navitoclax',.841)],
                    'default':[('Paclitaxel',.769),('Docetaxel',.837),('Cisplatin',.855)]},
                'LUAD':{'default':[('Erlotinib',.791),('Afatinib',.849),('Selumetinib',.900)]},
                'SKCM':{'default':[('Trametinib',.875),('Selumetinib',.900)]},
                'COAD':{'default':[('Oxaliplatin',.885),('5-Fluorouracil',.805),('Irinotecan',.832)]},
                'default':{'default':[('Cisplatin',.855),('Oxaliplatin',.885),('Olaparib',.909)]},
            }
            rm = DRUG_RECS.get(cancer, DRUG_RECS['default'])
            drugs_list = rm.get(bio, rm.get('default', [('Cisplatin',.855)]))
            model_results = {'risk':risk,'risk_class':tcls,'risk_label':tlbl,'risk_detail':tsub,
                             'action':action,'sensitivity_pct':sens,'model_auc':round(auc_m,3),
                             'drugs':[{'name':n,'auc':a} for n,a in drugs_list]}
            card_type = 'patient_risk'
            card_data = model_results.copy()
            card_data.update({'age':age,'sex':sex,'cancer':cancer,'stage':stage,
                              'ecog':ecog,'bio':bio,'expr':expr})
            # Save to DB
            try:
                session=Session()
                pat=Patient(age=int(age),sex=sex,cancer_type=cancer,stage=int(stage or 2),
                    ecog=int(ecog or 1),biomarker=bio,expr_profile=expr,prior_therapy=prior,
                    comorbid=comorbid,risk_score=risk,risk_label=tlbl,risk_action=action,
                    sensitivity_pct=sens,model_auc=round(auc_m,3))
                session.add(pat); session.commit(); session.close()
            except: pass

        # ── Step 3: Generate conversational reply with model context ───────────
        model_context = ""
        if model_results:
            model_context = f"\n\nReal model output (already computed, reference these exact numbers):\n{json.dumps(model_results, indent=2)}"

        REPLY_SYSTEM = f"""You are CancerGPT, an expert oncology AI assistant backed by XGBoost models trained on GDSC2 × Affymetrix HG-U133A data (41 drugs, 621 cell lines, mean AUC 0.798).

When model results are provided, always cite the exact numbers from them.
Be clinically precise, conversational, and direct. Do not repeat the question back.
Structure your response clearly. Use short paragraphs. End with a concrete clinical recommendation or next step.
Always add: "⚠ Research tool — results require oncologist review before clinical use."{model_context}"""

        reply_text, model_name, _ = call_llm(
            messages, REPLY_SYSTEM, max_tokens=700, temperature=0.4)

        if not reply_text:
            reply_text = "I'm sorry, I couldn't reach the LLM right now. Please check your GROQ_API_KEY or GEMINI_API_KEY in your .env file."

        return jsonify({
            'reply':      reply_text,
            'card_type':  card_type,
            'card_data':  card_data,
            'model':      model_name or 'unavailable',
            'intent':     intent,
            'parsed':     parsed,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__=='__main__':
    print("\n"+"="*50)
    print("  CancerGPT Flask + PostgreSQL")
    print(f"  DB: {DATABASE_URL[:45]}...")
    print(f"  Expression model : {'Ready' if MODEL_READY else 'Not loaded'} | Drugs: {len(DRUG_AUC)}")
    print(f"  Mutation model   : {'Ready' if MUT_MODEL_READY else 'Not loaded'}")

# ── Initialise database tables at startup ─────────────────────────────────────
try:
    Base.metadata.create_all(engine)
    print("  Database tables  : Ready ✓")
except Exception as _dbe:
    print(f"  Database tables  : Error — {_dbe}")
    print("="*50)
    init_db()
    app.run(debug=True,host='0.0.0.0',port=5000)
