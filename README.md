<div align="center">

<img src="https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge&logo=render" />
<img src="https://img.shields.io/badge/Model-XGBoost-orange?style=for-the-badge&logo=python" />
<img src="https://img.shields.io/badge/LLM-Groq_Llama_3.3_70b-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/Dataset-GDSC2-purple?style=for-the-badge" />
<img src="https://img.shields.io/badge/Mean_AUC-0.798-red?style=for-the-badge" />
<img src="https://img.shields.io/badge/Best_AUC-0.951_(Nutlin--3a)-darkred?style=for-the-badge" />

<br/><br/>

# 🧬 CancerGPT

### An Integrated Pharmacogenomic Decision Support System for Oncology  
### Using XGBoost Drug Sensitivity Prediction

<br/>

> **Final Year M.Sc. Bioinformatics Project**  
> MIT Art, Design and Technology University (MIT ADTU), Pune  
> In collaboration with **CSIR-National Chemical Laboratory (NCL), Pune**  
> Under the guidance of **Dr. M. Karthikeyan** (Senior Principal Scientist, CSIR-NCL)  
> Internal Guide: **Prof. Dr. Sanket Bapat** (MIT ADTU)  
> Academic Year: **2025–2026**

<br/>

**[🚀 Live Demo](https://cancergpt-api.onrender.com/ui)** &nbsp;·&nbsp;
**[📊 API Health](https://cancergpt-api.onrender.com/api/health)** &nbsp;·&nbsp;
**[💊 All Drugs](https://cancergpt-api.onrender.com/api/drugs)**

</div>

---

## 📖 About This Project

I developed Gen-AI ChatBot during my internship at **CSIR-National Chemical Laboratory, Pune** (January 2026 – June 2026) under the guidance of Dr. M. Karthikeyan.

The core idea came from Dr. Karthikeyan's explanation that cancer treatment cannot depend only on cancer type — two patients with the same cancer can respond differently to the same drug based on their genes and molecular profile. This motivated me to build a **pharmacogenomics-driven clinical decision support system** that combines gene expression data, mutation profiles, and machine learning to predict drug sensitivity.

ChatBot accepts natural language patient descriptions and runs real **XGBoost models trained on GDSC2 data** to predict drug sensitivity — all in a single conversational interaction like ChatGPT, but for oncology.

---

## 🔬 What I Built

| Component | Description |
|-----------|-------------|
| 🧠 **ML Models** | 41 XGBoost classifiers, one per drug, trained on GDSC2 × Affymetrix U133A |
| 🧬 **Multiomics Features** | 2,000 gene expression probes + 44 somatic mutation flags = 2,044 features |
| 📊 **Risk Scoring** | Integrated clinical risk algorithm (0–100 scale) across 6 clinical parameters |
| 💬 **Chat Interface** | Single-click conversational UI — describe patient in plain English |
| 🤖 **LLM Narratives** | AI-generated 3-paragraph clinical summaries via Groq/Gemini |
| 🗄️ **Database** | PostgreSQL with pgAdmin — stored 73 patient records during internship |
| 🐳 **Docker** | 3-container architecture: cancergpt_api · cancergpt_db · cancergpt_pgadmin |
| ☁️ **Deployment** | Render cloud hosting with automatic GitHub deployment |

---

## 📈 Model Performance

> Trained 41 independent XGBoost classifiers — one per drug in the GDSC2 dataset.  
> Mean ROC-AUC: **0.798** across all 41 drugs | Best: **Nutlin-3a AUC = 0.951**  
> 83% of models achieved AUC ≥ 0.75 | Cross-validation SD < 0.04 (stable)

```
Nutlin-3a (-)      ████████████████████  0.951   p53 / MDM2
Olaparib           ███████████████████   0.909   DNA repair / PARP
Selumetinib        ███████████████████   0.900   MAPK / MEK1/2
Oxaliplatin        ██████████████████    0.885   DNA damage
Trametinib         ██████████████████    0.875   MAPK / MEK1/2
Cisplatin          █████████████████     0.855   DNA damage
Afatinib           █████████████████     0.849   EGFR / ERBB2
Docetaxel          ████████████████      0.837   Microtubule
Navitoclax         ████████████████      0.841   Apoptosis / BCL2
Palbociclib        ████████████████      0.836   Cell cycle / CDK4/6
```

---

## 🛠️ Tech Stack

```
┌────────────────────────────────────────────────────────────────┐
│                   CancerGPT System Architecture                │
├──────────────────┬─────────────────────────────────────────────┤
│  Frontend        │  Single-page HTML + CSS + JavaScript        │
│  Backend         │  Python Flask REST API (Gunicorn)           │
│  ML Models       │  XGBoost + scikit-learn + imbalanced-learn  │
│  Feature Select  │  SelectKBest (F-statistic, k=50 per drug)   │
│  Class Balance   │  SMOTE (Synthetic Minority Oversampling)     │
│  LLM Primary     │  Groq API — Llama 3.3-70b-versatile         │
│  LLM Fallback    │  Google Gemini 2.0 Flash                    │
│  Database        │  PostgreSQL 15 + SQLAlchemy ORM             │
│  DB Management   │  pgAdmin 4                                  │
│  Container       │  Docker + Docker Compose (3 services)       │
│  Deployment      │  Render (Web Service + PostgreSQL)          │
│  Cheminformatics │  RDKit (rdkit_env conda environment)        │
└──────────────────┴─────────────────────────────────────────────┘
```

---

## 🔬 Methodology

### Dataset
- **GDSC2** (Genomics of Drug Sensitivity in Cancer v2) from Wellcome Sanger Institute
- Drug sensitivity for 250+ compounds tested across 969 cancer cell lines
- After merging with Affymetrix U133A expression data: **621 cell lines**, **41 drugs**, **147,490 drug-response rows**
- AUC binarised at threshold 0.8: below = Sensitive, above = Resistant

### Feature Engineering
- **Expression features:** Top 2,000 probes by variance (from 22,277 Affymetrix U133A probes)
- **Mutation features:** 44 binary somatic mutation flags (BRCA1/2, TP53, EGFR, KRAS etc.)
- **Total feature vector:** 2,044 dimensions per cell line

### Training Pipeline (per drug)
```
Step 1 → SelectKBest (F-statistic, k=50)     — top probes per drug
Step 2 → SMOTE                                — class imbalance handling
Step 3 → StandardScaler                       — normalise to zero mean/unit variance
Step 4 → XGBoost (n_est=200, depth=6, lr=0.05, subsample=0.8)
Step 5 → 5-fold stratified cross-validation   — ROC-AUC evaluation
```

### Overfitting Prevention
Four strategies working together: variance filtering → SelectKBest → XGBoost L1/L2 regularisation → 5-fold CV. Cross-validation SD remained below **0.04** for most models.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Model + database status |
| POST | `/api/chat` | Main conversational AI endpoint |
| POST | `/api/predict` | Drug sensitivity prediction |
| POST | `/api/patient` | Full patient risk assessment |
| POST | `/api/narrative` | LLM clinical narrative generation |
| POST | `/api/predict/stratified` | Mutation-stratified prediction |
| GET | `/api/drugs` | List all 41 trained drugs |
| GET | `/api/history/predictions` | Prediction history (PostgreSQL) |
| GET | `/api/history/patients` | Patient history (PostgreSQL) |
| GET | `/api/analytics` | Usage statistics |

---

## 🚀 Run Locally

### With Docker (recommended)

```bash
git clone https://github.com/SayaliKolhapure/xgboost-model.git
cd xgboost-model

# Create .env file with your API keys
echo "GROQ_API_KEY=your_key_here" > .env
echo "GEMINI_API_KEY=your_key_here" >> .env

docker-compose up --build
```

Open: `http://localhost:5000/ui`

### Without Docker

```bash
pip install -r requirements.txt
python app.py
```

---

## ⚙️ Environment Variables

```env
GROQ_API_KEY=your_groq_key        # Free at console.groq.com
GEMINI_API_KEY=your_gemini_key    # Free at aistudio.google.com
DATABASE_URL=your_postgresql_url  # Render PostgreSQL internal URL
```

---

## 📁 Repository Structure

```
xgboost-model/
│
├── app.py                           # Flask backend 
├── index.html                       # Conversational single-page chat UI
├── Dockerfile                       # Python 3.11-slim Docker build
├── docker-compose.yml               # 3-container: api + db + pgadmin
├── render.yaml                      # Render cloud deployment config
├── requirements.txt                 # Pinned dependencies
├── Procfile                         # Gunicorn process definition
│
├── xgboost_best_model.pkl           # Primary XGBoost classifier (41 drugs)
├── xgboost_probe_cols.pkl           # Affymetrix probe column names
├── xgboost_gdsc_results.csv         # Per-drug AUC results
├── xgboost_gdsc_features.csv        # Top predictive genes per drug
│
├── xgboost_mutation_model.pkl       # Mutation-stratified classifier
├── xgboost_mutation_results.csv     # Mutation model evaluation
└── xgboost_mutation_features.csv    # Mutation feature importance
```

---

## 🧬 Supported Cancer Types

```
BRCA  Breast           LUAD  Lung Adeno        LUSC  Lung Squamous
SKCM  Melanoma         COAD  Colorectal         GBM   Glioblastoma
OV    Ovarian          PRAD  Prostate           PAAD  Pancreatic
BLCA  Bladder          HNSC  Head & Neck        KIRC  Kidney
LIHC  Liver            STAD  Stomach            THCA  Thyroid
UCEC  Uterine          LAML  AML                ALL   ALL
CLL   CLL              MM    Myeloma            SCLC  Small Cell Lung
NB    Neuroblastoma    MESO  Mesothelioma
```

---

## 📋 Case Study Example

**Patient:** 54year male · SCLC Stage I (localised) · TP53 wild-type · ECOG 0

| Output | Value |
|--------|-------|
| Risk Score | 28/100 — Moderate Risk |
| Top Drug | Cisplatin (AUC 0.855) |
| 2nd Drug | Oxaliplatin (AUC 0.885) |
| Sensitivity | 94% predicted |
| Action | MDT review |

The AI narrative correctly identified TP53 wild-type significance, recommended platinum-based chemotherapy, and suggested CT restaging after 2–3 cycles (RECIST). ✅

---

## 📚 Bioinformatics Tools Used

During my internship at CSIR-NCL Pune, Dr. Karthikeyan introduced me to:

| Tool | Purpose |
|------|---------|
| **PubMed** | Pharmacogenomics literature search |
| **ChEMBL / PubChem** | Chemical compound and drug-target data |
| **DrugBank** | Drug mechanism and target information |
| **RDKit** | Cheminformatics and pharmacophore analysis |
| **pgAdmin** | PostgreSQL database administration |


---
## ⚠️ Disclaimer

> CancerGPT is a **research tool** trained on GDSC2 in vitro cancer cell line data.
> All predictions are population-level estimates from laboratory data and have not been
> clinically validated. **All outputs must be reviewed by a qualified oncologist.**
> Not intended for direct patient care.

---

## 👩‍💻 Author

<div align="center">

**Ms. Sayali Vidyasagar Kolhapure**  
M.Sc. Bioinformatics ·  
MIT ADTU School of Bioengineering Sciences & Research, Pune  
CSIR-National Chemical Laboratory (NCL), Pune · 2025–2026

[![GitHub](https://img.shields.io/badge/GitHub-SayaliKolhapure-black?style=flat&logo=github)](https://github.com/SayaliKolhapure)
[![Email](https://img.shields.io/badge/Email-sayalikolhapure01@gmail.com-red?style=flat&logo=gmail)](mailto:sayalikolhapure01@gmail.com)

</div>

---

## 📄 References

1. Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*
2. Iorio et al. (2016). A Landscape of Pharmacogenomic Interactions in Cancer. *Cell, 166(3)*
3. Yang et al. (2013). Genomics of Drug Sensitivity in Cancer (GDSC). *Nucleic Acids Research*
4. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR, 12*
5. Singhal et al. (2023). Large Language Models Encode Clinical Knowledge. *Nature, 620*

---

<div align="center">

*Built at CSIR-NCL Pune · MIT ADTU Bioinformatics · 2025–2026*  
*Research use only · Not validated for clinical use*

</div>
