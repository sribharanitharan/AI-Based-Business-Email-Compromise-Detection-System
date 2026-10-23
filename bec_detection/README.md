# 🛡️ BEC Shield — AI-Based Business Email Compromise Detection System v2.0

An enterprise-grade cyber threat intelligence (CTI) and machine learning solution for detecting, explaining, and responding to Business Email Compromise (BEC) attacks in real-time.

---

## 🚀 Key Features

- **Multi-Tier Model Ensemble**:
  - **Tier 1 Baselines**: Logistic Regression, Random Forest, Naive Bayes, Linear & RBF SVM
  - **Tier 2 Gradient Boosting**: XGBoost, LightGBM, CatBoost
  - **Tier 3 Deep Learning**: PyTorch Bi-LSTM with Attention, Kim TextCNN, and Transformer (BERT/DistilBERT)
  - **Tier 4 Anomaly Detection**: Isolation Forest, One-Class SVM, Deep Feature Reconstruction Autoencoder
  - **Tier 5 Graph Analytics**: Social interaction network topology analyzing VIP targeting & anomalous communication edges
  - **Tier 6 Semantic Intent**: Few-shot LLM intent reasoning for CEO impersonation, invoice diversion, and gift card fraud
- **Explainable AI (XAI)**: Local decision explainability via SHAP waterfall attributions and LIME feature importance.
- **Threat Intelligence**: Real-time enrichment via AbuseIPDB and VirusTotal with local caching.
- **SOC Analyst Operations**: Interactive web dashboard, human-in-the-loop active learning review queue, PDF incident reports, and CSV audit exports.
- **Incident Response**: Slack/Teams/SIEM webhook notifications and SMTP alerts.

---

## 📁 Repository Structure

```
bec_detection/
├── data/
│   ├── raw/                        # Raw email datasets
│   ├── processed/                  # Cleaned & feature-extracted data
│   ├── threat_intel/               # IOC feeds, dark web indicators
│   └── graph/                      # Email communication graphs
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_deep_learning.ipynb
│   ├── 06_graph_models.ipynb
│   └── 07_ensemble_results.ipynb
│
├── src/
│   ├── ingestion/
│   │   ├── email_parser.py
│   │   ├── imap_connector.py
│   │   └── threat_intel_fetcher.py
│   │
│   ├── features/
│   │   ├── text_features.py
│   │   ├── header_features.py
│   │   ├── url_features.py
│   │   ├── behavioral.py
│   │   ├── linguistic.py
│   │   ├── social_engineering.py
│   │   ├── metadata_forensics.py
│   │   ├── thread_context.py
│   │   ├── dark_web.py
│   │   └── extractors.py
│   │
│   ├── models/
│   │   ├── baseline/
│   │   ├── boosting/
│   │   ├── deep_learning/
│   │   ├── graph/
│   │   ├── anomaly/
│   │   ├── llm/
│   │   ├── automl/
│   │   ├── ensemble/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   ├── explainability/
│   │   ├── shap_explainer.py
│   │   └── lime_explainer.py
│   │
│   ├── api/
│   │   ├── app.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── middleware.py
│   │
│   └── alerts/
│       ├── alert_manager.py
│       ├── notifier.py
│       └── report_generator.py
│
├── dashboard/
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── tests/
├── requirements.txt
├── config.yaml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🛠️ Quick Start

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Train Models
```powershell
python train.py
```

### 3. Launch Dashboard & API Server
```powershell
python app.py
# Server starts at http://localhost:8000
```

### 4. Run Tests
```powershell
python -m pytest tests/
```

### 5. Docker Deployment
```powershell
docker compose up -d --build
```
