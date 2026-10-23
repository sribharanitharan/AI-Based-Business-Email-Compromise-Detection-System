# 🛡️ AI-Based Detection of Business Email Compromise (BEC)
## Enhanced Implementation Plan v2.0

---

## 📌 Project Overview

| Field | Details |
|---|---|
| **Project Name** | AI-Based BEC Detection System |
| **Domain** | Cybersecurity / Machine Learning |
| **Core Tech** | Python, NLP, Deep Learning, FastAPI |
| **Goal** | Detect fraudulent/suspicious emails using ensemble AI and alert users in real-time |
| **Version** | 2.0 — Enhanced Features + Extended Model Suite |

---

## 🏗️ Enhanced System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       BEC Detection System v2.0                          │
│                                                                          │
│  ┌─────────────┐   ┌──────────────────────────────────────────────────┐  │
│  │   Email     │──▶│              FEATURE EXTRACTION LAYER            │  │
│  │   Ingestion │   │  Content │ Header │ URL │ Behavioral │ Metadata  │  │
│  └─────────────┘   └────────────────────┬─────────────────────────────┘  │
│                                         │                                │
│  ┌──────────────────────────────────────▼─────────────────────────────┐  │
│  │                       ML / DL ENSEMBLE LAYER                       │  │
│  │  BERT │ XGBoost │ LSTM │ GNN │ IsolationForest │ LightGBM │ GPT   │  │
│  └────────────────────────────────────┬───────────────────────────────┘  │
│                                       │                                  │
│  ┌─────────────┐  ┌───────────────────▼──────────────┐  ┌────────────┐  │
│  │  SHAP / LIME│  │    Composite Risk Scoring Engine  │  │  Active    │  │
│  │  Explainer  │  │    + Threat Intelligence Feed     │  │  Learning  │  │
│  └─────────────┘  └──────────────────┬───────────────┘  └────────────┘  │
│                                      │                                   │
│  ┌─────────────────────────────────── ▼──────────────────────────────┐   │
│  │              Dashboard + Alert Engine + Reporting                 │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Enhanced Project Structure

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
│   │   └── threat_intel_fetcher.py  # NEW: Fetch IOC / dark web feeds
│   │
│   ├── features/
│   │   ├── text_features.py
│   │   ├── header_features.py
│   │   ├── url_features.py
│   │   ├── behavioral.py
│   │   ├── linguistic.py            # NEW: Deception & tone analysis
│   │   ├── social_engineering.py    # NEW: Manipulation pattern detector
│   │   ├── metadata_forensics.py    # NEW: Email metadata & EXIF analysis
│   │   ├── thread_context.py        # NEW: Thread-level anomaly detection
│   │   └── dark_web.py              # NEW: Dark web / IOC correlation
│   │
│   ├── models/
│   │   ├── baseline/
│   │   │   ├── logistic_regression.py
│   │   │   └── random_forest.py
│   │   ├── boosting/
│   │   │   ├── xgboost_model.py
│   │   │   ├── lightgbm_model.py    # NEW
│   │   │   └── catboost_model.py    # NEW
│   │   ├── deep_learning/
│   │   │   ├── bert_classifier.py
│   │   │   ├── lstm_classifier.py   # NEW
│   │   │   └── cnn_text.py          # NEW: 1D CNN for email text
│   │   ├── graph/
│   │   │   └── gnn_model.py         # NEW: Graph Neural Network
│   │   ├── anomaly/
│   │   │   ├── isolation_forest.py  # NEW
│   │   │   └── autoencoder.py       # NEW: Deep anomaly detection
│   │   ├── llm/
│   │   │   └── gpt_few_shot.py      # NEW: GPT few-shot BEC detector
│   │   ├── automl/
│   │   │   └── automl_pipeline.py   # NEW: AutoML with TPOT / AutoSklearn
│   │   ├── ensemble/
│   │   │   ├── stacking.py
│   │   │   └── weighted_voter.py    # NEW: Dynamic weight voting
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   ├── explainability/
│   │   ├── shap_explainer.py        # NEW: SHAP feature importance
│   │   └── lime_explainer.py        # NEW: LIME local explanations
│   │
│   ├── api/
│   │   ├── app.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── middleware.py            # NEW: Rate limiting, auth
│   │
│   └── alerts/
│       ├── alert_manager.py
│       ├── notifier.py
│       └── report_generator.py      # NEW: PDF/CSV report export
│
├── dashboard/
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── tests/
├── requirements.txt
├── config.yaml
└── README.md
```

---

## 📋 Enhanced Implementation Phases

### ✅ Phase 1 — Data Collection & Preparation (Week 1–2)

#### 1.1 Dataset Sources
| Dataset | Description | Size |
|---|---|---|
| **Enron Email Dataset** | Real corporate emails | 500K+ |
| **CEAS 2008 Spam Corpus** | Labeled spam/phishing | 100K+ |
| **APWG eCrime Dataset** | Phishing email samples | 50K+ |
| **TREC 2007 Spam Track** | Labeled spam | 75K+ |
| **SpamAssassin Public Corpus** | Spam + ham labeled | 6K+ |
| **PhishingCorpus** | Targeted phishing emails | Custom |
| **Synthetic BEC Dataset** | GPT-augmented BEC emails | Generated |
| **Dark Web IOC Feed** | Indicator of Compromise lists | Live feed |

#### 1.2 Data Augmentation
```python
# Augment limited BEC samples using:
augmentation_strategies = [
    "Back-translation (EN→FR→EN) to rephrase BEC emails",
    "Synonym substitution using WordNet",
    "GPT-3.5 prompt-based BEC email generation",
    "EDA (Easy Data Augmentation) — insertion, swap, delete",
    "SMOTE on extracted feature vectors for class balance"
]
```

---

### ✅ Phase 2 — Extended Feature Engineering (Week 2–3)

#### 2.1 Email Content Features *(Original)*
| Feature | Method |
|---|---|
| TF-IDF unigram + bigram vectors | `TfidfVectorizer(ngram_range=(1,2))` |
| BERT sentence embeddings | `sentence-transformers` |
| Urgency / pressure keywords | Custom lexicon |
| Financial keyword presence | Domain-specific regex |
| Impersonation cues | spaCy NER |

---

#### 2.2 🆕 Linguistic Deception Analysis
| Feature | Description | Tool |
|---|---|---|
| **Sentiment polarity shift** | Detect fear/urgency manipulation | `VADER`, `TextBlob` |
| **Certainty language score** | Detect over-confident phrasing | Custom lexicon |
| **Passive vs Active voice ratio** | BEC emails use commanding tone | spaCy POS tagging |
| **Readability score** | Attackers write unnaturally clean prose | `textstat` |
| **Punctuation anomaly** | Excess !!!, ALL CAPS detection | Regex |
| **Formality score** | Sudden formality shifts signal impersonation | `pyinflect` |
| **Hedging language ratio** | Legal/authentic emails use hedging | Custom regex |
| **Perplexity score** | Low perplexity → possibly AI-generated email | GPT-2 perplexity |

```python
# Example: Linguistic deception scoring
from textstat import flesch_reading_ease
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def extract_linguistic_features(text):
    sia = SentimentIntensityAnalyzer()
    return {
        'urgency_score': count_urgency_keywords(text),
        'sentiment_compound': sia.polarity_scores(text)['compound'],
        'readability': flesch_reading_ease(text),
        'caps_ratio': sum(1 for c in text if c.isupper()) / len(text),
        'exclamation_count': text.count('!'),
        'perplexity': compute_gpt2_perplexity(text)
    }
```

---

#### 2.3 🆕 Social Engineering Pattern Detection
| Pattern | Detection Method |
|---|---|
| **Authority invocation** | "The CEO has instructed...", "per board directive" | Regex + NER |
| **Scarcity / deadline pressure** | "Must be done by EOD", "Immediate action required" | Keyword classifier |
| **Secrecy request** | "Do not discuss", "Confidential – do not forward" | Phrase matching |
| **Relationship exploitation** | Using recipient's name, past interaction references | NER + name matching |
| **False urgency escalation** | Multiple urgency escalations within thread | Thread-level analysis |
| **Pretexting indicators** | Backstory setup before actual request | Sequence modeling (LSTM) |

---

#### 2.4 🆕 Email Metadata Forensics
| Feature | Description |
|---|---|
| **Message-ID format anomaly** | Legitimate MTA Message-IDs have specific patterns |
| **X-Mailer / User-Agent fingerprint** | Non-standard mail clients flag as suspicious |
| **Time zone inconsistency** | Sender claims to be in NY but sends from EU timezone |
| **MIME boundary anomaly** | Malformed MIME boundary = custom/malicious tool |
| **Encoding inconsistency** | Mixed UTF-8/Latin-1 encoding = obfuscation attempt |
| **Thread-ID mismatch** | Email claims to be a reply but has no thread parent |
| **Hop count anomaly** | Unusual number of Received headers |
| **Email size fingerprint** | Unusually minimal/padded emails |

---

#### 2.5 🆕 Email Thread Context Analysis
| Feature | Description |
|---|---|
| **Thread hijacking detection** | Email inserted into existing legitimate thread |
| **Response pattern break** | Sudden style change mid-thread |
| **New participant injection** | Unknown CC/BCC added to established thread |
| **Topic drift detection** | Conversation topic suddenly shifts to payment/credentials |
| **Reply velocity anomaly** | Unusually fast replies suggesting automation |

```python
def detect_thread_anomalies(thread_emails):
    """Analyze a complete email thread for BEC signals"""
    features = {}
    
    # Style consistency across thread
    embeddings = [get_bert_embedding(e['body']) for e in thread_emails]
    features['style_drift_score'] = compute_cosine_drift(embeddings)
    
    # New participant detection
    known_participants = get_established_participants(thread_emails[:-1])
    features['new_participant_flag'] = detect_new_participants(
        thread_emails[-1], known_participants
    )
    
    # Topic shift detection
    features['topic_shift_score'] = compute_topic_drift(thread_emails)
    
    return features
```

---

#### 2.6 🆕 Dark Web & Threat Intelligence Features
| Feature | Source |
|---|---|
| **Sender email in breach databases** | HaveIBeenPwned API |
| **Sender IP on threat blocklists** | AbuseIPDB, Spamhaus |
| **Domain on phishing feeds** | OpenPhish, PhishTank |
| **Similar attack campaigns detected** | MITRE ATT&CK pattern matching |
| **IOC (Indicator of Compromise) match** | AlienVault OTX |

```python
import requests

def check_threat_intel(sender_email, sender_ip, domain):
    flags = []
    
    # Check AbuseIPDB
    response = requests.get(
        f"https://api.abuseipdb.com/api/v2/check",
        params={"ipAddress": sender_ip, "maxAgeInDays": 90},
        headers={"Key": ABUSEIPDB_KEY}
    )
    if response.json()['data']['abuseConfidenceScore'] > 50:
        flags.append("IP on AbuseIPDB blocklist")
    
    # Check OpenPhish
    if domain in load_openphish_feed():
        flags.append("Domain on OpenPhish feed")
    
    return flags
```

---

#### 2.7 Email Header Features *(Enhanced)*
| Feature | Tool |
|---|---|
| SPF / DKIM / DMARC validation | `pyspf`, `dkimpy` |
| Sender domain age (WHOIS) | `python-whois` |
| Reply-To ≠ From mismatch | Header parsing |
| IP geolocation anomaly | `geoip2` |
| Lookalike domain (Levenshtein) | `editdistance` |
| Free email provider flag | Domain whitelist |
| **🆕 Domain registration < 30 days** | WHOIS timestamp |
| **🆕 MX record inconsistency** | `dnspython` |
| **🆕 ASN reputation score** | BGP routing tables |

#### 2.8 URL / Link Features *(Enhanced)*
| Feature | Tool |
|---|---|
| URL reputation score | VirusTotal API |
| Shortened URL detection | Regex |
| HTTPS vs HTTP | URL parsing |
| Domain typosquatting | `editdistance` |
| **🆕 URL entropy score** | High entropy = obfuscated URL |
| **🆕 Redirect chain depth** | Follow redirects, count hops |
| **🆕 Landing page similarity** | Compare to known legit site (SSIM) |
| **🆕 JavaScript obfuscation flag** | HTML parser + JS analysis |
| **🆕 IP-based URL** | `http://203.0.113.5/login` pattern |

---

### ✅ Phase 3 — Extended ML Model Suite (Week 3–6)

#### 3.1 Complete Model Inventory

```
TIER 1 — BASELINE
├── Logistic Regression (TF-IDF)
├── Naive Bayes (MultinomialNB)
└── Support Vector Machine (LinearSVC)

TIER 2 — BOOSTING ENSEMBLE
├── XGBoost
├── 🆕 LightGBM        ← Fast, handles large datasets well
└── 🆕 CatBoost        ← Excellent with categorical features

TIER 3 — DEEP LEARNING (TEXT)
├── Fine-tuned DistilBERT
├── Fine-tuned RoBERTa     ← 🆕 Better than BERT for email classification
├── 🆕 Bi-LSTM + Attention ← Sequential text modeling
└── 🆕 1D-CNN Text Classifier ← Fast local feature extraction

TIER 4 — ANOMALY / UNSUPERVISED
├── 🆕 Isolation Forest   ← Detect outlier email patterns
├── 🆕 DBSCAN Clustering  ← Group email campaigns
└── 🆕 Deep Autoencoder   ← Reconstruct normal emails; high error = BEC

TIER 5 — GRAPH-BASED
└── 🆕 Graph Neural Network (GNN) ← Model email communication networks

TIER 6 — LLM / FEW-SHOT
├── 🆕 GPT-4 Few-Shot Classifier ← Prompt-based detection
└── 🆕 SetFit (sentence-transformers) ← Few-shot with small labeled set

TIER 7 — AUTOML
└── 🆕 TPOT / Auto-Sklearn ← Automated pipeline search

FINAL — ENSEMBLE
└── Stacking Ensemble (Meta-learner = XGBoost)
    Inputs: BERT prob + LightGBM prob + LSTM prob + Isolation Forest score
```

---

#### 3.2 🆕 LightGBM Model

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': 10,    # Handle class imbalance
    'verbose': -1
}

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val)

model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
```

---

#### 3.3 🆕 CatBoost Model

```python
from catboost import CatBoostClassifier

cat_features_idx = [...]  # Indices of categorical columns (sender domain, country, etc.)

catboost_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='AUC',
    class_weights=[1, 10],    # Weight BEC class higher
    cat_features=cat_features_idx,
    early_stopping_rounds=50,
    verbose=100
)

catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val))
```

---

#### 3.4 🆕 Bi-LSTM with Attention

```python
import torch
import torch.nn as nn

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)          # (batch, seq, hidden*2)
        attn_weights = torch.softmax(
            self.attention(lstm_out), dim=1
        )                                          # Attention mechanism
        context = (lstm_out * attn_weights).sum(1) # Weighted sum
        return self.classifier(context)
```

---

#### 3.5 🆕 1D-CNN Text Classifier

```python
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=128, kernel_sizes=[2,3,4,5]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)   # (batch, embed, seq)
        pooled = [
            torch.relu(conv(x)).max(dim=2)[0]     # Max pooling per filter
            for conv in self.convs
        ]
        out = torch.cat(pooled, dim=1)
        return torch.sigmoid(self.fc(self.dropout(out)))
```

---

#### 3.6 🆕 Isolation Forest (Anomaly Detection)

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Train ONLY on legitimate emails (unsupervised)
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,    # Expected ~5% anomaly rate
    max_samples='auto',
    random_state=42
)
iso_forest.fit(X_legitimate_only)

# Predict: -1 = anomaly (potential BEC), 1 = normal
anomaly_scores = iso_forest.decision_function(X_test)
# Lower score = more anomalous
```

---

#### 3.7 🆕 Deep Autoencoder (Anomaly Detection)

```python
class EmailAutoencoder(nn.Module):
    """Trained on legitimate emails only. High reconstruction error → BEC"""
    def __init__(self, input_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, 32)         # Bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),        nn.ReLU(),
            nn.Linear(128, 256),       nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x):
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=1)  # Per-sample MSE
```

---

#### 3.8 🆕 Graph Neural Network (GNN)

```
Email Communication Graph:
  Nodes = Email addresses (senders + recipients)
  Edges = Email communications (weighted by frequency)
  Node features = Sending patterns, domain info, account age
  
BEC Signal: A compromised account shows abnormal edges
             (new connections, unusual recipients, time patterns)
```

```python
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool

class EmailGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.classifier(x))
```

---

#### 3.9 🆕 GPT Few-Shot Classifier

```python
from openai import OpenAI

client = OpenAI()

BEC_EXAMPLES = [
    {"email": "Please wire $85,000 ASAP...", "label": "BEC"},
    {"email": "Your invoice is attached...", "label": "LEGITIMATE"},
    # ... more examples
]

def classify_with_gpt(email_content):
    few_shot_prompt = build_few_shot_prompt(BEC_EXAMPLES, email_content)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a cybersecurity expert detecting Business Email Compromise attacks."},
            {"role": "user", "content": few_shot_prompt}
        ],
        temperature=0,
        max_tokens=200
    )
    
    return parse_gpt_response(response.choices[0].message.content)
    # Returns: {"label": "BEC", "confidence": 0.95, "reasons": [...]}
```

---

#### 3.10 🆕 RoBERTa Fine-tuning

```python
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# RoBERTa trained with more data and better preprocessing than BERT
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base', num_labels=2
)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Email-domain specific tokenization
def tokenize_email(email):
    # Combine subject + body with [SEP] token
    combined = f"Subject: {email['subject']} [SEP] {email['body'][:512]}"
    return tokenizer(
        combined,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
```

---

#### 3.11 🆕 Dynamic Weighted Ensemble

```python
class DynamicWeightedEnsemble:
    """Dynamically adjusts model weights based on recent performance"""
    
    def __init__(self):
        self.models = {
            'bert': bert_model,
            'roberta': roberta_model,
            'lightgbm': lgb_model,
            'catboost': cat_model,
            'lstm': lstm_model,
            'iso_forest': iso_model
        }
        self.weights = {k: 1/len(self.models) for k in self.models}
    
    def predict_proba(self, features):
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(features[name])
        
        # Weighted combination
        final_score = sum(
            self.weights[name] * pred
            for name, pred in predictions.items()
        )
        return final_score
    
    def update_weights(self, validation_results):
        """Update weights based on recent F1 scores"""
        total_f1 = sum(validation_results.values())
        self.weights = {
            name: f1 / total_f1
            for name, f1 in validation_results.items()
        }
```

---

### ✅ Phase 4 — Explainability Layer (Week 6)

#### 4.1 🆕 SHAP Explanations

```python
import shap

# For XGBoost / LightGBM
explainer = shap.TreeExplainer(lightgbm_model)
shap_values = explainer.shap_values(X_test)

# Feature importance plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Per-email explanation
def explain_prediction(email_features, model):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(email_features)
    
    top_factors = sorted(
        zip(feature_names, shap_vals[0]),
        key=lambda x: abs(x[1]), reverse=True
    )[:5]
    
    return [{"factor": f, "impact": v} for f, v in top_factors]
```

#### 4.2 🆕 LIME Explanations (for BERT/Neural Models)

```python
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['Legitimate', 'BEC'])

def explain_bert_prediction(email_text):
    def predict_fn(texts):
        return np.array([bert_predict(t) for t in texts])
    
    explanation = explainer.explain_instance(
        email_text,
        predict_fn,
        num_features=10,
        num_samples=500
    )
    return explanation.as_list()
    # Returns: [("wire transfer", 0.42), ("urgent", 0.38), ...]
```

---

### ✅ Phase 5 — Enhanced Risk Scoring (Updated)

```
╔═══════════════════════════════════════════════════════════╗
║              COMPOSITE RISK SCORE v2.0                    ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  Risk = 0.25 × Ensemble_ML_Score                          ║
║       + 0.20 × Linguistic_Deception_Score                 ║
║       + 0.15 × Header_Anomaly_Score                       ║
║       + 0.15 × Threat_Intel_Score     (NEW)               ║
║       + 0.10 × URL_Reputation_Score                       ║
║       + 0.10 × Thread_Context_Score   (NEW)               ║
║       + 0.05 × Behavioral_Score                           ║
║                                                           ║
╠═══════════════════════════════════════════════════════════╣
║  0.0 – 0.30 → ✅ Safe                                     ║
║  0.3 – 0.55 → ⚠️  Suspicious (Review Recommended)        ║
║  0.55 – 0.75 → 🟠 High Risk (Alert User)                  ║
║  0.75 – 1.0 → 🚨 Critical / Confirmed BEC (Block + Alert) ║
╚═══════════════════════════════════════════════════════════╝
```

---

### ✅ Phase 6 — API Development (Week 7)

#### 6.1 Enhanced API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analyze` | Analyze single email |
| `POST` | `/api/analyze/batch` | **🆕** Analyze multiple emails |
| `POST` | `/api/analyze/thread` | **🆕** Analyze full email thread |
| `GET` | `/api/history` | Get past analyzed emails |
| `GET` | `/api/stats` | Dashboard statistics |
| `POST` | `/api/feedback` | Submit analyst feedback |
| `GET` | `/api/explain/{email_id}` | **🆕** Get SHAP/LIME explanation |
| `GET` | `/api/report/{email_id}` | **🆕** Download PDF report |
| `GET` | `/api/threat-intel` | **🆕** Current threat feed status |
| `GET` | `/api/health` | Service health check |

#### 6.2 Enhanced Response Schema
```json
{
  "email_id": "uuid-1234",
  "risk_score": 0.87,
  "risk_level": "CRITICAL",
  "classification": "BEC",
  "confidence": 0.93,
  "model_scores": {
    "bert": 0.91,
    "lightgbm": 0.88,
    "lstm": 0.85,
    "isolation_forest": 0.79
  },
  "flags": [
    {"type": "header", "message": "Reply-To domain mismatch"},
    {"type": "linguistic", "message": "High urgency language score: 0.94"},
    {"type": "social_eng", "message": "Secrecy request detected"},
    {"type": "threat_intel", "message": "Sender IP on AbuseIPDB blocklist"},
    {"type": "domain", "message": "Lookalike domain: comp4ny.com (edit dist=1)"},
    {"type": "thread", "message": "Topic drift: financial request injected"}
  ],
  "top_shap_features": [
    {"feature": "wire_transfer_keyword", "impact": 0.42},
    {"feature": "reply_to_mismatch", "impact": 0.38},
    {"feature": "urgency_score", "impact": 0.31}
  ],
  "bec_type": "CEO Fraud",
  "recommended_action": "BLOCK + ALERT_ADMIN",
  "timestamp": "2026-10-15T14:23:06"
}
```

---

### ✅ Phase 7 — Active Learning Loop (Week 8)

```python
# Feedback-driven model improvement
class ActiveLearningPipeline:
    """
    Analyst reviews uncertain predictions (0.4–0.6 risk score)
    and labels them. These are fed back into training.
    """
    
    def get_uncertain_samples(self, threshold=(0.4, 0.6)):
        """Return emails where model is most uncertain"""
        predictions = self.model.predict_proba(unlabeled_emails)
        uncertain_mask = (predictions > threshold[0]) & (predictions < threshold[1])
        return unlabeled_emails[uncertain_mask]
    
    def retrain_with_feedback(self, labeled_feedback):
        """Incrementally retrain on analyst-labeled emails"""
        X_new, y_new = prepare_features(labeled_feedback)
        self.model.fit(
            np.vstack([X_existing, X_new]),
            np.hstack([y_existing, y_new])
        )
```

---

### ✅ Phase 8 — Dashboard & Alerts (Week 8–9)

#### 8.1 Dashboard Features
- 📊 Real-time email threat feed with live updates
- 🔴 4-tier risk level breakdown with drill-down
- 📈 Trends: BEC attempts, model performance over time
- 🔍 Email detail view with SHAP explanation waterfall chart
- 🧑‍💻 Analyst review queue for uncertain emails
- 🌐 Threat intelligence map (geolocation of suspicious IPs)
- 📋 Campaign detection: Group similar BEC attempts
- 📤 Export reports (PDF / CSV / STIX format)

---

### ✅ Phase 9 — Testing & Evaluation (Week 9–10)

#### 9.1 Model Comparison Table (Expected Targets)

| Model | Precision | Recall | F1 | AUC | Speed |
|---|---|---|---|---|---|
| Logistic Regression | 82% | 79% | 80% | 0.88 | ⚡ Fast |
| XGBoost | 89% | 87% | 88% | 0.94 | ⚡ Fast |
| **LightGBM** | 90% | 88% | 89% | 0.95 | ⚡ Fast |
| **CatBoost** | 89% | 89% | 89% | 0.95 | ⚡ Fast |
| DistilBERT | 91% | 90% | 91% | 0.96 | 🐢 Slow |
| **RoBERTa** | 93% | 92% | 92% | 0.97 | 🐢 Slow |
| **Bi-LSTM** | 88% | 87% | 87% | 0.93 | 🔸 Medium |
| **Isolation Forest** | — | 75% | — | 0.82 | ⚡ Fast |
| **GNN** | 87% | 86% | 86% | 0.92 | 🔸 Medium |
| **Ensemble (Final)** | **94%** | **93%** | **93%** | **0.98** | 🔸 Medium |

#### 9.2 Adversarial Testing
```python
adversarial_tests = [
    "BEC with perfect SPF/DKIM (compromised legitimate account)",
    "BEC with no financial keywords (indirect request)",
    "BEC using Unicode homoglyph domain (pаypal.com)",
    "BEC with legitimate-looking URLs (redirects to phishing page)",
    "AI-generated BEC email (natural language, no obvious patterns)",
    "BEC embedded in legitimate thread (thread hijacking)"
]
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **ML / Boosting** | scikit-learn, XGBoost, **LightGBM**, **CatBoost** |
| **Deep Learning** | PyTorch, HuggingFace Transformers |
| **Graph ML** | **PyTorch Geometric** |
| **NLP** | NLTK, spaCy, sentence-transformers |
| **Anomaly Detection** | scikit-learn IsolationForest, **custom Autoencoder** |
| **Explainability** | **SHAP**, **LIME** |
| **LLM** | **OpenAI GPT-4o-mini** (few-shot) |
| **AutoML** | **TPOT** or **Auto-Sklearn** |
| **Email Parsing** | Python `email`, `mailparser` |
| **Threat Intel** | **AbuseIPDB API**, **VirusTotal API**, **OpenPhish** |
| **API Framework** | FastAPI + Uvicorn |
| **Database** | SQLite (dev) / PostgreSQL (prod) |
| **Frontend** | HTML, CSS, JavaScript |
| **Visualization** | Chart.js, **D3.js** (graph viz) |
| **Containerization** | Docker + Docker Compose |

---

## 📦 Updated requirements.txt

```
# Core ML
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.1.0
catboost>=1.2.0
imbalanced-learn>=0.11.0
numpy>=1.24.0
pandas>=2.0.0

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
torch-geometric>=2.4.0

# NLP / Transformers
transformers>=4.35.0
sentence-transformers>=2.2.2
nltk>=3.8.1
spacy>=3.6.0
textstat>=0.7.3
vaderSentiment>=3.3.2

# Email Parsing
mailparser>=3.15.0
dnspython>=2.4.2
dkimpy>=1.1.4

# Explainability
shap>=0.43.0
lime>=0.2.0.1

# API
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.4.0

# Threat Intel / URL Analysis
requests>=2.31.0
python-whois>=0.8.0
editdistance>=0.6.3
geoip2>=4.7.0
openai>=1.3.0

# Utilities
python-dotenv>=1.0.0
pyyaml>=6.0.1
reportlab>=4.0.0    # PDF generation
tqdm>=4.66.0

# AutoML (Optional)
tpot>=0.12.0

# Testing
pytest>=7.4.0
httpx>=0.25.0
```

---

## 🗓️ Updated Development Timeline (10 Weeks)

| Week | Phase | Deliverable |
|---|---|---|
| Week 1 | Data Collection | Labeled dataset + augmented BEC samples |
| Week 2 | Feature Engineering | All 8 feature modules implemented |
| Week 3 | Baseline + Boosting Models | LR, XGBoost, LightGBM, CatBoost |
| Week 4 | Deep Learning Models | BERT, RoBERTa, Bi-LSTM, TextCNN |
| Week 5 | Anomaly + GNN Models | Isolation Forest, Autoencoder, GNN |
| Week 6 | Ensemble + Explainability | SHAP/LIME integrated, final ensemble |
| Week 7 | API Development | FastAPI with all endpoints |
| Week 8 | Active Learning | Feedback loop + model update pipeline |
| Week 9 | Dashboard + Alerts | Web UI, threat map, PDF reports |
| Week 10 | Testing + Documentation | Full test suite + final report + demo |

---

## 📊 BEC Attack Types Covered (Extended)

| BEC Type | Primary Detection Signal | Secondary Signal |
|---|---|---|
| CEO Fraud | Executive impersonation + wire request | Domain lookalike |
| Invoice Manipulation | Vendor domain mismatch + invoice keywords | New bank account |
| Account Compromise | Behavioral anomaly + GNN graph change | Geo-IP anomaly |
| Attorney Impersonation | Legal urgency + authority language | Free email provider |
| Data Theft | Credential request + attachment | Secrecy request |
| **🆕 Thread Hijacking** | Thread context drift + new participant | Style change |
| **🆕 AI-Generated BEC** | Low perplexity score + perfect grammar | No typos |
| **🆕 Supply Chain BEC** | Vendor email compromise + payment change | Domain age |
| **🆕 Payroll Diversion** | HR impersonation + bank detail change | Weekend send time |

---

> [!IMPORTANT]
> **Priority**: Recall > Precision for all models. Missing a BEC is far more costly than a false alarm.

> [!TIP]
> For college project scope: implement **Tier 1 + Tier 2 + DistilBERT + IsolationForest + SHAP**. The GNN and GPT modules can be presented as "future work" or bonus features.

> [!NOTE]
> **SHAP explainability** is what separates a research project from a production-ready system. Prioritize it — examiners love interpretable AI.

> [!WARNING]
> VirusTotal and OpenAI APIs have rate limits. Cache all API responses in a local SQLite database to avoid hitting limits during demos.
