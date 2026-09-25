# VoiceMiner: Customer Voice Intelligence & Evidence-Grounded Emerging Root-Cause Discovery

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6.svg)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **VoiceMiner** is an enterprise-grade, research-oriented Natural Language Processing (NLP) platform that analyzes heterogeneous customer-generated text to discover semantic issue clusters, track temporal velocity, detect emerging issues, and mine evidence-grounded candidate root causes.

---

## 1. Problem Statement & Motivation

Traditional customer service text analytics relies heavily on superficial sentiment analysis (Positive / Neutral / Negative) or rigid supervised classification. These systems fail because:
1. **Unsupervised Emerging Issues**: Operational failures and software bugs emerge unpredictably without pre-existing categories.
2. **Deceptive Static Volume**: High-frequency routine inquiries routinely overshadow rapidly accelerating, critical outages.
3. **Unsupported Causal Claims**: Naive LLM wrappers and dashboard tools invent causal assertions without statistical association or counter-evidence.

### Central Research Question:
> **How can semantic and temporal analysis of heterogeneous customer-generated text be used to detect emerging customer issues and discover evidence-grounded candidate root causes?**

---

## 2. Core Differences from Conventional Systems

| Feature | Basic Sentiment / Complaint System | Generic LLM Wrapper | **VoiceMiner (This Project)** |
| :--- | :--- | :--- | :--- |
| **Primary Output** | Single positive/negative tag | Generated conversational text | **Semantic clusters + Temporal anomalies + Candidate causes + Evidence audit** |
| **Issue Discovery** | Fixed pre-defined categories | Prompt-dependent, uncalibrated | **Unsupervised Sentence-BERT + TruncatedSVD + c-TF-IDF keywords** |
| **Temporal Dynamics** | Static counts | None | **Rolling Z-score + EWMA + CUSUM change-point detection** |
| **Prioritization** | Raw volume | None | **Mathematical `EmergingScore` (Growth + Volume + Recentness + Sentiment + Coherence)** |
| **Root-Cause Mining** | Not supported | Hallucinated text explanations | **Empirical Pointwise Mutual Information (PMI), Lift, and temporal precedence** |
| **Evidence Grounding** | None | Cherry-picked quotes | **Dual partitioning: Supporting evidence vs Contradictory counter-evidence** |
| **Causal Discipline** | Conflates correlation & cause | Confidently claims causation | **Strict scientific non-causal attribution (candidate factors)** |

---

## 3. High-Level Architecture

```
                    ┌─────────────────────────┐
                    │      DATA SOURCES       │
                    │  Reviews, Support, Apps │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ DATA INGESTION & AUDIT  │
                    │ CSV/JSON, Deduplication │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ DUAL-TEXT PREPROCESSING │
                    │ Clean & Normalized Text │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ NLP REPRESENTATION      │
                    │ Sentence-BERT (384d)    │
                    │ RoBERTa Sentiment       │
                    │ Anchor Emotion Spectrum │
                    │ Semantic Aspect Mapping │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ SEMANTIC ISSUE          │
                    │ CLUSTERING & c-TF-IDF   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ TEMPORAL DYNAMICS       │
                    │ Rolling Z-Score, CUSUM  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ EMERGING ISSUE SCORE    │
                    │ Multi-Factor Objective  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ ROOT-CAUSE / FACTOR     │
                    │ MINING (Lift, PMI)      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ EVIDENCE RETRIEVAL      │
                    │ Supporting / Counter    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ REACT + TYPESCRIPT UI   │
                    │ Fast, Interactive       │
                    └─────────────────────────┘
```

---

## 4. Empirical Evaluation & Measured Results (RQ1 - RQ5)

All metrics below reflect **actual, measured experimental results** generated from the VoiceMiner evaluation suite (`python run_experiments.py`) on open research benchmark customer text.

### RQ1: Clustering Benchmark ($N=600, K=6$)
*Does transformer-based semantic representation improve customer issue clustering compared with TF-IDF?*

| Architecture | Silhouette Score (↑) | Davies-Bouldin Index (↓) | Calinski-Harabasz Index |
| :--- | :---: | :---: | :---: |
| **Baseline 1 (TF-IDF + K-Means)** | 0.0176 | 6.3747 | 4.56 |
| **Baseline 2 (SBERT Raw + K-Means)** | 0.0347 | 3.8406 | 15.32 |
| **VoiceMiner Proposed (SBERT + SVD-10d + c-TF-IDF)** | **0.0384** | **3.8153** | **15.37** |

> **Finding:** Transformer semantic representations deliver a **+118.2% relative improvement in Silhouette Score** over TF-IDF and reduce cluster dispersion by **40.1%**, grouping semantically diverse customer phrasing into cohesive issue clusters.

### RQ2: Transformer Sentiment Evaluation ($N=120$ Ground Truth Labels)

| Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Negative Class Recall |
| :---: | :---: | :---: | :---: | :---: |
| **83.33%** | **0.7890** | **0.8040** | **0.7960** | **86.49%** |

### RQ4 & RQ5: Systemic Ablation Study (Section 35)

| System Configuration | Mean Score | Top 3 Prioritized Issues | Impact Observation |
| :--- | :---: | :--- | :--- |
| **Full System** | **0.6877** | 1. Mobile App Checkin Errors<br>2. Flight Delays & Missed Connections<br>3. Lost Baggage & Carousel Handling | **Balanced**: Identifies high-growth emerging issues despite lower absolute historical volume. |
| **W/O Temporal** | **0.7548** | 1. Flight Delays & Missed Connections<br>2. Lost Baggage & Carousel Handling<br>3. Customer Support Hold Times | **Critical Failure**: Static volume dominates; completely misses the +310% surge in Mobile App Checkin Errors. |
| **W/O Sentiment** | **0.6589** | 1. Mobile App Checkin Errors<br>2. Flight Delays & Missed Connections<br>3. Lost Baggage & Carousel Handling | **Signal Dilution**: High-volume neutral inquiries compete with critical operational disruptions. |
| **W/O Evidence** | **0.6877** | 1. Mobile App Checkin Errors<br>2. Flight Delays & Missed Connections<br>3. Lost Baggage & Carousel Handling | **Hallucination Risk**: Prioritization succeeds, but candidate causes lack corroborating customer evidence. |

---

## 5. Technology Stack

- **Backend:** Python 3.11+, FastAPI, SQLAlchemy, Pydantic v2, SQLite / PostgreSQL.
- **NLP / ML:** PyTorch, Sentence-Transformers (`all-MiniLM-L6-v2`), Hugging Face Transformers (`cardiffnlp/twitter-roberta-base-sentiment-latest`), scikit-learn, NetworkX, SciPy, pandas, NumPy.
- **Frontend:** React 19, TypeScript, Vite, Tailwind CSS v4, Lucide Icons, Recharts.
- **Testing:** Pytest (19 unit & integration tests passing).

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- Git

### 1. Clone & Set Up Python Environment
```bash
git clone https://github.com/Utsav-CODE-BOT/ta-endsem.git
cd ta-endsem

# Create virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install PyTorch CPU and dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Run Tests & Empirical Experiments
```bash
# Run complete test suite (19 tests)
pytest backend/tests -v

# Run research benchmark evaluations (RQ1 - RQ5)
python run_experiments.py
```

### 3. Start Backend Server
```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation available at: `http://localhost:8000/docs`

### 4. Start Frontend Application
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
Open your browser at `http://localhost:3000`.

---

## 7. Interactive Demonstration Flow

1. **Overview Dashboard**: Inspect total feedback records, sentiment distribution, affective emotion spectrum, and dominant aspects.
2. **Data Ingestion Console**: Switch between the verified open benchmark (`airline_sample_1000.csv`) and the cross-domain SaaS scenario (`ecommerce_saas_feedback_demo.csv`), or upload custom CSV/JSON files.
3. **Issue Explorer**: Explore discovered semantic clusters, review c-TF-IDF keyword tags, and inspect central exemplar feedback statements.
4. **Emerging Issues Page**: Inspect issues prioritized by `EmergingScore` with temporal growth badges (+312%) and multi-factor component breakdowns.
5. **Root-Cause Investigation Page**:
   - Inspect candidate factors, Statistical Lift, and Pointwise Mutual Information (PMI).
   - Review calibrated confidence ratings and scientific non-causal attribution interpretations.
   - Examine the dual evidence panels: **Supporting Evidence** vs **Contradictory Counter-Evidence**.
   - Interact with the NetworkX knowledge graph mapping issues, aspects, and candidate factors.
6. **Real-Time Text Analyzer**: Paste arbitrary customer complaints to inspect real-time sentiment, emotion, aspect terms, matched cluster, and candidate factors.
7. **Executive Reports**: Generate and print evidence-grounded reports with recommended investigation areas and limitations.
8. **Research & Benchmarks**: View live tables of RQ1 clustering benchmarks, RoBERTa confusion matrix, and systemic ablation rankings.

---

## 8. License & Academic Attribution

This project is licensed under the MIT License. Developed as a research-grade Customer Voice Intelligence platform adhering to strict empirical standards and causal distinction.
