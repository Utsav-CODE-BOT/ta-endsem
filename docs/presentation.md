# VoiceMiner: Customer Voice Intelligence and Evidence-Grounded Emerging Issue Discovery
**B.Tech End-Semester Project | Text Analytics**  
**Institution:** Amrita Vishwa Vidyapeetham  
**Team No.:** 21  

### Team Members:
- **Utsav D.** (BL.AI.U4AID24071)
- **Rishhav T.** (BL.AI.U4AID24070)
- **Narasimha Reddy** (BL.AI.U4AID24086)

**Direct Presentation File:** [`VoiceMiner_Presentation.pptx`](../VoiceMiner_Presentation.pptx)

---

## Slide 1: Title & Team Members
- **Title:** VoiceMiner: Customer Voice Intelligence and Evidence-Grounded Emerging Issue Discovery
- **Sub-Title:** B.Tech End-Semester Project | Text Analytics
- **Team No.:** 21
- **Members:**
  - Utsav D. — BL.AI.U4AID24071
  - Rishhav T. — BL.AI.U4AID24070
  - Narasimha Reddy — BL.AI.U4AID24086

---

## Slide 2: Introduction
### Background and Motivation
- Customer feedback originates across heterogeneous channels: social media, product reviews, and customer support tickets.
- The same operational issue or failure pattern appears across widely disparate vocabularies and paraphrases.
- A high-frequency issue is not necessarily the fastest-growing or most critical operational problem.
- Negative sentiment alone cannot diagnose which underlying operational factor or system component is involved.

> **VoiceMiner** combines unsupervised semantic issue discovery, time-based anomaly prioritization, and traceable evidence grounding into a unified end-to-end framework.

---

## Slide 3: Problem Statement
### Research Problem
To design an NLP-based system that discovers emerging issues in multi-source customer feedback, prioritizes them using temporal and sentiment signals, and identifies evidence-supported candidate factors.

### System Inputs & Outputs
- **Inputs:** Raw customer text feedback, available timestamps, source channels, and domain metadata.
- **Outputs:** Unsupervised semantic issue clusters, ranked emerging issues, candidate root-cause factors, and supporting / uncertain evidence audit trails.

---

## Slide 4: Summary of Literature
### Related Work and Adopted Components
| Study | Main Contribution | Role in VoiceMiner |
| :--- | :--- | :--- |
| **Reimers & Gurevych (2019)** | Sentence-BERT encodes dense semantic similarity. | Feedback text embeddings. |
| **Grootendorst (2022)** | c-TF-IDF extracts class-based topical keywords. | Interpretable issue-cluster labels. |
| **Page (1954)** | CUSUM detects sustained distribution shifts. | Temporal anomaly / surge change signals. |
| **Benjamini & Hochberg (1995)** | FDR adjustment for multiple hypothesis testing. | Candidate-factor statistical filtering. |

*These methods solve individual tasks; VoiceMiner integrates them into a single auditable and reproducible workflow.*

---

## Slide 5: Research Gap
### Limitations Addressed by VoiceMiner
| Common Approach | Primary Limitation | VoiceMiner Response |
| :--- | :--- | :--- |
| **Sentiment-only Monitoring** | Cannot group unknown issue types or detect domain shifts. | Unsupervised semantic clustering. |
| **Static Issue Counts** | Can miss small but rapidly increasing outages / emerging bugs. | Temporal analysis and EmergingScore. |
| **Co-occurrence Lists** | Promotes incidental associations and spurious correlations. | Fisher exact test, Lift, NPMI, and BH-FDR. |
| **Similarity-only Evidence** | Related text may not logically support or address the hypothesis. | Factor-specific stance verification & abstention. |

*Research focus: emerging issues and candidate factor associations, not unverified causal inference.*

---

## Slide 6: Project Objectives
1. **Preprocess** customer feedback while strictly preserving text, IDs, and timestamp provenance.
2. **Form issue clusters** from Sentence-BERT embeddings without relying on predefined issue taxonomies.
3. **Detect temporal increases** using rolling statistics, EWMA, and CUSUM change-point detection.
4. **Rank issues** with a multi-factor EmergingScore and sample confidence indicators.
5. **Mine statistically associated candidate factors** and retrieve hypothesis-grounded evidence.
6. **Evaluate the pipeline** through baselines, ablations (A0–A6), and an interactive web dashboard.

---

## Slide 7: Proposed Methodology
### System Architecture & Processing Flow
```
1. Sources (Airline / SaaS feedback)
   ↓
2. Preprocess (HTML clean, dates, IDs)
   ↓
3. NLP (SBERT dense embeddings + RoBERTa sentiment)
   ↓
4. Clustering (TruncatedSVD + K-Means + c-TF-IDF keywords)
   ↓
5. Trends (Rolling Z-score, EWMA, CUSUM, Bayesian surge)
   ↓
6. Ranking (Multi-factor EmergingScore)
   ↓
7. Factor Mining (Fisher exact test, Lift, NPMI, BH-FDR control)
   ↓
8. Evidence Audit (Hypothesis-conditioned stance verification)
   ↓
9. User Experience (FastAPI backend + SQLite + React TypeScript UI)
```

---

## Slide 8: Module-wise Methodology
| Module | Method Used | Result Produced |
| :--- | :--- | :--- |
| **Data Ingestion** | CSV / JSON validation, deduplication | Clean and traceable feedback records |
| **NLP Processing** | SBERT, RoBERTa, aspect extraction | Embeddings, sentiment, and aspect tags |
| **Issue Discovery** | TruncatedSVD (10d), K-Means, c-TF-IDF | Cluster assignments and topic names |
| **Temporal Detection** | Prior-only time windows, EWMA, CUSUM | Trend, velocity, and surge indicators |
| **Priority Ranking** | Weighted EmergingScore | Ranked issues and priority levels |
| **Evidence Assessment**| Fisher, BH-FDR, Lift / NPMI, stance | Candidate factors and audited quotes |

*Timestamp gaps are reported honestly; missing dates are never replaced with invented current dates.*

---

## Slide 9: EmergingScore and Evidence Logic
### Multi-Factor Scoring Formula
$$\text{Score} = 0.30 \cdot \text{Growth} + 0.20 \cdot \text{Volume} + 0.20 \cdot \text{Recentness} + 0.20 \cdot \text{Negative} + 0.10 \cdot \text{Coherence}$$

| Signal | Weight | Meaning / Definition |
| :--- | :---: | :--- |
| **Growth rate** | 30% | Rate of change in issue occurrence over baseline window. |
| **Issue volume** | 20% | Log-scaled cluster frequency relative to dataset volume. |
| **Recentness** | 20% | Proportion of cluster volume falling within recent window. |
| **Negative sentiment** | 20% | Fraction of cluster records exhibiting negative sentiment. |
| **Cluster coherence** | 10% | Semantic consistency and tightness of cluster members. |

### Factor Assessment Protocol:
$$\text{Fisher Exact Test} \longrightarrow \text{BH-FDR Correction} (\alpha=0.05) \longrightarrow \text{Association Strength} \longrightarrow \text{Stance Verification (Support / Contradict / Uncertain)}$$

---

## Slide 10: Data Description
### Datasets and Evaluation Splits
| Dataset | Records Used | Purpose |
| :--- | :---: | :--- |
| **Airline Twitter Feedback** | 500 inspected; 498 valid | Clustering benchmark and empirical A0–A6 ablations |
| **Airline Sentiment Ground-Truth** | 120 labeled texts | RoBERTa sentiment evaluation |
| **Synthetic SaaS / E-Commerce** | 500 generated entries | Cross-domain transfer and live application demo |

- **Typical Schema Fields:** Feedback text, record ID, timestamp (when available), source, sentiment label / rating, domain metadata.
- **Deterministic Sampling:** Deterministic slice of first 500 rows; 2 empty/short records filtered during preprocessing.

---

## Slide 11: Implementation Details
| Layer | Libraries / Frameworks | Responsibility |
| :--- | :--- | :--- |
| **Data Processing** | Python 3.11, pandas, NumPy | Validation, cleaning, feature alignment |
| **NLP & Clustering** | Transformers, Sentence-Transformers, scikit-learn | Inference and semantic issue discovery |
| **Statistical Analysis** | SciPy, NetworkX | Trend scoring, FDR, association graph |
| **Backend & Storage** | FastAPI, SQLAlchemy, SQLite | REST API endpoints, snapshot isolation |
| **Frontend UI** | React 19, TypeScript, Vite, Tailwind | Interactive issue, evidence, and research pages |

*Application Demo Flow:* Choose dataset $\to$ Run pipeline $\to$ Inspect issue rankings $\to$ Examine candidate factors $\to$ Audit evidence.

---

## Slide 12: Results and Analysis
### Clustering Benchmark — Airline Sample ($N = 498, K = 6$)
| Representation | Silhouette Score (↑) | Davies–Bouldin Index (↓) | Topic Diversity (↑) |
| :--- | :---: | :---: | :---: |
| **TF-IDF + K-Means** | 0.0176 | 6.3747 | 0.7500 |
| **SBERT + K-Means** | 0.0347 | 3.8406 | 0.7292 |
| **SBERT + SVD-10 + K-Means (Proposed)** | **0.0384** | **3.8153** | **0.7292** |

### Sentiment Model — 120 Labeled Airline Texts
- **Accuracy:** 83.33%
- **Macro-F1:** 0.7960
- **Negative Class Recall:** 86.49%

*Interpretation:* SBERT + SVD achieves superior silhouette (+118.2% relative) and lower DB index (-40.1% dispersion). Absolute silhouette remains modest as expected in dense semantic natural language spaces.

---

## Slide 13: Ablation Study (A0–A6)
### Empirical Comparison — Identical 498-Record Airline Sample
| Configuration | Change Tested | Key Measured Observation |
| :--- | :--- | :--- |
| **A0 — Reference** | Proposed full pipeline | Silhouette 0.0384; DB 3.8153 |
| **A1 — No temporal** | Static volume and sentiment only | Issue ranks shift by up to 2 positions |
| **A2 — No SBERT** | Lexical TF-IDF + K-Means | Silhouette 0.0176; DB 6.3747 |
| **A3 — No sentiment** | Remove negative-sentiment weight | Issue ranks shift by up to 1 position |
| **A4 — No FDR** | Raw vs adjusted candidate tests | 3 raw factors; 2 retained after FDR |
| **A5 — No stance** | Naive sentiment vs factor verification | 14 naive labels; 6 factor-matched; 9 uncertain |
| **A6 — No SVD** | Raw 384d SBERT + K-Means | Silhouette 0.0347; DB 3.8406 |

*A5 reports rule-based stance/abstention, not human-verified evidence accuracy.*

---

## Slide 14: Discussion and Limitations
### Interpretation of Results
- The combined dense representation provides genuine separation and compression benefits over lexical unigrams.
- Temporal and sentiment dynamics substantially alter issue priority rankings, preventing routine inquiries from overshadowing critical spikes.
- Evidence verification limits unsupported factor-specific claims through transparent model abstention.

### Academic Limitations
- Small evaluation sample ($N=498$); lacks independent historical incident outage labels for real-time detection validation.
- A5 evaluates rule-based stance criteria without human ground truth; statistical association does not prove causal intervention.
- A0 uses dataset sentiment labels; broader cross-domain production deployments require ongoing validation.

---

## Slide 15: References
1. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP-IJCNLP.
2. Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*. arXiv:2203.05794.
3. Page, E. S. (1954). *Continuous Inspection Schemes*. Biometrika, 41(1/2), 100–115.
4. Benjamini, Y., & Hochberg, Y. (1995). *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*. JRSS B, 57(1), 289–300.
5. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
6. VoiceMiner Source Code and Reproducible Experiments: [github.com/Utsav-CODE-BOT/ta-endsem](https://github.com/Utsav-CODE-BOT/ta-endsem)
