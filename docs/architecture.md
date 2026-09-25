# VoiceMiner System Architecture

## 1. Architectural Overview

VoiceMiner is architected as an end-to-end, evidence-grounded Customer Voice Intelligence and Emerging Root-Cause Discovery platform. Rather than serving as a superficial sentiment dashboard or conversational LLM wrapper, VoiceMiner implements an unsupervised, empirical NLP pipeline integrating:

1. **Multi-Source Data Ingestion & Quality Validation**: Ingests, validates, deduplicates, and audits customer feedback streams.
2. **Dual-Representation Text Preprocessor**: Maintains semantic fidelity for Transformer inference while creating normalized tokens for lexical baselines.
3. **Dense Semantic Embedding Engine**: Employs `all-MiniLM-L6-v2` with deterministic vector caching.
4. **Calibrated Sentiment & Emotion Enrichment**: Deploys CardiffNLP Twitter-RoBERTa for 3-class sentiment alongside anchor-grounded 7-emotion spectrum classification.
5. **Dynamic Semantic Aspect Normalizer**: Discovers and normalizes customer complaints into canonical and emerging aspects via embedding space similarity.
6. **Semantic Issue Discovery Engine**: Combines Sentence-BERT embeddings, TruncatedSVD dimensionality projection, and class-based TF-IDF (c-TF-IDF) keyword extraction.
7. **Temporal Change & Anomaly Analytics**: Computes rolling Z-scores, EWMA divergence, and CUSUM change-point statistics.
8. **Multi-Factor Emerging Issue Prioritizer**: Ranks issues by a weighted formulation of growth velocity, volume, recentness, negative sentiment, and coherence.
9. **Evidence-Supported Candidate Root-Cause Miner**: Discovers statistical factor associations via Pointwise Mutual Information (PMI), Lift, and temporal precedence.
10. **Dual Evidence Retrieval Engine**: Partitions grounded customer statements into supporting, contradicting, and neutral evidence to prevent confirmation bias and hallucination.
11. **NetworkX Multi-Relational Knowledge Graph**: Models issues, aspects, entities, and candidate causes as an interactive directed network.
12. **Enterprise React + TypeScript Dashboard**: Responsive, dark-themed UI built with Vite, Tailwind CSS, Recharts, and Lucide icons.

```
                    ┌─────────────────────────┐
                    │      DATA SOURCES       │
                    │                         │
                    │ Open Research Benchmark │
                    │ Support Tickets/Reviews │
                    │ Multi-Domain Feedback   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ DATA INGESTION &        │
                    │ VALIDATION AUDIT        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ TEXT PREPROCESSING      │
                    │                         │
                    │ NFKC Normalization      │
                    │ Entity & Whitespace     │
                    │ Dual Representation     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ NLP REPRESENTATION      │
                    │                         │
                    │ Sentence-BERT (384d)    │
                    │ RoBERTa Sentiment       │
                    │ Affective Emotion       │
                    │ Aspect Normalization    │
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
                    │ TEMPORAL CHANGE         │
                    │ & ANOMALY DETECTION     │
                    │                         │
                    │ Rolling Z-Score         │
                    │ EWMA Divergence         │
                    │ CUSUM Change-Points     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ EMERGING ISSUE SCORE    │
                    │ FORMULATION             │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ ROOT-CAUSE /            │
                    │ ASSOCIATION MINING      │
                    │ (PMI, Lift, Precedence) │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ EVIDENCE RETRIEVAL      │
                    │                         │
                    │ Supporting Statements   │
                    │ Contradictory Evidence  │
                    │ Neutral Evidence        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ FASTAPI REST BACKEND    │
                    │ & SQLITE / ORM STORE    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ ENTERPRISE REACT UI     │
                    │                         │
                    │ Overview Dashboard      │
                    │ Data Ingestion Console  │
                    │ Semantic Issue Explorer │
                    │ Emerging Issues (Flag)  │
                    │ Root Cause Audit (Flag) │
                    │ Real-Time Text Analyzer │
                    │ Longitudinal Analytics  │
                    │ Audit Report Generator  │
                    │ Empirical Evaluation RQ │
                    └─────────────────────────┘
```
