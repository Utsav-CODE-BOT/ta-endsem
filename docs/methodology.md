# VoiceMiner Mathematical & Scientific Methodology

## 1. Problem Formulation

Conventional text analytics in customer service typically stops at sentiment classification (Positive / Neutral / Negative) or supervised topic tagging. These approaches fail to address operational reality:
1. Emerging issues emerge unpredictably and lack pre-existing labels.
2. Volume alone is a deceptive indicator: a historically frequent inquiry can overshadow a rapidly surging critical outage.
3. Automated systems routinely hallucinate causal claims without verifying statistical co-occurrence or counter-evidence.

VoiceMiner addresses the core research question:
> **How can semantic and temporal analysis of heterogeneous customer-generated text be used to detect emerging customer issues and discover evidence-grounded candidate root causes?**

---

## 2. Text Representation & Aspect Normalization

### 2.1 Dense Semantic Embeddings
Customer feedback is projected into a dense continuous space using `SentenceTransformer("all-MiniLM-L6-v2")`:
$$e_i = f_\theta(x_i) \in \mathbb{R}^{384}, \quad \|e_i\|_2 = 1$$
Embeddings are normalized such that inner products compute exact cosine similarity:
$$\text{Sim}(x_i, x_j) = \langle e_i, e_j \rangle$$

### 2.2 Semantic Aspect Normalization
Candidate aspect phrases $p \in P$ (noun phrases, bigrams, domain terms) are extracted and mapped to canonical semantic aspect centroids $A_k$:
$$\text{Aspect}(p) = \arg\max_{k} \cos(e_p, e_{A_k})$$
When $\max_k \cos(e_p, e_{A_k}) < \tau_{\text{threshold}}$, novel open-domain aspects are dynamically formed.

---

## 3. Semantic Issue Discovery & c-TF-IDF Labeling

Issues are discovered via unsupervised clustering:
1. **Dimensionality Reduction**: Truncated Singular Value Decomposition (SVD) reduces the 384-dimensional dense space to 10 latent components while preserving semantic geometry.
2. **Cluster Formation**: K-Means / density-based partitioning discovers coherent customer pain-point clusters $C_1, \dots, C_K$.
### 3. Class-based TF-IDF (c-TF-IDF) & Empirical Coherence
1. **Class-based TF-IDF (c-TF-IDF)**:
   For term $t$ in cluster $c$:
   $$TF(t, c) = \frac{\text{count}(t, c)}{L_c}$$
   Where $L_c$ is the total token count of all documents belonging to cluster $c$.
   The Inverse Document Frequency across clusters is:
   $$IDF(t) = \log\left(1 + \frac{A}{T_t}\right)$$
   Where $A = \frac{1}{K} \sum_k L_k$ is the average document length across clusters, and $T_t = \sum_c \text{count}(t, c)$ is the global frequency of term $t$.
   The resulting weight is:
   $$c\text{-}TF\text{-}IDF(t, c) = TF(t, c) \times IDF(t)$$
   This produces distinctive, normalized cluster keywords free from corpus length distortion.

2. **Empirical Intra-Cluster Coherence**:
   Coherence is calculated directly from normalized member embeddings $\hat{e}_i$ and the normalized cluster centroid $\hat{C}_c$:
   $$\text{coherence}(c) = \frac{1}{|C_c|} \sum_{i \in C_c} (\hat{e}_i \cdot \hat{C}_c)$$
   If $|C_c| \le 1$, coherence is flagged as unreliably sparse (`coherence_reliable = False`).

3. **Open-Set Outlier / Unknown Detection**:
   For novel query or inference text with embedding $e_q$, if:
   $$\max_{c} \cos(e_q, \hat{C}_c) < \tau_{\text{outlier}} \quad (\tau = 0.25)$$
   The item is routed to `Cluster -1: Unknown / Emerging Outlier Issue` to prevent forced misclassification.

---

## 4. Temporal Dynamics & Anomaly Detection

To prevent calendar distortion and look-ahead bias, temporal analysis operates on **calendar-complete, zero-filled daily grids**:
- The date range $[\min(t), \max(t)]$ is fully materialized with zero counts for unobserved days across all clusters.
- Baseline metrics at day $t$ use **strictly prior observations** $x[\max(0, t-W) : t]$ with window $W=7$ days, eliminating future information leakage.

### 4.1 Prior-Only Rolling Z-Score
$$z_{t, c} = \frac{x_{t, c} - \mu_{t, \text{prior}}}{\max(\sigma_{t, \text{prior}}, 0.25 \cdot \max(1.0, \mu_{t, \text{prior}}))}$$
Guards against division by zero in quiescent periods and flags statistical volume spikes.

### 4.2 EWMA (Exponentially Weighted Moving Average)
$$S_{t, c} = \alpha x_{t, c} + (1 - \alpha) S_{t-1, c}, \quad \alpha = 0.3$$
Captures sudden divergence from historical baseline levels.

### 4.3 CUSUM Change-Point Detection
Calibrated strictly on baseline parameters:
$$S_{t, c}^+ = \max(0, S_{t-1, c}^+ + x_{t, c} - \mu_{\text{base}} - k)$$
Flags structural change-points when $S_{t, c}^+ > h$, recording the exact date of change-point onset.

### 4.4 Bayesian Beta-Binomial Surge Probability
Given prior $\alpha_0=1, \beta_0=1$:
$$P(\text{Surge} \mid D) = \frac{\alpha_0 + V_{\text{recent}}}{\alpha_0 + \beta_0 + V_{\text{total}}}$$
Quantifies posterior belief that an issue represents an accelerating cluster.

---

## 5. Emerging Issue Prioritization Formulation

To prevent arbitrary prioritization, VoiceMiner implements an explicit, normalized multi-factor objective:
$$\text{EmergingScore}(c) = \sum_{m} w_m \cdot f_m(c)$$
Subject to $\sum_m w_m = 1.0$:

1. **Normalized Growth Rate ($w=0.30$)**:
   $$\text{norm\_growth} = \frac{1}{1 + e^{-\text{growth\_pct} / 100}}$$
2. **Volume Ratio ($w=0.20$)**:
   $$\text{norm\_volume} = \frac{\log(1 + V_c)}{\log(1 + V_{\text{total}})}$$
3. **Recentness ($w=0.20$)**:
   $$\text{norm\_recent} = \frac{V_{c, \text{recent}}}{V_c}$$
4. **Negative Sentiment Ratio ($w=0.20$)**:
   $$\text{norm\_negative} = \frac{V_{c, \text{negative}}}{V_c}$$
5. **Cluster Coherence ($w=0.10$)**: Measured empirical intra-cluster cosine density.

---

## 6. Candidate Root-Cause Association Engine

In strict adherence to causal reasoning standards, VoiceMiner avoids unverified causal claims ("X caused Y") and frames findings as **empirical factor associations** governed by hypothesis testing.

### 6.1 Pointwise Mutual Information (PMI), NPMI & Lift
$$\text{Lift}(C, F) = \frac{P(C, F)}{P(C) \cdot P(F)}$$
$$\text{PMI}(C, F) = \log_2 \text{Lift}(C, F)$$
$$\text{NPMI}(C, F) = \frac{\text{PMI}(C, F)}{-\log_2(P(C, F))}$$
Normalized PMI bounds association strength to $[-1, +1]$, where $+1$ represents complete co-occurrence, $0$ independence, and $-1$ complete mutual exclusivity.

### 6.2 Fisher's Exact Test & Benjamini-Hochberg FDR
For each factor $F$ and cluster $C$, a $2 \times 2$ document contingency table is constructed:
$$\begin{pmatrix} n_{11} & n_{12} \\ n_{21} & n_{22} \end{pmatrix} = \begin{pmatrix} C \cap F & C \cap \neg F \\ \neg C \cap F & \neg C \cap \neg F \end{pmatrix}$$
- Exact two-sided $p$-value computed via `scipy.stats.fisher_exact`.
- Multiple testing corrected across all candidate factors using the Benjamini-Hochberg False Discovery Rate (FDR) procedure:
  $$p_{(i)} \le \frac{i}{M} \cdot \alpha \quad (\alpha = 0.05)$$
Only hypotheses maintaining adjusted $p \le 0.05$ are annotated as statistically significant.

### 6.3 Empirical Temporal Precedence
Classified by comparing the factor's earliest surge/mention date against the issue's historical inflection point:
- `PREPARES_OR_TRIGGERS_SURGE`: Factor emerges strictly before or during the issue change-point.
- `COINCIDES_WITH_SURGE`: Factor volume tracks the cluster onset simultaneously.
- `FOLLOWS_SURGE`: Factor emerges after issue volume has peaked (often a secondary symptom).

---

## 7. Evidence Retrieval & Hypothesis-Aware Stance Classification

To eliminate confirmation bias, semantic search partitions retrieved customer text into four distinct stance categories:
- **Supporting Evidence**: Feedback explicitly referencing the candidate factor and containing **negative sentiment / complaint cues** confirming that the factor is perceived as problematic.
- **Contradictory / Counter Evidence**: Feedback explicitly referencing the factor with **positive sentiment / resolution cues** indicating satisfaction or problem-free experiences.
- **Neutral Evidence**: Objective queries and informational comments mentioning the factor without strong affective stance.
- **Uncertain Evidence**: Semantically related feedback that fails lexical verification for the specific candidate factor entity.
