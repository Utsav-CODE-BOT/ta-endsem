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
3. **Class-based TF-IDF (c-TF-IDF)**:
   $$W_{t, c} = \text{TF}_{t, c} \times \log\left(1 + \frac{N}{\sum_{j} \text{TF}_{t, j}}\right)$$
   Where $\text{TF}_{t, c}$ is the frequency of term $t$ in cluster $c$, and $N$ is total word count across all clusters. This produces distinctive, human-interpretable issue labels.

---

## 4. Temporal Dynamics & Anomaly Detection

For each cluster $c$ across time windows $t = 1, \dots, T$:

### 4.1 Rolling Z-Score
$$z_{t, c} = \frac{x_{t, c} - \mu_{t-w, c}}{\sigma_{t-w, c} + \epsilon}$$
Identifies statistical volume spikes exceeding $2.0\sigma$.

### 4.2 EWMA (Exponentially Weighted Moving Average)
$$S_{t, c} = \alpha x_{t, c} + (1 - \alpha) S_{t-1, c}, \quad \alpha = 0.3$$
Captures sudden divergence from historical baseline levels.

### 4.3 CUSUM Change-Point Detection
$$S_{t, c}^+ = \max(0, S_{t-1, c}^+ + x_{t, c} - \mu_c - k)$$
Flags structural change-points when $S_{t, c}^+ > h$.

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
5. **Cluster Coherence ($w=0.10$)**: Intra-cluster cosine density.

---

## 6. Candidate Root-Cause Association Engine

In strict adherence to causal reasoning standards, VoiceMiner avoids unverified causal claims ("X caused Y") and frames findings as **empirical factor associations**.

### 6.1 Pointwise Mutual Information (PMI) & Lift
$$\text{Lift}(C, F) = \frac{P(C, F)}{P(C) \cdot P(F)}$$
$$\text{PMI}(C, F) = \log_2 \text{Lift}(C, F)$$
Where $C$ is the issue cluster and $F$ is a co-occurring candidate factor (entity, app update, weather event, airport gate).

### 6.2 Temporal Precedence Alignment
Evaluates whether factor mentions appear temporally before or coincide with the acceleration inflection point.

---

## 7. Evidence Retrieval & Counter-Evidence Partitioning

To prevent confirmation bias, semantic search partitions retrieved customer text into:
- **Supporting Evidence**: Semantically related feedback with **negative sentiment** confirming operational failure.
- **Contradictory / Counter Evidence**: Semantically related feedback with **positive sentiment** indicating normal, smooth customer experience despite the factor's presence.
- **Neutral Evidence**: Objective queries and informational comments.
