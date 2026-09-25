# VoiceMiner Scientific Limitations & Future Research

In accordance with Section 3, Section 45, and academic integrity guidelines, this document explicitly outlines the technical, statistical, and operational boundaries of the VoiceMiner system.

---

## 1. Association vs. True Causation (Section 3 Guardrail)

VoiceMiner operates as an **evidence-supported candidate root-cause discovery engine**. It does not perform formal causal inference because:
1. **Unobserved Confounders**: Observational customer text cannot control for hidden external variables (e.g., severe regional weather storms causing flight cancellations across multiple carriers simultaneously).
2. **Observational Nature**: In observational text streams without randomized A/B trials or counterfactual modeling, statistical co-occurrence (high Lift/PMI) and temporal precedence indicate **correlation**, not direct causal determination.
3. **Appropriate Terminology**: Outputs must consistently state *"Candidate factor"*, *"Suspected associated cause"*, or *"Evidence-supported hypothesis"*. The system strictly refrains from asserting unverified causal claims such as *"X caused Y"*.

---

## 2. Text Representation & Aspect Normalization Limitations

- **Colloquialism & Sarcasm**: Sarcastic remarks (e.g., *"Thanks for stranding me in the blizzard for 14 hours!"*) can challenge transformer sentiment polarity when contextual irony is subtle.
- **Short Texts**: Customer tweets under 5 words lack sufficient context for granular aspect attribution.
- **Aspect Granularity**: Embedding-based aspect mapping maps candidates to nearest semantic clusters, which may occasionally merge related but distinct sub-aspects (e.g., in-flight beverage service vs seat comfort).

---

## 3. Temporal Dynamics & Sampling Bias

- **Voluntary Response Bias**: Customer text predominantly over-indexes on negative experiences. Satisfied customers rarely contact support or tweet about routine successful flights.
- **Detection Lag**: Statistical change-point detection (CUSUM, EWMA) requires a minimum threshold of observations before a surge reaches statistical significance.

---

## 4. Recommended Future Research Directions

1. **DoWhy / Causal Graphical Models**: Integrating structural causal models (SCM) with observational text to estimate Average Treatment Effects (ATE) for feature releases.
2. **Semi-Supervised Active Learning**: Enabling human-in-the-loop annotators to confirm or reject candidate root causes, continuously updating confidence weights.
3. **Multi-Lingual Transformer Models**: Extending the embedding and aspect pipeline to XLM-RoBERTa for cross-lingual global customer support text.
