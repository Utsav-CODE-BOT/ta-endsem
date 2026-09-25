# VoiceMiner REST API Reference

The VoiceMiner FastAPI backend exposes clean, fully typed endpoints documented with OpenAPI / Swagger UI at `http://localhost:8000/docs`.

---

## Ingestion & Pipeline Management

### `POST /api/data/upload`
Uploads a custom customer text dataset (CSV, JSON, XLSX).
- **Body:** `multipart/form-data` with `file`.
- **Response:** `IngestionStats` (uploaded records, valid records, duplicates removed, missing timestamps, sources).

### `POST /api/analyze`
Triggers the full 10-stage VoiceMiner pipeline on a pre-existing dataset.
- **Request Body:**
  ```json
  {
    "dataset_name": "airline_sample_1000.csv",
    "n_clusters": 6,
    "similarity_threshold": 0.35
  }
  ```
- **Response:** Summary of pipeline execution, discovered clusters, and hypotheses.

---

## Intelligence & Discovery Endpoints

### `GET /api/dashboard`
Returns high-level KPI metrics, sentiment distribution, primary emotions, and source breakdown.

### `GET /api/issues`
Lists all discovered semantic issue clusters with 2D projection coordinates, volume, percentage, and c-TF-IDF terms.

### `GET /api/issues/{id}`
Returns detailed profile for a specific issue cluster, including central exemplar feedback statements.

### `GET /api/emerging-issues`
Returns prioritized list of emerging issues ranked by `EmergingScore`.
- **Response Schema:**
  ```json
  [
    {
      "issue_id": 0,
      "issue_name": "Flight Delays & Cancellations",
      "emerging_score": 0.842,
      "priority": "CRITICAL",
      "growth_rate_pct": 280.5,
      "total_volume": 340,
      "recent_volume": 140,
      "negative_sentiment_ratio": 0.88,
      "cluster_coherence": 0.78,
      "component_breakdown": { ... }
    }
  ]
  ```

---

## Root-Cause & Evidence Endpoints

### `GET /api/issues/{id}/root-causes`
Returns candidate root-cause hypotheses for an issue, including Lift, PMI, and calibrated confidence.

### `GET /api/issues/{id}/evidence`
Retrieves grounded customer feedback partitioned into **supporting**, **contradictory**, and **neutral** statements.

### `GET /api/graph`
Returns NetworkX multi-relational graph data linking issues, aspects, entities, and candidate causes for network visualization.

---

## Real-Time Testing & Reporting

### `POST /api/text/analyze`
Executes real-time inference on arbitrary customer text.
- **Request Body:**
  ```json
  {
    "text": "Payment failed three times at checkout right after updating the app!"
  }
  ```
- **Response:** Sentiment, emotion, aspect terms, matched cluster, candidate factor, and nearest customer statements.

### `POST /api/reports/generate`
Generates an executive, evidence-grounded report with recommendations and limitations.

### `GET /api/evaluation/metrics`
Returns empirical benchmark metrics (RQ1 clustering comparisons, sentiment evaluation, and ablation results).
