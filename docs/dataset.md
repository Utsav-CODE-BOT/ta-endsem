# VoiceMiner Dataset Strategy & Specifications

VoiceMiner implements a multi-source dataset strategy utilizing verified open research datasets and explicit cross-domain scenarios. In accordance with Section 7 and Section 49, no datasets, timestamps, or ground-truth annotations are fabricated.

---

## 1. Primary Benchmark Dataset: US Airline Twitter Customer Feedback

- **Source:** Crowdflower / Figure-Eight Open Research Benchmark (hosted on Hugging Face as `osanseviero/twitter-airline-sentiment`).
- **Domain:** Airline travel, flight operations, customer service, and ticketing.
- **Total Records:** 14,640 records (persisted in `data/raw/airline_customer_feedback.csv`).
- **Standard Balanced Sample:** 1,000 records (persisted in `data/samples/airline_sample_1000.csv`).
- **Fields:**
  - `tweet_id`: Unique record identifier
  - `text`: Raw customer feedback statement
  - `tweet_created`: Real timestamp formatted as `YYYY-MM-DD HH:MM:SS TZ`
  - `airline`: Source entity (Virgin America, United, US Airways, American, Delta, Southwest)
  - `airline_sentiment`: Annotated ground-truth sentiment (`negative`, `neutral`, `positive`)
  - `airline_sentiment_confidence`: Annotator agreement confidence
  - `negativereason`: Ground-truth issue attribution (`Late Flight`, `Customer Service Issue`, `Lost Luggage`, `Bad Flight`, `Cancelled Flight`, `Flight Attendant Complaints`, etc.)
- **License:** Open Academic / Research Public Domain.

---

## 2. Cross-Domain SaaS & E-Commerce Scenario (RQ5)

- **Identifier:** `data/samples/ecommerce_saas_feedback_demo.csv`
- **Explicit Label:** **DEMO / SYNTHETIC DATA** (Tagged in accordance with Section 49).
- **Domain:** Digital mobile apps, SaaS subscription platforms, payment gateways, and logistics.
- **Records:** 500 records spanning a 4-week window.
- **Injected Temporal Dynamic:** A sudden +310% surge in payment checkout failures coinciding with the deployment of "Mobile App v4.2" and "Payment Gateway 502" webhooks in week 3.
- **Purpose:** Used strictly to test cross-domain robustness (RQ5) and demonstrate multi-source ingestion. Never presented as real experimental benchmark data.

---

## 3. Data Ingestion Pipeline & Quality Audit

The ingestion module performs:
1. Format validation (CSV, JSON, JSONL, XLSX).
2. Automatic header normalization to lowercase snake_case.
3. Text column resolution across `text`, `comment`, `review`, `content`.
4. Timestamp parsing with UTC normalization and timezone stripping.
5. Exact text deduplication to eliminate automated bot retweets.
6. Validation metrics: character count, word count, non-empty text, and language detection.
