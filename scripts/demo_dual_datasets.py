import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

print("=======================================================")
print("VOICEMINER LIVE DUAL-DATASET END-TO-END APPLICATION DEMO")
print("=======================================================")

print("\n>>> [1/2] PROCESSING BENCHMARK DATASET: US AIRLINE FEEDBACK <<<")
res_air = client.post("/api/analyze", json={"dataset_name": "airline_sample_1000.csv", "n_clusters": 5})
print("1. Analysis Trigger:", res_air.status_code, res_air.json()["message"])
stats_air = res_air.json()["ingestion_stats"]
print(f"   Ingestion Stats: {stats_air['valid_records']} valid records, {stats_air['missing_timestamps']} missing timestamps, temporal_available={stats_air['temporal_available']}")

dash_air = client.get("/api/dashboard").json()
print(f"2. Executive Dashboard: Total Feedback={dash_air['total_feedback']}, Temporal Available={dash_air['temporal_available']}, Active Issues={dash_air['active_issues']}")
print(f"   Top Aspects: {[a['aspect'] for a in dash_air['top_aspects'][:3]]}")
print(f"   Sentiment Dist: Negative {dash_air['sentiment_distribution']['negative']}%, Neutral {dash_air['sentiment_distribution']['neutral']}%, Positive {dash_air['sentiment_distribution']['positive']}%")

issues_air = client.get("/api/issues").json()
print(f"3. Semantic Issues Discovered: {len(issues_air)} clusters")
for cl in issues_air:
    print(f"   - Issue {cl['issue_id']}: '{cl['name']}' ({cl['cluster_size']} records, {cl['percentage']}%) | Primary Aspect: {cl['primary_aspect']}")

em_air = client.get("/api/emerging-issues").json()
print(f"4. Emerging Issues Ranked by EmergingScore: {len(em_air)} issues")
for em in em_air:
    print(f"   - Issue {em['issue_id']}: '{em['issue_name']}' | Score: {em['emerging_score']} | Priority: {em['priority']} | Surge Prob: {em.get('surge_probability')}")

top_id_air = em_air[0]["issue_id"]
rc_air = client.get(f"/api/issues/{top_id_air}/root-causes").json()
print(f"5. Evidence-Grounded Root Causes for Issue {top_id_air} ({len(rc_air)} found):")
for rc in rc_air[:3]:
    print(f"   - Candidate Factor: '{rc['candidate_factor']}' | Type: {rc['factor_type']} | Lift: {rc['lift']} | Conf: {rc['confidence']} | FDR Sig: {rc.get('is_statistically_significant')}")

ev_air = client.get(f"/api/issues/{top_id_air}/evidence").json()
print(f"6. Evidence Audit for Issue {top_id_air}: Supporting={len(ev_air['supporting'])}, Contradicting={len(ev_air['contradicting'])}, Uncertain={len(ev_air.get('uncertain', []))}")

rt_air = client.post("/api/text/analyze", json={"text": "My flight was delayed 4 hours and bags were lost at the airport."}).json()
print("7. Real-Time Customer Text Matching:")
print(f"   - Input: 'My flight was delayed 4 hours and bags were lost at the airport.'")
print(f"   - Matched Issue: '{rt_air['matched_issue']['name']}' (Similarity: {rt_air['matched_issue']['similarity']})")
print(f"   - Sentiment: {rt_air['sentiment']['label']} ({rt_air['sentiment']['score']}) | Emotion: {rt_air['emotion']['primary_emotion']}")
print(f"   - Aspects Detected: {[a['aspect'] for a in rt_air['aspects']]}")


print("\n>>> [2/2] PROCESSING DOMAIN TRANSFER DATASET: ECOMMERCE SAAS SUPPORT <<<")
res_saas = client.post("/api/analyze", json={"dataset_name": "ecommerce_saas_feedback_demo.csv", "n_clusters": 4})
print("1. Analysis Trigger:", res_saas.status_code, res_saas.json()["message"])
stats_saas = res_saas.json()["ingestion_stats"]
print(f"   Ingestion Stats: {stats_saas['valid_records']} valid records, {stats_saas['missing_timestamps']} missing timestamps, temporal_available={stats_saas['temporal_available']}")

dash_saas = client.get("/api/dashboard").json()
print(f"2. Executive Dashboard: Total Feedback={dash_saas['total_feedback']}, Temporal Available={dash_saas['temporal_available']}, Active Issues={dash_saas['active_issues']}")
print(f"   Sentiment Dist: Negative {dash_saas['sentiment_distribution']['negative']}%, Neutral {dash_saas['sentiment_distribution']['neutral']}%, Positive {dash_saas['sentiment_distribution']['positive']}%")

issues_saas = client.get("/api/issues").json()
print(f"3. Semantic Issues Discovered: {len(issues_saas)} clusters")
for cl in issues_saas:
    print(f"   - Issue {cl['issue_id']}: '{cl['name']}' ({cl['cluster_size']} records, {cl['percentage']}%) | Primary Aspect: {cl['primary_aspect']}")

em_saas = client.get("/api/emerging-issues").json()
print(f"4. Emerging Issues Ranked by EmergingScore: {len(em_saas)} issues")
for em in em_saas:
    print(f"   - Issue {em['issue_id']}: '{em['issue_name']}' | Score: {em['emerging_score']} | Priority: {em['priority']} | Surge Prob: {em.get('surge_probability')}")

top_id_saas = em_saas[0]["issue_id"]
rc_saas = client.get(f"/api/issues/{top_id_saas}/root-causes").json()
print(f"5. Evidence-Grounded Root Causes for Issue {top_id_saas} ({len(rc_saas)} found):")
for rc in rc_saas[:3]:
    print(f"   - Candidate Factor: '{rc['candidate_factor']}' | Type: {rc['factor_type']} | Lift: {rc['lift']} | Conf: {rc['confidence']} | FDR Sig: {rc.get('is_statistically_significant')}")

ev_saas = client.get(f"/api/issues/{top_id_saas}/evidence").json()
print(f"6. Evidence Audit for Issue {top_id_saas}: Supporting={len(ev_saas['supporting'])}, Contradicting={len(ev_saas['contradicting'])}, Uncertain={len(ev_saas.get('uncertain', []))}")

rt_saas = client.post("/api/text/analyze", json={"text": "Payment checkout failed with error 500 when billing customer via stripe gateway"}).json()
print("7. Real-Time Customer Text Matching:")
print(f"   - Input: 'Payment checkout failed with error 500 when billing customer via stripe gateway'")
print(f"   - Matched Issue: '{rt_saas['matched_issue']['name']}' (Similarity: {rt_saas['matched_issue']['similarity']})")
print(f"   - Sentiment: {rt_saas['sentiment']['label']} ({rt_saas['sentiment']['score']}) | Emotion: {rt_saas['emotion']['primary_emotion']}")
print(f"   - Aspects Detected: {[a['aspect'] for a in rt_saas['aspects']]}")

print("\n=======================================================")
print("DEMO SUCCESS: BOTH DOMAINS EXECUTED AND AUDITED CLEANLY")
print("=======================================================")
