import pytest
import pandas as pd
from ml.rootcause.candidate_miner import CandidateRootCauseMiner, benjamini_hochberg


def test_benjamini_hochberg():
    p_vals = [0.001, 0.01, 0.04, 0.15, 0.80]
    adj, sig = benjamini_hochberg(p_vals, alpha=0.05)
    assert len(adj) == 5
    assert len(sig) == 5
    # The smallest p-value 0.001 must be statistically significant
    assert sig[0] is True
    # The largest p-value 0.80 must not be significant
    assert sig[4] is False
    # Adjusted p-values must be >= raw p-values
    for raw, a in zip(p_vals, adj):
        assert a >= raw - 1e-6


def test_npmi_and_fisher_exact():
    miner = CandidateRootCauseMiner(min_support=2, alpha_significance=0.05)
    # 20 documents: 8 in cluster 0, 12 in cluster 1
    # Factor "GATEWAY_TIMEOUT" appears 7 times in cluster 0 and 0 times in cluster 1 (strong statistical association)
    df = pd.DataFrame({"id": range(20)})
    cluster_labels = [0]*8 + [1]*12
    aspect_lists = []
    for i in range(20):
        if i < 7:
            aspect_lists.append([{"aspect": "GATEWAY_TIMEOUT"}])
        else:
            aspect_lists.append([{"aspect": "GENERAL"}])

    t0 = pd.Timestamp("2026-03-01 10:00:00", tz="UTC")
    timestamps = [t0 + pd.Timedelta(hours=i) for i in range(20)]

    hyps = miner.mine_candidates_for_cluster(
        cluster_id=0,
        cluster_name="Payment Gateway Outage",
        df=df,
        cluster_labels=cluster_labels,
        aspect_lists=aspect_lists,
        timestamps=timestamps,
    )

    assert len(hyps) > 0
    top = hyps[0]
    assert top["candidate_factor"] == "GATEWAY_TIMEOUT"
    assert top["lift"] > 1.5
    # NPMI must be in [-1, +1]
    assert -1.0 <= top["npmi"] <= 1.0
    # Strong association must have low Fisher p-value
    assert top["fisher_p_value"] < 0.01
    assert top["is_statistically_significant"] is True
    # Factor events occurred at indices 0-6 (early in cluster), so precedence must indicate precedes/triggers
    assert "Precedes" in top["temporal_relationship"] or "Coincides" in top["temporal_relationship"]
