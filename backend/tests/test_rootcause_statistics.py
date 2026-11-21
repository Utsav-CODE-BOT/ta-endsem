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
    # Factor events occurred at indices 0-6 (early in cluster), so precedence must indicate observed before/during
    assert "OBSERVED_BEFORE" in top["temporal_relationship"] or "OBSERVED_DURING" in top["temporal_relationship"]


def test_significant_only_filter_and_deduplication():
    miner = CandidateRootCauseMiner(min_support=2, alpha_significance=0.05)
    # 20 documents: 10 in cluster 0, 10 in cluster 1
    # Factor A is strongly associated with cluster 0 (8 times in cluster 0, 0 in cluster 1)
    # Factor B is uniform noise (2 times in cluster 0, 2 in cluster 1) -> not statistically significant
    df = pd.DataFrame({"id": range(20)})
    cluster_labels = [0]*10 + [1]*10
    aspect_lists = []
    entity_lists = []

    for i in range(20):
        asps = []
        ents = []
        if i < 8:
            asps.append({"aspect": "FACTOR_A"})
            # Duplicate FACTOR_A in entities for the same doc to test deduplication
            ents.append("FACTOR_A")
        if i in [0, 1, 10, 11]:
            asps.append({"aspect": "FACTOR_B"})
        aspect_lists.append(asps)
        entity_lists.append(ents)

    # When significant_only is False, both could be returned
    all_hyps = miner.mine_candidates_for_cluster(
        cluster_id=0,
        cluster_name="Test Cluster",
        df=df,
        cluster_labels=cluster_labels,
        aspect_lists=aspect_lists,
        entity_lists=entity_lists,
        significant_only=False,
    )
    # FACTOR_A total_mention_count must be 8, NOT 16 (deduplication worked!)
    factor_a_hyp = next(h for h in all_hyps if h["candidate_factor"] == "FACTOR_A")
    assert factor_a_hyp["total_mention_count"] == 8
    assert factor_a_hyp["association_level"] == "STATISTICALLY_SUPPORTED"

    # When significant_only is True, non-significant factor must be excluded
    sig_hyps = miner.mine_candidates_for_cluster(
        cluster_id=0,
        cluster_name="Test Cluster",
        df=df,
        cluster_labels=cluster_labels,
        aspect_lists=aspect_lists,
        entity_lists=entity_lists,
        significant_only=True,
    )
    assert all(h["is_statistically_significant"] for h in sig_hyps)
