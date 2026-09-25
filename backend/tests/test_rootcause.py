import pytest
import pandas as pd
from ml.rootcause.candidate_miner import CandidateRootCauseMiner
from ml.rootcause.relationship_graph import RelationshipGraphBuilder


def test_candidate_root_cause_mining():
    miner = CandidateRootCauseMiner(min_support=1)
    df = pd.DataFrame({"id": range(10)})
    cluster_labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    aspect_lists = [
        [{"aspect": "PAYMENT"}],
        [{"aspect": "PAYMENT"}],
        [{"aspect": "PAYMENT"}],
        [{"aspect": "PAYMENT"}],
        [{"aspect": "BAGGAGE"}],
        [{"aspect": "BAGGAGE"}],
        [{"aspect": "BAGGAGE"}],
        [{"aspect": "BAGGAGE"}],
        [{"aspect": "BAGGAGE"}],
        [{"aspect": "BAGGAGE"}],
    ]

    hyps = miner.mine_candidates_for_cluster(
        cluster_id=0,
        cluster_name="Payment Disruptions",
        df=df,
        cluster_labels=cluster_labels,
        aspect_lists=aspect_lists,
    )

    assert len(hyps) > 0
    top = hyps[0]
    assert top["candidate_factor"] == "PAYMENT"
    assert top["lift"] > 1.0
    assert "does not establish causality" in top["interpretation"]


def test_relationship_graph():
    builder = RelationshipGraphBuilder()
    clusters = [{"cluster_id": 0, "name": "Checkout Issues", "size": 100, "percentage": 50.0}]
    hypotheses = [{"cluster_id": 0, "candidate_factor": "Payment Gateway v4.2", "confidence": 0.85, "lift": 2.5}]
    graph = builder.build_graph(clusters, hypotheses)
    assert graph["total_nodes"] >= 2
    assert graph["total_edges"] >= 1
