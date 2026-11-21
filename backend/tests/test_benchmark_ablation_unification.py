"""
Tests for Unified Sample Selection, Cross-Benchmark Alignment, and A0-A6 Ablation Contracts.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np

from ml.evaluation.sample_selector import DeterministicSampleSelector, EvaluationSample
from ml.evaluation.benchmark_runner import BenchmarkRunner
from ml.evaluation.ablation_runner import AblationStudyRunner


@pytest.fixture
def sample_airline_df():
    sample_csv = "data/samples/airline_sample_1000.csv"
    if not os.path.exists(sample_csv):
        pytest.skip(f"Missing {sample_csv}")
    return pd.read_csv(sample_csv)


def test_sample_selector_deterministic_alignment(sample_airline_df):
    selector = DeterministicSampleSelector(min_word_count=3)
    sample1 = selector.select_sample(sample_airline_df, sample_size=500, random_seed=42)
    sample2 = selector.select_sample(sample_airline_df, sample_size=500, random_seed=42)

    # Identical selection
    assert sample1.raw_sample_size == 500
    assert sample1.post_filtering_sample_size == 498
    assert sample1.post_filtering_sample_size == sample2.post_filtering_sample_size
    assert sample1.record_ids == sample2.record_ids
    assert sample1.clean_texts == sample2.clean_texts
    assert sample1.sentiments == sample2.sentiments

    # 1-to-1 alignment between texts and metadata
    for i in range(len(sample1.clean_texts)):
        assert len(sample1.clean_texts[i].split()) >= 3
        assert sample1.sentiments[i] in ["negative", "neutral", "positive"]
        assert sample1.record_ids[i] is not None


def test_benchmark_and_ablation_identical_record_ids(sample_airline_df, tmp_path):
    selector = DeterministicSampleSelector(min_word_count=3)
    eval_sample = selector.select_sample(sample_airline_df, sample_size=500, random_seed=42)

    bench_runner = BenchmarkRunner(output_dir=str(tmp_path / "bench"))
    ablation_runner = AblationStudyRunner(output_dir=str(tmp_path / "ablation"))

    # Run clustering comparison
    bench_res = bench_runner.run_clustering_comparison(sample=eval_sample, n_clusters=6, random_state=42)
    ablation_res = ablation_runner.run_empirical_component_ablations(sample=eval_sample, n_clusters=6, random_seed=42)

    # Cross-benchmark population identity check
    assert bench_res["record_ids"] == ablation_res["record_ids"]
    assert bench_res["post_filtering_sample_size"] == ablation_res["post_filtering_sample_size"]
    assert bench_res["input_sample_size"] == ablation_res["input_sample_size"]

    # Silhouette score consistency on identical 498 records
    proposed_bench_sil = bench_res["baselines"]["proposed_sbert_reduced_kmeans"]["silhouette_score"]
    a0_sil = ablation_res["configurations"]["A0_full_system"]["clustering_silhouette"]
    assert proposed_bench_sil == a0_sil


def test_a0_full_system_execution_contract(sample_airline_df, tmp_path):
    selector = DeterministicSampleSelector(min_word_count=3)
    eval_sample = selector.select_sample(sample_airline_df, sample_size=500, random_seed=42)

    ablation_runner = AblationStudyRunner(output_dir=str(tmp_path / "ablation"))
    res = ablation_runner.run_empirical_component_ablations(sample=eval_sample, n_clusters=6, random_seed=42)

    a0 = res["configurations"]["A0_full_system"]
    assert a0["configuration"] == "A0_Full_System"
    assert a0["sentiment_input_mode"] == "DATASET_LABEL"
    assert "components_executed" in a0
    assert "preprocessing" in a0["components_executed"]
    assert "aspect_extraction" in a0["components_executed"]
    assert "candidate_factor_mining" in a0["components_executed"]
    assert "evidence_stance_verification" in a0["components_executed"]
    assert "top_prioritized_issue" in a0
    assert a0["top_issue_score"] > 0.0


def test_a1_and_a3_rank_displacement(sample_airline_df, tmp_path):
    selector = DeterministicSampleSelector(min_word_count=3)
    eval_sample = selector.select_sample(sample_airline_df, sample_size=500, random_seed=42)

    ablation_runner = AblationStudyRunner(output_dir=str(tmp_path / "ablation"))
    res = ablation_runner.run_empirical_component_ablations(sample=eval_sample, n_clusters=6, random_seed=42)

    a1 = res["configurations"]["A1_without_temporal"]
    assert a1["max_rank_displacement"] >= 0
    assert "rank_changes_by_cluster" in a1
    assert "top_prioritized_issue" in a1

    a3 = res["configurations"]["A3_without_sentiment"]
    assert a3["max_rank_displacement"] >= 0
    assert "rank_changes_by_cluster" in a3
    assert "top_prioritized_issue" in a3


def test_a4_fdr_candidate_counts(sample_airline_df, tmp_path):
    selector = DeterministicSampleSelector(min_word_count=3)
    eval_sample = selector.select_sample(sample_airline_df, sample_size=500, random_seed=42)

    ablation_runner = AblationStudyRunner(output_dir=str(tmp_path / "ablation"))
    res = ablation_runner.run_empirical_component_ablations(sample=eval_sample, n_clusters=6, random_seed=42)

    a4 = res["configurations"]["A4_without_statistical_filtering"]
    assert a4["candidates_without_fdr"] >= a4["candidates_with_fdr"]
    assert a4["fdr_pruned_count"] == a4["candidates_without_fdr"] - a4["candidates_with_fdr"]
    assert "statistical_procedure" in a4


def test_a5_identical_retrieved_documents_and_count_reconciliation(sample_airline_df, tmp_path):
    selector = DeterministicSampleSelector(min_word_count=3)
    eval_sample = selector.select_sample(sample_airline_df, sample_size=500, random_seed=42)

    ablation_runner = AblationStudyRunner(output_dir=str(tmp_path / "ablation"))
    res = ablation_runner.run_empirical_component_ablations(sample=eval_sample, n_clusters=6, random_seed=42)

    a5 = res["configurations"]["A5_without_evidence_verification"]
    k = a5["shared_document_count"]
    assert k == 15
    assert len(a5["shared_document_ids"]) == 15

    # Mathematical reconciliation on identical documents
    naive_total = a5["naive_supporting_count"] + a5["naive_contradicting_count"] + a5["naive_neutral_count"]
    assert naive_total == 15

    verified_total = a5["verified_supporting_count"] + a5["contradicting_count"] + a5["neutral_count"] + a5["uncertain_abstention_count"]
    assert verified_total == 15

    assert a5["unverified_items_abstained_from_support"] <= a5["naive_supporting_count"]
    assert a5["classification_changes_count"] >= 0


def test_frontend_empirical_json_compatibility(tmp_path):
    out_file = "experiments/ablation/empirical_ablation_results.json"
    if not os.path.exists(out_file):
        pytest.skip("experiments/ablation/empirical_ablation_results.json not generated yet")

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "configurations" in data
    configs = data["configurations"]
    for c_key in ["A0_full_system", "A1_without_temporal", "A2_without_semantic_embeddings", "A3_without_sentiment", "A4_without_statistical_filtering", "A5_without_evidence_verification", "A6_without_dimensionality_reduction"]:
        assert c_key in configs
        assert "description" in configs[c_key]
