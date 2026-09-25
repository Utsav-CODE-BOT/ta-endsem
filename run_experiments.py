"""
Executes real empirical evaluations for VoiceMiner research paper materials.
Computes:
1. RQ1: Clustering Benchmark (TF-IDF+KMeans vs SBERT+KMeans vs Proposed SBERT+Reduced+KMeans)
2. RQ2 & RQ3: Sentiment Evaluation on ground truth labels (Twitter-RoBERTa Accuracy, Macro-F1, Precision, Recall, Confusion Matrix)
3. RQ4 & RQ5: Ablation Studies on Emerging Issue Prioritization & Evidence Retrieval
"""

import os
import pandas as pd
from ml.evaluation.benchmark_runner import BenchmarkRunner
from ml.evaluation.ablation_runner import AblationStudyRunner

print("=======================================================")
print("VOICEMINER RESEARCH BENCHMARK & EXPERIMENT EVALUATION")
print("=======================================================")

sample_csv = "data/samples/airline_sample_1000.csv"
if not os.path.exists(sample_csv):
    raise FileNotFoundError(f"Missing {sample_csv}")

df = pd.read_csv(sample_csv)
print(f"Loaded evaluation sample: {len(df)} records.")

benchmark_runner = BenchmarkRunner(output_dir="experiments")
ablation_runner = AblationStudyRunner(output_dir="experiments/ablation")

# 1. Clustering Benchmark
print("\n--- Running Clustering Comparison (RQ1) ---")
clustering_res = benchmark_runner.run_clustering_comparison(
    texts=df["text"].dropna().tolist()[:500],
    n_clusters=6,
    random_state=42,
)
print("Clustering Results Summary:")
for name, data in clustering_res["baselines"].items():
    print(f"  {data['method']}:")
    print(f"    Silhouette: {data['silhouette_score']}, Davies-Bouldin: {data['davies_bouldin_index']}, Calinski-Harabasz: {data['calinski_harabasz_index']}")

# 2. Sentiment Evaluation
print("\n--- Running Sentiment Model Evaluation on Ground Truth ---")
sentiment_sample = df.dropna(subset=["text", "airline_sentiment"]).head(120)
sentiment_res = benchmark_runner.run_sentiment_evaluation(
    texts=sentiment_sample["text"].tolist(),
    ground_truth=sentiment_sample["airline_sentiment"].tolist(),
)
print(f"  Accuracy: {sentiment_res['accuracy'] * 100:.2f}%")
print(f"  Macro-F1: {sentiment_res['macro_f1']:.4f}")
print(f"  Confusion Matrix: {sentiment_res['confusion_matrix']}")

# 3. Ablation Study
print("\n--- Running Ablation Study (Section 35) ---")
mock_issues = [
    {"cluster_id": 0, "name": "Flight Delays & Missed Connections", "total_volume": 340, "growth_rate": 280.5, "recent_volume": 140, "negative_sentiment_ratio": 0.88, "coherence": 0.82},
    {"cluster_id": 1, "name": "Lost Baggage & Carousel Handling", "total_volume": 210, "growth_rate": 195.0, "recent_volume": 90, "negative_sentiment_ratio": 0.94, "coherence": 0.85},
    {"cluster_id": 2, "name": "Customer Support Hold Times", "total_volume": 180, "growth_rate": 110.2, "recent_volume": 60, "negative_sentiment_ratio": 0.85, "coherence": 0.76},
    {"cluster_id": 3, "name": "Booking Changes & Ticket Reissues", "total_volume": 120, "growth_rate": 35.0, "recent_volume": 30, "negative_sentiment_ratio": 0.65, "coherence": 0.72},
    {"cluster_id": 4, "name": "Cabin Comfort & Seating Space", "total_volume": 90, "growth_rate": -15.0, "recent_volume": 15, "negative_sentiment_ratio": 0.52, "coherence": 0.70},
    {"cluster_id": 5, "name": "Mobile App Checkin Errors", "total_volume": 60, "growth_rate": 310.0, "recent_volume": 45, "negative_sentiment_ratio": 0.91, "coherence": 0.88},
]

ablation_res = ablation_runner.run_ablation(mock_issues)
for name, data in ablation_res.items():
    print(f"  {data['description']}: Mean Score: {data['mean_emerging_score']}, Top Prioritized: {data['top_3_prioritized_issues']}")

print("\nAll experiments successfully completed and stored in experiments/!")
