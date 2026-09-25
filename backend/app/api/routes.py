import os
import json
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.schemas.schemas import (
    AnalyzeRequest,
    DashboardSummary,
    IssueClusterItem,
    IssueDetailResponse,
    EmergingIssueItem,
    RootCauseHypothesisItem,
    EvidenceResponse,
    TextAnalyzeRequest,
    TextAnalyzeResponse,
    ReportRequest,
    ReportResponse,
    IngestionStats,
)
from backend.app.services.pipeline_service import PipelineService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/data/upload", response_model=IngestionStats)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Ingests and validates uploaded CSV/JSON customer feedback files.
    """
    service = PipelineService.get_instance()
    upload_dir = "data/raw"
    os.makedirs(upload_dir, exist_ok=True)
    temp_path = os.path.join(upload_dir, file.filename)

    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        processed_df, stats = service.ingest_file(temp_path)
        # Trigger pipeline analysis on uploaded dataset
        service.run_full_pipeline(processed_df, db=db)
        return IngestionStats(**stats)
    except Exception as e:
        logger.error("Dataset upload/processing error: %s", e)
        raise HTTPException(status_code=400, detail=f"Failed to process dataset: {str(e)}")


@router.post("/analyze")
def trigger_analysis(
    req: AnalyzeRequest,
    db: Session = Depends(get_db),
):
    """
    Triggers analysis on an existing sample dataset (e.g. airline benchmark or SaaS demo).
    """
    service = PipelineService.get_instance()
    sample_path = os.path.join("data/samples", req.dataset_name)

    if not os.path.exists(sample_path):
        sample_path = os.path.join("data/raw", req.dataset_name)
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail=f"Dataset {req.dataset_name} not found.")

    try:
        processed_df, stats = service.ingest_file(sample_path)
        result = service.run_full_pipeline(
            processed_df,
            n_clusters=req.n_clusters,
            similarity_threshold=req.similarity_threshold,
            db=db,
        )
        return {"message": "Analysis completed successfully", "details": result, "ingestion_stats": stats}
    except Exception as e:
        logger.error("Pipeline analysis error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard", response_model=DashboardSummary)
def get_dashboard():
    """
    Returns executive overview metrics, sentiment, emotion distribution, and priority counts.
    """
    service = PipelineService.get_instance()
    return DashboardSummary(**service.get_dashboard_summary())


@router.get("/issues", response_model=List[IssueClusterItem])
def list_issues():
    """
    Lists all discovered semantic issue clusters with 2D scatter coordinates and top terms.
    """
    service = PipelineService.get_instance()
    items = []
    for cl in service.current_clusters:
        items.append(IssueClusterItem(
            issue_id=cl["cluster_id"],
            name=cl["name"],
            cluster_size=cl["size"],
            percentage=cl["percentage"],
            top_terms=cl["top_terms"],
            center_2d=cl["center_2d"],
            sentiment_distribution=cl.get("sentiment_distribution", {"negative": 60.0, "neutral": 25.0, "positive": 15.0}),
            primary_aspect=cl.get("primary_aspect", "GENERAL"),
        ))
    return items


@router.get("/issues/{issue_id}", response_model=IssueDetailResponse)
def get_issue_detail(issue_id: int):
    """
    Returns detailed profile and exemplar feedback statements for an issue cluster.
    """
    service = PipelineService.get_instance()
    for cl in service.current_clusters:
        if cl["cluster_id"] == issue_id:
            # Match metrics
            matched_metrics = None
            for em in service.current_emerging:
                if em["cluster_id"] == issue_id:
                    matched_metrics = em
                    break

            return IssueDetailResponse(
                issue_id=cl["cluster_id"],
                name=cl["name"],
                description=f"Semantic cluster with primary focus on {', '.join(cl['top_terms'][:3])}",
                cluster_size=cl["size"],
                percentage=cl["percentage"],
                top_terms=cl["top_terms"],
                center_2d=cl["center_2d"],
                exemplars=cl.get("exemplars", []),
                metrics=matched_metrics,
            )

    raise HTTPException(status_code=404, detail="Issue cluster not found.")


@router.get("/emerging-issues", response_model=List[EmergingIssueItem])
def list_emerging_issues():
    """
    Returns prioritized list of emerging issues ranked by EmergingScore.
    """
    service = PipelineService.get_instance()
    return [EmergingIssueItem(**item) for item in service.current_emerging]


@router.get("/issues/{issue_id}/root-causes", response_model=List[RootCauseHypothesisItem])
def get_issue_root_causes(issue_id: int):
    """
    Returns candidate root causes, Lift, PMI, and confidence for a given issue.
    """
    service = PipelineService.get_instance()
    results = []
    idx = 1
    for h in service.current_hypotheses:
        if h["cluster_id"] == issue_id:
            results.append(RootCauseHypothesisItem(
                id=idx,
                issue_id=h["cluster_id"],
                candidate_factor=h["candidate_factor"],
                factor_type=h.get("factor_type", "Associated Factor"),
                confidence=h["confidence"],
                lift=h["lift"],
                pmi=h["pmi"],
                temporal_relationship=h["temporal_relationship"],
                co_occurrence_count=h["co_occurrence_count"],
                total_mention_count=h["total_mention_count"],
                interpretation=h["interpretation"],
            ))
            idx += 1
    return results


@router.get("/issues/{issue_id}/evidence", response_model=EvidenceResponse)
def get_issue_evidence(
    issue_id: int,
    query: Optional[str] = None,
):
    """
    Retrieves supporting, contradictory, and neutral customer feedback evidence for an issue.
    """
    service = PipelineService.get_instance()
    if not service.is_ready or service.current_df is None:
        raise HTTPException(status_code=400, detail="Pipeline data not initialized.")

    # Find issue name
    issue_name = f"Issue Cluster {issue_id}"
    for cl in service.current_clusters:
        if cl["cluster_id"] == issue_id:
            issue_name = cl["name"]
            break

    search_query = query if query else f"{issue_name} problem complaint failure"

    evidence = service.evidence_retriever.retrieve_evidence(
        query=search_query,
        texts=service.current_df["clean_text"].tolist(),
        embeddings=service.current_embeddings,
        sentiments=service.current_df["sentiment"].tolist(),
        dates=service.current_df["timestamp"].dt.strftime("%Y-%m-%d").tolist(),
        sources=service.current_df["source"].tolist(),
    )

    return EvidenceResponse(**evidence)


@router.get("/graph")
def get_relationship_graph():
    """
    Returns the NetworkX multi-relational graph linking issues, aspects, and candidate factors.
    """
    service = PipelineService.get_instance()
    return service.current_graph


@router.post("/text/analyze", response_model=TextAnalyzeResponse)
def analyze_customer_text(req: TextAnalyzeRequest):
    """
    Real-time customer feedback text analyzer.
    """
    service = PipelineService.get_instance()
    res = service.analyze_realtime_text(req.text)
    return TextAnalyzeResponse(**res)


@router.get("/analytics/trends")
def get_analytics_trends():
    """
    Returns temporal issue volume and negative sentiment trajectories.
    """
    service = PipelineService.get_instance()
    if not service.is_ready or service.current_df is None:
        return []

    df = service.current_df
    df_copy = df.copy()
    df_copy["date"] = df_copy["timestamp"].dt.strftime("%Y-%m-%d")

    grouped = df_copy.groupby("date").agg(
        total_volume=("id", "count"),
        negative_volume=("sentiment", lambda s: (s == "negative").sum()),
        positive_volume=("sentiment", lambda s: (s == "positive").sum()),
        neutral_volume=("sentiment", lambda s: (s == "neutral").sum()),
    ).reset_index().sort_values("date")

    return grouped.to_dict(orient="records")


@router.get("/analytics/sentiment")
def get_analytics_sentiment_breakdown():
    """
    Returns sentiment and aspect matrix breakdowns.
    """
    service = PipelineService.get_instance()
    if not service.is_ready or service.current_df is None:
        return {}

    df = service.current_df
    by_source = df.groupby(["source", "sentiment"]).size().unstack(fill_value=0).to_dict(orient="index")
    by_emotion = df["emotion"].value_counts().to_dict()

    return {
        "by_source": by_source,
        "by_emotion": by_emotion,
    }


@router.post("/reports/generate", response_model=ReportResponse)
def generate_report(req: ReportRequest):
    """
    Generates an executive, evidence-grounded report with recommendations and limitations.
    """
    service = PipelineService.get_instance()
    rep = service.generate_report(include_raw_evidence=req.include_raw_evidence)
    return ReportResponse(**rep)


@router.get("/evaluation/metrics")
def get_evaluation_metrics():
    """
    Returns real benchmark evaluation numbers and ablation comparisons.
    """
    clustering_file = "experiments/clustering_benchmark.json"
    sentiment_file = "experiments/sentiment_evaluation.json"
    ablation_file = "experiments/ablation/ablation_results.json"

    clustering_data = {}
    if os.path.exists(clustering_file):
        with open(clustering_file, "r") as f:
            clustering_data = json.load(f)

    sentiment_data = {}
    if os.path.exists(sentiment_file):
        with open(sentiment_file, "r") as f:
            sentiment_data = json.load(f)

    ablation_data = {}
    if os.path.exists(ablation_file):
        with open(ablation_file, "r") as f:
            ablation_data = json.load(f)

    return {
        "clustering_benchmark": clustering_data,
        "sentiment_evaluation": sentiment_data,
        "ablation_study": ablation_data,
    }
