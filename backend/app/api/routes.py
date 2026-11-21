import os
import re
import json
import uuid
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

ALLOWED_EXTENSIONS = {".csv", ".json", ".jsonl", ".xlsx", ".xls"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB limit


@router.post("/data/upload", response_model=IngestionStats)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Ingests and validates uploaded CSV/JSON customer feedback files with strict
    filename sanitization, size limits, and path-traversal prevention.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")

    # 1. Extension validation
    raw_ext = os.path.splitext(file.filename)[1].lower()
    if raw_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{raw_ext}'. Supported formats: {sorted(list(ALLOWED_EXTENSIONS))}"
        )

    # 2. Safe server-generated filename preventing directory traversal
    base_name = re.sub(r"[^a-zA-Z0-9_\.-]", "_", os.path.basename(file.filename))
    safe_filename = f"upload_{uuid.uuid4().hex[:10]}_{base_name}"
    upload_dir = os.path.abspath("data/raw")
    os.makedirs(upload_dir, exist_ok=True)

    temp_path = os.path.abspath(os.path.join(upload_dir, safe_filename))
    # Directory containment validation
    if not temp_path.startswith(upload_dir):
        raise HTTPException(status_code=400, detail="Invalid file path detected.")

    # 3. Read content with size limit enforcement
    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File size exceeds maximum allowed limit (50 MB).")

    with open(temp_path, "wb") as f:
        f.write(content)

    service = PipelineService.get_instance()
    try:
        processed_df, stats = service.ingest_file(temp_path)
        # Trigger pipeline analysis on uploaded dataset
        pipeline_res = service.run_full_pipeline(processed_df, db=db)
        if pipeline_res.get("status") == "partial_success":
            stats["status"] = "partial_success"
            stats["warning"] = f"Analysis completed in-memory with warning: {pipeline_res.get('database_error', 'Database write issue')}"
        else:
            stats["status"] = "success"
        return IngestionStats(**stats)
    except Exception as e:
        logger.error("Dataset upload/processing error: %s", e)
        # Clean up failed upload file if possible
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=f"Failed to process dataset: {str(e)}")


@router.post("/analyze")
def trigger_analysis(
    req: AnalyzeRequest,
    db: Session = Depends(get_db),
):
    """
    Triggers analysis on an existing sample dataset with user-configured n_clusters
    and similarity threshold correctly propagated.
    """
    service = PipelineService.get_instance()
    safe_name = os.path.basename(req.dataset_name)
    sample_path = os.path.join("data/samples", safe_name)

    if not os.path.exists(sample_path):
        sample_path = os.path.join("data/raw", safe_name)
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
        is_full_success = result.get("status") == "success"
        msg = (
            "Analysis completed successfully"
            if is_full_success
            else f"Analysis completed in-memory with warning: {result.get('database_error', 'Database persistence failed')}"
        )
        return {"message": msg, "details": result, "ingestion_stats": stats}
    except Exception as e:
        logger.error("Pipeline analysis error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard", response_model=DashboardSummary)
def get_dashboard():
    """
    Returns executive overview metrics, sentiment, emotion distribution, and priority counts.
    """
    service = PipelineService.get_instance()
    snap = service.get_snapshot()
    return DashboardSummary(**service.get_dashboard_summary(snapshot=snap))


@router.get("/issues", response_model=List[IssueClusterItem])
def list_issues():
    """
    Lists all discovered semantic issue clusters with 2D scatter coordinates and top terms.
    """
    service = PipelineService.get_instance()
    snap = service.get_snapshot()
    items = []
    for cl in snap.clusters:
        items.append(IssueClusterItem(
            issue_id=cl["cluster_id"],
            name=cl["name"],
            cluster_size=cl["size"],
            percentage=cl["percentage"],
            top_terms=cl["top_terms"],
            center_2d=cl["center_2d"],
            sentiment_distribution=cl.get("sentiment_distribution", {"negative": 60.0, "neutral": 25.0, "positive": 15.0}),
            primary_aspect=cl.get("primary_aspect", "GENERAL"),
            cluster_coherence=cl.get("cluster_coherence"),
            coherence_reliable=cl.get("coherence_reliable"),
        ))
    return items


@router.get("/issues/{issue_id}", response_model=IssueDetailResponse)
def get_issue_detail(issue_id: int):
    """
    Returns detailed profile and exemplar feedback statements for an issue cluster.
    """
    service = PipelineService.get_instance()
    snap = service.get_snapshot()
    for cl in snap.clusters:
        if cl["cluster_id"] == issue_id:
            matched_metrics = None
            for em in snap.emerging:
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
                cluster_coherence=cl.get("cluster_coherence"),
                coherence_reliable=cl.get("coherence_reliable"),
            )

    raise HTTPException(status_code=404, detail="Issue cluster not found.")


@router.get("/emerging-issues", response_model=List[EmergingIssueItem])
def list_emerging_issues():
    """
    Returns prioritized list of emerging issues ranked by EmergingScore.
    """
    service = PipelineService.get_instance()
    snap = service.get_snapshot()
    return [EmergingIssueItem(**item) for item in snap.emerging]


@router.get("/issues/{issue_id}/root-causes", response_model=List[RootCauseHypothesisItem])
def get_issue_root_causes(issue_id: int):
    """
    Returns candidate root causes, Lift, PMI, NPMI, and association strength for a given issue.
    """
    service = PipelineService.get_instance()
    snap = service.get_snapshot()
    results = []
    idx = 1
    for h in snap.hypotheses:
        if h["cluster_id"] == issue_id:
            results.append(RootCauseHypothesisItem(
                id=idx,
                issue_id=h["cluster_id"],
                candidate_factor=h["candidate_factor"],
                factor_type=h.get("factor_type", "Associated Factor"),
                confidence=h["confidence"],
                lift=h["lift"],
                pmi=h["pmi"],
                npmi=h.get("npmi", 0.0),
                fisher_p_value=h.get("fisher_p_value"),
                adjusted_p_value=h.get("adjusted_p_value"),
                is_statistically_significant=h.get("is_statistically_significant", False),
                association_strength_score=h.get("association_strength_score", h["confidence"]),
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
    candidate_factor: Optional[str] = None,
):
    """
    Retrieves supporting, contradictory, neutral, and uncertain evidence for an issue hypothesis.
    """
    service = PipelineService.get_instance()
    snap = service.get_snapshot()
    if not snap.is_ready or snap.df is None or snap.embeddings is None:
        raise HTTPException(status_code=400, detail="Pipeline data not initialized.")

    issue_name = f"Issue Cluster {issue_id}"
    for cl in snap.clusters:
        if cl["cluster_id"] == issue_id:
            issue_name = cl["name"]
            break

    target_factor = candidate_factor
    if not target_factor:
        for h in snap.hypotheses:
            if h["cluster_id"] == issue_id:
                target_factor = h["candidate_factor"]
                break

    search_query = query if query else (f"{target_factor} {issue_name}" if target_factor else issue_name)

    dates_list = [
        t.strftime("%Y-%m-%d") if (pd.notna(t) and hasattr(t, "strftime")) else "N/A"
        for t in snap.df["timestamp"]
    ]

    evidence = service.evidence_retriever.retrieve_evidence(
        query=search_query,
        texts=snap.df["clean_text"].tolist(),
        embeddings=snap.embeddings,
        sentiments=snap.df["sentiment"].tolist(),
        candidate_factor=target_factor,
        dates=dates_list,
        sources=snap.df["source"].tolist(),
    )

    return EvidenceResponse(**evidence)


@router.get("/graph")
def get_relationship_graph():
    """
    Returns the NetworkX multi-relational graph linking issues, aspects, and candidate factors.
    """
    service = PipelineService.get_instance()
    snap = service.get_snapshot()
    return snap.graph


@router.post("/text/analyze", response_model=TextAnalyzeResponse)
def analyze_customer_text(req: TextAnalyzeRequest):
    """
    Real-time customer feedback text analyzer.
    """
    service = PipelineService.get_instance()
    res = service.analyze_realtime_text(
        raw_text=req.text,
        similarity_threshold=req.similarity_threshold,
    )
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
    valid_mask = df["timestamp"].notna()
    if not valid_mask.any():
        return []

    df_valid = df[valid_mask].copy()
    df_valid["date"] = df_valid["timestamp"].dt.strftime("%Y-%m-%d")

    grouped = df_valid.groupby("date").agg(
        total_volume=("id", "count"),
        negative_volume=("sentiment", lambda s: int((s == "negative").sum())),
        positive_volume=("sentiment", lambda s: int((s == "positive").sum())),
        neutral_volume=("sentiment", lambda s: int((s == "neutral").sum())),
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
    Returns real benchmark evaluation numbers, empirical component ablations (A0-A6),
    and controlled scoring demonstration comparisons.
    """
    clustering_file = "experiments/clustering_benchmark.json"
    sentiment_file = "experiments/sentiment_evaluation.json"
    ablation_file = "experiments/ablation/ablation_results.json"
    empirical_file = "experiments/ablation/empirical_ablation_results.json"

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

    empirical_data = {}
    if os.path.exists(empirical_file):
        with open(empirical_file, "r") as f:
            empirical_data = json.load(f)

    return {
        "clustering_benchmark": clustering_data,
        "sentiment_evaluation": sentiment_data,
        "ablation_study": ablation_data,
        "empirical_ablations": empirical_data,
    }


@router.get("/evaluation/empirical-ablations")
def get_empirical_ablations():
    """
    Returns RQ4 & RQ5 Empirical Component Ablation results (A0 - A6).
    """
    empirical_file = "experiments/ablation/empirical_ablation_results.json"
    if os.path.exists(empirical_file):
        with open(empirical_file, "r") as f:
            return json.load(f)
    return {"configurations": {}}
