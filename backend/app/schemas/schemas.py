from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator


class IngestionStats(BaseModel):
    records_uploaded: int
    valid_records: int
    invalid_records: int
    duplicates_removed: int
    missing_timestamps: int
    sources_detected: List[str]
    languages_detected: List[str]
    sample_columns: List[str]
    temporal_available: Optional[bool] = True
    status: Optional[str] = "success"
    warning: Optional[str] = None


class AnalyzeRequest(BaseModel):
    dataset_name: Optional[str] = "airline_sample_1000.csv"
    n_clusters: int = 6
    similarity_threshold: float = 0.35


class DashboardSummary(BaseModel):
    total_feedback: int
    sentiment_distribution: Dict[str, float]
    primary_emotions: Dict[str, int]
    active_issues: int
    emerging_issues_count: int
    critical_issues_count: int
    high_priority_count: int
    top_aspects: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    date_range: Dict[str, str]
    temporal_available: Optional[bool] = True


class IssueClusterItem(BaseModel):
    issue_id: int
    name: str
    cluster_size: int
    percentage: float
    top_terms: List[str]
    center_2d: List[float]
    sentiment_distribution: Dict[str, float]
    primary_aspect: str
    cluster_coherence: Optional[float] = None
    coherence_reliable: Optional[bool] = None


class IssueDetailResponse(BaseModel):
    issue_id: int
    name: str
    description: str
    cluster_size: int
    percentage: float
    top_terms: List[str]
    center_2d: List[float]
    exemplars: List[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]] = None
    cluster_coherence: Optional[float] = None
    coherence_reliable: Optional[bool] = None


class EmergingIssueItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    issue_id: int
    cluster_id: Optional[int] = None
    issue_name: str
    emerging_score: float
    priority: str
    growth_rate_pct: float
    total_volume: int
    recent_volume: int
    negative_sentiment_ratio: float
    cluster_coherence: float
    first_detected_date: str
    component_breakdown: Dict[str, float]
    confidence_level: Optional[str] = "MODERATE_STATISTICAL_SUPPORT"
    evidence_sufficient: Optional[bool] = True
    surge_probability: Optional[float] = None
    coherence_reliable: Optional[bool] = None
    trend_history: Optional[List[Dict[str, Any]]] = None
    data_sufficiency: Optional[str] = None
    cusum_change_detected: Optional[bool] = None
    cusum_change_date: Optional[str] = None
    temporal_available: Optional[bool] = None
    weights_used: Optional[Dict[str, float]] = None

    @model_validator(mode="before")
    @classmethod
    def reconcile_identifiers(cls, data: Any) -> Any:
        if isinstance(data, dict):
            c_id = data.get("issue_id", data.get("cluster_id"))
            if c_id is not None:
                if "issue_id" not in data:
                    data["issue_id"] = c_id
                if "cluster_id" not in data:
                    data["cluster_id"] = c_id
        return data


class RootCauseHypothesisItem(BaseModel):
    id: Optional[int] = None
    issue_id: int
    candidate_factor: str
    factor_type: str
    confidence: float
    lift: float
    pmi: float
    temporal_relationship: str
    co_occurrence_count: int
    total_mention_count: int
    interpretation: str
    npmi: Optional[float] = 0.0
    fisher_p_value: Optional[float] = None
    adjusted_p_value: Optional[float] = None
    is_statistically_significant: Optional[bool] = False
    association_strength_score: Optional[float] = None


class EvidenceItem(BaseModel):
    index: int
    text: str
    similarity: float
    sentiment: str
    date: str
    source: str
    stance: Optional[str] = None
    factor_verified: Optional[bool] = None


class EvidenceResponse(BaseModel):
    query: str
    supporting: List[EvidenceItem]
    contradicting: List[EvidenceItem]
    neutral: List[EvidenceItem]
    uncertain: Optional[List[EvidenceItem]] = []
    candidate_factor: Optional[str] = None
    stats: Dict[str, Any]


class TextAnalyzeRequest(BaseModel):
    text: str
    similarity_threshold: Optional[float] = None


class TextAnalyzeResponse(BaseModel):
    original_text: str
    clean_text: str
    sentiment: Dict[str, Union[str, float]]
    emotion: Dict[str, Any]
    aspects: List[Dict[str, Any]]
    matched_issue: Optional[Dict[str, Any]] = None
    candidate_cause: Optional[Dict[str, Any]] = None
    similar_evidence: List[EvidenceItem] = []


class TrendDataPoint(BaseModel):
    date: str
    total_volume: int
    negative_volume: int
    top_issue: str


class ReportRequest(BaseModel):
    include_raw_evidence: bool = True
    format: str = "json"  # "json", "markdown"


class ReportResponse(BaseModel):
    title: str
    generated_at: str
    executive_summary: str
    emerging_issues_summary: List[Dict[str, Any]]
    candidate_root_causes: List[Dict[str, Any]]
    evidence_audit: Dict[str, Any]
    limitations: List[str]
    recommended_investigation_areas: List[str]
