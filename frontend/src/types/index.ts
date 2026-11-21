export interface DashboardSummary {
  total_feedback: number;
  sentiment_distribution: {
    negative: number;
    neutral: number;
    positive: number;
  };
  primary_emotions: Record<string, number>;
  active_issues: number;
  emerging_issues_count: number;
  critical_issues_count: number;
  high_priority_count: number;
  top_aspects: Array<{ aspect: string; count: number; percentage: number }>;
  sources: Array<{ source: string; count: number; percentage: number }>;
  date_range: { start: string; end: string };
  temporal_available?: boolean;
  topic_diversity?: number;
}

export interface IssueClusterItem {
  issue_id: number;
  name: string;
  cluster_size: number;
  percentage: number;
  top_terms: string[];
  center_2d: [number, number];
  sentiment_distribution: {
    negative: number;
    neutral: number;
    positive: number;
  };
  primary_aspect: string;
  cluster_coherence?: number;
  coherence_reliable?: boolean;
}

export interface ExemplarItem {
  index: number;
  text: string;
  similarity: number;
}

export interface IssueDetailResponse {
  issue_id: number;
  name: string;
  description: string;
  cluster_size: number;
  percentage: number;
  top_terms: string[];
  center_2d: [number, number];
  exemplars: ExemplarItem[];
  metrics?: EmergingIssueItem;
}

export interface EmergingIssueItem {
  issue_id: number;
  issue_name: string;
  emerging_score: number;
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  growth_rate_pct: number;
  total_volume: number;
  recent_volume: number;
  negative_sentiment_ratio: number;
  cluster_coherence: number;
  coherence_reliable?: boolean;
  surge_probability?: number;
  confidence_level?: string;
  evidence_sufficient?: boolean;
  first_detected_date: string;
  component_breakdown: {
    norm_growth: number;
    norm_volume: number;
    norm_recentness: number;
    norm_negative_sentiment: number;
    norm_coherence: number;
  };
  trend_history?: Array<{
    date: string;
    count: number;
    z_score: number;
    ewma: number;
    negative_ratio: number;
  }>;
}

export interface RootCauseHypothesisItem {
  id: number;
  issue_id: number;
  candidate_factor: string;
  factor_type: string;
  confidence: number;
  lift: number;
  pmi: number;
  npmi?: number;
  fisher_p_value?: number;
  adjusted_p_value?: number;
  is_statistically_significant?: boolean;
  temporal_relationship: string;
  co_occurrence_count: number;
  total_mention_count: number;
  interpretation: string;
}

export interface EvidenceItem {
  index: number;
  text: string;
  similarity: number;
  sentiment: string;
  stance?: 'SUPPORTING' | 'CONTRADICTING' | 'NEUTRAL' | 'UNCERTAIN';
  factor_verified?: boolean;
  date: string;
  source: string;
}

export interface EvidenceResponse {
  query: string;
  supporting: EvidenceItem[];
  contradicting: EvidenceItem[];
  neutral: EvidenceItem[];
  uncertain?: EvidenceItem[];
  stats: {
    total_qualifying_statements: number;
    supporting_count: number;
    contradicting_count: number;
    neutral_count: number;
    uncertain_count?: number;
    evidence_balance_ratio: number;
    factor_verification_rate?: number;
  };
}

export interface TextAnalyzeResponse {
  original_text: string;
  clean_text: string;
  sentiment: { label: string; score: number };
  emotion: {
    primary_emotion: string;
    score: number;
    scores: Record<string, number>;
  };
  aspects: Array<{
    aspect: string;
    matched_term: string;
    confidence: number;
    type: string;
  }>;
  matched_issue?: {
    issue_id: number;
    name: string;
    similarity: number;
    primary_aspect: string;
  };
  candidate_cause?: {
    candidate_factor: string;
    confidence: number;
    lift: number;
    interpretation: string;
  };
  similar_evidence: EvidenceItem[];
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  size: number;
  group: string;
  metadata: Record<string, any>;
}

export interface GraphLink {
  source: string;
  target: string;
  relationship: string;
  weight: number;
  label: string;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
  total_nodes: number;
  total_edges: number;
}

export interface TrendPoint {
  date: string;
  total_volume: number;
  negative_volume: number;
  positive_volume: number;
  neutral_volume: number;
}

export interface ReportResponse {
  title: string;
  generated_at: string;
  executive_summary: string;
  emerging_issues_summary: Array<{
    issue_name: string;
    emerging_score: number;
    priority: string;
    growth_rate_pct: number;
    volume: number;
    negative_sentiment_ratio: number;
  }>;
  candidate_root_causes: Array<{
    associated_issue: string;
    candidate_factor: string;
    confidence: number;
    lift: number;
    temporal_relationship: string;
    interpretation: string;
  }>;
  evidence_audit: Record<string, any>;
  limitations: string[];
  recommended_investigation_areas: string[];
}

export interface EvaluationMetrics {
  clustering_benchmark: {
    n_samples?: number;
    baselines?: Record<string, {
      method: string;
      silhouette_score: number;
      davies_bouldin_index: number;
      calinski_harabasz_index: number;
      topic_diversity?: number;
    }>;
  };
  sentiment_evaluation: {
    accuracy?: number;
    macro_precision?: number;
    macro_recall?: number;
    macro_f1?: number;
    confusion_matrix?: Record<string, Record<string, number>>;
    classes?: string[];
  };
  ablation_study: Record<string, {
    description: string;
    top_3_prioritized_issues: string[];
    mean_emerging_score: number;
    priority_distribution: Record<string, number>;
  }>;
}
