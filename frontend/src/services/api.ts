import {
  DashboardSummary,
  IssueClusterItem,
  IssueDetailResponse,
  EmergingIssueItem,
  RootCauseHypothesisItem,
  EvidenceResponse,
  TextAnalyzeResponse,
  GraphData,
  TrendPoint,
  ReportResponse,
  EvaluationMetrics,
} from '../types';

const API_BASE = '/api';

export const api = {
  async getDashboard(): Promise<DashboardSummary> {
    const res = await fetch(`${API_BASE}/dashboard`);
    if (!res.ok) throw new Error('Failed to load dashboard summary');
    return res.json();
  },

  async getIssues(): Promise<IssueClusterItem[]> {
    const res = await fetch(`${API_BASE}/issues`);
    if (!res.ok) throw new Error('Failed to load issue clusters');
    return res.json();
  },

  async getIssueDetail(issueId: number): Promise<IssueDetailResponse> {
    const res = await fetch(`${API_BASE}/issues/${issueId}`);
    if (!res.ok) throw new Error(`Failed to load issue ${issueId}`);
    return res.json();
  },

  async getEmergingIssues(): Promise<EmergingIssueItem[]> {
    const res = await fetch(`${API_BASE}/emerging-issues`);
    if (!res.ok) throw new Error('Failed to load emerging issues');
    return res.json();
  },

  async getRootCauses(issueId: number): Promise<RootCauseHypothesisItem[]> {
    const res = await fetch(`${API_BASE}/issues/${issueId}/root-causes`);
    if (!res.ok) throw new Error(`Failed to load root causes for issue ${issueId}`);
    return res.json();
  },

  async getEvidence(issueId: number, query?: string): Promise<EvidenceResponse> {
    const url = query
      ? `${API_BASE}/issues/${issueId}/evidence?query=${encodeURIComponent(query)}`
      : `${API_BASE}/issues/${issueId}/evidence`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to load evidence for issue ${issueId}`);
    return res.json();
  },

  async getGraph(): Promise<GraphData> {
    const res = await fetch(`${API_BASE}/graph`);
    if (!res.ok) throw new Error('Failed to load relationship graph');
    return res.json();
  },

  async analyzeText(text: string): Promise<TextAnalyzeResponse> {
    const res = await fetch(`${API_BASE}/text/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error('Failed to analyze text');
    return res.json();
  },

  async getTrends(): Promise<TrendPoint[]> {
    const res = await fetch(`${API_BASE}/analytics/trends`);
    if (!res.ok) throw new Error('Failed to load trend analytics');
    return res.json();
  },

  async getSentimentBreakdown(): Promise<any> {
    const res = await fetch(`${API_BASE}/analytics/sentiment`);
    if (!res.ok) throw new Error('Failed to load sentiment analytics');
    return res.json();
  },

  async generateReport(): Promise<ReportResponse> {
    const res = await fetch(`${API_BASE}/reports/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ include_raw_evidence: true, format: 'json' }),
    });
    if (!res.ok) throw new Error('Failed to generate report');
    return res.json();
  },

  async getEvaluationMetrics(): Promise<EvaluationMetrics> {
    const res = await fetch(`${API_BASE}/evaluation/metrics`);
    if (!res.ok) throw new Error('Failed to load evaluation metrics');
    return res.json();
  },

  async triggerAnalysis(datasetName: string): Promise<any> {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_name: datasetName, n_clusters: 6 }),
    });
    if (!res.ok) throw new Error('Failed to switch dataset');
    return res.json();
  },

  async uploadDataset(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/data/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('Failed to upload dataset');
    return res.json();
  },
};
