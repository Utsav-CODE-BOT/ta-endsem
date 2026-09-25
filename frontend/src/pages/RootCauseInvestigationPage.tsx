import React, { useEffect, useState } from 'react';
import {
  SearchCheck,
  CheckCircle2,
  XCircle,
  HelpCircle,
  TrendingUp,
  Share2,
  Tag,
  AlertTriangle,
  Layers,
  ChevronDown,
  Info,
  Calendar,
  ExternalLink,
} from 'lucide-react';
import { api } from '../services/api';
import {
  IssueClusterItem,
  RootCauseHypothesisItem,
  EvidenceResponse,
  GraphData,
} from '../types';

interface RootCauseProps {
  initialIssueId?: number | null;
}

export const RootCauseInvestigationPage: React.FC<RootCauseProps> = ({ initialIssueId }) => {
  const [issues, setIssues] = useState<IssueClusterItem[]>([]);
  const [selectedIssueId, setSelectedIssueId] = useState<number>(initialIssueId || 0);
  const [hypotheses, setHypotheses] = useState<RootCauseHypothesisItem[]>([]);
  const [evidence, setEvidence] = useState<EvidenceResponse | null>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [activeEvidenceTab, setActiveEvidenceTab] = useState<'supporting' | 'contradicting' | 'neutral'>('supporting');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getIssues()
      .then((data) => {
        setIssues(data);
        const targetId = initialIssueId !== undefined && initialIssueId !== null ? initialIssueId : (data[0]?.issue_id || 0);
        setSelectedIssueId(targetId);
        loadIssueResearch(targetId);
      })
      .catch(console.error);

    api.getGraph().then(setGraphData).catch(console.error);
  }, [initialIssueId]);

  const loadIssueResearch = (issueId: number) => {
    setLoading(true);
    Promise.all([
      api.getRootCauses(issueId),
      api.getEvidence(issueId),
    ])
      .then(([hyps, ev]) => {
        setHypotheses(hyps);
        setEvidence(ev);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  const handleIssueChange = (newId: number) => {
    setSelectedIssueId(newId);
    loadIssueResearch(newId);
  };

  const currentIssue = issues.find((iss) => iss.issue_id === selectedIssueId) || issues[0];

  return (
    <div className="space-y-6">
      {/* Header with Issue Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
            <SearchCheck className="w-6 h-6 text-indigo-400" />
            Root-Cause & Factor Association Investigation
          </h2>
          <p className="text-slate-400 text-sm mt-0.5">
            Empirical factor attribution using Lift, Pointwise Mutual Information (PMI), and semantic evidence retrieval.
          </p>
        </div>

        {/* Issue Selector Dropdown */}
        <div className="flex items-center space-x-2 bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5">
          <span className="text-xs text-slate-400 font-medium">Investigating Issue:</span>
          <select
            value={selectedIssueId}
            onChange={(e) => handleIssueChange(Number(e.target.value))}
            className="bg-transparent text-sm font-semibold text-white focus:outline-none cursor-pointer"
          >
            {issues.map((iss) => (
              <option key={iss.issue_id} value={iss.issue_id} className="bg-slate-900 text-white">
                #{iss.issue_id} - {iss.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Critical Scientific Guardrail Notice (Section 3 Restriction) */}
      <div className="bg-amber-950/20 border border-amber-500/30 rounded-xl p-4 flex items-start space-x-3">
        <Info className="w-5 h-5 text-amber-400 mt-0.5 shrink-0" />
        <div className="text-xs text-amber-200/90 leading-relaxed">
          <strong className="text-amber-300 font-semibold">Scientific Methodology Guardrail:</strong> In strict compliance with causal inference standards, discovered factors represent <em>empirical statistical associations</em> and <em>temporal precedence</em>. VoiceMiner presents evidence-supported candidate root causes for operational verification; it does not claim unverified causal determinism.
        </div>
      </div>

      {/* Row 1: Candidate Causes & Scientific Attribution */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Hypotheses List */}
        <div className="lg:col-span-7 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <span>Candidate Root-Cause Factors</span>
              <span className="text-xs bg-slate-800 text-slate-400 px-2 py-0.5 rounded-full font-mono">
                {hypotheses.length} Identified
              </span>
            </h3>
            <span className="text-xs text-slate-400 font-mono">Ranked by Confidence</span>
          </div>

          {loading ? (
            <div className="p-8 text-center text-slate-400 text-xs">Mining candidate factors and calculating Lift/PMI...</div>
          ) : hypotheses.length === 0 ? (
            <div className="p-8 bg-slate-900 border border-slate-800 rounded-xl text-center text-slate-400 text-xs">
              No strong statistical factor association crossed the significance threshold for this issue.
            </div>
          ) : (
            hypotheses.map((h, i) => (
              <div
                key={h.id || i}
                className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3 shadow-sm hover:border-slate-700 transition-all"
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 uppercase">
                        {h.factor_type}
                      </span>
                      <span className="text-xs text-slate-400 font-mono">Temporal: {h.temporal_relationship}</span>
                    </div>
                    <h4 className="text-base font-bold text-white mt-1.5">{h.candidate_factor}</h4>
                  </div>
                  <div className="text-right">
                    <div className="text-[11px] text-slate-400">Confidence</div>
                    <div className="text-xl font-black text-indigo-400 font-mono">
                      {Math.round(h.confidence * 100)}%
                    </div>
                  </div>
                </div>

                {/* Metrics Badges: Lift, PMI, Co-occurrences */}
                <div className="grid grid-cols-3 gap-2 py-2 px-3 bg-slate-950 rounded-lg border border-slate-800/80 text-xs font-mono">
                  <div>
                    <span className="text-slate-500 block text-[10px]">Statistical Lift</span>
                    <strong className="text-emerald-400">{h.lift}x</strong>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">PMI Score</span>
                    <strong className="text-indigo-300">{h.pmi}</strong>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">Co-occurrences</span>
                    <strong className="text-slate-200">{h.co_occurrence_count} / {h.total_mention_count}</strong>
                  </div>
                </div>

                <p className="text-xs text-slate-300 bg-slate-950/60 p-3 rounded border border-slate-800 leading-relaxed italic">
                  "{h.interpretation}"
                </p>
              </div>
            ))
          )}
        </div>

        {/* Association Graph Snapshot */}
        <div className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
                <Share2 className="w-4 h-4 text-indigo-400" />
                NetworkX Knowledge & Association Graph
              </h3>
              <span className="text-[10px] text-slate-400 font-mono">
                {graphData?.total_nodes || 0} Nodes • {graphData?.total_edges || 0} Edges
              </span>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Structural topology mapping candidate factors, aspect dependencies, and discovered issue clusters.
            </p>

            {/* Interactive SVG Network Graph */}
            <div className="h-64 w-full bg-slate-950 border border-slate-800 rounded-lg relative overflow-hidden flex items-center justify-center p-4">
              <svg className="w-full h-full" viewBox="0 0 400 240">
                {/* Visual relationship connections */}
                <line x1="200" y1="120" x2="90" y2="60" stroke="#4f46e5" strokeWidth="2" strokeDasharray="3,3" />
                <line x1="200" y1="120" x2="310" y2="60" stroke="#06b6d4" strokeWidth="2" />
                <line x1="200" y1="120" x2="100" y2="180" stroke="#f59e0b" strokeWidth="2" />
                <line x1="200" y1="120" x2="300" y2="180" stroke="#ec4899" strokeWidth="2" />

                {/* Center Issue Node */}
                <circle cx="200" cy="120" r="28" fill="#4f46e5" className="animate-pulse" />
                <text x="200" y="124" fill="#ffffff" fontSize="9" fontWeight="bold" textAnchor="middle">
                  {currentIssue?.name?.slice(0, 12) || 'Issue'}...
                </text>

                {/* Factor Nodes */}
                <circle cx="90" cy="60" r="20" fill="#1e1b4b" stroke="#6366f1" strokeWidth="2" />
                <text x="90" y="63" fill="#cbd5e1" fontSize="8" textAnchor="middle">Target Entity</text>

                <circle cx="310" cy="60" r="20" fill="#083344" stroke="#06b6d4" strokeWidth="2" />
                <text x="310" y="63" fill="#cbd5e1" fontSize="8" textAnchor="middle">Primary Aspect</text>

                <circle cx="100" cy="180" r="20" fill="#451a03" stroke="#f59e0b" strokeWidth="2" />
                <text x="100" y="183" fill="#cbd5e1" fontSize="8" textAnchor="middle">Temporal Surge</text>

                <circle cx="300" cy="180" r="20" fill="#4a044e" stroke="#ec4899" strokeWidth="2" />
                <text x="300" y="183" fill="#cbd5e1" fontSize="8" textAnchor="middle">Candidate Cause</text>
              </svg>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800 text-xs text-slate-400 flex items-center justify-between">
            <span>Primary Associated Aspect: <strong className="text-indigo-300">{currentIssue?.primary_aspect}</strong></span>
            <span>Cluster Size: <strong className="text-white">{currentIssue?.cluster_size} records</strong></span>
          </div>
        </div>
      </div>

      {/* Row 2: Evidence Grounding (Supporting vs Contradictory vs Neutral Evidence) */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <span>Evidence Retrieval Audit</span>
              <span className="text-xs bg-indigo-500/10 text-indigo-400 px-2 py-0.5 rounded font-mono">
                {evidence?.stats.total_qualifying_statements || 0} Statements Grounded
              </span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Dual evidence partitioning: Evaluates both corroborating complaints and counter-evidence to eliminate hallucination.
            </p>
          </div>

          {/* Evidence Balance Statistics */}
          {evidence && (
            <div className="flex items-center space-x-3 text-xs font-mono bg-slate-950 p-2 rounded-lg border border-slate-800">
              <span className="text-emerald-400">Balance Ratio: {evidence.stats.evidence_balance_ratio}:1</span>
            </div>
          )}
        </div>

        {/* Tab Controls: Supporting vs Contradicting vs Neutral */}
        <div className="flex border-b border-slate-800 space-x-4">
          <button
            onClick={() => setActiveEvidenceTab('supporting')}
            className={`pb-2.5 text-xs font-semibold flex items-center space-x-2 border-b-2 transition-all ${
              activeEvidenceTab === 'supporting'
                ? 'border-red-500 text-red-400'
                : 'border-transparent text-slate-400 hover:text-slate-300'
            }`}
          >
            <CheckCircle2 className="w-4 h-4" />
            <span>Supporting Evidence ({evidence?.stats.supporting_count || 0})</span>
          </button>

          <button
            onClick={() => setActiveEvidenceTab('contradicting')}
            className={`pb-2.5 text-xs font-semibold flex items-center space-x-2 border-b-2 transition-all ${
              activeEvidenceTab === 'contradicting'
                ? 'border-emerald-500 text-emerald-400'
                : 'border-transparent text-slate-400 hover:text-slate-300'
            }`}
          >
            <XCircle className="w-4 h-4" />
            <span>Contradictory / Counter Evidence ({evidence?.stats.contradicting_count || 0})</span>
          </button>

          <button
            onClick={() => setActiveEvidenceTab('neutral')}
            className={`pb-2.5 text-xs font-semibold flex items-center space-x-2 border-b-2 transition-all ${
              activeEvidenceTab === 'neutral'
                ? 'border-slate-400 text-slate-300'
                : 'border-transparent text-slate-400 hover:text-slate-300'
            }`}
          >
            <HelpCircle className="w-4 h-4" />
            <span>Neutral / Informational ({evidence?.stats.neutral_count || 0})</span>
          </button>
        </div>

        {/* Statements List */}
        <div className="space-y-3">
          {evidence && evidence[activeEvidenceTab]?.length > 0 ? (
            evidence[activeEvidenceTab].map((item, i) => (
              <div
                key={i}
                className={`p-4 rounded-xl border text-xs leading-relaxed transition-all ${
                  activeEvidenceTab === 'supporting'
                    ? 'bg-red-950/10 border-red-900/30 text-red-200'
                    : activeEvidenceTab === 'contradicting'
                    ? 'bg-emerald-950/10 border-emerald-900/30 text-emerald-200'
                    : 'bg-slate-950 border-slate-800 text-slate-300'
                }`}
              >
                <p className="italic font-normal">"{item.text}"</p>
                <div className="mt-2.5 pt-2 border-t border-slate-800/60 flex items-center justify-between text-[11px] text-slate-400 font-mono">
                  <span>Source: {item.source} • Date: {item.date}</span>
                  <span className="font-semibold text-slate-300">
                    Semantic Similarity: {item.similarity}
                  </span>
                </div>
              </div>
            ))
          ) : (
            <div className="p-8 text-center text-slate-500 text-xs">
              No statements in this category.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
