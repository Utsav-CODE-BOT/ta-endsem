import React, { useEffect, useState } from 'react';
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

const SectionTitle: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <h3 style={{ margin: '0 0 14px', fontSize: '14px', fontWeight: 700, color: 'var(--text-main)', paddingBottom: '8px', borderBottom: '1px solid var(--border)' }}>
    {children}
  </h3>
);

const MiniLabel: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <span style={{ fontSize: '9.5px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-light)' }}>
    {children}
  </span>
);

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
        const targetId = (initialIssueId !== undefined && initialIssueId !== null)
          ? initialIssueId
          : (data[0]?.issue_id || 0);
        setSelectedIssueId(targetId);
        loadIssueResearch(targetId);
      })
      .catch(console.error);
    api.getGraph().then(setGraphData).catch(console.error);
  }, [initialIssueId]);

  const loadIssueResearch = (issueId: number) => {
    setLoading(true);
    Promise.all([api.getRootCauses(issueId), api.getEvidence(issueId)])
      .then(([hyps, ev]) => { setHypotheses(hyps); setEvidence(ev); })
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  const handleIssueChange = (newId: number) => {
    setSelectedIssueId(newId);
    loadIssueResearch(newId);
  };

  const currentIssue = issues.find((iss) => iss.issue_id === selectedIssueId) || issues[0];

  const EV_TABS = [
    { id: 'supporting',    label: `Supporting (${evidence?.stats.supporting_count || 0})`,    color: 'var(--red)' },
    { id: 'contradicting', label: `Contradicting (${evidence?.stats.contradicting_count || 0})`, color: 'var(--green)' },
    { id: 'neutral',       label: `Neutral (${evidence?.stats.neutral_count || 0})`,           color: 'var(--text-muted)' },
  ] as const;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>

      {/* ── Header ───────────────────────────────────────── */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '22px', fontWeight: 700, color: 'var(--text-main)' }}>
            Root-Cause & Factor Association Investigation
          </h2>
          <p style={{ margin: '5px 0 0', fontSize: '13px', color: 'var(--text-muted)' }}>
            Empirical factor attribution using Statistical Lift, Pointwise Mutual Information (PMI), and semantic evidence retrieval.
          </p>
        </div>

        {/* Issue selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'var(--paper)', border: '1px solid var(--border)', borderRadius: '4px', padding: '6px 12px', fontSize: '13px' }}>
          <span style={{ color: 'var(--text-light)', fontWeight: 500 }}>Issue:</span>
          <select
            value={selectedIssueId}
            onChange={(e) => handleIssueChange(Number(e.target.value))}
            style={{ background: 'transparent', border: 'none', fontSize: '13px', fontWeight: 600, color: 'var(--text-main)', cursor: 'pointer', outline: 'none' }}
          >
            {issues.map((iss) => (
              <option key={iss.issue_id} value={iss.issue_id}>
                #{iss.issue_id} – {iss.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* ── Scientific guardrail notice ───────────────────── */}
      <div style={{
        background: '#fefaf0',
        border: '1px solid #e8c67a',
        borderRadius: '5px',
        padding: '12px 16px',
        fontSize: '12px',
        color: '#7a5a10',
        lineHeight: 1.6,
      }}>
        <strong>⚠ Scientific Methodology Guardrail:</strong> Discovered factors represent <em>empirical statistical associations</em> and <em>temporal precedence</em>. VoiceMiner presents candidate root causes for operational verification; it does not claim causal determinism.
      </div>

      {/* ── Row 1: Hypotheses + Graph ─────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '7fr 5fr', gap: '16px', alignItems: 'flex-start' }}>

        {/* Hypotheses */}
        <div style={{ background: 'var(--paper)', border: '1px solid var(--border)', borderRadius: '6px', padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', paddingBottom: '8px', borderBottom: '1px solid var(--border)' }}>
            <h3 style={{ margin: 0, fontSize: '14px', fontWeight: 700, color: 'var(--text-main)' }}>
              Candidate Root-Cause Factors
            </h3>
            <span style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-light)' }}>
              {hypotheses.length} identified · ranked by confidence
            </span>
          </div>

          {loading ? (
            <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
              Mining candidate factors…
            </div>
          ) : hypotheses.length === 0 ? (
            <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
              No strong statistical factor associations crossed the significance threshold.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {hypotheses.map((h, i) => (
                <div key={h.id || i} style={{
                  border: '1px solid var(--border)',
                  borderRadius: '5px',
                  padding: '14px 16px',
                  background: 'var(--cream)',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px' }}>
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '5px' }}>
                        <span style={{
                          fontSize: '9.5px', fontWeight: 700, textTransform: 'uppercase',
                          padding: '2px 6px', borderRadius: '3px',
                          background: 'var(--teal-light)', color: 'var(--teal)',
                          border: '1px solid #9ecfcf',
                        }}>
                          {h.factor_type}
                        </span>
                        <span style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-light)' }}>
                          Temporal: {h.temporal_relationship}
                        </span>
                      </div>
                      <h4 style={{ margin: 0, fontSize: '14.5px', fontWeight: 600, color: 'var(--text-main)' }}>
                        {h.candidate_factor}
                      </h4>
                    </div>
                    <div style={{ textAlign: 'right', flexShrink: 0 }}>
                      <div style={{ fontSize: '10px', color: 'var(--text-light)' }}>Confidence</div>
                      <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'monospace', color: 'var(--teal)' }}>
                        {Math.round(h.confidence * 100)}%
                      </div>
                    </div>
                  </div>

                  {/* Metrics */}
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr 1fr',
                    gap: '6px',
                    marginTop: '10px',
                    background: 'var(--paper)',
                    border: '1px solid var(--border)',
                    borderRadius: '4px',
                    padding: '8px 10px',
                    fontSize: '12px',
                    fontFamily: 'monospace',
                  }}>
                    <div>
                      <MiniLabel>Lift</MiniLabel>
                      <div style={{ fontWeight: 700, color: 'var(--green)' }}>{h.lift}×</div>
                    </div>
                    <div>
                      <MiniLabel>PMI</MiniLabel>
                      <div style={{ fontWeight: 700, color: 'var(--teal)' }}>{h.pmi}</div>
                    </div>
                    <div>
                      <MiniLabel>Co-occurrences</MiniLabel>
                      <div style={{ fontWeight: 700, color: 'var(--text-main)' }}>{h.co_occurrence_count} / {h.total_mention_count}</div>
                    </div>
                  </div>

                  <p style={{
                    margin: '10px 0 0',
                    fontSize: '12px',
                    color: 'var(--text-muted)',
                    fontStyle: 'italic',
                    background: 'var(--paper)',
                    padding: '8px 10px',
                    borderRadius: '3px',
                    border: '1px solid var(--border)',
                    lineHeight: 1.6,
                  }}>
                    "{h.interpretation}"
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Graph snapshot */}
        <div style={{ background: 'var(--paper)', border: '1px solid var(--border)', borderRadius: '6px', padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <h3 style={{ margin: 0, fontSize: '14px', fontWeight: 700, color: 'var(--text-main)' }}>
              Association Graph Snapshot
            </h3>
            <span style={{ fontSize: '10.5px', fontFamily: 'monospace', color: 'var(--text-light)' }}>
              {graphData?.total_nodes || 0}N · {graphData?.total_edges || 0}E
            </span>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', margin: '0 0 12px' }}>
            NetworkX topology mapping candidate factors, aspect dependencies, and issue clusters.
          </p>

          <div style={{
            height: '220px',
            background: 'var(--cream)',
            border: '1px solid var(--border)',
            borderRadius: '4px',
            overflow: 'hidden',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            <svg width="100%" height="100%" viewBox="0 0 360 220">
              {/* Edges */}
              <line x1="180" y1="110" x2="80"  y2="55"  stroke="var(--teal)"  strokeWidth="1.5" strokeDasharray="4,3" />
              <line x1="180" y1="110" x2="280" y2="55"  stroke="var(--amber)" strokeWidth="1.5" />
              <line x1="180" y1="110" x2="80"  y2="165" stroke="var(--red)"   strokeWidth="1.5" />
              <line x1="180" y1="110" x2="280" y2="165" stroke="#7c5cbf"      strokeWidth="1.5" strokeDasharray="4,3" />

              {/* Center node */}
              <circle cx="180" cy="110" r="26" fill="var(--teal)" />
              <text x="180" y="114" fill="#fff" fontSize="8" fontWeight="bold" textAnchor="middle">
                {currentIssue?.name?.slice(0, 10) || 'Issue'}
              </text>

              {/* Satellite nodes */}
              <circle cx="80"  cy="55"  r="18" fill="#e0f0f0" stroke="var(--teal)"  strokeWidth="1.5" />
              <text x="80"  y="58"  fill="var(--teal)"  fontSize="7.5" textAnchor="middle">Entity</text>

              <circle cx="280" cy="55"  r="18" fill="#fef3e2" stroke="var(--amber)" strokeWidth="1.5" />
              <text x="280" y="58"  fill="var(--amber)" fontSize="7.5" textAnchor="middle">Aspect</text>

              <circle cx="80"  cy="165" r="18" fill="#fdf0ee" stroke="var(--red)"   strokeWidth="1.5" />
              <text x="80"  y="168" fill="var(--red)"   fontSize="7.5" textAnchor="middle">Surge</text>

              <circle cx="280" cy="165" r="18" fill="#f0eefa" stroke="#7c5cbf"      strokeWidth="1.5" />
              <text x="280" y="168" fill="#7c5cbf"      fontSize="7.5" textAnchor="middle">Candidate</text>
            </svg>
          </div>

          <div style={{ marginTop: '12px', paddingTop: '10px', borderTop: '1px solid var(--border)', fontSize: '11.5px', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
            <span>Primary Aspect: <strong style={{ color: 'var(--teal)' }}>{currentIssue?.primary_aspect || '—'}</strong></span>
            <span>Cluster Size: <strong>{currentIssue?.cluster_size || '—'}</strong></span>
          </div>
        </div>
      </div>

      {/* ── Row 2: Evidence ───────────────────────────────── */}
      <div style={{ background: 'var(--paper)', border: '1px solid var(--border)', borderRadius: '6px', padding: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
          <div>
            <h3 style={{ margin: '0 0 4px', fontSize: '14px', fontWeight: 700, color: 'var(--text-main)' }}>
              Evidence Retrieval Audit
            </h3>
            <p style={{ margin: 0, fontSize: '12px', color: 'var(--text-muted)' }}>
              Dual evidence partitioning: corroborating complaints & counter-evidence to avoid hallucination.
            </p>
          </div>
          {evidence && (
            <span style={{ fontSize: '12px', fontFamily: 'monospace', color: 'var(--teal)', background: 'var(--teal-light)', border: '1px solid #9ecfcf', borderRadius: '4px', padding: '3px 10px' }}>
              Balance ratio: {evidence.stats.evidence_balance_ratio}:1 · {evidence.stats.total_qualifying_statements} statements
            </span>
          )}
        </div>

        {/* Tab bar */}
        <div style={{ display: 'flex', borderBottom: '1px solid var(--border)', marginBottom: '14px', gap: '2px' }}>
          {EV_TABS.map((tab) => {
            const active = activeEvidenceTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveEvidenceTab(tab.id)}
                style={{
                  padding: '7px 14px',
                  fontSize: '12px',
                  fontWeight: active ? 600 : 400,
                  border: 'none',
                  borderBottom: active ? `2px solid ${tab.color}` : '2px solid transparent',
                  background: 'none',
                  color: active ? tab.color : 'var(--text-muted)',
                  cursor: 'pointer',
                  transition: 'color 0.1s',
                  marginBottom: '-1px',
                }}
              >
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Statements */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {evidence && (evidence as any)[activeEvidenceTab]?.length > 0 ? (
            (evidence as any)[activeEvidenceTab].map((item: any, i: number) => {
              const bgMap: Record<string, string> = {
                supporting: '#fdf6f5', contradicting: '#f2fbf7', neutral: 'var(--cream)',
              };
              const borderMap: Record<string, string> = {
                supporting: '#f0c0bb', contradicting: '#a8d9c0', neutral: 'var(--border)',
              };
              return (
                <div key={i} style={{
                  background: bgMap[activeEvidenceTab],
                  border: `1px solid ${borderMap[activeEvidenceTab]}`,
                  borderRadius: '5px',
                  padding: '12px 14px',
                  fontSize: '12.5px',
                  lineHeight: 1.65,
                  color: 'var(--text-main)',
                }}>
                  <p style={{ margin: 0, fontStyle: 'italic' }}>"{item.text}"</p>
                  <div style={{ marginTop: '8px', paddingTop: '6px', borderTop: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-light)' }}>
                    <span>Source: {item.source} · Date: {item.date}</span>
                    <span style={{ fontWeight: 600, color: 'var(--teal)' }}>Similarity: {item.similarity}</span>
                  </div>
                </div>
              );
            })
          ) : (
            <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
              No statements in this category.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
