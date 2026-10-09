import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { EmergingIssueItem } from '../types';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts';

const PRIORITY_STYLES: Record<string, { border: string; bg: string; badge: string; badgeText: string; badgeBorder: string }> = {
  CRITICAL: { border: '#e8a09a', bg: '#fdf6f5', badge: '#fbd5d0', badgeText: 'var(--red)', badgeBorder: '#f5a89e' },
  HIGH:     { border: '#e8c67a', bg: '#fefaf0', badge: '#fdecc8', badgeText: '#8a5c0a', badgeBorder: '#f0d28a' },
  MEDIUM:   { border: '#9ecfcf', bg: '#f3fbfb', badge: 'var(--teal-light)', badgeText: 'var(--teal)', badgeBorder: '#8bbfbf' },
  LOW:      { border: 'var(--border)', bg: 'var(--paper)', badge: '#ede8e1', badgeText: 'var(--text-muted)', badgeBorder: 'var(--border)' },
};

const SparklineTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: 'var(--paper)', border: '1px solid var(--border)', borderRadius: '3px', padding: '5px 9px', fontSize: '11px' }}>
      {payload[0]?.value} mentions
    </div>
  );
};

interface EmergingIssuesProps {
  onInvestigate: (issueId: number) => void;
}

export const EmergingIssuesPage: React.FC<EmergingIssuesProps> = ({ onInvestigate }) => {
  const [issues, setIssues] = useState<EmergingIssueItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getEmergingIssues()
      .then(setIssues)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>

      {/* ── Header ─────────────────────────────────── */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '22px', fontWeight: 700, color: 'var(--text-main)' }}>
            Emerging Customer Issues
          </h2>
          <p style={{ margin: '5px 0 0', fontSize: '13px', color: 'var(--text-muted)', maxWidth: '580px' }}>
            Ranked by multi-factor scoring formulation: Growth (30%) + Volume (20%) + Recentness (20%) + Negative Sentiment (20%) + Coherence (10%).
          </p>
        </div>
        <div style={{
          fontSize: '11.5px',
          fontFamily: 'monospace',
          color: 'var(--text-muted)',
          border: '1px solid var(--border)',
          borderRadius: '4px',
          padding: '5px 12px',
          background: 'var(--paper)',
        }}>
          EmergingScore ∈ [0, 1.0]
        </div>
      </div>

      {/* ── Scoring key ──────────────────────────────── */}
      <div style={{
        background: '#f0fafa',
        border: '1px solid #b2d8d8',
        borderRadius: '5px',
        padding: '10px 16px',
        fontSize: '12px',
        color: 'var(--text-muted)',
        display: 'flex',
        flexWrap: 'wrap',
        gap: '16px',
      }}>
        <span><strong style={{ color: 'var(--teal)' }}>Growth ×0.30</strong></span>
        <span><strong style={{ color: 'var(--teal)' }}>Volume ×0.20</strong></span>
        <span><strong style={{ color: 'var(--teal)' }}>Recentness ×0.20</strong></span>
        <span><strong style={{ color: 'var(--teal)' }}>Neg. Sentiment ×0.20</strong></span>
        <span><strong style={{ color: 'var(--teal)' }}>Coherence ×0.10</strong></span>
      </div>

      {/* ── Issue cards ─────────────────────────────── */}
      {loading ? (
        <div style={{ padding: '48px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
          Evaluating temporal issue dynamics…
        </div>
      ) : issues.length === 0 ? (
        <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
          No issues currently flagged as emerging.
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(460px, 1fr))', gap: '16px' }}>
          {issues.map((item) => {
            const p = PRIORITY_STYLES[item.priority] || PRIORITY_STYLES.LOW;
            return (
              <div key={item.issue_id} style={{
                background: p.bg,
                border: `1px solid ${p.border}`,
                borderRadius: '6px',
                padding: '18px 20px',
                display: 'flex',
                flexDirection: 'column',
                gap: '14px',
              }}>
                {/* Title row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '5px' }}>
                      <span style={{
                        fontSize: '9.5px',
                        fontWeight: 700,
                        textTransform: 'uppercase',
                        letterSpacing: '0.06em',
                        padding: '2px 6px',
                        borderRadius: '3px',
                        background: p.badge,
                        color: p.badgeText,
                        border: `1px solid ${p.badgeBorder}`,
                      }}>
                        {item.priority}
                      </span>
                      <span style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-light)' }}>
                        Cluster #{item.issue_id}
                      </span>
                    </div>
                    <h3 style={{ margin: 0, fontSize: '15px', fontWeight: 600, color: 'var(--text-main)' }}>
                      {item.issue_name}
                    </h3>
                  </div>
                  <div style={{ textAlign: 'right', flexShrink: 0, marginLeft: '12px' }}>
                    <div style={{ fontSize: '10px', color: 'var(--text-light)', marginBottom: '2px' }}>Score</div>
                    <div style={{
                      fontSize: '22px',
                      fontWeight: 700,
                      fontFamily: 'monospace',
                      color: 'var(--amber)',
                    }}>
                      {item.emerging_score.toFixed(3)}
                    </div>
                  </div>
                </div>

                {/* Metric row */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr 1fr',
                  gap: '8px',
                  background: 'var(--paper)',
                  border: '1px solid var(--border)',
                  borderRadius: '4px',
                  padding: '10px 12px',
                  textAlign: 'center',
                }}>
                  <div>
                    <div style={{ fontSize: '10px', color: 'var(--text-light)', marginBottom: '2px' }}>Growth</div>
                    <div style={{
                      fontFamily: 'monospace',
                      fontWeight: 700,
                      fontSize: '13px',
                      color: item.growth_rate_pct > 0 ? 'var(--red)' : 'var(--text-muted)',
                    }}>
                      {item.growth_rate_pct > 0 ? `+${item.growth_rate_pct}%` : `${item.growth_rate_pct}%`}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '10px', color: 'var(--text-light)', marginBottom: '2px' }}>Volume</div>
                    <div style={{ fontFamily: 'monospace', fontWeight: 700, fontSize: '13px', color: 'var(--text-main)' }}>
                      {item.total_volume}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '10px', color: 'var(--text-light)', marginBottom: '2px' }}>Neg. Ratio</div>
                    <div style={{ fontFamily: 'monospace', fontWeight: 700, fontSize: '13px', color: 'var(--red)' }}>
                      {Math.round(item.negative_sentiment_ratio * 100)}%
                    </div>
                  </div>
                </div>

                {/* Sparkline */}
                {item.trend_history && item.trend_history.length > 2 && (
                  <div>
                    <div style={{ fontSize: '10px', color: 'var(--text-light)', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600, marginBottom: '4px' }}>
                      Daily Volume Trend
                    </div>
                    <div style={{ height: '56px', width: '100%' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={item.trend_history}>
                          <XAxis dataKey="date" hide />
                          <YAxis hide />
                          <Tooltip content={<SparklineTooltip />} />
                          <Line type="monotone" dataKey="count" stroke="var(--teal)" strokeWidth={2} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                {/* Component breakdown */}
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', borderTop: '1px solid var(--border)', paddingTop: '10px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Growth factor (w=0.30)</span>
                    <span style={{ fontFamily: 'monospace', color: 'var(--text-main)' }}>{item.component_breakdown.norm_growth}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Volume ratio (w=0.20)</span>
                    <span style={{ fontFamily: 'monospace', color: 'var(--text-main)' }}>{item.component_breakdown.norm_volume}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Neg. sentiment (w=0.20)</span>
                    <span style={{ fontFamily: 'monospace', color: 'var(--text-main)' }}>{item.component_breakdown.norm_negative_sentiment}</span>
                  </div>
                </div>

                {/* Investigate button */}
                <button
                  onClick={() => onInvestigate(item.issue_id)}
                  style={{
                    width: '100%',
                    padding: '9px 16px',
                    borderRadius: '4px',
                    border: '1px solid var(--teal)',
                    background: 'var(--teal)',
                    color: '#fff',
                    fontSize: '12.5px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                    transition: 'background 0.12s',
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = '#0a5f62')}
                  onMouseLeave={(e) => (e.currentTarget.style.background = 'var(--teal)')}
                >
                  Investigate Root Causes & Evidence →
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
