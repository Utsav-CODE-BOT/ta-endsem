import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { DashboardSummary } from '../types';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

/* ─── tiny shared primitives ─────────────────────────── */
const Card: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div style={{
    background: 'var(--paper)',
    border: '1px solid var(--border)',
    borderRadius: '6px',
    padding: '20px',
    ...style,
  }}>
    {children}
  </div>
);

const Label: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div style={{
    fontSize: '10px',
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    color: 'var(--text-light)',
    marginBottom: '6px',
  }}>
    {children}
  </div>
);

const BigNum: React.FC<{ children: React.ReactNode; color?: string }> = ({ children, color }) => (
  <div style={{
    fontSize: '30px',
    fontWeight: 700,
    color: color || 'var(--text-main)',
    lineHeight: 1.15,
    fontFamily: 'JetBrains Mono, monospace',
  }}>
    {children}
  </div>
);

/* ─── custom tooltip ─────────────────────────────────── */
const ChartTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'var(--paper)',
      border: '1px solid var(--border)',
      borderRadius: '4px',
      padding: '8px 12px',
      fontSize: '12px',
      color: 'var(--text-main)',
    }}>
      <strong>{label || payload[0]?.name}</strong>: {payload[0]?.value}
      {payload[0]?.unit}
    </div>
  );
};

/* ─── main component ─────────────────────────────────── */
export const OverviewDashboard: React.FC<{ onNavigateToEmerging: () => void }> = ({
  onNavigateToEmerging,
}) => {
  const [data, setData] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getDashboard()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div style={{ padding: '60px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '14px' }}>
        <div style={{ fontSize: '24px', marginBottom: '12px' }}>⧗</div>
        Loading intelligence data…
      </div>
    );
  }

  if (!data) {
    return <div style={{ padding: '32px', color: 'var(--red)' }}>Failed to load overview data.</div>;
  }

  const sentimentData = [
    { name: 'Negative', value: data.sentiment_distribution.negative, color: '#c0392b' },
    { name: 'Neutral',  value: data.sentiment_distribution.neutral,  color: '#9c958e' },
    { name: 'Positive', value: data.sentiment_distribution.positive, color: '#1e6b4a' },
  ];

  const emotionData = Object.entries(data.primary_emotions || {}).map(([emotion, count]) => ({
    emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
    count,
  }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

      {/* ── Page header ─────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '22px', fontWeight: 700, color: 'var(--text-main)' }}>
            Intelligence Overview
          </h2>
          <p style={{ margin: '4px 0 0', fontSize: '13px', color: 'var(--text-muted)' }}>
            Real-time synthesis across multi-source customer feedback, extracted aspects, and temporal issue clusters.
          </p>
        </div>
        <div style={{
          fontSize: '12px',
          color: 'var(--text-muted)',
          background: 'var(--paper)',
          border: '1px solid var(--border)',
          borderRadius: '4px',
          padding: '6px 12px',
          fontFamily: 'monospace',
        }}>
          Observation window: <strong>{data.date_range.start}</strong> → <strong>{data.date_range.end}</strong>
        </div>
      </div>

      {/* ── KPI row ─────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px' }}>

        {/* 1. Total Feedback */}
        <Card>
          <Label>Total Feedback</Label>
          <BigNum>{data.total_feedback.toLocaleString()}</BigNum>
          <div style={{ marginTop: '6px', fontSize: '11px', color: 'var(--green)', fontWeight: 500 }}>
            ✓ Multi-domain ingestion verified
          </div>
        </Card>

        {/* 2. Negative Sentiment */}
        <Card>
          <Label>Negative Sentiment</Label>
          <BigNum color="var(--red)">{data.sentiment_distribution.negative}%</BigNum>
          <div style={{ marginTop: '6px', fontSize: '11px', color: 'var(--text-light)' }}>
            {data.sentiment_distribution.positive}% pos · {data.sentiment_distribution.neutral}% neu
          </div>
        </Card>

        {/* 3. Active Issue Clusters */}
        <Card>
          <Label>Active Issue Clusters</Label>
          <BigNum>{data.active_issues}</BigNum>
          <div style={{ marginTop: '6px', fontSize: '11px', color: 'var(--text-light)' }}>
            Discovered via SBERT + c-TF-IDF
          </div>
        </Card>

        {/* 4. Emerging Issues — clickable */}
        <Card
          style={{
            cursor: 'pointer',
            borderColor: '#e2a84b',
            background: 'var(--amber-bg)',
          }}
        >
          <div onClick={onNavigateToEmerging}>
            <Label>Emerging Issues ↗</Label>
            <BigNum color="var(--amber)">{data.emerging_issues_count}</BigNum>
            <div style={{ marginTop: '6px', fontSize: '11px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <span style={{
                background: '#fbd5d0',
                color: 'var(--red)',
                border: '1px solid #f5a89e',
                borderRadius: '3px',
                padding: '1px 6px',
                fontWeight: 600,
                fontSize: '10px',
              }}>
                {data.critical_issues_count} CRITICAL
              </span>
              <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>Click to inspect →</span>
            </div>
          </div>
        </Card>
      </div>

      {/* ── Row 2: Charts ───────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px' }}>

        {/* Sentiment bar chart */}
        <Card>
          <div style={{ fontWeight: 600, fontSize: '13px', marginBottom: '14px' }}>
            Transformer Sentiment Breakdown
          </div>
          <div style={{ height: '160px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sentimentData} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
                <XAxis type="number" unit="%" tick={{ fontSize: 11, fill: 'var(--text-light)' }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} axisLine={false} tickLine={false} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="value" radius={[0, 3, 3, 0]} maxBarSize={18}>
                  {sentimentData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', marginTop: '10px', paddingTop: '10px', borderTop: '1px solid var(--border)', textAlign: 'center', gap: '4px' }}>
            {sentimentData.map((s) => (
              <div key={s.name}>
                <div style={{ fontSize: '10px', color: 'var(--text-light)' }}>{s.name}</div>
                <div style={{ fontSize: '14px', fontWeight: 700, color: s.color }}>{s.value}%</div>
              </div>
            ))}
          </div>
        </Card>

        {/* Top Aspects */}
        <Card>
          <div style={{ fontWeight: 600, fontSize: '13px', marginBottom: '14px', display: 'flex', justifyContent: 'space-between' }}>
            <span>Dominant Customer Aspects</span>
            <span style={{ fontWeight: 400, fontSize: '11px', color: 'var(--text-light)' }}>Semantic N-gram</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {data.top_aspects.map((asp, idx) => (
              <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
                <span style={{ width: '14px', color: 'var(--text-light)', textAlign: 'right', fontFamily: 'monospace', flexShrink: 0 }}>
                  {idx + 1}.
                </span>
                <span style={{ fontWeight: 500, color: 'var(--text-main)', flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {asp.aspect}
                </span>
                <div style={{ width: '70px', background: 'var(--border)', borderRadius: '2px', height: '5px', flexShrink: 0 }}>
                  <div style={{
                    width: `${Math.min(100, asp.percentage * 2.5)}%`,
                    height: '5px',
                    background: 'var(--teal)',
                    borderRadius: '2px',
                  }} />
                </div>
                <span style={{ fontSize: '10px', color: 'var(--text-light)', fontFamily: 'monospace', flexShrink: 0, width: '36px', textAlign: 'right' }}>
                  {asp.count}
                </span>
              </div>
            ))}
          </div>
        </Card>

        {/* Emotion bar chart */}
        <Card>
          <div style={{ fontWeight: 600, fontSize: '13px', marginBottom: '14px' }}>
            Affective Emotion Spectrum
          </div>
          <div style={{ height: '160px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={emotionData} margin={{ top: 4, right: 6, left: -20, bottom: 24 }}>
                <XAxis dataKey="emotion" tick={{ fontSize: 9, fill: 'var(--text-light)' }} angle={-30} textAnchor="end" axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: 'var(--text-light)' }} axisLine={false} tickLine={false} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="count" fill="var(--teal-mid)" radius={[3, 3, 0, 0]} maxBarSize={20} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p style={{ fontSize: '10.5px', color: 'var(--text-light)', marginTop: '6px', textAlign: 'center' }}>
            7-emotion inference via anchor cosine similarity
          </p>
        </Card>
      </div>

      {/* ── Row 3: Source distribution ──────────────── */}
      <Card>
        <div style={{ fontWeight: 600, fontSize: '13px', marginBottom: '14px' }}>
          Multi-Source / Entity Contribution
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: '10px' }}>
          {data.sources.map((src, i) => (
            <div key={i} style={{
              background: 'var(--cream)',
              border: '1px solid var(--border)',
              borderRadius: '5px',
              padding: '12px 14px',
              textAlign: 'center',
            }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {src.source}
              </div>
              <div style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-main)', marginTop: '4px', fontFamily: 'monospace' }}>
                {src.count}
              </div>
              <div style={{ fontSize: '10px', color: 'var(--teal)', fontFamily: 'monospace', marginTop: '2px' }}>
                {src.percentage}% of total
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};
