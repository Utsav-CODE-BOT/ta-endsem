import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { TrendPoint } from '../types';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';

export const AnalyticsPage: React.FC = () => {
  const [trends, setTrends] = useState<TrendPoint[]>([]);
  const [sentimentData, setSentimentData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.getTrends(), api.getSentimentBreakdown()])
      .then(([tData, sData]) => {
        setTrends(tData);
        setSentimentData(sData);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const sourceData = sentimentData?.by_source
    ? Object.entries(sentimentData.by_source).map(([source, sents]: [string, any]) => ({
        source,
        negative: sents.negative || 0,
        neutral: sents.neutral || 0,
        positive: sents.positive || 0,
      }))
    : [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
      <div>
        <h2 style={{ margin: 0, fontSize: '22px', fontWeight: 700, color: 'var(--text-main)' }}>
          Customer Voice Analytics & Temporal Dynamics
        </h2>
        <p style={{ margin: '5px 0 0', fontSize: '13px', color: 'var(--text-muted)' }}>
          Longitudinal trend monitoring, sentiment volume trajectory, and source distribution matrices.
        </p>
      </div>

      {loading ? (
        <div style={{ padding: '48px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
          Aggregating temporal analytics…
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Timeline Volume Trajectory */}
          <div style={{
            background: 'var(--paper)',
            border: '1px solid var(--border)',
            borderRadius: '6px',
            padding: '20px',
          }}>
            <h3 style={{ margin: '0 0 2px', fontSize: '14px', fontWeight: 600, color: 'var(--text-main)' }}>
              Temporal Feedback Volume & Negative Trajectory
            </h3>
            <p style={{ margin: '0 0 14px', fontSize: '12px', color: 'var(--text-muted)' }}>
              Daily total mentions vs negative complaint volume
            </p>

            <div style={{ height: '260px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trends} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#0d7377" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#0d7377" stopOpacity={0.0} />
                    </linearGradient>
                    <linearGradient id="colorNegative" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#c0392b" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#c0392b" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e8e2d8" vertical={false} />
                  <XAxis dataKey="date" stroke="var(--border-md)" tick={{ fontSize: 10, fill: 'var(--text-light)' }} tickLine={false} axisLine={false} />
                  <YAxis stroke="var(--border-md)" tick={{ fontSize: 10, fill: 'var(--text-light)' }} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'var(--paper)',
                      border: '1px solid var(--border)',
                      borderRadius: '4px',
                      fontSize: '12px',
                      color: 'var(--text-main)',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                  <Area
                    type="monotone"
                    dataKey="total_volume"
                    name="Total Feedback"
                    stroke="#0d7377"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#colorTotal)"
                  />
                  <Area
                    type="monotone"
                    dataKey="negative_volume"
                    name="Negative Complaints"
                    stroke="#c0392b"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#colorNegative)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Sentiment by Source Comparison */}
          <div style={{
            background: 'var(--paper)',
            border: '1px solid var(--border)',
            borderRadius: '6px',
            padding: '20px',
          }}>
            <h3 style={{ margin: '0 0 2px', fontSize: '14px', fontWeight: 600, color: 'var(--text-main)' }}>
              Sentiment Distribution by Entity / Channel
            </h3>
            <p style={{ margin: '0 0 14px', fontSize: '12px', color: 'var(--text-muted)' }}>
              Cross-comparison across airline carriers or touchpoints
            </p>

            <div style={{ height: '240px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={sourceData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e8e2d8" vertical={false} />
                  <XAxis dataKey="source" stroke="var(--border-md)" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickLine={false} axisLine={false} />
                  <YAxis stroke="var(--border-md)" tick={{ fontSize: 10, fill: 'var(--text-light)' }} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'var(--paper)',
                      border: '1px solid var(--border)',
                      borderRadius: '4px',
                      fontSize: '12px',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '12px' }} />
                  <Bar dataKey="negative" name="Negative" fill="#c0392b" stackId="a" />
                  <Bar dataKey="neutral" name="Neutral" fill="#9c958e" stackId="a" />
                  <Bar dataKey="positive" name="Positive" fill="#1e6b4a" stackId="a" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
