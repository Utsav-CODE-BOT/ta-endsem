import React, { useEffect, useState } from 'react';
import { BarChart3, TrendingUp, Calendar, Layers, ShieldCheck } from 'lucide-react';
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
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
          <BarChart3 className="w-6 h-6 text-indigo-400" />
          Customer Voice Analytics & Temporal Dynamics
        </h2>
        <p className="text-slate-400 text-sm mt-0.5">
          Longitudinal trend monitoring, sentiment volume trajectory, and source distribution matrices.
        </p>
      </div>

      {loading ? (
        <div className="p-12 text-center text-slate-400 text-sm">Aggregating temporal analytics...</div>
      ) : (
        <div className="space-y-6">
          {/* Timeline Volume Trajectory */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-2">Temporal Feedback Volume & Negative Trajectory</h3>
            <p className="text-xs text-slate-400 mb-4">Daily total mentions vs negative complaint volume</p>
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trends} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6366f1" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                    </linearGradient>
                    <linearGradient id="colorNegative" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.5} />
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                  <Area
                    type="monotone"
                    dataKey="total_volume"
                    name="Total Feedback"
                    stroke="#6366f1"
                    fillOpacity={1}
                    fill="url(#colorTotal)"
                  />
                  <Area
                    type="monotone"
                    dataKey="negative_volume"
                    name="Negative Complaints"
                    stroke="#ef4444"
                    fillOpacity={1}
                    fill="url(#colorNegative)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Sentiment by Source Comparison */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-2">Sentiment Distribution by Entity / Channel</h3>
            <p className="text-xs text-slate-400 mb-4">Direct cross-comparison across airlines or SaaS touchpoints</p>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={sourceData} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="source" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '12px' }} />
                  <Bar dataKey="negative" name="Negative" fill="#ef4444" stackId="a" />
                  <Bar dataKey="neutral" name="Neutral" fill="#64748b" stackId="a" />
                  <Bar dataKey="positive" name="Positive" fill="#10b981" stackId="a" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
