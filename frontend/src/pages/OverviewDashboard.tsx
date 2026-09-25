import React, { useEffect, useState } from 'react';
import {
  MessageSquare,
  AlertTriangle,
  Flame,
  CheckCircle2,
  TrendingDown,
  TrendingUp,
  Tag,
  Share2,
  Calendar,
} from 'lucide-react';
import { api } from '../services/api';
import { DashboardSummary } from '../types';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

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
      <div className="flex items-center justify-center h-full min-h-[400px]">
        <div className="flex flex-col items-center space-y-3">
          <div className="w-8 h-8 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-slate-400 text-sm">Loading VoiceMiner Intelligence...</p>
        </div>
      </div>
    );
  }

  if (!data) return <div className="p-8 text-slate-400">Failed to load overview data.</div>;

  const sentimentData = [
    { name: 'Negative', value: data.sentiment_distribution.negative, color: '#ef4444' },
    { name: 'Neutral', value: data.sentiment_distribution.neutral, color: '#64748b' },
    { name: 'Positive', value: data.sentiment_distribution.positive, color: '#10b981' },
  ];

  const emotionData = Object.entries(data.primary_emotions || {}).map(([emotion, count]) => ({
    emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
    count,
  }));

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Intelligence Overview</h2>
          <p className="text-slate-400 text-sm mt-0.5">
            Real-time synthesis across multi-source customer feedback, aspects, and temporal issues.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs bg-slate-900 border border-slate-800 px-3.5 py-2 rounded-lg text-slate-300">
          <Calendar className="w-4 h-4 text-indigo-400" />
          <span>Observation Window: <strong className="text-white">{data.date_range.start}</strong> to <strong className="text-white">{data.date_range.end}</strong></span>
        </div>
      </div>

      {/* KPI Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Total Feedback</span>
            <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400">
              <MessageSquare className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-3xl font-extrabold text-white tracking-tight">
              {data.total_feedback.toLocaleString()}
            </div>
            <p className="text-xs text-slate-400 mt-1 flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              <span>Multi-domain ingestion verified</span>
            </p>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Negative Sentiment</span>
            <div className="p-2 rounded-lg bg-red-500/10 text-red-400">
              <TrendingDown className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-3xl font-extrabold text-red-400 tracking-tight">
              {data.sentiment_distribution.negative}%
            </div>
            <p className="text-xs text-slate-400 mt-1">
              {data.sentiment_distribution.positive}% positive • {data.sentiment_distribution.neutral}% neutral
            </p>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Active Issue Clusters</span>
            <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400">
              <Share2 className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-3xl font-extrabold text-white tracking-tight">
              {data.active_issues}
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Discovered via SBERT + c-TF-IDF
            </p>
          </div>
        </div>

        <div
          onClick={onNavigateToEmerging}
          className="bg-gradient-to-br from-amber-950/40 via-slate-900 to-slate-900 border border-amber-500/30 rounded-xl p-5 shadow-sm cursor-pointer hover:border-amber-500/60 transition-all group"
        >
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-amber-400 uppercase tracking-wider flex items-center gap-1.5">
              <Flame className="w-4 h-4 text-amber-500 animate-bounce" />
              Emerging Issues
            </span>
            <div className="p-2 rounded-lg bg-amber-500/20 text-amber-400 group-hover:bg-amber-500 group-hover:text-black transition-all">
              <AlertTriangle className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-3xl font-extrabold text-amber-400 tracking-tight flex items-baseline gap-2">
              {data.emerging_issues_count}
              <span className="text-xs font-semibold text-red-400 bg-red-500/10 px-2 py-0.5 rounded border border-red-500/20">
                {data.critical_issues_count} CRITICAL
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">Click to inspect unusual surges →</p>
          </div>
        </div>
      </div>

      {/* Row 2: Sentiment Bar & Aspects */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sentiment Distribution */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-white mb-4">Transformer Sentiment Breakdown</h3>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sentimentData} layout="vertical" margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                <XAxis type="number" unit="%" stroke="#64748b" />
                <YAxis type="category" dataKey="name" stroke="#94a3b8" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                  formatter={(val: any) => [`${val}%`, 'Proportion']}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {sentimentData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-3 gap-2 mt-4 pt-3 border-t border-slate-800 text-center">
            <div>
              <div className="text-xs text-slate-400">Negative</div>
              <div className="text-sm font-bold text-red-400">{data.sentiment_distribution.negative}%</div>
            </div>
            <div>
              <div className="text-xs text-slate-400">Neutral</div>
              <div className="text-sm font-bold text-slate-300">{data.sentiment_distribution.neutral}%</div>
            </div>
            <div>
              <div className="text-xs text-slate-400">Positive</div>
              <div className="text-sm font-bold text-emerald-400">{data.sentiment_distribution.positive}%</div>
            </div>
          </div>
        </div>

        {/* Top Extracted Aspects */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white">Dominant Customer Aspects</h3>
            <span className="text-xs text-slate-400">Semantic N-gram Normalization</span>
          </div>
          <div className="space-y-3">
            {data.top_aspects.map((asp, idx) => (
              <div key={idx} className="flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2">
                  <Tag className="w-3.5 h-3.5 text-indigo-400" />
                  <span className="font-semibold text-slate-200">{asp.aspect}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-24 bg-slate-800 rounded-full h-2 overflow-hidden">
                    <div
                      className="bg-indigo-500 h-full rounded-full"
                      style={{ width: `${Math.min(100, asp.percentage * 2.5)}%` }}
                    ></div>
                  </div>
                  <span className="text-slate-400 font-mono w-12 text-right">{asp.count} ({asp.percentage}%)</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Emotion Spectrum */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-white mb-4">Affective Emotion Spectrum</h3>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={emotionData} margin={{ top: 5, right: 10, left: -20, bottom: 20 }}>
                <XAxis dataKey="emotion" stroke="#64748b" tick={{ fontSize: 10 }} angle={-25} textAnchor="end" />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                />
                <Bar dataKey="count" fill="#818cf8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-slate-400 mt-2 text-center">
            Grounded 7-emotion affective inference mapped via anchor cosine similarity.
          </p>
        </div>
      </div>

      {/* Row 3: Sources Distribution */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white mb-3">Multi-Source / Entity Contribution</h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
          {data.sources.map((src, i) => (
            <div key={i} className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-3 text-center">
              <div className="text-xs text-slate-400 font-medium truncate">{src.source}</div>
              <div className="text-lg font-bold text-white mt-1">{src.count}</div>
              <div className="text-[11px] text-indigo-400 font-mono">{src.percentage}% of total</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
