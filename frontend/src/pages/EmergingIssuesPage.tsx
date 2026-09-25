import React, { useEffect, useState } from 'react';
import {
  Flame,
  TrendingUp,
  AlertTriangle,
  Calendar,
  Layers,
  ArrowRight,
  ShieldAlert,
  Activity,
  BarChart2,
} from 'lucide-react';
import { api } from '../services/api';
import { EmergingIssueItem } from '../types';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts';

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

  const getPriorityBadge = (priority: string) => {
    switch (priority) {
      case 'CRITICAL':
        return 'bg-red-500/20 text-red-400 border-red-500/30 animate-pulse';
      case 'HIGH':
        return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
      case 'MEDIUM':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      default:
        return 'bg-slate-700/40 text-slate-400 border-slate-600/30';
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
            <Flame className="w-6 h-6 text-amber-500 animate-bounce" />
            Emerging Customer Issues
          </h2>
          <p className="text-slate-400 text-sm mt-0.5">
            Ranked by multi-factor mathematical formulation: Growth Surge (30%) + Volume (20%) + Recentness (20%) + Negative Sentiment (20%) + Coherence (10%).
          </p>
        </div>
        <div className="text-xs bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-lg text-slate-300">
          Scoring Formulation: <strong className="text-indigo-400 font-mono">EmergingScore ∈ [0, 1.0]</strong>
        </div>
      </div>

      {loading ? (
        <div className="p-12 text-center text-slate-400 text-sm">Evaluating temporal issue dynamics...</div>
      ) : issues.length === 0 ? (
        <div className="p-8 text-center text-slate-400">No issues currently flagged as emerging.</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {issues.map((item) => (
            <div
              key={item.issue_id}
              className="bg-slate-900 border border-slate-800 hover:border-slate-700 rounded-xl p-5 shadow-sm space-y-4 transition-all"
            >
              {/* Header with Priority and Emerging Score */}
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center space-x-2">
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded border uppercase ${getPriorityBadge(item.priority)}`}>
                      {item.priority} PRIORITY
                    </span>
                    <span className="text-xs font-mono text-slate-400">Cluster #{item.issue_id}</span>
                  </div>
                  <h3 className="text-base font-bold text-white mt-1.5">{item.issue_name}</h3>
                </div>

                <div className="text-right">
                  <div className="text-xs text-slate-400 font-medium">Emerging Score</div>
                  <div className="text-2xl font-black text-amber-400 font-mono tracking-tight">
                    {item.emerging_score.toFixed(3)}
                  </div>
                </div>
              </div>

              {/* Core Metric Highlights */}
              <div className="grid grid-cols-3 gap-2 p-3 bg-slate-950 rounded-lg border border-slate-800/80 text-center">
                <div>
                  <span className="text-[11px] text-slate-400 flex items-center justify-center gap-1">
                    <TrendingUp className="w-3 h-3 text-red-400" />
                    Growth
                  </span>
                  <span className={`text-sm font-bold font-mono mt-0.5 block ${item.growth_rate_pct > 0 ? 'text-red-400' : 'text-slate-400'}`}>
                    {item.growth_rate_pct > 0 ? `+${item.growth_rate_pct}%` : `${item.growth_rate_pct}%`}
                  </span>
                </div>
                <div>
                  <span className="text-[11px] text-slate-400">Volume</span>
                  <span className="text-sm font-bold text-white font-mono mt-0.5 block">
                    {item.total_volume} ({item.recent_volume} rec.)
                  </span>
                </div>
                <div>
                  <span className="text-[11px] text-slate-400">Negative Ratio</span>
                  <span className="text-sm font-bold text-red-400 font-mono mt-0.5 block">
                    {Math.round(item.negative_sentiment_ratio * 100)}%
                  </span>
                </div>
              </div>

              {/* Sparkline Time Series if available */}
              {item.trend_history && item.trend_history.length > 2 && (
                <div>
                  <span className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">
                    Temporal Velocity Trend (Daily Volume):
                  </span>
                  <div className="h-16 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={item.trend_history}>
                        <XAxis dataKey="date" hide />
                        <YAxis hide />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px', borderRadius: '6px' }}
                          formatter={(v: any) => [`${v} mentions`, 'Volume']}
                        />
                        <Line type="monotone" dataKey="count" stroke="#f59e0b" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Normalized Multi-Factor Breakdown */}
              <div className="space-y-1.5 pt-2 border-t border-slate-800 text-[11px]">
                <div className="flex justify-between text-slate-400">
                  <span>Growth Factor (w=0.30)</span>
                  <span className="font-mono text-slate-300">{item.component_breakdown.norm_growth}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Volume Ratio (w=0.20)</span>
                  <span className="font-mono text-slate-300">{item.component_breakdown.norm_volume}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Negative Sentiment (w=0.20)</span>
                  <span className="font-mono text-slate-300">{item.component_breakdown.norm_negative_sentiment}</span>
                </div>
              </div>

              {/* Action Button */}
              <button
                onClick={() => onInvestigate(item.issue_id)}
                className="w-full py-2.5 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center justify-center gap-2 shadow-md shadow-indigo-600/30 transition-all"
              >
                <span>Investigate Root Causes & Evidence</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
