import React, { useEffect, useState } from 'react';
import {
  Compass,
  Tag,
  MessageSquare,
  Search,
  ExternalLink,
  ChevronRight,
  TrendingDown,
  Layers,
  Sparkles,
} from 'lucide-react';
import { api } from '../services/api';
import { IssueClusterItem, IssueDetailResponse } from '../types';
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, Tooltip, Cell, ZAxis } from 'recharts';

interface IssueExplorerProps {
  onInvestigateIssue: (issueId: number) => void;
}

export const IssueExplorerPage: React.FC<IssueExplorerProps> = ({ onInvestigateIssue }) => {
  const [issues, setIssues] = useState<IssueClusterItem[]>([]);
  const [selectedIssue, setSelectedIssue] = useState<IssueDetailResponse | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadIssues();
  }, []);

  const loadIssues = () => {
    setLoading(true);
    api.getIssues()
      .then((data) => {
        setIssues(data);
        if (data.length > 0) {
          handleSelectIssue(data[0].issue_id);
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  const handleSelectIssue = (id: number) => {
    api.getIssueDetail(id).then(setSelectedIssue).catch(console.error);
  };

  const filteredIssues = issues.filter((iss) =>
    iss.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    iss.top_terms.some((t) => t.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  // Scatter plot data for 2D semantic projection
  const scatterData = issues.map((iss) => ({
    x: iss.center_2d[0],
    y: iss.center_2d[1],
    z: iss.cluster_size,
    name: iss.name,
    id: iss.issue_id,
    aspect: iss.primary_aspect,
  }));

  const COLORS = ['#6366f1', '#06b6d4', '#f59e0b', '#ec4899', '#10b981', '#8b5cf6'];

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Semantic Issue Discovery Explorer</h2>
          <p className="text-slate-400 text-sm mt-0.5">
            Unsupervised semantic clustering via Sentence-BERT embeddings, dimensionality projection, and c-TF-IDF keyword extraction.
          </p>
        </div>
        <div className="relative">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Filter clusters or terms..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-9 pr-4 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 w-64"
          />
        </div>
      </div>

      {/* 2D Semantic Projection Scatter Visualizer */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-2">
          <div>
            <h3 className="text-sm font-semibold text-white">2D Semantic Projection Space (PCA / TruncatedSVD)</h3>
            <p className="text-xs text-slate-400">
              Clusters positioned by semantic similarity. Bubble size corresponds to feedback volume.
            </p>
          </div>
          <span className="text-[11px] text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20 font-mono">
            {issues.length} Discovered Clusters
          </span>
        </div>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <XAxis type="number" dataKey="x" name="Semantic Dim 1" stroke="#475569" tick={{ fontSize: 10 }} />
              <YAxis type="number" dataKey="y" name="Semantic Dim 2" stroke="#475569" tick={{ fontSize: 10 }} />
              <ZAxis type="number" dataKey="z" range={[80, 500]} name="Volume" />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                content={({ payload }) => {
                  if (payload && payload.length) {
                    const item = payload[0].payload;
                    return (
                      <div className="bg-slate-950 border border-slate-800 p-2.5 rounded-lg shadow-xl text-xs">
                        <p className="font-bold text-white">{item.name}</p>
                        <p className="text-indigo-400 mt-1">Aspect: {item.aspect}</p>
                        <p className="text-slate-400">Volume: {item.z} records</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Scatter
                name="Issue Clusters"
                data={scatterData}
                onClick={(node: any) => handleSelectIssue(node.id)}
                className="cursor-pointer"
              >
                {scatterData.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Main Grid: Clusters List + Detail Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Issue Cards Column */}
        <div className="lg:col-span-6 space-y-3">
          <h3 className="text-sm font-semibold text-white">Discovered Semantic Clusters</h3>
          {filteredIssues.map((iss, index) => {
            const isSelected = selectedIssue?.issue_id === iss.issue_id;
            return (
              <div
                key={iss.issue_id}
                onClick={() => handleSelectIssue(iss.issue_id)}
                className={`p-4 rounded-xl border transition-all cursor-pointer ${
                  isSelected
                    ? 'bg-slate-900 border-indigo-500 ring-1 ring-indigo-500 shadow-md shadow-indigo-500/10'
                    : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: COLORS[index % COLORS.length] }}
                    ></span>
                    <span className="text-[11px] font-mono text-slate-400">Cluster #{iss.issue_id}</span>
                  </div>
                  <span className="text-xs font-semibold px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                    {iss.cluster_size} records ({iss.percentage}%)
                  </span>
                </div>

                <h4 className="text-sm font-bold text-white mt-2">{iss.name}</h4>

                {/* Keywords */}
                <div className="flex flex-wrap gap-1.5 mt-2.5">
                  {iss.top_terms.slice(0, 5).map((term, tIdx) => (
                    <span
                      key={tIdx}
                      className="text-[10px] bg-slate-800 text-slate-300 px-2 py-0.5 rounded border border-slate-700/60 font-mono"
                    >
                      {term}
                    </span>
                  ))}
                </div>

                {/* Sentiment Bar */}
                <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
                  <span className="flex items-center gap-1 text-red-400">
                    <TrendingDown className="w-3.5 h-3.5" />
                    {iss.sentiment_distribution.negative}% Negative
                  </span>
                  <span className="text-indigo-400 flex items-center gap-1 font-medium">
                    View Exemplars
                    <ChevronRight className="w-3.5 h-3.5" />
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Selected Issue Detail & Exemplars */}
        <div className="lg:col-span-6">
          {selectedIssue ? (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-5 sticky top-6">
              <div className="flex items-start justify-between">
                <div>
                  <span className="text-xs font-mono text-indigo-400 uppercase tracking-wider font-semibold">
                    Cluster Details • #{selectedIssue.issue_id}
                  </span>
                  <h3 className="text-lg font-bold text-white mt-1">{selectedIssue.name}</h3>
                </div>
                <button
                  onClick={() => onInvestigateIssue(selectedIssue.issue_id)}
                  className="px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center gap-1.5 shadow-md shadow-indigo-600/30 transition-all"
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                  Root Cause Audit
                </button>
              </div>

              {/* c-TF-IDF Distinctive Terms */}
              <div>
                <span className="text-xs text-slate-400 font-semibold uppercase">c-TF-IDF Distinctive Keywords:</span>
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {selectedIssue.top_terms.map((term, i) => (
                    <span
                      key={i}
                      className="text-xs bg-indigo-950/60 text-indigo-300 border border-indigo-800/50 px-2.5 py-1 rounded-md font-mono"
                    >
                      {term}
                    </span>
                  ))}
                </div>
              </div>

              {/* Representative Customer Feedback Statements */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-slate-400 font-semibold uppercase flex items-center gap-1.5">
                    <MessageSquare className="w-3.5 h-3.5 text-indigo-400" />
                    Central Exemplar Feedback (Closest to Centroid)
                  </span>
                </div>
                <div className="space-y-2.5">
                  {selectedIssue.exemplars.map((ex, i) => (
                    <div
                      key={i}
                      className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 leading-relaxed"
                    >
                      <p className="italic">"{ex.text}"</p>
                      <div className="mt-2 flex items-center justify-between text-[11px] text-slate-500 font-mono">
                        <span>Record #{ex.index}</span>
                        <span className="text-indigo-400">Cosine Sim to Center: {ex.similarity}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 text-center text-slate-400 text-sm">
              Select an issue cluster to inspect representative feedback.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
