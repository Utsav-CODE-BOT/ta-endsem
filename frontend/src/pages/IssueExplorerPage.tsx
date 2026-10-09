import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { IssueClusterItem, IssueDetailResponse } from '../types';
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, Tooltip, Cell, ZAxis } from 'recharts';

interface IssueExplorerProps {
  onInvestigateIssue: (issueId: number) => void;
}

const COLORS = ['#0d7377', '#c07c2a', '#c0392b', '#1e6b4a', '#5b4a8a', '#2b5c8f'];

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

  const scatterData = issues.map((iss) => ({
    x: iss.center_2d[0],
    y: iss.center_2d[1],
    z: iss.cluster_size,
    name: iss.name,
    id: iss.issue_id,
    aspect: iss.primary_aspect,
  }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
      {/* Header & Filter */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '22px', fontWeight: 700, color: 'var(--text-main)' }}>
            Semantic Issue Discovery Explorer
          </h2>
          <p style={{ margin: '5px 0 0', fontSize: '13px', color: 'var(--text-muted)' }}>
            Unsupervised semantic clustering via Sentence-BERT embeddings, dimensionality projection, and c-TF-IDF keyword extraction.
          </p>
        </div>

        <input
          type="text"
          placeholder="Filter clusters or terms…"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          style={{
            padding: '7px 12px',
            background: 'var(--paper)',
            border: '1px solid var(--border)',
            borderRadius: '4px',
            fontSize: '12px',
            color: 'var(--text-main)',
            width: '240px',
            outline: 'none',
          }}
          onFocus={(e) => (e.currentTarget.style.borderColor = 'var(--teal)')}
          onBlur={(e) => (e.currentTarget.style.borderColor = 'var(--border)')}
        />
      </div>

      {/* 2D Semantic Projection Scatter */}
      <div style={{
        background: 'var(--paper)',
        border: '1px solid var(--border)',
        borderRadius: '6px',
        padding: '20px',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: 'var(--text-main)' }}>
              2D Semantic Projection Space (PCA / TruncatedSVD)
            </h3>
            <p style={{ margin: '2px 0 0', fontSize: '11.5px', color: 'var(--text-muted)' }}>
              Clusters positioned by semantic similarity. Bubble size corresponds to feedback volume.
            </p>
          </div>
          <span style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--teal)', background: 'var(--teal-light)', padding: '3px 8px', borderRadius: '4px', border: '1px solid #9ecfcf' }}>
            {issues.length} Clusters
          </span>
        </div>

        <div style={{ height: '240px', width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 15, right: 15, bottom: 15, left: 15 }}>
              <XAxis type="number" dataKey="x" name="Semantic Dim 1" stroke="var(--border-md)" tick={{ fontSize: 10, fill: 'var(--text-light)' }} tickLine={false} axisLine={false} />
              <YAxis type="number" dataKey="y" name="Semantic Dim 2" stroke="var(--border-md)" tick={{ fontSize: 10, fill: 'var(--text-light)' }} tickLine={false} axisLine={false} />
              <ZAxis type="number" dataKey="z" range={[90, 480]} name="Volume" />
              <Tooltip
                cursor={{ strokeDasharray: '3 3', stroke: 'var(--border-md)' }}
                content={({ payload }) => {
                  if (payload && payload.length) {
                    const item = payload[0].payload;
                    return (
                      <div style={{
                        background: 'var(--paper)',
                        border: '1px solid var(--border)',
                        padding: '10px 12px',
                        borderRadius: '4px',
                        fontSize: '12px',
                        color: 'var(--text-main)',
                      }}>
                        <div style={{ fontWeight: 700 }}>{item.name}</div>
                        <div style={{ color: 'var(--teal)', fontSize: '11px', marginTop: '2px' }}>Aspect: {item.aspect}</div>
                        <div style={{ color: 'var(--text-muted)', fontSize: '11px' }}>Volume: {item.z} records</div>
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
                cursor="pointer"
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
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', alignItems: 'flex-start' }}>
        {/* Left: Clusters List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-light)' }}>
            Discovered Clusters ({filteredIssues.length})
          </div>

          {filteredIssues.map((iss, index) => {
            const isSelected = selectedIssue?.issue_id === iss.issue_id;
            const dotColor = COLORS[index % COLORS.length];
            return (
              <div
                key={iss.issue_id}
                onClick={() => handleSelectIssue(iss.issue_id)}
                style={{
                  background: isSelected ? 'var(--paper)' : 'var(--cream)',
                  border: isSelected ? '2px solid var(--teal)' : '1px solid var(--border)',
                  borderRadius: '6px',
                  padding: '14px 16px',
                  cursor: 'pointer',
                  transition: 'all 0.12s ease',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: '9px', height: '9px', borderRadius: '50%', background: dotColor }} />
                    <span style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-light)' }}>
                      Cluster #{iss.issue_id}
                    </span>
                  </div>
                  <span style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-muted)' }}>
                    {iss.cluster_size} records ({iss.percentage}%)
                  </span>
                </div>

                <h4 style={{ margin: '8px 0 6px', fontSize: '14px', fontWeight: 600, color: 'var(--text-main)' }}>
                  {iss.name}
                </h4>

                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', margin: '6px 0 8px' }}>
                  {iss.top_terms.slice(0, 5).map((term, tIdx) => (
                    <span
                      key={tIdx}
                      style={{
                        fontSize: '10px',
                        fontFamily: 'monospace',
                        background: 'var(--paper)',
                        border: '1px solid var(--border)',
                        borderRadius: '3px',
                        padding: '1px 6px',
                        color: 'var(--text-muted)',
                      }}
                    >
                      {term}
                    </span>
                  ))}
                </div>

                <div style={{ paddingTop: '8px', borderTop: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
                  <span style={{ color: 'var(--red)', fontWeight: 500 }}>
                    {iss.sentiment_distribution.negative}% Negative
                  </span>
                  <span style={{ color: 'var(--teal)', fontWeight: 600 }}>
                    Inspect →
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Right: Selected Detail & Exemplars */}
        <div>
          {selectedIssue ? (
            <div style={{
              background: 'var(--paper)',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              padding: '20px',
              position: 'sticky',
              top: '72px',
              display: 'flex',
              flexDirection: 'column',
              gap: '16px',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '10px' }}>
                <div>
                  <span style={{ fontSize: '10px', fontFamily: 'monospace', color: 'var(--teal)', fontWeight: 700, textTransform: 'uppercase' }}>
                    Cluster #{selectedIssue.issue_id}
                  </span>
                  <h3 style={{ margin: '4px 0 0', fontSize: '16px', fontWeight: 700, color: 'var(--text-main)' }}>
                    {selectedIssue.name}
                  </h3>
                </div>
                <button
                  onClick={() => onInvestigateIssue(selectedIssue.issue_id)}
                  style={{
                    padding: '6px 12px',
                    borderRadius: '4px',
                    border: '1px solid var(--teal)',
                    background: 'var(--teal)',
                    color: '#fff',
                    fontSize: '12px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    whiteSpace: 'nowrap',
                  }}
                >
                  Root Cause Audit →
                </button>
              </div>

              {/* Distinctive Keywords */}
              <div>
                <div style={{ fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-light)', marginBottom: '6px' }}>
                  c-TF-IDF Distinctive Keywords
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
                  {selectedIssue.top_terms.map((term, i) => (
                    <span
                      key={i}
                      style={{
                        fontSize: '11px',
                        fontFamily: 'monospace',
                        background: 'var(--teal-light)',
                        color: 'var(--teal)',
                        border: '1px solid #9ecfcf',
                        borderRadius: '3px',
                        padding: '2px 8px',
                        fontWeight: 500,
                      }}
                    >
                      {term}
                    </span>
                  ))}
                </div>
              </div>

              {/* Exemplar Feedback */}
              <div>
                <div style={{ fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-light)', marginBottom: '8px' }}>
                  Central Exemplar Feedback (Closest to Centroid)
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {selectedIssue.exemplars.map((ex, i) => (
                    <div
                      key={i}
                      style={{
                        background: 'var(--cream)',
                        border: '1px solid var(--border)',
                        borderRadius: '4px',
                        padding: '10px 12px',
                        fontSize: '12px',
                        lineHeight: 1.6,
                      }}
                    >
                      <p style={{ margin: 0, fontStyle: 'italic', color: 'var(--text-main)' }}>
                        "{ex.text}"
                      </p>
                      <div style={{ marginTop: '6px', paddingTop: '5px', borderTop: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', fontSize: '10.5px', fontFamily: 'monospace', color: 'var(--text-light)' }}>
                        <span>Record #{ex.index}</span>
                        <span style={{ color: 'var(--teal)', fontWeight: 600 }}>Sim: {ex.similarity}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div style={{ background: 'var(--paper)', border: '1px solid var(--border)', borderRadius: '6px', padding: '32px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
              Select an issue cluster to inspect representative feedback.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
