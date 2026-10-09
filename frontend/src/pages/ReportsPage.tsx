import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { ReportResponse } from '../types';

export const ReportsPage: React.FC = () => {
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.generateReport()
      .then(setReport)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const handlePrint = () => {
    window.print();
  };

  const handleDownloadJSON = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `VoiceMiner_Audit_Report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loading) return <div style={{ padding: '48px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>Compiling intelligence audit report…</div>;
  if (!report) return <div style={{ padding: '32px', color: 'var(--red)' }}>Failed to generate report.</div>;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
      {/* Top action bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '22px', fontWeight: 700, color: 'var(--text-main)' }}>
            Intelligence Audit & Evaluation Report
          </h2>
          <p style={{ margin: '4px 0 0', fontSize: '12px', fontFamily: 'monospace', color: 'var(--text-light)' }}>
            Generated at: {report.generated_at}
          </p>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={handleDownloadJSON}
            style={{
              padding: '7px 14px',
              borderRadius: '4px',
              border: '1px solid var(--border)',
              background: 'var(--paper)',
              color: 'var(--text-main)',
              fontSize: '12px',
              fontWeight: 500,
              cursor: 'pointer',
            }}
          >
            Download JSON
          </button>
          <button
            onClick={handlePrint}
            style={{
              padding: '7px 16px',
              borderRadius: '4px',
              border: '1px solid var(--teal)',
              background: 'var(--teal)',
              color: '#fff',
              fontSize: '12px',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Print Report
          </button>
        </div>
      </div>

      {/* Main Printable Document Card */}
      <div style={{
        background: 'var(--paper)',
        border: '1px solid var(--border)',
        borderRadius: '6px',
        padding: '36px 40px',
        display: 'flex',
        flexDirection: 'column',
        gap: '28px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
      }}>
        {/* Document Header */}
        <div style={{ borderBottom: '2px solid var(--text-main)', paddingBottom: '16px' }}>
          <div style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700, color: 'var(--teal)' }}>
            VoiceMiner Research Platform • Technical Audit
          </div>
          <h1 style={{ margin: '8px 0 4px', fontSize: '24px', fontWeight: 700, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
            Customer Voice Intelligence & Emerging Issue Report
          </h1>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Systematic NLP pipeline synthesis across multi-source textual streams
          </div>
        </div>

        {/* Section 1: Executive Summary */}
        <div>
          <h3 style={{ margin: '0 0 8px', fontSize: '14px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-main)' }}>
            1. Executive Summary
          </h3>
          <p style={{ margin: 0, fontSize: '13.5px', lineHeight: 1.7, color: 'var(--text-main)' }}>
            {report.executive_summary}
          </p>
        </div>

        {/* Section 2: Prioritized Emerging Issues */}
        <div>
          <h3 style={{ margin: '0 0 10px', fontSize: '14px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-main)' }}>
            2. Prioritized Emerging Issues Summary
          </h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1.5px solid var(--border-md)', color: 'var(--text-muted)', fontFamily: 'monospace', fontSize: '11px' }}>
                  <th style={{ padding: '8px 10px' }}>Issue Cluster</th>
                  <th style={{ padding: '8px 10px' }}>Emerging Score</th>
                  <th style={{ padding: '8px 10px' }}>Priority</th>
                  <th style={{ padding: '8px 10px' }}>Growth Rate</th>
                  <th style={{ padding: '8px 10px' }}>Negative Ratio</th>
                </tr>
              </thead>
              <tbody>
                {report.emerging_issues_summary.map((iss, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '10px', fontWeight: 600, color: 'var(--text-main)' }}>{iss.issue_name}</td>
                    <td style={{ padding: '10px', fontFamily: 'monospace', fontWeight: 700, color: 'var(--amber)' }}>{iss.emerging_score.toFixed(3)}</td>
                    <td style={{ padding: '10px' }}>
                      <span style={{
                        padding: '2px 6px',
                        borderRadius: '3px',
                        fontSize: '9.5px',
                        fontWeight: 700,
                        textTransform: 'uppercase',
                        background: iss.priority === 'CRITICAL' ? 'var(--red-bg)' : 'var(--amber-bg)',
                        color: iss.priority === 'CRITICAL' ? 'var(--red)' : 'var(--amber)',
                        border: `1px solid ${iss.priority === 'CRITICAL' ? '#f5a89e' : '#f0d28a'}`,
                      }}>
                        {iss.priority}
                      </span>
                    </td>
                    <td style={{ padding: '10px', fontFamily: 'monospace', color: 'var(--red)', fontWeight: 600 }}>+{iss.growth_rate_pct}%</td>
                    <td style={{ padding: '10px', fontFamily: 'monospace' }}>{Math.round(iss.negative_sentiment_ratio * 100)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Section 3: Candidate Root Causes */}
        <div>
          <h3 style={{ margin: '0 0 10px', fontSize: '14px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-main)' }}>
            3. Candidate Root Causes & Empirical Associations
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {report.candidate_root_causes.map((c, i) => (
              <div key={i} style={{
                background: 'var(--cream)',
                border: '1px solid var(--border)',
                borderRadius: '4px',
                padding: '12px 14px',
                fontSize: '12px',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                  <span style={{ fontWeight: 700, fontSize: '13px', color: 'var(--text-main)' }}>{c.candidate_factor}</span>
                  <span style={{ fontFamily: 'monospace', color: 'var(--teal)', fontWeight: 600 }}>
                    Confidence: {Math.round(c.confidence * 100)}% · Lift: {c.lift}×
                  </span>
                </div>
                <div style={{ fontSize: '11px', color: 'var(--text-light)', marginBottom: '6px' }}>
                  Associated with: <strong style={{ color: 'var(--text-muted)' }}>{c.associated_issue}</strong>
                </div>
                <p style={{ margin: 0, fontStyle: 'italic', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                  "{c.interpretation}"
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Section 4: Recommended Investigation Areas */}
        <div>
          <h3 style={{ margin: '0 0 10px', fontSize: '14px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-main)' }}>
            4. Recommended Investigation Areas
          </h3>
          <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px', lineHeight: 1.8, color: 'var(--text-main)' }}>
            {report.recommended_investigation_areas.map((rec, i) => (
              <li key={i}>{rec}</li>
            ))}
          </ul>
        </div>

        {/* Section 5: Methodological Limitations */}
        <div style={{
          background: 'var(--cream)',
          border: '1px solid var(--border)',
          borderRadius: '4px',
          padding: '16px 18px',
        }}>
          <h3 style={{ margin: '0 0 8px', fontSize: '12.5px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#7a5a10' }}>
            5. Methodological & Scientific Limitations (Section 3 Restriction)
          </h3>
          <ul style={{ margin: 0, paddingLeft: '18px', fontSize: '12px', lineHeight: 1.7, color: 'var(--text-muted)' }}>
            {report.limitations.map((lim, i) => (
              <li key={i}>{lim}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};
