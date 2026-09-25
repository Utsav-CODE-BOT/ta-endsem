import React, { useEffect, useState } from 'react';
import { FileText, Download, Printer, ShieldCheck, AlertTriangle, CheckCircle2 } from 'lucide-react';
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

  if (loading) return <div className="p-12 text-center text-slate-400 text-sm">Compiling intelligence audit report...</div>;
  if (!report) return <div className="p-8 text-slate-400">Failed to generate report.</div>;

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Header with actions */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
            <FileText className="w-6 h-6 text-indigo-400" />
            Executive Intelligence & Root-Cause Audit Report
          </h2>
          <p className="text-slate-400 text-xs mt-0.5 font-mono">Generated: {report.generated_at}</p>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={handleDownloadJSON}
            className="px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold flex items-center gap-2 border border-slate-700 transition-all"
          >
            <Download className="w-4 h-4" />
            <span>Download JSON</span>
          </button>
          <button
            onClick={handlePrint}
            className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center gap-2 shadow-md shadow-indigo-600/30 transition-all"
          >
            <Printer className="w-4 h-4" />
            <span>Print Report</span>
          </button>
        </div>
      </div>

      {/* Main Printable Document Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 space-y-6 shadow-sm">
        {/* Executive Summary */}
        <div className="border-b border-slate-800 pb-5">
          <h3 className="text-sm font-bold text-indigo-400 uppercase tracking-wider mb-2">1. Executive Summary</h3>
          <p className="text-sm text-slate-200 leading-relaxed font-sans">{report.executive_summary}</p>
        </div>

        {/* Prioritized Emerging Issues */}
        <div className="border-b border-slate-800 pb-5">
          <h3 className="text-sm font-bold text-amber-400 uppercase tracking-wider mb-3">
            2. Prioritized Emerging Issues Summary
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950 text-slate-400 uppercase font-mono border-b border-slate-800">
                <tr>
                  <th className="p-3">Issue Cluster</th>
                  <th className="p-3">Emerging Score</th>
                  <th className="p-3">Priority</th>
                  <th className="p-3">Growth Rate</th>
                  <th className="p-3">Negative Ratio</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 text-slate-300">
                {report.emerging_issues_summary.map((iss, i) => (
                  <tr key={i}>
                    <td className="p-3 font-semibold text-white">{iss.issue_name}</td>
                    <td className="p-3 font-mono text-amber-400 font-bold">{iss.emerging_score.toFixed(3)}</td>
                    <td className="p-3">
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-red-500/10 text-red-400 border border-red-500/20 uppercase">
                        {iss.priority}
                      </span>
                    </td>
                    <td className="p-3 font-mono text-red-400">+{iss.growth_rate_pct}%</td>
                    <td className="p-3 font-mono">{Math.round(iss.negative_sentiment_ratio * 100)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Candidate Root Causes */}
        <div className="border-b border-slate-800 pb-5">
          <h3 className="text-sm font-bold text-indigo-400 uppercase tracking-wider mb-3">
            3. Candidate Root Causes & Empirical Associations
          </h3>
          <div className="space-y-3">
            {report.candidate_root_causes.map((c, i) => (
              <div key={i} className="p-4 bg-slate-950 rounded-lg border border-slate-800 text-xs">
                <div className="flex items-center justify-between font-mono mb-1">
                  <span className="font-bold text-white text-sm">{c.candidate_factor}</span>
                  <span className="text-indigo-400">Confidence: {Math.round(c.confidence * 100)}% (Lift: {c.lift}x)</span>
                </div>
                <div className="text-slate-400 mb-2">Target Issue: <strong>{c.associated_issue}</strong></div>
                <p className="text-slate-300 italic">"{c.interpretation}"</p>
              </div>
            ))}
          </div>
        </div>

        {/* Recommended Investigation Areas */}
        <div className="border-b border-slate-800 pb-5">
          <h3 className="text-sm font-bold text-emerald-400 uppercase tracking-wider mb-2">
            4. Recommended Investigation Areas
          </h3>
          <ul className="space-y-2 text-xs text-slate-300">
            {report.recommended_investigation_areas.map((rec, i) => (
              <li key={i} className="flex items-start space-x-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 shrink-0" />
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Methodological Limitations */}
        <div>
          <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-2">
            5. Methodological & Scientific Limitations (Section 3 Restriction)
          </h3>
          <ul className="space-y-2 text-xs text-slate-400">
            {report.limitations.map((lim, i) => (
              <li key={i} className="flex items-start space-x-2">
                <AlertTriangle className="w-4 h-4 text-amber-400 mt-0.5 shrink-0" />
                <span>{lim}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};
