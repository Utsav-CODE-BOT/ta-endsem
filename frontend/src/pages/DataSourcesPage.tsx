import React, { useState } from 'react';
import {
  Upload,
  CheckCircle,
  AlertCircle,
  FileSpreadsheet,
  Database,
  Layers,
  ArrowRight,
  RefreshCw,
} from 'lucide-react';
import { api } from '../services/api';

interface DataSourcesPageProps {
  onDatasetReloaded: () => void;
}

export const DataSourcesPage: React.FC<DataSourcesPageProps> = ({ onDatasetReloaded }) => {
  const [selectedPreset, setSelectedPreset] = useState<string>('airline_sample_1000.csv');
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [lastStats, setLastStats] = useState<any>(null);

  const presets = [
    {
      id: 'airline_sample_1000.csv',
      name: 'US Airline Twitter Sentiment & Operations (Research Benchmark)',
      records: 1000,
      type: 'REAL BENCHMARK',
      description: 'Peer-reviewed research dataset with real timestamps, multi-airline entities, flight operations, and negative reason ground truth.',
    },
    {
      id: 'ecommerce_saas_feedback_demo.csv',
      name: 'Digital SaaS & Mobile App Feedback (Cross-Domain Scenario)',
      records: 500,
      type: 'DEMO / SYNTHETIC DATA',
      description: 'Multi-domain software scenario with simulated v4.2 mobile updates, payment gateway 502 timeouts, and auth crashes.',
    },
  ];

  const handleSelectPreset = async (presetId: string) => {
    setSelectedPreset(presetId);
    setIsProcessing(true);
    setUploadStatus(null);
    try {
      const res = await api.triggerAnalysis(presetId);
      setLastStats(res.ingestion_stats);
      setUploadStatus(`Successfully switched and analyzed dataset: ${presetId}`);
      onDatasetReloaded();
    } catch (err: any) {
      setUploadStatus(`Error: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsProcessing(true);
    setUploadStatus(null);
    try {
      const stats = await api.uploadDataset(file);
      setLastStats(stats);
      setUploadStatus(`Successfully ingested ${file.name} (${stats.valid_records} valid records).`);
      onDatasetReloaded();
    } catch (err: any) {
      setUploadStatus(`Upload error: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight">Data Ingestion & Dataset Selection</h2>
        <p className="text-slate-400 text-sm mt-0.5">
          Select standard open research datasets or upload custom CSV/JSON multi-source feedback for automated ingestion audit.
        </p>
      </div>

      {/* Dataset Presets */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {presets.map((p) => {
          const isSelected = selectedPreset === p.id;
          return (
            <div
              key={p.id}
              onClick={() => !isProcessing && handleSelectPreset(p.id)}
              className={`p-5 rounded-xl border transition-all cursor-pointer ${
                isSelected
                  ? 'bg-slate-900 border-indigo-500 shadow-md shadow-indigo-500/10 ring-1 ring-indigo-500'
                  : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <span
                  className={`text-[10px] font-bold px-2 py-0.5 rounded border uppercase ${
                    p.type === 'REAL BENCHMARK'
                      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                      : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                  }`}
                >
                  {p.type}
                </span>
                <span className="text-xs text-slate-400 font-mono">{p.records} records</span>
              </div>
              <h3 className="text-base font-bold text-white mt-2.5">{p.name}</h3>
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">{p.description}</p>
              <div className="mt-4 flex items-center justify-between text-xs">
                <span className="text-indigo-400 font-medium flex items-center gap-1">
                  {isSelected ? 'Currently Loaded' : 'Click to Load & Analyze'}
                  <ArrowRight className="w-3.5 h-3.5" />
                </span>
                {isSelected && <CheckCircle className="w-4 h-4 text-emerald-400" />}
              </div>
            </div>
          );
        })}
      </div>

      {/* Upload Custom File */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h3 className="text-base font-semibold text-white mb-1">Upload Custom Customer Text Dataset</h3>
        <p className="text-xs text-slate-400 mb-4">
          Accepts CSV, JSON, or Excel. Automatic schema detection identifies text, timestamps, sources, and sentiment.
        </p>

        <label className="border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-colors bg-slate-950/40">
          <Upload className="w-8 h-8 text-indigo-400 mb-2" />
          <span className="text-sm font-semibold text-white">Click or drag file to upload</span>
          <span className="text-xs text-slate-500 mt-1">Supports UTF-8 CSV, JSON, JSONL, XLSX (Max 50MB)</span>
          <input
            type="file"
            accept=".csv,.json,.jsonl,.xlsx,.xls"
            onChange={handleFileUpload}
            disabled={isProcessing}
            className="hidden"
          />
        </label>

        {isProcessing && (
          <div className="mt-4 p-4 rounded-lg bg-indigo-950/40 border border-indigo-800/40 flex items-center gap-3">
            <RefreshCw className="w-5 h-5 text-indigo-400 animate-spin" />
            <div className="text-xs text-indigo-200">
              Processing dataset: Cleaning • Embeddings • Transformer Sentiment • Aspect Discovery • Clustering...
            </div>
          </div>
        )}

        {uploadStatus && (
          <div className="mt-4 p-4 rounded-lg bg-slate-800 border border-slate-700 text-xs text-slate-200 flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-emerald-400" />
            <span>{uploadStatus}</span>
          </div>
        )}
      </div>

      {/* Ingestion Audit Log */}
      {lastStats && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
          <h3 className="text-base font-semibold text-white mb-4 flex items-center gap-2">
            <FileSpreadsheet className="w-4 h-4 text-indigo-400" />
            Data Ingestion & Quality Audit (Section 8 Verification)
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <div className="text-xs text-slate-400">Records Uploaded</div>
              <div className="text-xl font-bold text-white mt-1">{lastStats.records_uploaded}</div>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <div className="text-xs text-slate-400">Valid Records</div>
              <div className="text-xl font-bold text-emerald-400 mt-1">{lastStats.valid_records}</div>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <div className="text-xs text-slate-400">Duplicates Removed</div>
              <div className="text-xl font-bold text-amber-400 mt-1">{lastStats.duplicates_removed}</div>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <div className="text-xs text-slate-400">Missing Timestamps</div>
              <div className="text-xl font-bold text-slate-300 mt-1">{lastStats.missing_timestamps}</div>
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-slate-800 flex flex-wrap gap-4 text-xs text-slate-400">
            <div>
              <strong>Sources Detected:</strong> {lastStats.sources_detected?.join(', ') || 'N/A'}
            </div>
            <div>
              <strong>Languages Detected:</strong> {lastStats.languages_detected?.join(', ') || 'en'}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
