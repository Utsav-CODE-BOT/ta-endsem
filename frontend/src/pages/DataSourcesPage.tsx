import React, { useState } from 'react';
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
      {/* Header */}
      <div>
        <h2 style={{ margin: 0, fontSize: '22px', fontWeight: 700, color: 'var(--text-main)' }}>
          Data Ingestion & Dataset Selection
        </h2>
        <p style={{ margin: '5px 0 0', fontSize: '13px', color: 'var(--text-muted)' }}>
          Select standard open research datasets or upload custom customer feedback for automated ingestion audit.
        </p>
      </div>

      {/* Dataset Presets */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '14px' }}>
        {presets.map((p) => {
          const isSelected = selectedPreset === p.id;
          return (
            <div
              key={p.id}
              onClick={() => !isProcessing && handleSelectPreset(p.id)}
              style={{
                background: isSelected ? 'var(--paper)' : 'var(--cream)',
                border: isSelected ? '2px solid var(--teal)' : '1px solid var(--border)',
                borderRadius: '6px',
                padding: '18px 20px',
                cursor: isProcessing ? 'not-allowed' : 'pointer',
                transition: 'all 0.15s ease',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
              }}
            >
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <span
                    style={{
                      fontSize: '9.5px',
                      fontWeight: 700,
                      textTransform: 'uppercase',
                      letterSpacing: '0.06em',
                      padding: '2px 7px',
                      borderRadius: '3px',
                      background: p.type === 'REAL BENCHMARK' ? 'var(--green-bg)' : 'var(--amber-bg)',
                      color: p.type === 'REAL BENCHMARK' ? 'var(--green)' : 'var(--amber)',
                      border: `1px solid ${p.type === 'REAL BENCHMARK' ? '#b2d9c5' : '#f0d28a'}`,
                    }}
                  >
                    {p.type}
                  </span>
                  <span style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-light)' }}>
                    {p.records} records
                  </span>
                </div>
                <h3 style={{ margin: '4px 0 6px', fontSize: '15px', fontWeight: 600, color: 'var(--text-main)' }}>
                  {p.name}
                </h3>
                <p style={{ margin: 0, fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                  {p.description}
                </p>
              </div>

              <div style={{ marginTop: '14px', paddingTop: '10px', borderTop: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '12px' }}>
                <span style={{ color: isSelected ? 'var(--teal)' : 'var(--text-muted)', fontWeight: isSelected ? 600 : 400 }}>
                  {isSelected ? '● Currently Loaded' : 'Click to Load & Analyze →'}
                </span>
                {isSelected && (
                  <span style={{ color: 'var(--green)', fontWeight: 700 }}>Active</span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Upload Custom File */}
      <div style={{
        background: 'var(--paper)',
        border: '1px solid var(--border)',
        borderRadius: '6px',
        padding: '22px',
      }}>
        <h3 style={{ margin: '0 0 4px', fontSize: '15px', fontWeight: 600, color: 'var(--text-main)' }}>
          Upload Custom Customer Text Dataset
        </h3>
        <p style={{ margin: '0 0 16px', fontSize: '12.5px', color: 'var(--text-muted)' }}>
          Accepts CSV, JSON, or Excel. Automatic schema detection identifies text, timestamps, sources, and sentiment.
        </p>

        <label style={{
          border: '2px dashed var(--border-md)',
          borderRadius: '6px',
          padding: '28px 20px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: isProcessing ? 'not-allowed' : 'pointer',
          background: 'var(--cream)',
          transition: 'border-color 0.15s ease',
        }}>
          <span style={{ fontSize: '24px', marginBottom: '6px', color: 'var(--teal)' }}>↑</span>
          <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-main)' }}>
            Select or drop file to upload
          </span>
          <span style={{ fontSize: '11px', color: 'var(--text-light)', marginTop: '3px' }}>
            Supports UTF-8 CSV, JSON, JSONL, XLSX (Max 50MB)
          </span>
          <input
            type="file"
            accept=".csv,.json,.jsonl,.xlsx,.xls"
            onChange={handleFileUpload}
            disabled={isProcessing}
            style={{ display: 'none' }}
          />
        </label>

        {isProcessing && (
          <div style={{
            marginTop: '14px',
            padding: '12px 16px',
            borderRadius: '4px',
            background: 'var(--teal-light)',
            border: '1px solid #9ecfcf',
            color: 'var(--teal)',
            fontSize: '12px',
            fontFamily: 'monospace',
          }}>
            ⧗ Ingestion in progress: Cleaning • Embeddings • CardiffNLP Sentiment • Aspects • Semantic Clustering…
          </div>
        )}

        {uploadStatus && (
          <div style={{
            marginTop: '14px',
            padding: '10px 14px',
            borderRadius: '4px',
            background: uploadStatus.startsWith('Error') ? 'var(--red-bg)' : 'var(--green-bg)',
            border: `1px solid ${uploadStatus.startsWith('Error') ? '#f5a89e' : '#b2d9c5'}`,
            color: uploadStatus.startsWith('Error') ? 'var(--red)' : 'var(--green)',
            fontSize: '12px',
            fontWeight: 500,
          }}>
            {uploadStatus}
          </div>
        )}
      </div>

      {/* Ingestion Audit Log */}
      {lastStats && (
        <div style={{
          background: 'var(--paper)',
          border: '1px solid var(--border)',
          borderRadius: '6px',
          padding: '20px',
        }}>
          <h3 style={{ margin: '0 0 14px', fontSize: '14px', fontWeight: 600, color: 'var(--text-main)', borderBottom: '1px solid var(--border)', paddingBottom: '8px' }}>
            Data Ingestion & Quality Audit (Section 8 Verification)
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '10px' }}>
            <div style={{ background: 'var(--cream)', border: '1px solid var(--border)', borderRadius: '4px', padding: '12px' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Uploaded</div>
              <div style={{ fontSize: '18px', fontWeight: 700, fontFamily: 'monospace', color: 'var(--text-main)', marginTop: '4px' }}>
                {lastStats.records_uploaded}
              </div>
            </div>
            <div style={{ background: 'var(--cream)', border: '1px solid var(--border)', borderRadius: '4px', padding: '12px' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Valid Records</div>
              <div style={{ fontSize: '18px', fontWeight: 700, fontFamily: 'monospace', color: 'var(--green)', marginTop: '4px' }}>
                {lastStats.valid_records}
              </div>
            </div>
            <div style={{ background: 'var(--cream)', border: '1px solid var(--border)', borderRadius: '4px', padding: '12px' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Duplicates Removed</div>
              <div style={{ fontSize: '18px', fontWeight: 700, fontFamily: 'monospace', color: 'var(--amber)', marginTop: '4px' }}>
                {lastStats.duplicates_removed}
              </div>
            </div>
            <div style={{ background: 'var(--cream)', border: '1px solid var(--border)', borderRadius: '4px', padding: '12px' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Missing Timestamps</div>
              <div style={{ fontSize: '18px', fontWeight: 700, fontFamily: 'monospace', color: 'var(--text-main)', marginTop: '4px' }}>
                {lastStats.missing_timestamps}
              </div>
            </div>
          </div>

          <div style={{ marginTop: '14px', paddingTop: '10px', borderTop: '1px solid var(--border)', display: 'flex', flexWrap: 'wrap', gap: '20px', fontSize: '11.5px', color: 'var(--text-muted)' }}>
            <div><strong>Sources detected:</strong> {lastStats.sources_detected?.join(', ') || 'N/A'}</div>
            <div><strong>Languages detected:</strong> {lastStats.languages_detected?.join(', ') || 'en'}</div>
          </div>
        </div>
      )}
    </div>
  );
};
