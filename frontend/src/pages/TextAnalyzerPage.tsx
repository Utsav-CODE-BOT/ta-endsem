import React, { useState } from 'react';
import { api } from '../services/api';
import { TextAnalyzeResponse } from '../types';

const SAMPLES = [
  'Payment failed three times at checkout with error 502 right after the latest app update!',
  'Baggage was lost on flight 492 and customer service representative hung up on me.',
  'Flight was delayed 6 hours on the tarmac with zero water or food provided.',
  'Amazing experience! Boarding was seamless and flight attendants were wonderful.',
];

const SENTIMENT_COLORS: Record<string, string> = {
  negative: 'var(--red)',
  positive: 'var(--green)',
  neutral: 'var(--text-muted)',
};

const Card: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div style={{ background: 'var(--paper)', border: '1px solid var(--border)', borderRadius: '6px', padding: '18px 20px', ...style }}>
    {children}
  </div>
);

const SubLabel: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div style={{ fontSize: '9.5px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-light)', marginBottom: '6px' }}>
    {children}
  </div>
);

export const TextAnalyzerPage: React.FC = () => {
  const [inputText, setInputText] = useState(SAMPLES[0]);
  const [result, setResult] = useState<TextAnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    try {
      const res = await api.analyzeText(inputText);
      setResult(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>

      {/* Header */}
      <div>
        <h2 style={{ margin: 0, fontSize: '22px', fontWeight: 700, color: 'var(--text-main)' }}>
          Real-Time Customer Text Analyzer
        </h2>
        <p style={{ margin: '5px 0 0', fontSize: '13px', color: 'var(--text-muted)' }}>
          Paste arbitrary customer feedback to test live inference across sentiment, 7-emotion spectrum, aspect extraction, semantic cluster matching, and evidence retrieval.
        </p>
      </div>

      {/* Input card */}
      <Card>
        <label style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          Customer Text Input
        </label>
        <textarea
          rows={4}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Paste customer review, complaint, or support message here…"
          style={{
            width: '100%',
            resize: 'vertical',
            background: 'var(--cream)',
            border: '1px solid var(--border)',
            borderRadius: '4px',
            padding: '10px 12px',
            fontSize: '13.5px',
            color: 'var(--text-main)',
            fontFamily: 'Inter, sans-serif',
            lineHeight: 1.6,
            outline: 'none',
            boxSizing: 'border-box',
          }}
          onFocus={(e) => (e.currentTarget.style.borderColor = 'var(--teal)')}
          onBlur={(e) => (e.currentTarget.style.borderColor = 'var(--border)')}
        />

        <div style={{ marginTop: '12px', display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '11.5px', color: 'var(--text-light)' }}>Try a sample:</span>
            {SAMPLES.map((q, idx) => (
              <button
                key={idx}
                onClick={() => setInputText(q)}
                style={{
                  fontSize: '11px',
                  padding: '3px 9px',
                  border: '1px solid var(--border)',
                  borderRadius: '3px',
                  background: 'var(--cream)',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  transition: 'background 0.1s',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = '#ede8e1')}
                onMouseLeave={(e) => (e.currentTarget.style.background = 'var(--cream)')}
              >
                S{idx + 1}
              </button>
            ))}
          </div>

          <button
            onClick={handleAnalyze}
            disabled={loading}
            style={{
              padding: '9px 20px',
              background: loading ? '#9ecfcf' : 'var(--teal)',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              fontSize: '13px',
              fontWeight: 600,
              cursor: loading ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '7px',
              transition: 'background 0.12s',
            }}
            onMouseEnter={(e) => { if (!loading) (e.currentTarget.style.background = '#0a5f62'); }}
            onMouseLeave={(e) => { if (!loading) (e.currentTarget.style.background = 'var(--teal)'); }}
          >
            {loading ? '⧗ Analyzing…' : '⌖ Analyze Customer Voice'}
          </button>
        </div>
      </Card>

      {/* Results */}
      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>

          {/* Row 1: Sentiment / Emotion / Aspects */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
            <Card>
              <SubLabel>Transformer Sentiment</SubLabel>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginTop: '4px' }}>
                <span style={{ fontSize: '24px', fontWeight: 700, textTransform: 'capitalize', color: SENTIMENT_COLORS[result.sentiment.label] || 'var(--text-main)' }}>
                  {result.sentiment.label}
                </span>
                <span style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-light)' }}>
                  {Math.round(result.sentiment.score * 100)}% conf.
                </span>
              </div>
            </Card>

            <Card>
              <SubLabel>Affective Emotion</SubLabel>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginTop: '4px' }}>
                <span style={{ fontSize: '24px', fontWeight: 700, textTransform: 'capitalize', color: 'var(--amber)' }}>
                  {result.emotion.primary_emotion}
                </span>
                <span style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-light)' }}>
                  {Math.round(result.emotion.score * 100)}%
                </span>
              </div>
            </Card>

            <Card>
              <SubLabel>Extracted Aspects</SubLabel>
              <div style={{ marginTop: '6px', display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
                {result.aspects.length > 0 ? result.aspects.map((asp, i) => (
                  <span key={i} style={{
                    fontSize: '11.5px',
                    padding: '2px 8px',
                    borderRadius: '3px',
                    background: 'var(--teal-light)',
                    color: 'var(--teal)',
                    border: '1px solid #9ecfcf',
                    fontFamily: 'monospace',
                    fontWeight: 600,
                  }}>
                    {asp.aspect}
                  </span>
                )) : (
                  <span style={{ fontSize: '12px', color: 'var(--text-light)' }}>General aspect</span>
                )}
              </div>
            </Card>
          </div>

          {/* Row 2: Matched cluster + Candidate cause */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <Card>
              <SubLabel>Matched Semantic Issue Cluster</SubLabel>
              {result.matched_issue ? (
                <div>
                  <h4 style={{ margin: '6px 0 4px', fontSize: '14.5px', fontWeight: 600, color: 'var(--text-main)' }}>
                    {result.matched_issue.name}
                  </h4>
                  <div style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-light)', display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                    <span>Cluster #{result.matched_issue.issue_id}</span>
                    <span>Similarity: {result.matched_issue.similarity}</span>
                    <span>Aspect: {result.matched_issue.primary_aspect}</span>
                  </div>
                </div>
              ) : (
                <p style={{ margin: '6px 0 0', fontSize: '12.5px', color: 'var(--text-light)' }}>No active cluster matched.</p>
              )}
            </Card>

            <Card>
              <SubLabel>Suspected Associated Factor</SubLabel>
              {result.candidate_cause ? (
                <div>
                  <h4 style={{ margin: '6px 0 4px', fontSize: '14.5px', fontWeight: 600, color: 'var(--amber)' }}>
                    {result.candidate_cause.candidate_factor}
                  </h4>
                  <div style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-light)', marginBottom: '8px', display: 'flex', gap: '12px' }}>
                    <span>Confidence: {Math.round(result.candidate_cause.confidence * 100)}%</span>
                    <span>Lift: {result.candidate_cause.lift}×</span>
                  </div>
                  <p style={{ margin: 0, fontSize: '12px', color: 'var(--text-muted)', fontStyle: 'italic', background: 'var(--cream)', padding: '7px 10px', borderRadius: '3px', border: '1px solid var(--border)', lineHeight: 1.6 }}>
                    "{result.candidate_cause.interpretation}"
                  </p>
                </div>
              ) : (
                <p style={{ margin: '6px 0 0', fontSize: '12.5px', color: 'var(--text-light)' }}>No candidate factor found for this cluster.</p>
              )}
            </Card>
          </div>

          {/* Row 3: Similar evidence */}
          {result.similar_evidence.length > 0 && (
            <Card>
              <h3 style={{ margin: '0 0 12px', fontSize: '13.5px', fontWeight: 700, color: 'var(--text-main)' }}>
                Nearest Historical Feedback (Cosine Similarity)
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {result.similar_evidence.map((ev, i) => (
                  <div key={i} style={{
                    background: 'var(--cream)',
                    border: '1px solid var(--border)',
                    borderRadius: '4px',
                    padding: '10px 12px',
                    fontSize: '12.5px',
                  }}>
                    <p style={{ margin: 0, fontStyle: 'italic', color: 'var(--text-main)', lineHeight: 1.6 }}>"{ev.text}"</p>
                    <div style={{ marginTop: '6px', paddingTop: '5px', borderTop: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', fontSize: '10.5px', fontFamily: 'monospace', color: 'var(--text-light)' }}>
                      <span>Source: {ev.source} · Date: {ev.date}</span>
                      <span style={{ color: 'var(--teal)', fontWeight: 600 }}>cos = {ev.similarity}</span>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  );
};
