import React, { useState } from 'react';
import {
  FileSearch,
  Sparkles,
  TrendingDown,
  TrendingUp,
  Tag,
  Share2,
  AlertCircle,
  CheckCircle2,
  ArrowRight,
} from 'lucide-react';
import { api } from '../services/api';
import { TextAnalyzeResponse } from '../types';

export const TextAnalyzerPage: React.FC = () => {
  const [inputText, setInputText] = useState(
    'Payment failed three times at checkout with error 502 right after the latest app update!'
  );
  const [result, setResult] = useState<TextAnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const sampleQueries = [
    'Payment failed three times at checkout with error 502 right after the latest app update!',
    'Baggage was lost on flight 492 and customer service representative hung up on me.',
    'Flight was delayed 6 hours on the tarmac with zero water or food provided.',
    'Amazing experience! Boarding was seamless and flight attendants were wonderful.',
  ];

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
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
          <FileSearch className="w-6 h-6 text-indigo-400" />
          Real-Time Customer Text & Root-Cause Analyzer
        </h2>
        <p className="text-slate-400 text-sm mt-0.5">
          Paste arbitrary customer feedback to test real-time inference across sentiment, 7-emotion spectrum, aspect extraction, semantic cluster matching, and evidence retrieval.
        </p>
      </div>

      {/* Input Form */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <label className="text-xs font-semibold text-slate-300 block uppercase">
          Customer Text Input:
        </label>
        <textarea
          rows={3}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Paste customer review, complaint, or support message here..."
          className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 leading-relaxed font-sans"
        />

        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-1.5 text-xs text-slate-400">
            <span>Try sample queries:</span>
            {sampleQueries.map((q, idx) => (
              <button
                key={idx}
                onClick={() => setInputText(q)}
                className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors text-[11px]"
              >
                Sample {idx + 1}
              </button>
            ))}
          </div>

          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-semibold text-xs flex items-center gap-2 shadow-lg shadow-indigo-600/30 transition-all"
          >
            {loading ? (
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <Sparkles className="w-4 h-4" />
            )}
            <span>Analyze Customer Voice</span>
          </button>
        </div>
      </div>

      {/* Result Cards */}
      {result && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Sentiment */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <span className="text-xs font-semibold text-slate-400 uppercase">Transformer Sentiment</span>
              <div className="mt-2 flex items-baseline gap-2">
                <span
                  className={`text-2xl font-black capitalize ${
                    result.sentiment.label === 'negative'
                      ? 'text-red-400'
                      : result.sentiment.label === 'positive'
                      ? 'text-emerald-400'
                      : 'text-slate-300'
                  }`}
                >
                  {result.sentiment.label}
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  Confidence: {Math.round(result.sentiment.score * 100)}%
                </span>
              </div>
            </div>

            {/* Emotion */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <span className="text-xs font-semibold text-slate-400 uppercase">Affective Emotion</span>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-2xl font-black text-amber-400 capitalize">
                  {result.emotion.primary_emotion}
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  Score: {Math.round(result.emotion.score * 100)}%
                </span>
              </div>
            </div>

            {/* Extracted Aspects */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <span className="text-xs font-semibold text-slate-400 uppercase">Extracted Aspects</span>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {result.aspects.length > 0 ? (
                  result.aspects.map((asp, i) => (
                    <span
                      key={i}
                      className="text-xs px-2 py-0.5 rounded bg-indigo-950/80 text-indigo-300 border border-indigo-800/60 font-mono font-semibold"
                    >
                      {asp.aspect} ({asp.matched_term})
                    </span>
                  ))
                ) : (
                  <span className="text-xs text-slate-500">General Aspect</span>
                )}
              </div>
            </div>
          </div>

          {/* Matched Semantic Issue & Candidate Cause */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
              <span className="text-xs font-semibold text-slate-400 uppercase flex items-center gap-1.5">
                <Share2 className="w-4 h-4 text-cyan-400" />
                Matched Semantic Issue Cluster
              </span>
              {result.matched_issue ? (
                <div>
                  <h4 className="text-base font-bold text-white">{result.matched_issue.name}</h4>
                  <div className="mt-2 flex items-center gap-3 text-xs text-slate-400 font-mono">
                    <span>Cluster ID: #{result.matched_issue.issue_id}</span>
                    <span>Similarity: {result.matched_issue.similarity}</span>
                    <span>Aspect: {result.matched_issue.primary_aspect}</span>
                  </div>
                </div>
              ) : (
                <p className="text-xs text-slate-500">No active cluster matched.</p>
              )}
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
              <span className="text-xs font-semibold text-slate-400 uppercase flex items-center gap-1.5">
                <AlertCircle className="w-4 h-4 text-amber-400" />
                Suspected Associated Factor
              </span>
              {result.candidate_cause ? (
                <div>
                  <h4 className="text-base font-bold text-amber-400">{result.candidate_cause.candidate_factor}</h4>
                  <div className="mt-2 flex items-center gap-3 text-xs text-slate-400 font-mono">
                    <span>Confidence: {Math.round(result.candidate_cause.confidence * 100)}%</span>
                    <span>Lift: {result.candidate_cause.lift}x</span>
                  </div>
                  <p className="text-xs text-slate-300 mt-2 italic bg-slate-950 p-2.5 rounded border border-slate-800">
                    "{result.candidate_cause.interpretation}"
                  </p>
                </div>
              ) : (
                <p className="text-xs text-slate-500">No candidate factor associated with this cluster yet.</p>
              )}
            </div>
          </div>

          {/* Similar Evidence Statements */}
          {result.similar_evidence.length > 0 && (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
              <h3 className="text-sm font-semibold text-white">Nearest Historical Customer Feedback</h3>
              <div className="space-y-2">
                {result.similar_evidence.map((ev, i) => (
                  <div key={i} className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-xs">
                    <p className="text-slate-200 italic">"{ev.text}"</p>
                    <div className="mt-2 flex items-center justify-between text-[11px] text-slate-500 font-mono">
                      <span>Source: {ev.source} • Date: {ev.date}</span>
                      <span className="text-indigo-400">Cosine Similarity: {ev.similarity}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
