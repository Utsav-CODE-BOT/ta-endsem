import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { OverviewDashboard } from './pages/OverviewDashboard';
import { DataSourcesPage } from './pages/DataSourcesPage';
import { IssueExplorerPage } from './pages/IssueExplorerPage';
import { EmergingIssuesPage } from './pages/EmergingIssuesPage';
import { RootCauseInvestigationPage } from './pages/RootCauseInvestigationPage';
import { TextAnalyzerPage } from './pages/TextAnalyzerPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { ReportsPage } from './pages/ReportsPage';
import { ResearchEvaluationPage } from './pages/ResearchEvaluationPage';

const TAB_LABELS: Record<string, string> = {
  overview: 'Overview',
  sources: 'Data Sources',
  issues: 'Issue Explorer',
  emerging: 'Emerging Issues',
  rootcause: 'Root-Cause Investigation',
  analyzer: 'Text Analyzer',
  analytics: 'Analytics',
  reports: 'Reports',
  research: 'Evaluation & RQ',
};

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<string>('overview');
  const [investigatingIssueId, setInvestigatingIssueId] = useState<number | null>(null);
  const [reloadKey, setReloadKey] = useState<number>(0);

  const handleNavigateToEmerging = () => setCurrentTab('emerging');

  const handleInvestigateIssue = (issueId: number) => {
    setInvestigatingIssueId(issueId);
    setCurrentTab('rootcause');
  };

  const handleDatasetReloaded = () => setReloadKey((prev) => prev + 1);

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--cream)' }}>
      <Sidebar
        currentTab={currentTab}
        setCurrentTab={(tab) => {
          if (tab !== 'rootcause') setInvestigatingIssueId(null);
          setCurrentTab(tab);
        }}
      />

      {/* Main area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {/* Top bar */}
        <header style={{
          height: '52px',
          borderBottom: '1px solid var(--border)',
          background: 'var(--paper)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 28px',
          position: 'sticky',
          top: 0,
          zIndex: 20,
          flexShrink: 0,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: 'var(--text-muted)' }}>
            <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>VoiceMiner</span>
            <span style={{ color: 'var(--border-md)' }}>/</span>
            <span style={{ color: 'var(--teal)', fontWeight: 500 }}>{TAB_LABELS[currentTab] || currentTab}</span>
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '12px',
            color: 'var(--green)',
            background: 'var(--green-bg)',
            border: '1px solid #b2d9c5',
            borderRadius: '4px',
            padding: '3px 10px',
            fontFamily: 'var(--font-mono, monospace)',
            fontWeight: 500,
          }}>
            <span style={{
              width: '7px', height: '7px',
              borderRadius: '50%',
              background: 'var(--green)',
              display: 'inline-block',
            }} />
            API Online
          </div>
        </header>

        {/* Page content */}
        <main style={{ flex: 1, overflowY: 'auto', padding: '28px 32px' }}>
          <div key={reloadKey} style={{ maxWidth: '1160px', margin: '0 auto' }}>
            {currentTab === 'overview' && <OverviewDashboard onNavigateToEmerging={handleNavigateToEmerging} />}
            {currentTab === 'sources' && <DataSourcesPage onDatasetReloaded={handleDatasetReloaded} />}
            {currentTab === 'issues' && <IssueExplorerPage onInvestigateIssue={handleInvestigateIssue} />}
            {currentTab === 'emerging' && <EmergingIssuesPage onInvestigate={handleInvestigateIssue} />}
            {currentTab === 'rootcause' && <RootCauseInvestigationPage initialIssueId={investigatingIssueId} />}
            {currentTab === 'analyzer' && <TextAnalyzerPage />}
            {currentTab === 'analytics' && <AnalyticsPage />}
            {currentTab === 'reports' && <ReportsPage />}
            {currentTab === 'research' && <ResearchEvaluationPage />}
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;
