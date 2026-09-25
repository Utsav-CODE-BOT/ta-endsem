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

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<string>('overview');
  const [investigatingIssueId, setInvestigatingIssueId] = useState<number | null>(null);
  const [reloadKey, setReloadKey] = useState<number>(0);

  const handleNavigateToEmerging = () => {
    setCurrentTab('emerging');
  };

  const handleInvestigateIssue = (issueId: number) => {
    setInvestigatingIssueId(issueId);
    setCurrentTab('rootcause');
  };

  const handleDatasetReloaded = () => {
    setReloadKey((prev) => prev + 1);
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* Sidebar Navigation */}
      <Sidebar
        currentTab={currentTab}
        setCurrentTab={(tab) => {
          if (tab !== 'rootcause') {
            setInvestigatingIssueId(null);
          }
          setCurrentTab(tab);
        }}
      />

      {/* Main Workspace Area */}
      <main className="flex-1 flex flex-col h-screen overflow-y-auto">
        {/* Global Enterprise Topbar */}
        <header className="h-14 border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-20">
          <div className="flex items-center space-x-2 text-xs text-slate-400">
            <span className="font-semibold text-slate-300">VoiceMiner Intelligence Console</span>
            <span>/</span>
            <span className="capitalize text-indigo-400 font-medium">{currentTab.replace('-', ' ')}</span>
          </div>

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-full font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
              <span>FastAPI & Transformers Online</span>
            </div>
          </div>
        </header>

        {/* Tab Content Container */}
        <div key={reloadKey} className="p-6 md:p-8 max-w-7xl w-full mx-auto flex-1">
          {currentTab === 'overview' && (
            <OverviewDashboard onNavigateToEmerging={handleNavigateToEmerging} />
          )}

          {currentTab === 'sources' && (
            <DataSourcesPage onDatasetReloaded={handleDatasetReloaded} />
          )}

          {currentTab === 'issues' && (
            <IssueExplorerPage onInvestigateIssue={handleInvestigateIssue} />
          )}

          {currentTab === 'emerging' && (
            <EmergingIssuesPage onInvestigate={handleInvestigateIssue} />
          )}

          {currentTab === 'rootcause' && (
            <RootCauseInvestigationPage initialIssueId={investigatingIssueId} />
          )}

          {currentTab === 'analyzer' && <TextAnalyzerPage />}

          {currentTab === 'analytics' && <AnalyticsPage />}

          {currentTab === 'reports' && <ReportsPage />}

          {currentTab === 'research' && <ResearchEvaluationPage />}
        </div>
      </main>
    </div>
  );
};

export default App;
