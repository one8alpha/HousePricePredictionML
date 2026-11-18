import React, { useMemo, useState } from 'react';
import { Download } from 'lucide-react';
import { DataEdaView } from './components/DataEdaView';
import { ExperimentsView } from './components/ExperimentsView';
import { ModelLabView } from './components/ModelLabView';
import { PredictorView } from './components/PredictorView';
import { RepositoryView } from './components/RepositoryView';
import {
  DEFAULT_PIPELINE_CONFIG,
  runFullPipeline,
  runImprovementExperiments,
} from './ml/pipeline';
import { PipelineConfig } from './ml/types';

type ActiveTab = 'predictor' | 'data' | 'models' | 'experiments' | 'repo';

const NAV_ITEMS: Array<{ id: ActiveTab; label: string }> = [
  { id: 'predictor', label: 'Predictor' },
  { id: 'data', label: 'Data & EDA' },
  { id: 'models', label: 'Model Lab' },
  { id: 'experiments', label: 'Experiments' },
  { id: 'repo', label: 'Repository' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('predictor');
  const [config, setConfig] = useState<PipelineConfig>(DEFAULT_PIPELINE_CONFIG);

  const baselineArtifacts = useMemo(
    () => runFullPipeline(DEFAULT_PIPELINE_CONFIG),
    []
  );
  const artifacts = useMemo(() => runFullPipeline(config), [config]);
  const experiments = useMemo(() => runImprovementExperiments(), []);

  const handleExportModel = () => {
    const best = artifacts.models[artifacts.bestModelId];
    const payload = {
      project: 'House Price Prediction (Roll No. 1)',
      ml_type: 'Regression',
      selected_model: best.name,
      hyperparameters: best.hyperparameters,
      test_metrics: best.testMetrics,
      train_metrics: best.trainMetrics,
      feature_importances: best.featureImportances,
      scaler_params: artifacts.preprocessingReport.scalerParams,
      split: artifacts.splitInfo,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'house_price_final_model_roll_1.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      {/* Top Bar Contract: [Brand] — [5 single-line nav links] — [1 primary action] */}
      <header className="sticky top-0 z-30 bg-white/95 backdrop-blur border-b border-slate-200">
        <div className="max-w-[1280px] mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-8">
          {/* Zone 1: Brand Wordmark */}
          <button
            type="button"
            onClick={() => setActiveTab('predictor')}
            className="text-base font-bold font-display tracking-tight text-slate-900 whitespace-nowrap shrink-0 cursor-pointer"
          >
            House Price ML
          </button>

          {/* Zone 2: 5 Single-line Navigation Links */}
          <nav
            aria-label="Primary Navigation"
            className="flex items-center gap-5 sm:gap-7 overflow-x-auto no-scrollbar"
          >
            {NAV_ITEMS.map((item) => {
              const active = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setActiveTab(item.id)}
                  className={`py-4 text-sm font-medium whitespace-nowrap shrink-0 border-b-2 transition-colors cursor-pointer ${
                    active
                      ? 'border-blue-700 text-slate-900'
                      : 'border-transparent text-slate-600 hover:text-slate-900 hover:border-slate-300'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Zone 3: 1 Primary Action */}
          <div className="hidden sm:flex items-center shrink-0">
            <button
              type="button"
              onClick={handleExportModel}
              className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-medium text-white bg-slate-900 hover:bg-slate-800 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              Export Model
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="flex-1 max-w-[1280px] w-full mx-auto px-4 sm:px-6 py-8">
        {activeTab === 'predictor' && (
          <PredictorView artifacts={artifacts} onNavigate={setActiveTab} />
        )}
        {activeTab === 'data' && <DataEdaView artifacts={artifacts} />}
        {activeTab === 'models' && (
          <ModelLabView artifacts={artifacts} config={config} />
        )}
        {activeTab === 'experiments' && (
          <ExperimentsView
            artifacts={artifacts}
            baselineArtifacts={baselineArtifacts}
            experiments={experiments}
            config={config}
            onUpdateConfig={setConfig}
          />
        )}
        {activeTab === 'repo' && <RepositoryView artifacts={artifacts} />}
      </main>

      {/* Quiet Footer */}
      <footer className="border-t border-slate-200 bg-white">
        <div className="max-w-[1280px] mx-auto px-4 sm:px-6 py-5 flex flex-wrap items-center justify-between gap-4 text-xs text-slate-500">
          <div>
            TY B.Tech. CSE · Machine Learning Individual Project (2026–27) · Roll No. 1: House Price Prediction
          </div>
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={() => setActiveTab('repo')}
              className="hover:text-slate-900 transition-colors"
            >
              View Repository Files
            </button>
            <button
              type="button"
              onClick={handleExportModel}
              className="hover:text-slate-900 transition-colors"
            >
              Download Model JSON
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
