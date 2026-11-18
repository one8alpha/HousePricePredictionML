import React from 'react';
import { RotateCcw } from 'lucide-react';
import { FEATURE_METADATA, formatINR } from '../ml/dataset';
import { DEFAULT_PIPELINE_CONFIG } from '../ml/pipeline';
import {
  ExperimentResult,
  FeatureKey,
  PipelineArtifacts,
  PipelineConfig,
  ScalingMethod,
} from '../ml/types';

export function ExperimentsView({
  artifacts,
  baselineArtifacts,
  experiments,
  config,
  onUpdateConfig,
}: {
  artifacts: PipelineArtifacts;
  baselineArtifacts: PipelineArtifacts;
  experiments: ExperimentResult[];
  config: PipelineConfig;
  onUpdateConfig: (next: PipelineConfig) => void;
}) {
  const toggleFeature = (key: FeatureKey) => {
    const exists = config.selectedFeatures.includes(key);
    if (exists && config.selectedFeatures.length <= 1) return; // keep at least 1 feature
    const nextFeatures = exists
      ? config.selectedFeatures.filter((f) => f !== key)
      : [...config.selectedFeatures, key];
    onUpdateConfig({ ...config, selectedFeatures: nextFeatures });
  };

  const bestModel = artifacts.models[artifacts.bestModelId];
  const baseRf = baselineArtifacts.models.random_forest;
  const currRf = artifacts.models.random_forest;
  const r2Delta = currRf.testMetrics.r2 - baseRf.testMetrics.r2;

  return (
    <div className="space-y-8">
      {/* Header */}
      <section className="border-b border-slate-200 pb-5">
        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 font-medium mb-1.5">
          <span>Parts K & L</span>
          <span aria-hidden="true">·</span>
          <span>Model Improvement Experiments</span>
          <span aria-hidden="true">·</span>
          <span>Live Viva Examiner Sandbox</span>
        </div>
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-slate-900">
          Model Improvement, Viva Simulator & Final Interpretation
        </h1>
        <p className="mt-1.5 text-sm text-slate-600 max-w-3xl">
          Recorded hyperparameter, preprocessing, and split experiments alongside an interactive Viva Simulator that dynamically retrains all three algorithms when parameters or features are modified.
        </p>
      </section>

      {/* Part K: Documented Model Improvement Experiments */}
      <section className="bg-white border border-slate-200 rounded-lg p-6 space-y-5">
        <div className="pb-4 border-b border-slate-200">
          <h2 className="text-lg font-semibold text-slate-900">
            09. Model Improvement Experiments (Part K)
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Each experiment executes the real ML pipeline before and after modification, recording Change → Result → Observation → Conclusion.
          </p>
        </div>

        <div className="divide-y divide-slate-200">
          {experiments.map((exp, idx) => {
            const improved = exp.deltaR2 >= 0;
            return (
              <div key={exp.id} className="py-5 first:pt-2 last:pb-2 space-y-3">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <div>
                    <span className="text-xs font-mono text-slate-500 mr-2">
                      Experiment {idx + 1} · {exp.category}
                    </span>
                    <h3 className="text-base font-semibold text-slate-900 inline">
                      {exp.title}
                    </h3>
                  </div>
                  <div
                    className={`text-xs font-mono tabular-nums font-semibold ${
                      improved ? 'text-emerald-700' : 'text-amber-700'
                    }`}
                  >
                    Δ Test R²: {improved ? '+' : ''}
                    {exp.deltaR2.toFixed(4)}
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 text-xs">
                  <div className="lg:col-span-5 space-y-2">
                    <div>
                      <span className="font-semibold text-slate-800">Change Applied: </span>
                      <span className="text-slate-600">{exp.change}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-3 pt-1 font-mono tabular-nums">
                      <div className="p-2.5 rounded border border-slate-200 bg-slate-50">
                        <div className="text-[11px] text-slate-500 font-sans">Before</div>
                        <div className="text-slate-800 font-semibold mt-0.5">
                          R² = {exp.baselineR2.toFixed(4)}
                        </div>
                        <div className="text-[11px] text-slate-600">
                          RMSE: {formatINR(exp.baselineRmse)}
                        </div>
                      </div>
                      <div className="p-2.5 rounded border border-blue-200 bg-blue-50/50">
                        <div className="text-[11px] text-blue-700 font-sans">After</div>
                        <div className="text-slate-900 font-semibold mt-0.5">
                          R² = {exp.modifiedR2.toFixed(4)}
                        </div>
                        <div className="text-[11px] text-slate-700">
                          RMSE: {formatINR(exp.modifiedRmse)}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="lg:col-span-7 space-y-2 text-slate-600 leading-relaxed">
                    <div>
                      <span className="font-semibold text-slate-800">Observation: </span>
                      {exp.observation}
                    </div>
                    <div>
                      <span className="font-semibold text-slate-800">Conclusion: </span>
                      {exp.conclusion}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Live Viva Examiner Sandbox */}
      <section className="bg-white border border-slate-200 rounded-lg p-6 space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-200">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              10. Interactive Viva Experiment Sandbox
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Modify hyperparameters, preprocessing toggles, split ratio, or drop features live to demonstrate pipeline behavior during viva.
            </p>
          </div>
          <button
            type="button"
            onClick={() => onUpdateConfig(DEFAULT_PIPELINE_CONFIG)}
            className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-700 hover:text-slate-900 px-3 py-1.5 rounded border border-slate-200 hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Restore Baseline Defaults
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Controls */}
          <div className="lg:col-span-7 space-y-5">
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">
                  Test Split Ratio ({Math.round(config.testRatio * 100)}%)
                </label>
                <input
                  type="range"
                  min={0.15}
                  max={0.4}
                  step={0.05}
                  value={config.testRatio}
                  onChange={(e) =>
                    onUpdateConfig({ ...config, testRatio: Number(e.target.value) })
                  }
                  className="w-full accent-blue-700"
                />
                <div className="flex justify-between text-[11px] font-mono text-slate-500">
                  <span>15%</span>
                  <span>20% (Std)</span>
                  <span>40%</span>
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">
                  Decision Tree max_depth ({config.dtMaxDepth})
                </label>
                <input
                  type="range"
                  min={2}
                  max={12}
                  step={1}
                  value={config.dtMaxDepth}
                  onChange={(e) =>
                    onUpdateConfig({ ...config, dtMaxDepth: Number(e.target.value) })
                  }
                  className="w-full accent-blue-700"
                />
                <div className="flex justify-between text-[11px] font-mono text-slate-500">
                  <span>2 (Underfit)</span>
                  <span>5</span>
                  <span>12 (Overfit)</span>
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">
                  Random Forest Trees ({config.rfEstimators})
                </label>
                <input
                  type="range"
                  min={3}
                  max={40}
                  step={1}
                  value={config.rfEstimators}
                  onChange={(e) =>
                    onUpdateConfig({ ...config, rfEstimators: Number(e.target.value) })
                  }
                  className="w-full accent-blue-700"
                />
                <div className="flex justify-between text-[11px] font-mono text-slate-500">
                  <span>3</span>
                  <span>25</span>
                  <span>40</span>
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">
                  Feature Scaling
                </label>
                <select
                  value={config.scaling}
                  onChange={(e) =>
                    onUpdateConfig({
                      ...config,
                      scaling: e.target.value as ScalingMethod,
                    })
                  }
                  className="w-full px-2.5 py-1.5 text-xs bg-white border border-slate-300 rounded"
                >
                  <option value="standard">StandardScaler (z-score)</option>
                  <option value="minmax">MinMaxScaler (0–1)</option>
                  <option value="none">None (Raw Unscaled)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">
                  Random State Seed
                </label>
                <input
                  type="number"
                  min={1}
                  max={9999}
                  value={config.randomState}
                  onChange={(e) =>
                    onUpdateConfig({
                      ...config,
                      randomState: Math.max(1, Number(e.target.value) || 42),
                    })
                  }
                  className="w-full px-2.5 py-1.5 text-xs font-mono tabular-nums bg-white border border-slate-300 rounded"
                />
              </div>

              <div className="flex flex-col justify-end">
                <label className="inline-flex items-center gap-2 text-xs font-medium text-slate-700 cursor-pointer py-1.5">
                  <input
                    type="checkbox"
                    checked={config.handleOutliers}
                    onChange={(e) =>
                      onUpdateConfig({ ...config, handleOutliers: e.target.checked })
                    }
                    className="rounded border-slate-300 text-blue-700 focus:ring-blue-600"
                  />
                  Remove IQR Outliers
                </label>
              </div>
            </div>

            {/* Feature Toggle Matrix */}
            <div className="pt-3 border-t border-slate-100">
              <div className="text-xs font-medium text-slate-700 mb-2">
                Active Input Features in X ({config.selectedFeatures.length} / 12) — Click to Drop/Include
              </div>
              <div className="flex flex-wrap gap-2">
                {FEATURE_METADATA.map((f) => {
                  const active = config.selectedFeatures.includes(f.key);
                  return (
                    <button
                      key={f.key}
                      type="button"
                      onClick={() => toggleFeature(f.key)}
                      className={`px-2.5 py-1 text-xs rounded border transition-colors ${
                        active
                          ? 'bg-blue-700 text-white border-blue-700'
                          : 'bg-white text-slate-500 border-slate-200 hover:text-slate-900'
                      }`}
                    >
                      {f.shortLabel}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Live Retrained Results Comparison */}
          <div className="lg:col-span-5 border border-slate-200 rounded-md p-4 bg-slate-50/70 space-y-3">
            <div className="flex items-center justify-between border-b border-slate-200 pb-2">
              <span className="text-xs font-semibold text-slate-900">
                Live Pipeline Metrics vs. Baseline
              </span>
              <span
                className={`text-xs font-mono tabular-nums font-semibold ${
                  r2Delta >= 0 ? 'text-emerald-700' : 'text-red-600'
                }`}
              >
                RF ΔR²: {r2Delta >= 0 ? '+' : ''}
                {r2Delta.toFixed(4)}
              </span>
            </div>

            <div className="space-y-2.5 text-xs">
              {(['linear', 'decision_tree', 'random_forest'] as const).map((id) => {
                const curr = artifacts.models[id];
                const base = baselineArtifacts.models[id];
                return (
                  <div
                    key={id}
                    className="p-2.5 bg-white border border-slate-200 rounded space-y-1"
                  >
                    <div className="flex justify-between font-medium text-slate-900">
                      <span>{curr.shortName}</span>
                      <span className="font-mono tabular-nums">
                        Test R²: {curr.testMetrics.r2.toFixed(4)} (base {base.testMetrics.r2.toFixed(4)})
                      </span>
                    </div>
                    <div className="flex justify-between text-[11px] font-mono tabular-nums text-slate-600">
                      <span>Train R²: {curr.trainMetrics.r2.toFixed(4)}</span>
                      <span>RMSE: {formatINR(curr.testMetrics.rmse)}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      {/* Part L: Final Model Selection & 5 Mandatory Interpretation Questions */}
      <section className="bg-white border border-slate-200 rounded-lg p-6 space-y-5">
        <div className="pb-4 border-b border-slate-200">
          <h2 className="text-lg font-semibold text-slate-900">
            11. Final Model Selection & Interpretation (Part L)
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Comprehensive answers to the five final evaluation questions required for submission and viva defense.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-slate-700 leading-relaxed">
          <div className="space-y-4">
            <div>
              <h3 className="text-xs font-semibold text-slate-900 mb-1">
                1. Which model did you finally select?
              </h3>
              <p>
                <strong>{bestModel.name}</strong> was selected as the final deployed model based on live holdout test evaluation.
              </p>
            </div>
            <div>
              <h3 className="text-xs font-semibold text-slate-900 mb-1">
                2. Why was it selected over the alternatives?
              </h3>
              <p>
                It achieved the highest unseen Test R² (<span className="font-mono tabular-nums font-medium text-slate-900">{bestModel.testMetrics.r2.toFixed(4)}</span>) and lowest Test RMSE (<span className="font-mono tabular-nums font-medium text-slate-900">{formatINR(bestModel.testMetrics.rmse)}</span>).{' '}
                {artifacts.bestModelId === 'random_forest'
                  ? 'Unlike Linear Regression, Random Forest captures non-linear area tier jumps and amenity interaction premiums (e.g., AC + Preferred Locality), while bootstrap averaging eliminates single Decision Tree overfitting.'
                  : artifacts.bestModelId === 'linear'
                    ? 'Under the current feature subset and parameters, Multiple Linear Regression captures the dominant additive trend cleanly with low variance and full coefficient interpretability.'
                    : 'Under the current configuration, the pruned CART Decision Tree provides clear interpretable split thresholds with strong test generalization.'}
              </p>
            </div>
            <div>
              <h3 className="text-xs font-semibold text-slate-900 mb-1">
                3. What were the most important features?
              </h3>
              <p>
                Across both correlation analysis and tree impurity reduction, the dominant features are{' '}
                <span className="font-mono text-xs text-slate-900">
                  {bestModel.featureImportances
                    .slice(0, 5)
                    .map((f) => `${f.label} (${(f.importance * 100).toFixed(1)}%)`)
                    .join(', ')}
                </span>
                .
              </p>
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <h3 className="text-xs font-semibold text-slate-900 mb-1">
                4. What limitations does your model have?
              </h3>
              <p>
                Tree ensembles cannot extrapolate linearly beyond the maximum house area or price observed in the training set (~10,500 sq.ft.). Additionally, the dataset lacks exact GPS latitude/longitude coordinates, property age, and builder reputation, which influence real-world micro-market valuations.
              </p>
            </div>
            <div>
              <h3 className="text-xs font-semibold text-slate-900 mb-1">
                5. What could be improved in the future?
              </h3>
              <p>
                Future iterations can integrate Gradient Boosted Trees (XGBoost / LightGBM), 5-fold cross-validation for hyperparameter grid search, spatial neighborhood coordinates, and periodic retraining pipelines adjusted for real-estate inflation indices.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
