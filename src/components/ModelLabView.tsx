import React, { useState } from 'react';
import { FEATURE_METADATA, formatINR } from '../ml/dataset';
import { PipelineArtifacts, PipelineConfig } from '../ml/types';
import {
  ActualVsPredictedChart,
  DecisionTreeDiagram,
  FeatureImportanceChart,
  ResidualPlotChart,
} from './Charts';

export function ModelLabView({
  artifacts,
  config,
}: {
  artifacts: PipelineArtifacts;
  config: PipelineConfig;
}) {
  const [activeModelId, setActiveModelId] = useState<'linear' | 'decision_tree' | 'random_forest'>(
    artifacts.bestModelId
  );

  const modelList = [
    artifacts.models.linear,
    artifacts.models.decision_tree,
    artifacts.models.random_forest,
  ];
  const activeModel = artifacts.models[activeModelId];

  return (
    <div className="space-y-8">
      {/* Header */}
      <section className="border-b border-slate-200 pb-5">
        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 font-medium mb-1.5">
          <span>Parts E, F, G, H, I & J</span>
          <span aria-hidden="true">·</span>
          <span>{config.selectedFeatures.length} Selected Features</span>
          <span aria-hidden="true">·</span>
          <span>
            Split {artifacts.splitInfo.trainRatio}% / {artifacts.splitInfo.testRatio}% (random_state={artifacts.splitInfo.randomState})
          </span>
        </div>
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-slate-900">
          Model Training, Evaluation & Diagnostics
        </h1>
        <p className="mt-1.5 text-sm text-slate-600 max-w-3xl">
          Feature and target specification, reproducible train-test partitioning, side-by-side comparison of three regression algorithms, and residual/tree visualizations.
        </p>
      </section>

      {/* Part E & F: Feature Selection + Train-Test Split */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-lg p-6 space-y-4">
          <div className="pb-3 border-b border-slate-200">
            <h2 className="text-base font-semibold text-slate-900">
              05. Feature & Target Selection (Part E)
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Explicit separation of independent predictors (X) and continuous target variable (y)
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
            <div>
              <div className="font-semibold text-slate-900 mb-1.5">
                Input Features Matrix (X) — {config.selectedFeatures.length} Columns
              </div>
              <div className="flex flex-wrap gap-1.5 font-mono text-[11px] text-slate-700">
                {config.selectedFeatures.map((f, idx) => (
                  <span key={f}>
                    {f}
                    {idx < config.selectedFeatures.length - 1 ? ',' : ''}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <div className="font-semibold text-slate-900 mb-1.5">
                Target Output Vector (y) — 1 Column
              </div>
              <p className="text-slate-600 leading-relaxed">
                <code className="font-mono text-slate-900 bg-slate-100 px-1 py-0.5 rounded">price</code> (Selling price in INR). Excluded ID column and raw unencoded strings to prevent spurious correlation.
              </p>
            </div>
          </div>
          <div className="pt-2 border-t border-slate-100 text-xs text-slate-600 leading-relaxed">
            <strong>Justification:</strong> All {config.selectedFeatures.length} active features reflect tangible real-estate value drivers—built-up space (<code className="font-mono">area</code>, <code className="font-mono">stories</code>), room capacity (<code className="font-mono">bedrooms</code>, <code className="font-mono">bathrooms</code>), utility amenities (<code className="font-mono">airconditioning</code>, <code className="font-mono">parking</code>, <code className="font-mono">hotwaterheating</code>), and location (<code className="font-mono">mainroad</code>, <code className="font-mono">prefarea</code>).
          </div>
        </div>

        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-lg p-6 space-y-4">
          <div className="pb-3 border-b border-slate-200">
            <h2 className="text-base font-semibold text-slate-900">
              06. Train-Test Split Specification (Part F)
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Unbiased holdout validation with deterministic seeding
            </p>
          </div>
          <div className="grid grid-cols-3 gap-3 py-1">
            <div>
              <div className="text-xs text-slate-500">Total Clean</div>
              <div className="text-lg font-bold font-mono tabular-nums text-slate-900">
                {artifacts.splitInfo.totalRows}
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-500">Training ({artifacts.splitInfo.trainRatio}%)</div>
              <div className="text-lg font-bold font-mono tabular-nums text-blue-700">
                {artifacts.splitInfo.trainRows}
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-500">Testing ({artifacts.splitInfo.testRatio}%)</div>
              <div className="text-lg font-bold font-mono tabular-nums text-emerald-700">
                {artifacts.splitInfo.testRows}
              </div>
            </div>
          </div>
          <ul className="space-y-1.5 text-xs text-slate-600 leading-relaxed pt-2 border-t border-slate-100">
            <li>
              <strong>Why Divide:</strong> Evaluating on unseen test records measures generalization and catches overfitting that training accuracy hides.
            </li>
            <li>
              <strong>Role of <code className="font-mono text-slate-900">random_state={artifacts.splitInfo.randomState}</code>:</strong> Locks the Fisher-Yates shuffle seed so all three algorithms train and evaluate on the exact same partitions every run.
            </li>
          </ul>
        </div>
      </section>

      {/* Part G, H, I: Model Comparison Table & Rationale */}
      <section className="bg-white border border-slate-200 rounded-lg p-6 space-y-6">
        <div className="pb-4 border-b border-slate-200">
          <h2 className="text-lg font-semibold text-slate-900">
            07. Algorithm Selection, Training & Evaluation Comparison (Parts G–I)
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Live evaluation metrics computed on {artifacts.splitInfo.trainRows} training samples and {artifacts.splitInfo.testRows} unseen test samples
          </p>
        </div>

        <div className="overflow-x-auto border border-slate-200 rounded-md">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-slate-600">
                <th className="py-2.5 px-3 font-medium">Algorithm</th>
                <th className="py-2.5 px-3 font-medium text-right">Train R²</th>
                <th className="py-2.5 px-3 font-medium text-right">Test R²</th>
                <th className="py-2.5 px-3 font-medium text-right">Test MAE</th>
                <th className="py-2.5 px-3 font-medium text-right">Test RMSE</th>
                <th className="py-2.5 px-3 font-medium text-right">Test MSE</th>
                <th className="py-2.5 px-3 font-medium">Key Parameters</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {modelList.map((m) => {
                const isBest = m.id === artifacts.bestModelId;
                return (
                  <tr
                    key={m.id}
                    className={isBest ? 'bg-blue-50/50 font-medium' : 'hover:bg-slate-50'}
                  >
                    <td className="py-3 px-3 text-slate-900">
                      {m.name}
                      {isBest && (
                        <span className="ml-2 text-[11px] font-semibold text-blue-700">
                          (Best Test R²)
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-700">
                      {m.trainMetrics.r2.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums font-semibold text-blue-700">
                      {m.testMetrics.r2.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-800">
                      {formatINR(m.testMetrics.mae)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-900">
                      {formatINR(m.testMetrics.rmse)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-600">
                      {(m.testMetrics.mse / 1e10).toFixed(2)} × 10¹⁰
                    </td>
                    <td className="py-3 px-3 font-mono text-[11px] text-slate-600">
                      {Object.entries(m.hyperparameters)
                        .map(([k, v]) => `${k}=${v}`)
                        .join(', ')}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* How Each Model Trains & Result Interpretation */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
          <div className="space-y-1.5 text-xs text-slate-600 leading-relaxed">
            <h3 className="text-sm font-semibold text-slate-900">
              1. Multiple Linear Regression
            </h3>
            <p>
              <strong>How it trains:</strong> Solves the Normal Equation <code className="font-mono">(XᵀX + λI)⁻¹Xᵀy</code> to learn one linear weight per standardized feature plus an intercept bias.
            </p>
            <p>
              <strong>Interpretation:</strong> Strong interpretable baseline (Test R² = <span className="font-mono tabular-nums text-slate-900">{artifacts.models.linear.testMetrics.r2.toFixed(4)}</span>, RMSE = <span className="font-mono tabular-nums text-slate-900">{formatINR(artifacts.models.linear.testMetrics.rmse)}</span>). Works best on additive trends, though pure linear weights cannot model step-wise tier premiums directly.
            </p>
          </div>

          <div className="space-y-1.5 text-xs text-slate-600 leading-relaxed">
            <h3 className="text-sm font-semibold text-slate-900">
              2. Decision Tree Regression (CART)
            </h3>
            <p>
              <strong>How it trains:</strong> Recursively partitions training samples using feature thresholds that maximize Mean Squared Error (MSE) reduction, constrained by <code className="font-mono">max_depth={config.dtMaxDepth}</code>.
            </p>
            <p>
              <strong>Interpretation:</strong> Captures non-linear thresholds naturally, though a single tree produces step-wise predictions with higher variance on unseen boundaries (Test R² = <span className="font-mono tabular-nums text-slate-900">{artifacts.models.decision_tree.testMetrics.r2.toFixed(4)}</span>, RMSE = <span className="font-mono tabular-nums text-slate-900">{formatINR(artifacts.models.decision_tree.testMetrics.rmse)}</span>).
            </p>
          </div>

          <div className="space-y-1.5 text-xs text-slate-600 leading-relaxed">
            <h3 className="text-sm font-semibold text-slate-900">
              3. Random Forest Regression
            </h3>
            <p>
              <strong>How it trains:</strong> Trains {config.rfEstimators} bootstrap CART trees and averages predictions across all trees to smooth step boundaries.
            </p>
            <p>
              <strong>Interpretation:</strong> Bagging cuts single-tree variance dramatically (Test R² = <span className="font-mono tabular-nums text-slate-900">{artifacts.models.random_forest.testMetrics.r2.toFixed(4)}</span>, RMSE = <span className="font-mono tabular-nums text-slate-900">{formatINR(artifacts.models.random_forest.testMetrics.rmse)}</span>), capturing both non-linear thresholds and smooth averages.
            </p>
          </div>
        </div>
      </section>

      {/* Part J: Model Visualizations */}
      <section className="space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              08. Model Diagnostics & Visualizations (Part J)
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Select any trained algorithm to inspect its Actual vs. Predicted fit, Residual errors, Feature Importances, and learned parameters.
            </p>
          </div>
          <div className="flex items-center gap-1 p-1 bg-slate-200/70 rounded-md">
            {modelList.map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => setActiveModelId(m.id)}
                className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap ${
                  activeModelId === m.id
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {m.shortName}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white border border-slate-200 rounded-lg p-6 space-y-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">
                Actual vs. Predicted House Prices ({activeModel.shortName})
              </h3>
              <p className="text-xs text-slate-500">
                Closer clustering along the dashed y = x diagonal indicates higher predictive accuracy
              </p>
            </div>
            <ActualVsPredictedChart model={activeModel} />
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-6 space-y-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">
                Residual Error Plot ({activeModel.shortName})
              </h3>
              <p className="text-xs text-slate-500">
                Random scatter centered around 0L confirms unbiased predictions without systematic drift
              </p>
            </div>
            <ResidualPlotChart model={activeModel} />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-5 bg-white border border-slate-200 rounded-lg p-6 space-y-4">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">
                Learned Feature Importances ({activeModel.shortName})
              </h3>
              <p className="text-xs text-slate-500">
                Relative contribution of each predictor to house valuation
              </p>
            </div>
            <FeatureImportanceChart model={activeModel} />
          </div>

          <div className="lg:col-span-7 bg-white border border-slate-200 rounded-lg p-6 space-y-4">
            {activeModel.id === 'linear' && activeModel.coefficients ? (
              <>
                <div>
                  <h3 className="text-sm font-semibold text-slate-900">
                    Learned Linear Regression Weights (Standardized)
                  </h3>
                  <p className="text-xs text-slate-500">
                    Intercept β₀ = {formatINR(activeModel.bias ?? 0)} · Each coefficient βⱼ shows price change per +1σ shift
                  </p>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {activeModel.coefficients.map((c) => {
                    const meta = FEATURE_METADATA.find((m) => m.key === c.feature);
                    return (
                      <div
                        key={c.feature}
                        className="p-2.5 border border-slate-200 rounded bg-slate-50/70 text-xs"
                      >
                        <div className="text-slate-600 truncate">{meta?.shortLabel ?? c.label}</div>
                        <div className="mt-0.5 font-mono font-semibold tabular-nums text-slate-900">
                          {c.weight >= 0 ? '+' : '−'}
                          {formatINR(Math.abs(c.weight))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </>
            ) : (
              <>
                <div>
                  <h3 className="text-sm font-semibold text-slate-900">
                    Learned CART Decision Tree Structure (Top 3 Levels)
                  </h3>
                  <p className="text-xs text-slate-500">
                    Visualizing actual MSE-optimal threshold splits learned from the training partition
                  </p>
                </div>
                <DecisionTreeDiagram
                  root={artifacts.models.decision_tree.treeRoot}
                />
              </>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
