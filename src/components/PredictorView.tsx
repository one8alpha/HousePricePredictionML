import React, { useMemo, useState } from 'react';
import { ArrowRight, RotateCcw } from 'lucide-react';
import { formatINR, formatLakhs } from '../ml/dataset';
import { PipelineArtifacts, PredictionInput } from '../ml/types';

const DEFAULT_INPUT: PredictionInput = {
  area: 1500,
  bedrooms: 3,
  bathrooms: 2,
  stories: 2,
  mainroad: 'yes',
  guestroom: 'no',
  basement: 'no',
  hotwaterheating: 'no',
  airconditioning: 'yes',
  parking: 1,
  prefarea: 'yes',
  furnishingstatus: 'semi-furnished',
};

const PRESETS: Array<{ label: string; input: PredictionInput }> = [
  {
    label: 'PDF Reference (1500 sq.ft., 3BHK)',
    input: DEFAULT_INPUT,
  },
  {
    label: 'Compact Starter (1200 sq.ft., 2BHK)',
    input: {
      area: 1200,
      bedrooms: 2,
      bathrooms: 1,
      stories: 1,
      mainroad: 'yes',
      guestroom: 'no',
      basement: 'no',
      hotwaterheating: 'no',
      airconditioning: 'no',
      parking: 0,
      prefarea: 'no',
      furnishingstatus: 'unfurnished',
    },
  },
  {
    label: 'Family Residence (4200 sq.ft., 4BHK)',
    input: {
      area: 4200,
      bedrooms: 4,
      bathrooms: 3,
      stories: 2,
      mainroad: 'yes',
      guestroom: 'yes',
      basement: 'yes',
      hotwaterheating: 'no',
      airconditioning: 'yes',
      parking: 2,
      prefarea: 'yes',
      furnishingstatus: 'furnished',
    },
  },
];

export function PredictorView({
  artifacts,
  onNavigate,
}: {
  artifacts: PipelineArtifacts;
  onNavigate: (tab: 'data' | 'models' | 'experiments' | 'repo') => void;
}) {
  const [formInput, setFormInput] = useState<PredictionInput>(DEFAULT_INPUT);
  const [submittedInput, setSubmittedInput] = useState<PredictionInput>(DEFAULT_INPUT);
  const [selectedModel, setSelectedModel] = useState<'linear' | 'decision_tree' | 'random_forest'>(
    artifacts.bestModelId
  );

  const prediction = useMemo(
    () => artifacts.predict(selectedModel, submittedInput),
    [artifacts, selectedModel, submittedInput]
  );

  const allModelComparison = useMemo(() => {
    return (['linear', 'decision_tree', 'random_forest'] as const).map((id) => ({
      id,
      name: artifacts.models[id].shortName,
      r2: artifacts.models[id].testMetrics.r2,
      predicted: artifacts.predict(id, submittedInput).predictedPrice,
    }));
  }, [artifacts, submittedInput]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmittedInput({ ...formInput });
  };

  const handlePreset = (presetInput: PredictionInput) => {
    setFormInput(presetInput);
    setSubmittedInput(presetInput);
  };

  const activeEval = artifacts.models[selectedModel];

  return (
    <div className="space-y-8">
      {/* Page Lead */}
      <section className="border-b border-slate-200 pb-6">
        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 font-medium mb-2">
          <span>Roll No. 1</span>
          <span aria-hidden="true">·</span>
          <span>Supervised Regression</span>
          <span aria-hidden="true">·</span>
          <span>545 Raw Records ({artifacts.cleanedData.length} Cleaned)</span>
          <span aria-hidden="true">·</span>
          <span>3 Trained Models</span>
        </div>
        <h1
          className="text-3xl md:text-4xl font-bold tracking-tight text-slate-900 max-w-3xl"
          style={{ textWrap: 'balance' }}
        >
          House Price Prediction System
        </h1>
        <p className="mt-2 text-slate-600 text-base max-w-3xl leading-relaxed">
          End-to-end Machine Learning regression pipeline predicting residential property valuation in Indian Rupees (INR) from structural dimensions and neighborhood amenities.
        </p>
      </section>

      {/* Main 2-Column Prediction Application (Stage 13: Enter Input -> Submit -> Predict -> Display Output) */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left Column: User Input Form */}
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-lg p-6">
          <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-200">
            <div>
              <h2 className="text-lg font-semibold text-slate-900">
                Property Input Parameters
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Enter property details and submit to compute the valuation using the trained pipeline.
              </p>
            </div>
            <button
              type="button"
              onClick={() => handlePreset(DEFAULT_INPUT)}
              className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-600 hover:text-slate-900 px-2.5 py-1.5 rounded border border-slate-200 hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset Default
            </button>
          </div>

          {/* Quick Presets */}
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-500 font-medium mr-1">Sample Inputs:</span>
            {PRESETS.map((p) => (
              <button
                key={p.label}
                type="button"
                onClick={() => handlePreset(p.input)}
                className="text-xs px-2.5 py-1 rounded border border-slate-200 bg-slate-50 text-slate-700 hover:bg-slate-100 hover:text-slate-900 transition-colors whitespace-nowrap shrink-0"
              >
                {p.label}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} className="mt-5 space-y-5">
            {/* Numerical Inputs */}
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1.5">
                  Area (sq.ft.)
                </label>
                <input
                  type="number"
                  min={500}
                  max={15000}
                  step={50}
                  required
                  value={formInput.area}
                  onChange={(e) =>
                    setFormInput((prev) => ({
                      ...prev,
                      area: Math.max(500, Number(e.target.value) || 500),
                    }))
                  }
                  className="w-full px-3 py-2 text-sm font-mono tabular-nums bg-white border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-600 focus:border-blue-600"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1.5">
                  Bedrooms
                </label>
                <select
                  value={formInput.bedrooms}
                  onChange={(e) =>
                    setFormInput((prev) => ({ ...prev, bedrooms: Number(e.target.value) }))
                  }
                  className="w-full px-3 py-2 text-sm font-mono tabular-nums bg-white border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-600"
                >
                  {[1, 2, 3, 4, 5, 6].map((n) => (
                    <option key={n} value={n}>
                      {n} {n === 1 ? 'Bedroom' : 'Bedrooms'}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1.5">
                  Bathrooms
                </label>
                <select
                  value={formInput.bathrooms}
                  onChange={(e) =>
                    setFormInput((prev) => ({ ...prev, bathrooms: Number(e.target.value) }))
                  }
                  className="w-full px-3 py-2 text-sm font-mono tabular-nums bg-white border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-600"
                >
                  {[1, 2, 3, 4].map((n) => (
                    <option key={n} value={n}>
                      {n} {n === 1 ? 'Bathroom' : 'Bathrooms'}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1.5">
                  Stories / Floors
                </label>
                <select
                  value={formInput.stories}
                  onChange={(e) =>
                    setFormInput((prev) => ({ ...prev, stories: Number(e.target.value) }))
                  }
                  className="w-full px-3 py-2 text-sm font-mono tabular-nums bg-white border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-600"
                >
                  {[1, 2, 3, 4].map((n) => (
                    <option key={n} value={n}>
                      {n} {n === 1 ? 'Story' : 'Stories'}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1.5">
                  Parking Spaces
                </label>
                <select
                  value={formInput.parking}
                  onChange={(e) =>
                    setFormInput((prev) => ({ ...prev, parking: Number(e.target.value) }))
                  }
                  className="w-full px-3 py-2 text-sm font-mono tabular-nums bg-white border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-600"
                >
                  {[0, 1, 2, 3].map((n) => (
                    <option key={n} value={n}>
                      {n} Parking
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1.5">
                  Furnishing Status
                </label>
                <select
                  value={formInput.furnishingstatus}
                  onChange={(e) =>
                    setFormInput((prev) => ({
                      ...prev,
                      furnishingstatus: e.target.value as PredictionInput['furnishingstatus'],
                    }))
                  }
                  className="w-full px-3 py-2 text-sm bg-white border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-600"
                >
                  <option value="unfurnished">Unfurnished</option>
                  <option value="semi-furnished">Semi-Furnished</option>
                  <option value="furnished">Furnished</option>
                </select>
              </div>
            </div>

            {/* Binary Amenity Selectors */}
            <div className="pt-3 border-t border-slate-200">
              <div className="text-xs font-medium text-slate-700 mb-3">
                Location & Interior Amenities
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {(
                  [
                    { key: 'mainroad', label: 'Main Road' },
                    { key: 'airconditioning', label: 'Air Conditioning' },
                    { key: 'prefarea', label: 'Preferred Area' },
                    { key: 'guestroom', label: 'Guest Room' },
                    { key: 'basement', label: 'Basement' },
                    { key: 'hotwaterheating', label: 'Hot Water' },
                  ] as const
                ).map(({ key, label }) => {
                  const val = formInput[key];
                  return (
                    <div
                      key={key}
                      className="flex items-center justify-between gap-2 px-3 py-2 rounded-md border border-slate-200 bg-slate-50/70"
                    >
                      <span className="text-xs font-medium text-slate-700 truncate">{label}</span>
                      <div className="flex items-center gap-1 bg-slate-200/70 p-0.5 rounded shrink-0">
                        <button
                          type="button"
                          onClick={() => setFormInput((prev) => ({ ...prev, [key]: 'yes' }))}
                          className={`px-2 py-0.5 text-[11px] font-medium rounded transition-colors ${
                            val === 'yes'
                              ? 'bg-white text-slate-900 shadow-xs'
                              : 'text-slate-600 hover:text-slate-900'
                          }`}
                        >
                          Yes
                        </button>
                        <button
                          type="button"
                          onClick={() => setFormInput((prev) => ({ ...prev, [key]: 'no' }))}
                          className={`px-2 py-0.5 text-[11px] font-medium rounded transition-colors ${
                            val === 'no'
                              ? 'bg-white text-slate-900 shadow-xs'
                              : 'text-slate-600 hover:text-slate-900'
                          }`}
                        >
                          No
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="pt-2 flex flex-wrap items-center justify-between gap-4">
              <div className="text-xs text-slate-500">
                Pipeline passes inputs through fitted scaler before model inference.
              </div>
              <button
                type="submit"
                className="px-5 py-2.5 bg-blue-700 hover:bg-blue-800 text-white text-sm font-medium rounded-md transition-colors inline-flex items-center gap-2 whitespace-nowrap shrink-0 cursor-pointer"
              >
                Predict Price
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </form>
        </div>

        {/* Right Column: Prediction Output Display */}
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-lg p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-2 pb-4 border-b border-slate-200">
              <h2 className="text-lg font-semibold text-slate-900">Valuation Output</h2>
              <span className="text-xs text-slate-500 font-mono tabular-nums">
                R² = {activeEval.testMetrics.r2.toFixed(4)}
              </span>
            </div>

            {/* Active Model Selector */}
            <div className="mt-4">
              <label className="block text-xs font-medium text-slate-600 mb-1.5">
                Inference Model
              </label>
              <div className="grid grid-cols-3 gap-1 p-1 bg-slate-100 rounded-md">
                {(
                  [
                    { id: 'random_forest', label: 'Random Forest' },
                    { id: 'decision_tree', label: 'Decision Tree' },
                    { id: 'linear', label: 'Linear Reg.' },
                  ] as const
                ).map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setSelectedModel(m.id)}
                    className={`px-2 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap truncate ${
                      selectedModel === m.id
                        ? 'bg-white text-slate-900 shadow-xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Primary Predicted Price Readout */}
            <div className="mt-6 py-5 border-y border-slate-200">
              <div className="text-xs font-medium text-slate-500">
                Predicted House Price ({activeEval.shortName})
              </div>
              <div className="mt-1.5 text-3xl md:text-4xl font-bold font-mono tabular-nums text-slate-900 tracking-tight">
                {formatINR(prediction.predictedPrice)}
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-slate-600 font-mono tabular-nums">
                <span>{formatLakhs(prediction.predictedPrice)}</span>
                <span aria-hidden="true">·</span>
                <span>{formatINR(prediction.pricePerSqFt)} / sq.ft.</span>
              </div>
              <div className="mt-3 text-xs text-slate-500 font-mono tabular-nums">
                Estimated Range: {formatLakhs(prediction.lowerBound)} –{' '}
                {formatLakhs(prediction.upperBound)}
              </div>
            </div>

            {/* Cross-Model Comparison for Current Input */}
            <div className="mt-5">
              <div className="text-xs font-medium text-slate-700 mb-2">
                All Models on Submitted Input
              </div>
              <div className="divide-y divide-slate-100 border border-slate-200 rounded-md">
                {allModelComparison.map((row) => (
                  <div
                    key={row.id}
                    className={`flex items-center justify-between px-3 py-2 text-xs ${
                      row.id === selectedModel ? 'bg-blue-50/60 font-medium text-slate-900' : 'text-slate-600'
                    }`}
                  >
                    <span>
                      {row.name}
                      {row.id === artifacts.bestModelId ? ' (Final)' : ''}
                    </span>
                    <span className="font-mono tabular-nums text-slate-900">
                      {formatINR(row.predicted)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Top Feature Drivers */}
            <div className="mt-5">
              <div className="text-xs font-medium text-slate-700 mb-2">
                Key Valuation Drivers (Relative to Dataset Mean)
              </div>
              <div className="space-y-1.5">
                {prediction.contributions.slice(0, 4).map((c) => {
                  const positive = c.impact >= 0;
                  return (
                    <div
                      key={c.label}
                      className="flex items-center justify-between text-xs py-1 border-b border-slate-100 last:border-0"
                    >
                      <span className="text-slate-700">{c.label}</span>
                      <span
                        className={`font-mono tabular-nums font-medium ${
                          positive ? 'text-emerald-700' : 'text-red-600'
                        }`}
                      >
                        {positive ? '+' : '−'}
                        {formatLakhs(Math.abs(c.impact))}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Full-Width 4-Column Pipeline Summary Strip */}
      <section className="grid grid-cols-2 lg:grid-cols-4 border border-slate-200 bg-white rounded-lg divide-y lg:divide-y-0 lg:divide-x divide-slate-200">
        <div className="p-5">
          <div className="text-xs text-slate-500">Final Selected Model</div>
          <div className="mt-1 text-base font-semibold text-slate-900">
            {artifacts.models[artifacts.bestModelId].shortName}
          </div>
          <div className="mt-1 text-xs text-slate-500">Highest test R² & stability</div>
        </div>
        <div className="p-5">
          <div className="text-xs text-slate-500">Test Set R² Score</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-blue-700">
            {artifacts.models[artifacts.bestModelId].testMetrics.r2.toFixed(4)}
          </div>
          <div className="mt-1 text-xs text-slate-500">
            Explains {(artifacts.models[artifacts.bestModelId].testMetrics.r2 * 100).toFixed(1)}% price variance
          </div>
        </div>
        <div className="p-5">
          <div className="text-xs text-slate-500">Test Set RMSE</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900">
            {formatINR(artifacts.models[artifacts.bestModelId].testMetrics.rmse)}
          </div>
          <div className="mt-1 text-xs text-slate-500">
            MAE: {formatINR(artifacts.models[artifacts.bestModelId].testMetrics.mae)}
          </div>
        </div>
        <div className="p-5">
          <div className="text-xs text-slate-500">Dataset Split (80/20)</div>
          <div className="mt-1 text-xl font-bold font-mono tabular-nums text-slate-900">
            {artifacts.splitInfo.trainRows} / {artifacts.splitInfo.testRows}
          </div>
          <div className="mt-1 text-xs text-slate-500">
            random_state = {artifacts.splitInfo.randomState}
          </div>
        </div>
      </section>

      {/* Part A: Problem Definition, Aim & Objectives */}
      <section className="bg-white border border-slate-200 rounded-lg p-6 space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-200">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              01. Problem Definition, Aim & Objectives (Part A)
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Formal specification for Roll No. 1 — House Price Prediction
            </p>
          </div>
          <button
            type="button"
            onClick={() => onNavigate('data')}
            className="text-xs font-medium text-blue-700 hover:text-blue-800 inline-flex items-center gap-1 whitespace-nowrap shrink-0"
          >
            Inspect Dataset & EDA
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-4 text-sm text-slate-700 leading-relaxed">
            <div>
              <h3 className="text-xs font-semibold text-slate-900 mb-1">
                Problem Statement & Real-World Requirement
              </h3>
              <p>
                Residential property buyers, sellers, and mortgage lenders require objective, data-driven property appraisals. Manual valuation varies widely based on subjective estimates. This system predicts a house&apos;s fair market selling price using 12 measurable structural and amenity characteristics.
              </p>
            </div>
            <div>
              <h3 className="text-xs font-semibold text-slate-900 mb-1">
                Why Supervised Regression Is Appropriate
              </h3>
              <p>
                The expected output variable (<code className="font-mono text-xs bg-slate-100 px-1 py-0.5 rounded">price</code>) is a <strong>continuous numerical quantity</strong> measured in INR rather than a discrete class label or unlabeled cluster. Therefore, supervised regression algorithms (Linear Regression, Decision Tree Regression, Random Forest Regression) evaluated via MAE, MSE, RMSE, and R² are the mathematically appropriate choice.
              </p>
            </div>
            <div className="grid grid-cols-2 gap-4 pt-2 border-t border-slate-100">
              <div>
                <div className="text-xs font-semibold text-slate-900">Input Variables (X)</div>
                <div className="text-xs text-slate-600 mt-1">
                  12 features: Area, Bedrooms, Bathrooms, Stories, Parking, Furnishing, Main Road, Guest Room, Basement, Hot Water, AC, Preferred Area.
                </div>
              </div>
              <div>
                <div className="text-xs font-semibold text-slate-900">Target Output (y)</div>
                <div className="text-xs text-slate-600 mt-1">
                  Continuous Selling Price in Indian Rupees (INR), ranging from ₹17.5 Lakhs to ₹1.35 Crores after IQR outlier filtering.
                </div>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-xs font-semibold text-slate-900 mb-2">
              Project Aim & Specific Objectives
            </h3>
            <p className="text-sm text-slate-700 mb-3">
              <strong>Aim:</strong> To develop, evaluate, and deploy a Machine Learning regression model that accurately predicts the selling price of a house based on its characteristics.
            </p>
            <ol className="space-y-1.5 text-xs text-slate-600 list-decimal list-inside">
              <li>Understand a continuous supervised regression problem and real-world valuation criteria.</li>
              <li>Inspect and analyze raw housing data dimensions, types, missing entries, duplicates, and outliers.</li>
              <li>Apply justified preprocessing: median imputation, deduplication, IQR filtering, encoding, and StandardScaler.</li>
              <li>Perform Exploratory Data Analysis (EDA) across 5 statistical visualizations with documented observations.</li>
              <li>Select input features (X) and target (y), and partition into 80% Train / 20% Test (<code className="font-mono">random_state=42</code>).</li>
              <li>Implement and compare Linear Regression, Decision Tree Regression, and Random Forest Regression.</li>
              <li>Conduct model improvement experiments (pruning, scaling, feature selection, ensemble tuning) and interpret results.</li>
              <li>Deploy an interactive Front-End prediction UI and organize complete GitHub repository artifacts.</li>
            </ol>
          </div>
        </div>
      </section>
    </div>
  );
}
