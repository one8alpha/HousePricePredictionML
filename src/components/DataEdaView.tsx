import React, { useState } from 'react';
import { formatINR, formatLakhs } from '../ml/dataset';
import { PipelineArtifacts } from '../ml/types';
import {
  AmenityBarChart,
  AreaVsPriceScatterChart,
  CorrelationHeatmap,
  FurnishingBoxPlot,
  PriceDistributionChart,
} from './Charts';

export function DataEdaView({ artifacts }: { artifacts: PipelineArtifacts }) {
  const [tableMode, setTableMode] = useState<'raw' | 'cleaned'>('raw');
  const { inspection, preprocessingReport, rawData, cleanedData, correlationMatrix } =
    artifacts;

  const totalImputed = Object.values(preprocessingReport.imputedCounts).reduce(
    (a, b) => a + b,
    0
  );

  return (
    <div className="space-y-8">
      {/* Header */}
      <section className="border-b border-slate-200 pb-5">
        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 font-medium mb-1.5">
          <span>Parts B, C & D</span>
          <span aria-hidden="true">·</span>
          <span>UCI / Kaggle Style Housing Dataset</span>
          <span aria-hidden="true">·</span>
          <span>{inspection.rawRows} Raw → {preprocessingReport.finalRows} Cleaned Rows</span>
        </div>
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-slate-900">
          Dataset Understanding, Preprocessing & EDA
        </h1>
        <p className="mt-1.5 text-sm text-slate-600 max-w-3xl">
          Complete inspection of raw dataset anomalies, step-by-step preprocessing transformations with justifications, and 5 statistical visualizations with analytical observations.
        </p>
      </section>

      {/* Part B: Dataset Understanding */}
      <section className="bg-white border border-slate-200 rounded-lg p-6 space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-200">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              02. Dataset Understanding & Schema Inspection (Part B)
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Source: Housing Prices Dataset (13 attributes: 5 numerical inputs, 7 categorical inputs, 1 continuous target)
            </p>
          </div>
          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md">
            <button
              type="button"
              onClick={() => setTableMode('raw')}
              className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap ${
                tableMode === 'raw'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Raw First 5 Records
            </button>
            <button
              type="button"
              onClick={() => setTableMode('cleaned')}
              className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap ${
                tableMode === 'cleaned'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Cleaned & Encoded (Top 5)
            </button>
          </div>
        </div>

        {/* Dimensions & Anomaly Summary Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pb-4 border-b border-slate-100">
          <div>
            <div className="text-xs text-slate-500">Raw Dimensions</div>
            <div className="mt-0.5 text-lg font-semibold font-mono tabular-nums text-slate-900">
              {inspection.rawRows} × {inspection.rawCols}
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-500">Missing Values Detected</div>
            <div className="mt-0.5 text-lg font-semibold font-mono tabular-nums text-amber-600">
              {inspection.totalMissing} cells
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-500">Duplicate Records</div>
            <div className="mt-0.5 text-lg font-semibold font-mono tabular-nums text-amber-600">
              {inspection.duplicateCount} rows
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-500">IQR Outliers (Price / Area)</div>
            <div className="mt-0.5 text-lg font-semibold font-mono tabular-nums text-red-600">
              {inspection.outlierCountPrice} / {inspection.outlierCountArea}
            </div>
          </div>
        </div>

        {/* First 5 Records Preview Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500">
                <th className="py-2 pr-3 font-medium">ID</th>
                <th className="py-2 px-3 font-medium text-right">Price (y)</th>
                <th className="py-2 px-3 font-medium text-right">Area</th>
                <th className="py-2 px-3 font-medium text-right">Bed</th>
                <th className="py-2 px-3 font-medium text-right">Bath</th>
                <th className="py-2 px-3 font-medium text-right">Stories</th>
                <th className="py-2 px-3 font-medium">Main Road</th>
                <th className="py-2 px-3 font-medium">AC</th>
                <th className="py-2 px-3 font-medium text-right">Parking</th>
                <th className="py-2 px-3 font-medium">Pref. Area</th>
                <th className="py-2 pl-3 font-medium">Furnishing</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono tabular-nums">
              {(tableMode === 'raw' ? rawData.slice(0, 5) : cleanedData.slice(0, 5)).map(
                (row) => (
                  <tr key={row.id} className="hover:bg-slate-50">
                    <td className="py-2 pr-3 text-slate-500">#{row.id}</td>
                    <td className="py-2 px-3 text-right font-medium text-slate-900">
                      {row.price !== null ? formatINR(row.price) : 'NaN'}
                    </td>
                    <td className="py-2 px-3 text-right text-slate-700">
                      {row.area !== null ? `${row.area}` : 'NaN'}
                    </td>
                    <td className="py-2 px-3 text-right text-slate-700">
                      {row.bedrooms ?? 'NaN'}
                    </td>
                    <td className="py-2 px-3 text-right text-slate-700">
                      {row.bathrooms ?? 'NaN'}
                    </td>
                    <td className="py-2 px-3 text-right text-slate-700">{row.stories}</td>
                    <td className="py-2 px-3 text-slate-700">{row.mainroad}</td>
                    <td className="py-2 px-3 text-slate-700">{row.airconditioning}</td>
                    <td className="py-2 px-3 text-right text-slate-700">
                      {row.parking ?? 'NaN'}
                    </td>
                    <td className="py-2 px-3 text-slate-700">{row.prefarea}</td>
                    <td className="py-2 pl-3 text-slate-700">{row.furnishingstatus}</td>
                  </tr>
                )
              )}
            </tbody>
          </table>
        </div>

        {/* Column Schema & Summary Statistics */}
        <div className="pt-2">
          <h3 className="text-sm font-semibold text-slate-900 mb-3">
            Feature Types, Missing Counts & Descriptive Statistics
          </h3>
          <div className="overflow-x-auto border border-slate-200 rounded-md">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-slate-600">
                  <th className="py-2.5 px-3 font-medium">Column</th>
                  <th className="py-2.5 px-3 font-medium">Data Type</th>
                  <th className="py-2.5 px-3 font-medium">Category</th>
                  <th className="py-2.5 px-3 font-medium text-right">Missing</th>
                  <th className="py-2.5 px-3 font-medium text-right">Unique</th>
                  <th className="py-2.5 px-3 font-medium text-right">Min</th>
                  <th className="py-2.5 px-3 font-medium text-right">Median</th>
                  <th className="py-2.5 px-3 font-medium text-right">Max</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {inspection.columns.map((col) => (
                  <tr key={col.name} className="hover:bg-slate-50/70">
                    <td className="py-2 px-3 font-mono font-medium text-slate-900">
                      {col.name}
                    </td>
                    <td className="py-2 px-3 font-mono text-slate-600">{col.dtype}</td>
                    <td className="py-2 px-3 text-slate-700">{col.kind}</td>
                    <td
                      className={`py-2 px-3 text-right font-mono tabular-nums ${
                        col.missingCount > 0 ? 'text-amber-700 font-semibold' : 'text-slate-500'
                      }`}
                    >
                      {col.missingCount}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-700">
                      {col.uniqueCount}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-700">
                      {col.min !== undefined
                        ? col.name === 'price'
                          ? formatLakhs(col.min)
                          : col.min.toLocaleString()
                        : '—'}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-700">
                      {col.median !== undefined
                        ? col.name === 'price'
                          ? formatLakhs(col.median)
                          : col.median.toLocaleString()
                        : '—'}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-700">
                      {col.max !== undefined
                        ? col.name === 'price'
                          ? formatLakhs(col.max)
                          : col.max.toLocaleString()
                        : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Part C: Data Preprocessing Pipeline & Justifications */}
      <section className="bg-white border border-slate-200 rounded-lg p-6 space-y-5">
        <div className="pb-4 border-b border-slate-200">
          <h2 className="text-lg font-semibold text-slate-900">
            03. Data Preprocessing & Engineering Decisions (Part C)
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Every transformation applied to the dataset, its quantitative impact, and the statistical justification.
          </p>
        </div>

        <div className="divide-y divide-slate-200">
          {[
            {
              step: '1. Missing Value Imputation',
              impact: `${totalImputed} missing cells imputed (area: ${preprocessingReport.imputedCounts.area}, bathrooms: ${preprocessingReport.imputedCounts.bathrooms}, bedrooms: ${preprocessingReport.imputedCounts.bedrooms}, parking: ${preprocessingReport.imputedCounts.parking})`,
              justification:
                'Used column Median instead of Mean because housing prices and plot areas are right-skewed. Median imputation preserves sample size (545 rows) without dragging central tendency toward extreme estates.',
            },
            {
              step: '2. Duplicate Record Removal',
              impact: `${preprocessingReport.duplicatesRemoved} exact duplicate rows removed (${preprocessingReport.afterMissingRows} → ${preprocessingReport.afterMissingRows - preprocessingReport.duplicatesRemoved} rows)`,
              justification:
                'Duplicate listings bias model weights toward repeated properties and cause data leakage if one copy lands in Train and the other in Test, artificially inflating R².',
            },
            {
              step: '3. Outlier Treatment (1.5 × IQR Rule)',
              impact: `${preprocessingReport.outliersRemoved} extreme outliers removed (Price upper bound: ${formatLakhs(inspection.priceBounds.upper)}, Area upper bound: ${Math.round(inspection.areaBounds.upper).toLocaleString()} sq.ft.)`,
              justification:
                'Ordinary Least Squares minimizes squared errors; extreme 13,000–16,200 sq.ft. luxury outliers exert disproportionate leverage on linear regression slopes. Filtering via IQR stabilizes predictions for residential homes.',
            },
            {
              step: '4. Categorical Variable Encoding',
              impact: '6 binary attributes mapped (yes=1, no=0); furnishingstatus ordinal-encoded (unfurnished=0, semi-furnished=1, furnished=2)',
              justification:
                'Regression equations and CART splits require numeric matrices. Binary mapping preserves interpretable slope weights; ordinal encoding reflects the natural monotonic progression of furnishing value.',
            },
            {
              step: '5. Feature Scaling (StandardScaler)',
              impact: `Fitted strictly on ${artifacts.splitInfo.trainRows} training rows (z = (x − μ) / σ), then applied to test & inference inputs`,
              justification:
                'Area ranges from 1,050–9,500 while bedrooms range from 1–5. Standardizing ensures numerical stability in matrix inversion and makes Linear Regression coefficients directly comparable in magnitude.',
            },
          ].map((item) => (
            <div key={item.step} className="py-4 grid grid-cols-1 md:grid-cols-12 gap-4">
              <div className="md:col-span-3">
                <div className="text-sm font-semibold text-slate-900">{item.step}</div>
                <div className="text-xs font-mono text-blue-700 mt-1">{item.impact}</div>
              </div>
              <div className="md:col-span-9 text-sm text-slate-600 leading-relaxed">
                {item.justification}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Part D: Exploratory Data Analysis (5 Visualizations + Observations) */}
      <section className="space-y-6">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">
            04. Exploratory Data Analysis & Visual Observations (Part D)
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Five statistical visualizations generated directly from the cleaned housing dataset ({cleanedData.length} records).
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* EDA 1: Price Distribution */}
          <div className="bg-white border border-slate-200 rounded-lg p-6 space-y-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">
                Visualization 1: House Price Distribution Histogram
              </h3>
              <p className="text-xs text-slate-500">Frequency distribution across INR Lakhs bins</p>
            </div>
            <PriceDistributionChart data={cleanedData} />
            <div className="pt-2 border-t border-slate-100 text-xs text-slate-700 leading-relaxed">
              <strong>Observation 1:</strong> House prices follow a mildly right-skewed unimodal distribution where the Mean sits slightly above the Median, concentrated between ₹55L and ₹95L after IQR filtering.
            </div>
          </div>

          {/* EDA 2: Area vs Price Scatter */}
          <div className="bg-white border border-slate-200 rounded-lg p-6 space-y-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">
                Visualization 2: Area vs. Price Scatter & Trendline
              </h3>
              <p className="text-xs text-slate-500">Grouped by Air Conditioning status with OLS fit</p>
            </div>
            <AreaVsPriceScatterChart data={cleanedData} />
            <div className="pt-2 border-t border-slate-100 text-xs text-slate-700 leading-relaxed">
              <strong>Observation 2:</strong> Strong positive linear correlation between plot area and selling price; properties with Air Conditioning (blue) consistently cluster above the regression line at comparable square footages.
            </div>
          </div>

          {/* EDA 3: Box Plot by Furnishing */}
          <div className="bg-white border border-slate-200 rounded-lg p-6 space-y-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">
                Visualization 3: Price Spread by Furnishing Status (Box Plot)
              </h3>
              <p className="text-xs text-slate-500">Quartiles, median, whiskers, and mild upper outliers</p>
            </div>
            <FurnishingBoxPlot data={cleanedData} />
            <div className="pt-2 border-t border-slate-100 text-xs text-slate-700 leading-relaxed">
              <strong>Observation 3:</strong> Median valuation steps up progressively from Unfurnished to Semi-Furnished to Furnished homes, validating our ordinal encoding (<code className="font-mono">0 &lt; 1 &lt; 2</code>).
            </div>
          </div>

          {/* EDA 4: Amenities Bar Chart */}
          <div className="bg-white border border-slate-200 rounded-lg p-6 space-y-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">
                Visualization 4: Mean Price Impact of Key Amenities
              </h3>
              <p className="text-xs text-slate-500">Average selling price across AC, Locality, and Parking tiers</p>
            </div>
            <AmenityBarChart data={cleanedData} />
            <div className="pt-2 border-t border-slate-100 text-xs text-slate-700 leading-relaxed">
              <strong>Observation 4:</strong> Homes with Air Conditioning, Preferred Locality status, and 2+ parking spaces command substantial average price premiums (~₹15L–₹25L higher on average).
            </div>
          </div>
        </div>

        {/* EDA 5: Full Correlation Matrix Heatmap */}
        <div className="bg-white border border-slate-200 rounded-lg p-6 space-y-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">
              Visualization 5: Feature Correlation Matrix Heatmap
            </h3>
            <p className="text-xs text-slate-500">
              Pairwise Pearson correlation (r) across target price and all 12 encoded features
            </p>
          </div>
          <CorrelationHeatmap matrix={correlationMatrix} />
          <div className="pt-2 border-t border-slate-100 text-xs text-slate-700 leading-relaxed">
            <strong>Observation 5:</strong> Target <code className="font-mono">price</code> exhibits the strongest positive correlation with <code className="font-mono">area</code>, followed by <code className="font-mono">bathrooms</code>, <code className="font-mono">airconditioning</code>, <code className="font-mono">stories</code>, and <code className="font-mono">parking</code>, confirming high predictive signal with low multicollinearity across predictors.
          </div>
        </div>
      </section>
    </div>
  );
}
