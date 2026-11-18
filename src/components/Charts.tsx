import React, { useMemo, useState } from 'react';
import { formatINR, formatLakhs } from '../ml/dataset';
import { CleanHousingRecord, ModelEvaluation, TreeNode } from '../ml/types';

// 1. Price Distribution Histogram (EDA Plot 1)
export function PriceDistributionChart({ data }: { data: CleanHousingRecord[] }) {
  const [hoveredBin, setHoveredBin] = useState<number | null>(null);

  const { bins, maxCount, meanPrice, medianPrice } = useMemo(() => {
    const prices = data.map((d) => d.price / 100000).sort((a, b) => a - b);
    const min = Math.floor(prices[0] / 10) * 10;
    const max = Math.ceil(prices[prices.length - 1] / 10) * 10;
    const numBins = 14;
    const step = (max - min) / numBins || 10;

    const histogram = Array.from({ length: numBins }, (_, i) => {
      const start = min + i * step;
      const end = start + step;
      const count = prices.filter((p) =>
        i === numBins - 1 ? p >= start && p <= end : p >= start && p < end
      ).length;
      return { start, end, count };
    });

    const mean = prices.reduce((a, b) => a + b, 0) / (prices.length || 1);
    const med = prices[Math.floor(prices.length / 2)] || 0;
    const peak = Math.max(...histogram.map((b) => b.count), 1);

    return { bins: histogram, maxCount: peak, meanPrice: mean, medianPrice: med };
  }, [data]);

  const W = 560;
  const H = 250;
  const pad = { top: 24, right: 20, bottom: 38, left: 44 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const minX = bins[0]?.start ?? 0;
  const maxX = bins[bins.length - 1]?.end ?? 100;
  const xScale = (val: number) =>
    pad.left + ((val - minX) / (maxX - minX || 1)) * plotW;

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto overflow-visible">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const y = pad.top + plotH * (1 - t);
          const tickVal = Math.round(maxCount * t);
          return (
            <g key={t}>
              <line
                x1={pad.left}
                x2={W - pad.right}
                y1={y}
                y2={y}
                stroke="#e2e8f0"
                strokeDasharray={t === 0 ? undefined : '3 3'}
              />
              <text
                x={pad.left - 8}
                y={y + 4}
                textAnchor="end"
                className="fill-slate-500 text-[10px] font-mono tabular-nums"
              >
                {tickVal}
              </text>
            </g>
          );
        })}

        {bins.map((b, idx) => {
          const barW = plotW / bins.length - 4;
          const x = pad.left + idx * (plotW / bins.length) + 2;
          const barH = (b.count / maxCount) * plotH;
          const y = pad.top + plotH - barH;
          const active = hoveredBin === idx;

          return (
            <g
              key={idx}
              onMouseEnter={() => setHoveredBin(idx)}
              onMouseLeave={() => setHoveredBin(null)}
              className="cursor-pointer"
            >
              <rect
                x={x}
                y={y}
                width={Math.max(4, barW)}
                height={barH}
                rx={2}
                fill={active ? '#1d4ed8' : '#3b82f6'}
                fillOpacity={active ? 0.95 : 0.82}
              />
              {idx % 2 === 0 && (
                <text
                  x={x + barW / 2}
                  y={H - 14}
                  textAnchor="middle"
                  className="fill-slate-500 text-[10px] font-mono tabular-nums"
                >
                  {Math.round(b.start)}L
                </text>
              )}
            </g>
          );
        })}

        {/* Mean & Median Lines */}
        <line
          x1={xScale(meanPrice)}
          x2={xScale(meanPrice)}
          y1={pad.top}
          y2={pad.top + plotH}
          stroke="#dc2626"
          strokeWidth={1.5}
          strokeDasharray="4 3"
        />
        <text
          x={xScale(meanPrice) + 4}
          y={pad.top + 10}
          className="fill-red-600 text-[10px] font-mono font-medium"
        >
          Mean {meanPrice.toFixed(1)}L
        </text>

        <line
          x1={xScale(medianPrice)}
          x2={xScale(medianPrice)}
          y1={pad.top}
          y2={pad.top + plotH}
          stroke="#059669"
          strokeWidth={1.5}
          strokeDasharray="4 3"
        />
        <text
          x={xScale(medianPrice) - 4}
          y={pad.top + 22}
          textAnchor="end"
          className="fill-emerald-700 text-[10px] font-mono font-medium"
        >
          Median {medianPrice.toFixed(1)}L
        </text>
      </svg>

      {hoveredBin !== null && bins[hoveredBin] && (
        <div className="mt-2 flex items-center justify-between text-xs text-slate-600 font-mono tabular-nums bg-slate-50 border border-slate-200 rounded px-3 py-1.5">
          <span>
            Range: ₹{bins[hoveredBin].start.toFixed(0)}L – ₹{bins[hoveredBin].end.toFixed(0)}L
          </span>
          <span className="font-semibold text-slate-900">
            {bins[hoveredBin].count} properties ({((bins[hoveredBin].count / data.length) * 100).toFixed(1)}%)
          </span>
        </div>
      )}
    </div>
  );
}

// 2. Area vs Price Scatter Plot with OLS Regression Line (EDA Plot 2)
export function AreaVsPriceScatterChart({ data }: { data: CleanHousingRecord[] }) {
  const [hovered, setHovered] = useState<CleanHousingRecord | null>(null);

  const { minA, maxA, minP, maxP, slope, intercept } = useMemo(() => {
    const areas = data.map((d) => d.area);
    const prices = data.map((d) => d.price / 100000);
    const mnA = Math.min(...areas);
    const mxA = Math.max(...areas);
    const mnP = Math.min(...prices);
    const mxP = Math.max(...prices);

    const n = data.length || 1;
    const meanA = areas.reduce((a, b) => a + b, 0) / n;
    const meanP = prices.reduce((a, b) => a + b, 0) / n;
    let num = 0;
    let den = 0;
    for (let i = 0; i < n; i++) {
      num += (areas[i] - meanA) * (prices[i] - meanP);
      den += (areas[i] - meanA) ** 2;
    }
    const m = den === 0 ? 0 : num / den;
    const b = meanP - m * meanA;

    return { minA: mnA, maxA: mxA, minP: mnP, maxP: mxP, slope: m, intercept: b };
  }, [data]);

  const W = 560;
  const H = 250;
  const pad = { top: 20, right: 20, bottom: 38, left: 48 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const xScale = (a: number) =>
    pad.left + ((a - minA) / (maxA - minA || 1)) * plotW;
  const yScale = (p: number) =>
    pad.top + plotH - ((p - minP) / (maxP - minP || 1)) * plotH;

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto overflow-visible">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const val = minP + (maxP - minP) * t;
          const y = yScale(val);
          return (
            <g key={t}>
              <line
                x1={pad.left}
                x2={W - pad.right}
                y1={y}
                y2={y}
                stroke="#e2e8f0"
                strokeDasharray={t === 0 ? undefined : '3 3'}
              />
              <text
                x={pad.left - 8}
                y={y + 4}
                textAnchor="end"
                className="fill-slate-500 text-[10px] font-mono tabular-nums"
              >
                ₹{Math.round(val)}L
              </text>
            </g>
          );
        })}

        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const val = minA + (maxA - minA) * t;
          const x = xScale(val);
          return (
            <text
              key={t}
              x={x}
              y={H - 14}
              textAnchor="middle"
              className="fill-slate-500 text-[10px] font-mono tabular-nums"
            >
              {Math.round(val).toLocaleString()}
            </text>
          );
        })}

        {data.map((d) => {
          const cx = xScale(d.area);
          const cy = yScale(d.price / 100000);
          const isHovered = hovered?.id === d.id;
          return (
            <circle
              key={d.id}
              cx={cx}
              cy={cy}
              r={isHovered ? 5.5 : 3}
              fill={d.airconditioning === 1 ? '#1d4ed8' : '#64748b'}
              fillOpacity={isHovered ? 1 : 0.6}
              stroke={isHovered ? '#0f172a' : 'none'}
              strokeWidth={1.5}
              onMouseEnter={() => setHovered(d)}
              onMouseLeave={() => setHovered(null)}
              className="cursor-pointer"
            />
          );
        })}

        {/* Regression Line */}
        <line
          x1={xScale(minA)}
          y1={yScale(slope * minA + intercept)}
          x2={xScale(maxA)}
          y2={yScale(slope * maxA + intercept)}
          stroke="#dc2626"
          strokeWidth={2}
        />
      </svg>

      <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600">
        <div className="flex items-center gap-4">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-700 inline-block" />
            With Air Conditioning
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-slate-500 inline-block" />
            No AC
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-4 h-0.5 bg-red-600 inline-block" />
            OLS Trendline
          </span>
        </div>
        {hovered && (
          <span className="font-mono tabular-nums text-slate-900">
            Area: {hovered.area.toLocaleString()} sq.ft. · Price: {formatLakhs(hovered.price)} · {hovered.bedrooms} Bed
          </span>
        )}
      </div>
    </div>
  );
}

// 3. Box Plot by Furnishing Status (EDA Plot 3)
export function FurnishingBoxPlot({ data }: { data: CleanHousingRecord[] }) {
  const groups = useMemo(() => {
    const categories: Array<{ key: CleanHousingRecord['furnishingLabel']; label: string }> = [
      { key: 'unfurnished', label: 'Unfurnished' },
      { key: 'semi-furnished', label: 'Semi-Furnished' },
      { key: 'furnished', label: 'Furnished' },
    ];

    return categories.map((cat) => {
      const vals = data
        .filter((d) => d.furnishingLabel === cat.key)
        .map((d) => d.price / 100000)
        .sort((a, b) => a - b);

      const q = (p: number) => vals[Math.floor((vals.length - 1) * p)] || 0;
      const q1 = q(0.25);
      const med = q(0.5);
      const q3 = q(0.75);
      const iqr = q3 - q1;
      const lowerW = Math.max(vals[0] || 0, q1 - 1.5 * iqr);
      const upperW = Math.min(vals[vals.length - 1] || 0, q3 + 1.5 * iqr);
      const outliers = vals.filter((v) => v < lowerW || v > upperW);

      return { ...cat, count: vals.length, q1, med, q3, lowerW, upperW, outliers };
    });
  }, [data]);

  const maxP = Math.max(...groups.map((g) => Math.max(g.upperW, ...g.outliers, 100)));
  const minP = Math.min(...groups.map((g) => Math.min(g.lowerW, 15)));

  const W = 520;
  const H = 240;
  const pad = { top: 20, right: 20, bottom: 36, left: 48 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const yScale = (v: number) =>
    pad.top + plotH - ((v - minP) / (maxP - minP || 1)) * plotH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto overflow-visible">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => {
        const val = minP + (maxP - minP) * t;
        const y = yScale(val);
        return (
          <g key={t}>
            <line
              x1={pad.left}
              x2={W - pad.right}
              y1={y}
              y2={y}
              stroke="#e2e8f0"
              strokeDasharray={t === 0 ? undefined : '3 3'}
            />
            <text
              x={pad.left - 8}
              y={y + 4}
              textAnchor="end"
              className="fill-slate-500 text-[10px] font-mono tabular-nums"
            >
              ₹{Math.round(val)}L
            </text>
          </g>
        );
      })}

      {groups.map((g, i) => {
        const cx = pad.left + (i + 0.5) * (plotW / groups.length);
        const boxW = 54;
        return (
          <g key={g.key}>
            {/* Whisker stem */}
            <line
              x1={cx}
              x2={cx}
              y1={yScale(g.upperW)}
              y2={yScale(g.lowerW)}
              stroke="#475569"
              strokeWidth={1.5}
            />
            {/* Whisker caps */}
            <line
              x1={cx - 14}
              x2={cx + 14}
              y1={yScale(g.upperW)}
              y2={yScale(g.upperW)}
              stroke="#475569"
              strokeWidth={1.5}
            />
            <line
              x1={cx - 14}
              x2={cx + 14}
              y1={yScale(g.lowerW)}
              y2={yScale(g.lowerW)}
              stroke="#475569"
              strokeWidth={1.5}
            />
            {/* IQR Box */}
            <rect
              x={cx - boxW / 2}
              y={yScale(g.q3)}
              width={boxW}
              height={Math.max(2, yScale(g.q1) - yScale(g.q3))}
              fill="#dbeafe"
              stroke="#1d4ed8"
              strokeWidth={1.5}
              rx={3}
            />
            {/* Median Line */}
            <line
              x1={cx - boxW / 2}
              x2={cx + boxW / 2}
              y1={yScale(g.med)}
              y2={yScale(g.med)}
              stroke="#0f172a"
              strokeWidth={2}
            />
            {/* Median Label */}
            <text
              x={cx + boxW / 2 + 6}
              y={yScale(g.med) + 3}
              className="fill-slate-700 text-[10px] font-mono tabular-nums"
            >
              {g.med.toFixed(1)}L
            </text>
            {/* Outliers */}
            {g.outliers.slice(0, 6).map((ov, oi) => (
              <circle
                key={oi}
                cx={cx}
                cy={yScale(ov)}
                r={2.5}
                fill="#d97706"
              />
            ))}
            <text
              x={cx}
              y={H - 12}
              textAnchor="middle"
              className="fill-slate-700 text-[11px] font-medium"
            >
              {g.label} ({g.count})
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// 4. Average Price by Key Categorical Features Bar Chart (EDA Plot 4)
export function AmenityBarChart({ data }: { data: CleanHousingRecord[] }) {
  const bars = useMemo(() => {
    const avg = (arr: CleanHousingRecord[]) =>
      arr.length ? arr.reduce((a, b) => a + b.price / 100000, 0) / arr.length : 0;

    return [
      { label: 'No AC', val: avg(data.filter((d) => d.airconditioning === 0)), tone: '#94a3b8' },
      { label: 'With AC', val: avg(data.filter((d) => d.airconditioning === 1)), tone: '#1d4ed8' },
      { label: 'Standard Area', val: avg(data.filter((d) => d.prefarea === 0)), tone: '#94a3b8' },
      { label: 'Pref. Area', val: avg(data.filter((d) => d.prefarea === 1)), tone: '#059669' },
      { label: '0 Parking', val: avg(data.filter((d) => d.parking === 0)), tone: '#94a3b8' },
      { label: '2+ Parking', val: avg(data.filter((d) => d.parking >= 2)), tone: '#0f172a' },
    ];
  }, [data]);

  const maxVal = Math.max(...bars.map((b) => b.val), 100);
  const W = 520;
  const H = 240;
  const pad = { top: 24, right: 16, bottom: 40, left: 48 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto overflow-visible">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => {
        const val = maxVal * t;
        const y = pad.top + plotH * (1 - t);
        return (
          <g key={t}>
            <line
              x1={pad.left}
              x2={W - pad.right}
              y1={y}
              y2={y}
              stroke="#e2e8f0"
              strokeDasharray={t === 0 ? undefined : '3 3'}
            />
            <text
              x={pad.left - 8}
              y={y + 4}
              textAnchor="end"
              className="fill-slate-500 text-[10px] font-mono tabular-nums"
            >
              ₹{Math.round(val)}L
            </text>
          </g>
        );
      })}

      {bars.map((b, i) => {
        const slotW = plotW / bars.length;
        const barW = Math.min(42, slotW - 14);
        const x = pad.left + i * slotW + (slotW - barW) / 2;
        const barH = (b.val / maxVal) * plotH;
        const y = pad.top + plotH - barH;
        return (
          <g key={b.label}>
            <rect
              x={x}
              y={y}
              width={barW}
              height={barH}
              rx={3}
              fill={b.tone}
            />
            <text
              x={x + barW / 2}
              y={y - 6}
              textAnchor="middle"
              className="fill-slate-800 text-[10px] font-mono font-semibold tabular-nums"
            >
              {b.val.toFixed(1)}L
            </text>
            <text
              x={x + barW / 2}
              y={H - 14}
              textAnchor="middle"
              className="fill-slate-600 text-[10px]"
            >
              {b.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// 5. Pearson Correlation Heatmap (EDA Plot 5)
export function CorrelationHeatmap({
  matrix,
}: {
  matrix: { labels: string[]; keys: string[]; values: number[][] };
}) {
  const [cell, setCell] = useState<{ i: number; j: number; val: number } | null>(
    null
  );

  const colorForCorr = (r: number) => {
    if (r >= 0) {
      const alpha = Math.min(0.92, Math.abs(r) * 0.9 + 0.05);
      return `rgba(29, 78, 216, ${alpha})`; // Blue scale for positive correlation
    }
    const alpha = Math.min(0.92, Math.abs(r) * 0.9 + 0.05);
    return `rgba(220, 38, 38, ${alpha})`; // Crimson scale for negative correlation
  };

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-[11px] font-mono tabular-nums">
          <thead>
            <tr>
              <th className="p-1.5 text-left text-slate-500 font-medium">Feature</th>
              {matrix.labels.map((lbl) => (
                <th
                  key={lbl}
                  className="p-1 text-center text-slate-600 font-medium whitespace-nowrap"
                >
                  {lbl}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.labels.map((rowLbl, i) => (
              <tr key={rowLbl}>
                <td className="p-1.5 text-slate-700 font-medium whitespace-nowrap">
                  {rowLbl}
                </td>
                {matrix.values[i].map((val, j) => {
                  const strong = Math.abs(val) > 0.48;
                  return (
                    <td
                      key={`${i}-${j}`}
                      onMouseEnter={() => setCell({ i, j, val })}
                      onMouseLeave={() => setCell(null)}
                      style={{ backgroundColor: colorForCorr(val) }}
                      className={`p-1 text-center border border-white/60 cursor-crosshair transition-opacity ${
                        strong ? 'text-white font-semibold' : 'text-slate-900'
                      }`}
                    >
                      {val.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-2.5 flex items-center justify-between text-xs text-slate-600">
        <span>Pearson Correlation Coefficient (r): −1.00 (Crimson) to +1.00 (Blue)</span>
        {cell && (
          <span className="font-mono font-medium text-slate-900 tabular-nums">
            corr({matrix.labels[cell.i]}, {matrix.labels[cell.j]}) = {cell.val.toFixed(3)}
          </span>
        )}
      </div>
    </div>
  );
}

// 6. Actual vs Predicted Plot (Model Visualization 1)
export function ActualVsPredictedChart({ model }: { model: ModelEvaluation }) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const pts = model.predictionsTest;
  const allVals = pts.flatMap((p) => [p.actual / 100000, p.predicted / 100000]);
  const minV = Math.floor(Math.min(...allVals) / 10) * 10;
  const maxV = Math.ceil(Math.max(...allVals) / 10) * 10;

  const W = 520;
  const H = 260;
  const pad = { top: 20, right: 20, bottom: 38, left: 48 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const xScale = (v: number) =>
    pad.left + ((v - minV) / (maxV - minV || 1)) * plotW;
  const yScale = (v: number) =>
    pad.top + plotH - ((v - minV) / (maxV - minV || 1)) * plotH;

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto overflow-visible">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const val = minV + (maxV - minV) * t;
          return (
            <g key={t}>
              <line
                x1={pad.left}
                x2={W - pad.right}
                y1={yScale(val)}
                y2={yScale(val)}
                stroke="#e2e8f0"
                strokeDasharray={t === 0 ? undefined : '3 3'}
              />
              <text
                x={pad.left - 8}
                y={yScale(val) + 4}
                textAnchor="end"
                className="fill-slate-500 text-[10px] font-mono tabular-nums"
              >
                {Math.round(val)}L
              </text>
              <text
                x={xScale(val)}
                y={H - 14}
                textAnchor="middle"
                className="fill-slate-500 text-[10px] font-mono tabular-nums"
              >
                {Math.round(val)}L
              </text>
            </g>
          );
        })}

        {/* Ideal y = x line */}
        <line
          x1={xScale(minV)}
          y1={yScale(minV)}
          x2={xScale(maxV)}
          y2={yScale(maxV)}
          stroke="#0f172a"
          strokeWidth={1.5}
          strokeDasharray="5 4"
        />

        {pts.map((p, idx) => {
          const cx = xScale(p.actual / 100000);
          const cy = yScale(p.predicted / 100000);
          const active = hoveredIdx === idx;
          return (
            <circle
              key={idx}
              cx={cx}
              cy={cy}
              r={active ? 5.5 : 3.4}
              fill="#1d4ed8"
              fillOpacity={active ? 1 : 0.72}
              stroke={active ? '#0f172a' : 'none'}
              strokeWidth={1.5}
              onMouseEnter={() => setHoveredIdx(idx)}
              onMouseLeave={() => setHoveredIdx(null)}
              className="cursor-pointer"
            />
          );
        })}
      </svg>
      <div className="mt-2 flex items-center justify-between text-xs text-slate-600">
        <span>X: Actual Price · Y: Predicted Price · Dashed: Ideal Fit (y = x)</span>
        {hoveredIdx !== null && pts[hoveredIdx] && (
          <span className="font-mono tabular-nums text-slate-900">
            Actual {formatLakhs(pts[hoveredIdx].actual)} vs Pred {formatLakhs(pts[hoveredIdx].predicted)}
          </span>
        )}
      </div>
    </div>
  );
}

// 7. Residual Plot (Model Visualization 2)
export function ResidualPlotChart({ model }: { model: ModelEvaluation }) {
  const pts = model.predictionsTest;
  const preds = pts.map((p) => p.predicted / 100000);
  const resids = pts.map((p) => p.residual / 100000);

  const minX = Math.min(...preds);
  const maxX = Math.max(...preds);
  const maxAbsR = Math.max(...resids.map((r) => Math.abs(r)), 5);

  const W = 520;
  const H = 260;
  const pad = { top: 20, right: 20, bottom: 38, left: 48 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const xScale = (v: number) =>
    pad.left + ((v - minX) / (maxX - minX || 1)) * plotW;
  const yScale = (r: number) =>
    pad.top + plotH - ((r + maxAbsR) / (2 * maxAbsR || 1)) * plotH;

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto overflow-visible">
        {[-maxAbsR, -maxAbsR / 2, 0, maxAbsR / 2, maxAbsR].map((val, i) => (
          <g key={i}>
            <line
              x1={pad.left}
              x2={W - pad.right}
              y1={yScale(val)}
              y2={yScale(val)}
              stroke={val === 0 ? '#0f172a' : '#e2e8f0'}
              strokeWidth={val === 0 ? 1.5 : 1}
              strokeDasharray={val === 0 ? '4 3' : '3 3'}
            />
            <text
              x={pad.left - 8}
              y={yScale(val) + 4}
              textAnchor="end"
              className="fill-slate-500 text-[10px] font-mono tabular-nums"
            >
              {val > 0 ? `+${val.toFixed(0)}L` : `${val.toFixed(0)}L`}
            </text>
          </g>
        ))}

        {pts.map((p, idx) => {
          const cx = xScale(p.predicted / 100000);
          const cy = yScale(p.residual / 100000);
          const large = Math.abs(p.residual) > model.testMetrics.rmse * 1.3;
          return (
            <circle
              key={idx}
              cx={cx}
              cy={cy}
              r={3.4}
              fill={large ? '#dc2626' : '#059669'}
              fillOpacity={0.72}
            />
          );
        })}
      </svg>
      <div className="mt-2 flex items-center justify-between text-xs text-slate-600">
        <span>Residual Error (Actual − Predicted) centered around 0L</span>
        <span className="font-mono tabular-nums text-slate-700">
          Test RMSE: {formatINR(model.testMetrics.rmse)}
        </span>
      </div>
    </div>
  );
}

// 8. Decision Tree Structure SVG (Model Visualization 3 - Top 3 levels)
export function DecisionTreeDiagram({ root }: { root?: TreeNode }) {
  if (!root) return null;

  const nodes: Array<{
    node: TreeNode;
    x: number;
    y: number;
    parentX?: number;
    parentY?: number;
    edgeLabel?: string;
  }> = [];

  const W = 720;
  const levelH = 82;

  function traverse(
    n: TreeNode,
    x: number,
    y: number,
    spread: number,
    parentX?: number,
    parentY?: number,
    edgeLabel?: string
  ) {
    nodes.push({ node: n, x, y, parentX, parentY, edgeLabel });
    if (n.depth >= 2 || !n.left || !n.right) return;
    traverse(n.left, x - spread, y + levelH, spread * 0.5, x, y, '≤');
    traverse(n.right, x + spread, y + levelH, spread * 0.5, x, y, '>');
  }

  traverse(root, W / 2, 34, 175);

  const fmtThreshold = (n: TreeNode) => {
    if (n.threshold === undefined) return '';
    if (n.feature === 'area') return `${Math.round(n.threshold).toLocaleString()} sq.ft.`;
    return n.threshold.toFixed(1);
  };

  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${W} 255`} className="w-full min-w-[620px] h-auto">
        {nodes.map((item) =>
          item.parentX !== undefined && item.parentY !== undefined ? (
            <g key={`edge-${item.node.id}`}>
              <line
                x1={item.parentX}
                y1={item.parentY + 24}
                x2={item.x}
                y2={item.y - 20}
                stroke="#cbd5e1"
                strokeWidth={1.5}
              />
              <text
                x={(item.parentX + item.x) / 2}
                y={(item.parentY + 24 + item.y - 20) / 2 - 2}
                textAnchor="middle"
                className="fill-slate-500 text-[10px] font-mono"
              >
                {item.edgeLabel}
              </text>
            </g>
          ) : null
        )}

        {nodes.map(({ node, x, y }) => {
          const isSplit = Boolean(node.feature && node.depth < 2 && node.left);
          return (
            <g key={node.id}>
              <rect
                x={x - 76}
                y={y - 20}
                width={152}
                height={44}
                rx={5}
                fill={isSplit ? '#eff6ff' : '#f8fafc'}
                stroke={isSplit ? '#2563eb' : '#94a3b8'}
                strokeWidth={1.2}
              />
              <text
                x={x}
                y={y - 3}
                textAnchor="middle"
                className="fill-slate-900 text-[10.5px] font-semibold font-mono"
              >
                {node.featureLabel
                  ? `${node.featureLabel} <= ${fmtThreshold(node)}`
                  : 'Leaf Node'}
              </text>
              <text
                x={x}
                y={y + 13}
                textAnchor="middle"
                className="fill-slate-600 text-[10px] font-mono tabular-nums"
              >
                n={node.samples} · val={formatLakhs(node.value)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// 9. Feature Importance Horizontal Bar Chart (Model Visualization 4)
export function FeatureImportanceChart({ model }: { model: ModelEvaluation }) {
  const items = model.featureImportances.slice(0, 8);
  const maxImp = Math.max(...items.map((i) => i.importance), 0.01);

  return (
    <div className="space-y-2.5">
      {items.map((item, idx) => {
        const pct = (item.importance * 100).toFixed(1);
        const widthPct = (item.importance / maxImp) * 100;
        return (
          <div key={item.feature} className="grid grid-cols-12 items-center gap-3 text-xs">
            <div className="col-span-4 text-slate-700 font-medium truncate">
              {idx + 1}. {item.label}
            </div>
            <div className="col-span-6 h-2.5 bg-slate-100 rounded-sm overflow-hidden">
              <div
                className="h-full bg-blue-700 rounded-sm"
                style={{ width: `${Math.max(2, widthPct)}%` }}
              />
            </div>
            <div className="col-span-2 text-right font-mono tabular-nums text-slate-800 font-medium">
              {pct}%
            </div>
          </div>
        );
      })}
    </div>
  );
}
