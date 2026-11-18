import { FEATURE_METADATA } from './dataset';
import {
  CleanHousingRecord,
  ColumnSummary,
  DatasetInspection,
  FeatureKey,
  PreprocessingReport,
  RawHousingRecord,
  ScalingMethod,
} from './types';

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return quantile(sorted, 0.5);
}

function meanAndStd(values: number[]): { mean: number; std: number } {
  if (values.length === 0) return { mean: 0, std: 1 };
  const mean = values.reduce((acc, v) => acc + v, 0) / values.length;
  const variance =
    values.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) / values.length;
  return { mean, std: Math.sqrt(variance) || 1 };
}

function rowSignature(r: RawHousingRecord): string {
  return [
    r.price,
    r.area,
    r.bedrooms,
    r.bathrooms,
    r.stories,
    r.mainroad,
    r.guestroom,
    r.basement,
    r.hotwaterheating,
    r.airconditioning,
    r.parking,
    r.prefarea,
    r.furnishingstatus,
  ].join('|');
}

export function inspectRawDataset(rawRows: RawHousingRecord[]): DatasetInspection {
  const seen = new Set<string>();
  let duplicateCount = 0;
  for (const r of rawRows) {
    const sig = rowSignature(r);
    if (seen.has(sig)) duplicateCount++;
    else seen.add(sig);
  }

  const missingByColumn: Record<string, number> = {
    price: 0,
    area: 0,
    bedrooms: 0,
    bathrooms: 0,
    stories: 0,
    mainroad: 0,
    guestroom: 0,
    basement: 0,
    hotwaterheating: 0,
    airconditioning: 0,
    parking: 0,
    prefarea: 0,
    furnishingstatus: 0,
  };

  for (const r of rawRows) {
    if (r.price === null) missingByColumn.price++;
    if (r.area === null) missingByColumn.area++;
    if (r.bedrooms === null) missingByColumn.bedrooms++;
    if (r.bathrooms === null) missingByColumn.bathrooms++;
    if (r.parking === null) missingByColumn.parking++;
  }

  const totalMissing = Object.values(missingByColumn).reduce((a, b) => a + b, 0);

  const prices = rawRows
    .map((r) => r.price)
    .filter((v): v is number => v !== null)
    .sort((a, b) => a - b);
  const areas = rawRows
    .map((r) => r.area)
    .filter((v): v is number => v !== null)
    .sort((a, b) => a - b);

  const pQ1 = quantile(prices, 0.25);
  const pQ3 = quantile(prices, 0.75);
  const pIqr = pQ3 - pQ1;
  const priceBounds = {
    q1: pQ1,
    q3: pQ3,
    iqr: pIqr,
    lower: pQ1 - 1.5 * pIqr,
    upper: pQ3 + 1.5 * pIqr,
  };

  const aQ1 = quantile(areas, 0.25);
  const aQ3 = quantile(areas, 0.75);
  const aIqr = aQ3 - aQ1;
  const areaBounds = {
    q1: aQ1,
    q3: aQ3,
    iqr: aIqr,
    lower: aQ1 - 1.5 * aIqr,
    upper: aQ3 + 1.5 * aIqr,
  };

  const outlierCountPrice = prices.filter(
    (p) => p < priceBounds.lower || p > priceBounds.upper
  ).length;
  const outlierCountArea = areas.filter(
    (a) => a < areaBounds.lower || a > areaBounds.upper
  ).length;

  const numKeys: Array<{ key: keyof RawHousingRecord; kind: ColumnSummary['kind']; dtype: ColumnSummary['dtype'] }> = [
    { key: 'price', kind: 'Target (Numerical)', dtype: 'int64' },
    { key: 'area', kind: 'Numerical', dtype: 'float64' },
    { key: 'bedrooms', kind: 'Numerical', dtype: 'float64' },
    { key: 'bathrooms', kind: 'Numerical', dtype: 'float64' },
    { key: 'stories', kind: 'Numerical', dtype: 'int64' },
    { key: 'parking', kind: 'Numerical', dtype: 'float64' },
  ];

  const columns: ColumnSummary[] = numKeys.map(({ key, kind, dtype }) => {
    const vals = rawRows
      .map((r) => r[key] as number | null)
      .filter((v): v is number => v !== null);
    const sorted = [...vals].sort((a, b) => a - b);
    const { mean, std } = meanAndStd(vals);
    return {
      name: String(key),
      dtype,
      kind,
      missingCount: missingByColumn[String(key)] || 0,
      uniqueCount: new Set(vals).size,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      mean,
      median: quantile(sorted, 0.5),
      std,
    };
  });

  const catKeys: Array<keyof RawHousingRecord> = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea',
    'furnishingstatus',
  ];

  for (const key of catKeys) {
    const vals = rawRows.map((r) => String(r[key]));
    columns.push({
      name: String(key),
      dtype: 'object (categorical)',
      kind: 'Categorical',
      missingCount: missingByColumn[String(key)] || 0,
      uniqueCount: new Set(vals).size,
    });
  }

  return {
    rawRows: rawRows.length,
    rawCols: 13,
    duplicateCount,
    totalMissing,
    missingByColumn,
    outlierCountPrice,
    outlierCountArea,
    priceBounds,
    areaBounds,
    columns,
  };
}

export function cleanAndEncodeDataset(
  rawRows: RawHousingRecord[],
  inspection: DatasetInspection,
  options: {
    handleMissing: boolean;
    removeDuplicates: boolean;
    handleOutliers: boolean;
  }
): { cleaned: CleanHousingRecord[]; report: Omit<PreprocessingReport, 'scalerParams'> } {
  const areaMed = median(
    rawRows.map((r) => r.area).filter((v): v is number => v !== null)
  );
  const bedMed = median(
    rawRows.map((r) => r.bedrooms).filter((v): v is number => v !== null)
  );
  const bathMed = median(
    rawRows.map((r) => r.bathrooms).filter((v): v is number => v !== null)
  );
  const parkMed = median(
    rawRows.map((r) => r.parking).filter((v): v is number => v !== null)
  );

  const imputedCounts: Record<string, number> = {
    area: 0,
    bedrooms: 0,
    bathrooms: 0,
    parking: 0,
  };

  // Step 1: Missing Value Handling (Impute or Drop)
  let working: RawHousingRecord[] = [];
  if (options.handleMissing) {
    working = rawRows.map((r) => {
      const next = { ...r };
      if (next.area === null) {
        next.area = areaMed;
        imputedCounts.area++;
      }
      if (next.bedrooms === null) {
        next.bedrooms = bedMed;
        imputedCounts.bedrooms++;
      }
      if (next.bathrooms === null) {
        next.bathrooms = bathMed;
        imputedCounts.bathrooms++;
      }
      if (next.parking === null) {
        next.parking = parkMed;
        imputedCounts.parking++;
      }
      return next;
    });
  } else {
    working = rawRows.filter(
      (r) =>
        r.price !== null &&
        r.area !== null &&
        r.bedrooms !== null &&
        r.bathrooms !== null &&
        r.parking !== null
    );
  }
  const afterMissingRows = working.length;

  // Step 2: Duplicate Removal
  let duplicatesRemoved = 0;
  if (options.removeDuplicates) {
    const seen = new Set<string>();
    const deduped: RawHousingRecord[] = [];
    for (const r of working) {
      const sig = rowSignature(r);
      if (seen.has(sig)) {
        duplicatesRemoved++;
      } else {
        seen.add(sig);
        deduped.push(r);
      }
    }
    working = deduped;
  }

  // Step 3: Outlier Removal (IQR bounds on price and area)
  let outliersRemoved = 0;
  if (options.handleOutliers) {
    const filtered: RawHousingRecord[] = [];
    for (const r of working) {
      const priceOut =
        (r.price as number) < inspection.priceBounds.lower ||
        (r.price as number) > inspection.priceBounds.upper;
      const areaOut =
        (r.area as number) < inspection.areaBounds.lower ||
        (r.area as number) > inspection.areaBounds.upper;
      if (priceOut || areaOut) {
        outliersRemoved++;
      } else {
        filtered.push(r);
      }
    }
    working = filtered;
  }

  // Step 4: Categorical Encoding
  const cleaned: CleanHousingRecord[] = working.map((r) => ({
    id: r.id,
    price: r.price as number,
    area: r.area as number,
    bedrooms: r.bedrooms as number,
    bathrooms: r.bathrooms as number,
    stories: r.stories,
    mainroad: r.mainroad === 'yes' ? 1 : 0,
    guestroom: r.guestroom === 'yes' ? 1 : 0,
    basement: r.basement === 'yes' ? 1 : 0,
    hotwaterheating: r.hotwaterheating === 'yes' ? 1 : 0,
    airconditioning: r.airconditioning === 'yes' ? 1 : 0,
    parking: r.parking as number,
    prefarea: r.prefarea === 'yes' ? 1 : 0,
    furnishingstatus:
      r.furnishingstatus === 'furnished'
        ? 2
        : r.furnishingstatus === 'semi-furnished'
          ? 1
          : 0,
    furnishingLabel: r.furnishingstatus,
  }));

  return {
    cleaned,
    report: {
      initialRows: rawRows.length,
      afterMissingRows,
      imputedCounts,
      duplicatesRemoved,
      outliersRemoved,
      finalRows: cleaned.length,
    },
  };
}

export interface ScalerParams {
  means: Record<FeatureKey, number>;
  stds: Record<FeatureKey, number>;
  mins: Record<FeatureKey, number>;
  maxs: Record<FeatureKey, number>;
}

export function fitScaler(trainRecords: CleanHousingRecord[]): ScalerParams {
  const means = {} as Record<FeatureKey, number>;
  const stds = {} as Record<FeatureKey, number>;
  const mins = {} as Record<FeatureKey, number>;
  const maxs = {} as Record<FeatureKey, number>;

  for (const meta of FEATURE_METADATA) {
    const key = meta.key;
    const vals = trainRecords.map((r) => r[key]);
    const { mean, std } = meanAndStd(vals);
    means[key] = mean;
    stds[key] = std || 1;
    mins[key] = Math.min(...vals);
    maxs[key] = Math.max(...vals);
  }

  return { means, stds, mins, maxs };
}

export function scaleValue(
  value: number,
  key: FeatureKey,
  scaler: ScalerParams,
  method: ScalingMethod
): number {
  if (method === 'standard') {
    return (value - scaler.means[key]) / (scaler.stds[key] || 1);
  }
  if (method === 'minmax') {
    const range = scaler.maxs[key] - scaler.mins[key];
    return range === 0 ? 0 : (value - scaler.mins[key]) / range;
  }
  return value;
}

export function unscaleValue(
  scaled: number,
  key: FeatureKey,
  scaler: ScalerParams,
  method: ScalingMethod
): number {
  if (method === 'standard') {
    return scaled * (scaler.stds[key] || 1) + scaler.means[key];
  }
  if (method === 'minmax') {
    const range = scaler.maxs[key] - scaler.mins[key];
    return scaled * range + scaler.mins[key];
  }
  return scaled;
}

export function extractFeatureMatrix(
  records: CleanHousingRecord[],
  features: FeatureKey[],
  scaler: ScalerParams,
  method: ScalingMethod
): number[][] {
  return records.map((r) =>
    features.map((f) => scaleValue(r[f], f, scaler, method))
  );
}

export function computeCorrelationMatrix(records: CleanHousingRecord[]): {
  labels: string[];
  keys: string[];
  values: number[][];
} {
  const keys = ['price', ...FEATURE_METADATA.map((f) => f.key)];
  const labels = ['Price', ...FEATURE_METADATA.map((f) => f.shortLabel)];
  const n = records.length;

  const series = keys.map((k) =>
    records.map((r) => r[k as keyof CleanHousingRecord] as number)
  );
  const means = series.map((arr) => arr.reduce((a, b) => a + b, 0) / (n || 1));

  const values: number[][] = keys.map((_, i) =>
    keys.map((__, j) => {
      if (i === j) return 1;
      let num = 0;
      let denA = 0;
      let denB = 0;
      for (let idx = 0; idx < n; idx++) {
        const da = series[i][idx] - means[i];
        const db = series[j][idx] - means[j];
        num += da * db;
        denA += da * da;
        denB += db * db;
      }
      const den = Math.sqrt(denA * denB);
      return den === 0 ? 0 : Number((num / den).toFixed(3));
    })
  );

  return { labels, keys, values };
}

export function cleanedDatasetToCSV(rows: CleanHousingRecord[]): string {
  const header =
    'price,area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus';
  const lines = rows.map((r) =>
    [
      r.price,
      r.area,
      r.bedrooms,
      r.bathrooms,
      r.stories,
      r.mainroad,
      r.guestroom,
      r.basement,
      r.hotwaterheating,
      r.airconditioning,
      r.parking,
      r.prefarea,
      r.furnishingstatus,
    ].join(',')
  );
  return [header, ...lines].join('\n');
}
