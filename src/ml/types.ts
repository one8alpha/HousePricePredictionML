export type YesNo = 'yes' | 'no';
export type FurnishingStatus = 'furnished' | 'semi-furnished' | 'unfurnished';

export interface RawHousingRecord {
  id: number;
  price: number | null; // Target in INR (e.g., 6850000 = ₹68,50,000)
  area: number | null; // Square feet
  bedrooms: number | null;
  bathrooms: number | null;
  stories: number;
  mainroad: YesNo;
  guestroom: YesNo;
  basement: YesNo;
  hotwaterheating: YesNo;
  airconditioning: YesNo;
  parking: number | null;
  prefarea: YesNo;
  furnishingstatus: FurnishingStatus;
}

export interface CleanHousingRecord {
  id: number;
  price: number;
  area: number;
  bedrooms: number;
  bathrooms: number;
  stories: number;
  mainroad: number; // 1 | 0
  guestroom: number; // 1 | 0
  basement: number; // 1 | 0
  hotwaterheating: number; // 1 | 0
  airconditioning: number; // 1 | 0
  parking: number;
  prefarea: number; // 1 | 0
  furnishingstatus: number; // 0 = unfurnished, 1 = semi-furnished, 2 = furnished
  furnishingLabel: FurnishingStatus;
}

export type FeatureKey =
  | 'area'
  | 'bedrooms'
  | 'bathrooms'
  | 'stories'
  | 'mainroad'
  | 'guestroom'
  | 'basement'
  | 'hotwaterheating'
  | 'airconditioning'
  | 'parking'
  | 'prefarea'
  | 'furnishingstatus';

export interface FeatureMeta {
  key: FeatureKey;
  label: string;
  shortLabel: string;
  type: 'numerical' | 'categorical-binary' | 'categorical-ordinal';
  unit?: string;
  description: string;
}

export type ScalingMethod = 'standard' | 'minmax' | 'none';

export interface PipelineConfig {
  handleMissing: boolean;
  removeDuplicates: boolean;
  handleOutliers: boolean;
  scaling: ScalingMethod;
  testRatio: number; // e.g., 0.20 for 80/20 split
  randomState: number;
  selectedFeatures: FeatureKey[];
  dtMaxDepth: number;
  dtMinSamplesSplit: number;
  rfEstimators: number;
  rfMaxDepth: number;
}

export interface ColumnSummary {
  name: string;
  dtype: 'float64' | 'int64' | 'object (categorical)';
  kind: 'Numerical' | 'Categorical' | 'Target (Numerical)';
  missingCount: number;
  uniqueCount: number;
  min?: number;
  max?: number;
  mean?: number;
  median?: number;
  std?: number;
}

export interface DatasetInspection {
  rawRows: number;
  rawCols: number;
  duplicateCount: number;
  totalMissing: number;
  missingByColumn: Record<string, number>;
  outlierCountPrice: number;
  outlierCountArea: number;
  priceBounds: { q1: number; q3: number; iqr: number; lower: number; upper: number };
  areaBounds: { q1: number; q3: number; iqr: number; lower: number; upper: number };
  columns: ColumnSummary[];
}

export interface PreprocessingReport {
  initialRows: number;
  afterMissingRows: number;
  imputedCounts: Record<string, number>;
  duplicatesRemoved: number;
  outliersRemoved: number;
  finalRows: number;
  scalerParams: {
    means: Record<FeatureKey, number>;
    stds: Record<FeatureKey, number>;
    mins: Record<FeatureKey, number>;
    maxs: Record<FeatureKey, number>;
  };
}

export interface RegressionMetrics {
  mae: number;
  mse: number;
  rmse: number;
  r2: number;
}

export interface TreeNode {
  id: string;
  depth: number;
  samples: number;
  mse: number;
  value: number; // mean predicted price (INR)
  feature?: FeatureKey;
  featureLabel?: string;
  threshold?: number; // original unscaled threshold for interpretability
  scaledThreshold?: number;
  left?: TreeNode;
  right?: TreeNode;
}

export interface ModelEvaluation {
  id: 'linear' | 'decision_tree' | 'random_forest';
  name: string;
  shortName: string;
  trainMetrics: RegressionMetrics;
  testMetrics: RegressionMetrics;
  predictionsTest: Array<{
    actual: number;
    predicted: number;
    residual: number;
    area: number;
  }>;
  featureImportances: Array<{
    feature: FeatureKey;
    label: string;
    importance: number; // normalized 0..1
  }>;
  coefficients?: Array<{
    feature: FeatureKey;
    label: string;
    weight: number;
  }>;
  bias?: number;
  treeRoot?: TreeNode;
  hyperparameters: Record<string, string | number>;
}

export interface ExperimentResult {
  id: string;
  title: string;
  category: string;
  change: string;
  baselineModel: string;
  baselineRmse: number;
  baselineR2: number;
  modifiedModel: string;
  modifiedRmse: number;
  modifiedR2: number;
  deltaR2: number;
  observation: string;
  conclusion: string;
}

export interface PredictionInput {
  area: number;
  bedrooms: number;
  bathrooms: number;
  stories: number;
  mainroad: YesNo;
  guestroom: YesNo;
  basement: YesNo;
  hotwaterheating: YesNo;
  airconditioning: YesNo;
  parking: number;
  prefarea: YesNo;
  furnishingstatus: FurnishingStatus;
}

export interface PipelineArtifacts {
  rawData: RawHousingRecord[];
  cleanedData: CleanHousingRecord[];
  inspection: DatasetInspection;
  preprocessingReport: PreprocessingReport;
  correlationMatrix: {
    labels: string[];
    keys: string[];
    values: number[][];
  };
  splitInfo: {
    totalRows: number;
    trainRows: number;
    testRows: number;
    trainRatio: number;
    testRatio: number;
    randomState: number;
  };
  models: {
    linear: ModelEvaluation;
    decision_tree: ModelEvaluation;
    random_forest: ModelEvaluation;
  };
  bestModelId: 'linear' | 'decision_tree' | 'random_forest';
  predict: (
    modelId: 'linear' | 'decision_tree' | 'random_forest',
    input: PredictionInput
  ) => {
    predictedPrice: number;
    lowerBound: number;
    upperBound: number;
    pricePerSqFt: number;
    contributions: Array<{ label: string; impact: number }>;
  };
}
