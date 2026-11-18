import { createRng, FEATURE_METADATA } from './dataset';
import { ScalerParams, unscaleValue } from './preprocessing';
import {
  CleanHousingRecord,
  FeatureKey,
  ModelEvaluation,
  RegressionMetrics,
  ScalingMethod,
  TreeNode,
} from './types';

function featureLabel(key: FeatureKey): string {
  return FEATURE_METADATA.find((f) => f.key === key)?.shortLabel ?? key;
}

export function trainTestSplit<T>(
  rows: T[],
  testRatio: number,
  randomState: number
): { train: T[]; test: T[] } {
  const rng = createRng(randomState);
  const indices = rows.map((_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  const testCount = Math.max(10, Math.round(rows.length * testRatio));
  const testIdx = new Set(indices.slice(0, testCount));
  const train: T[] = [];
  const test: T[] = [];
  rows.forEach((row, idx) => {
    if (testIdx.has(idx)) test.push(row);
    else train.push(row);
  });
  return { train, test };
}

export function computeMetrics(
  yTrue: number[],
  yPred: number[]
): RegressionMetrics {
  const n = yTrue.length || 1;
  let absSum = 0;
  let sqSum = 0;
  const meanY = yTrue.reduce((a, b) => a + b, 0) / n;
  let totalVar = 0;

  for (let i = 0; i < n; i++) {
    const err = yTrue[i] - yPred[i];
    absSum += Math.abs(err);
    sqSum += err * err;
    totalVar += (yTrue[i] - meanY) * (yTrue[i] - meanY);
  }

  const mae = absSum / n;
  const mse = sqSum / n;
  const rmse = Math.sqrt(mse);
  const r2 = totalVar === 0 ? 0 : 1 - sqSum / totalVar;

  return { mae, mse, rmse, r2 };
}

// Solve linear system A * x = b via Gauss-Jordan elimination with partial pivoting
function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = b.length;
  const aug = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) {
        maxRow = row;
      }
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-10) {
      aug[col][col] = 1e-6;
    }
    const diag = aug[col][col];
    for (let j = col; j <= n; j++) {
      aug[col][j] /= diag;
    }

    for (let row = 0; row < n; row++) {
      if (row !== col) {
        const factor = aug[row][col];
        for (let j = col; j <= n; j++) {
          aug[row][j] -= factor * aug[col][j];
        }
      }
    }
  }

  return aug.map((row) => row[n]);
}

export interface TrainedLinearModel {
  evaluation: ModelEvaluation;
  predictVector: (x: number[]) => number;
  weights: number[];
  bias: number;
}

export function trainLinearRegression(
  XTrain: number[][],
  yTrain: number[],
  XTest: number[][],
  yTest: number[],
  testRecords: CleanHousingRecord[],
  features: FeatureKey[]
): TrainedLinearModel {
  const n = XTrain.length;
  const p = features.length;
  const dim = p + 1; // bias + weights

  // Build X_b^T X_b + ridge * I (do not regularize bias at index 0)
  const XtX: number[][] = Array.from({ length: dim }, () => Array(dim).fill(0));
  const Xty: number[] = Array(dim).fill(0);

  for (let i = 0; i < n; i++) {
    const row = [1, ...XTrain[i]];
    const yi = yTrain[i];
    for (let j = 0; j < dim; j++) {
      Xty[j] += row[j] * yi;
      for (let k = j; k < dim; k++) {
        XtX[j][k] += row[j] * row[k];
      }
    }
  }

  for (let j = 0; j < dim; j++) {
    for (let k = 0; k < j; k++) {
      XtX[j][k] = XtX[k][j];
    }
    if (j > 0) XtX[j][j] += 1e-3; // tiny ridge term for numerical stability
  }

  const theta = solveLinearSystem(XtX, Xty);
  const bias = theta[0];
  const weights = theta.slice(1);

  const predictVector = (x: number[]) => {
    let sum = bias;
    for (let j = 0; j < p; j++) sum += weights[j] * x[j];
    return Math.max(800000, sum);
  };

  const trainPreds = XTrain.map(predictVector);
  const testPreds = XTest.map(predictVector);

  // Compute standardized contribution importance for Linear Regression
  const featureStds = features.map((_, j) => {
    const col = XTrain.map((r) => r[j]);
    const mean = col.reduce((a, b) => a + b, 0) / (col.length || 1);
    const v = col.reduce((a, b) => a + (b - mean) ** 2, 0) / (col.length || 1);
    return Math.sqrt(v) || 1;
  });

  const rawImp = weights.map((w, j) => Math.abs(w * featureStds[j]));
  const totalImp = rawImp.reduce((a, b) => a + b, 0) || 1;

  const featureImportances = features
    .map((f, j) => ({
      feature: f,
      label: featureLabel(f),
      importance: rawImp[j] / totalImp,
    }))
    .sort((a, b) => b.importance - a.importance);

  return {
    bias,
    weights,
    predictVector,
    evaluation: {
      id: 'linear',
      name: 'Multiple Linear Regression',
      shortName: 'Linear Regression',
      trainMetrics: computeMetrics(yTrain, trainPreds),
      testMetrics: computeMetrics(yTest, testPreds),
      predictionsTest: testPreds.map((pred, i) => ({
        actual: yTest[i],
        predicted: pred,
        residual: yTest[i] - pred,
        area: testRecords[i].area,
      })),
      featureImportances,
      coefficients: features.map((f, j) => ({
        feature: f,
        label: featureLabel(f),
        weight: weights[j],
      })),
      bias,
      hyperparameters: {
        fit_intercept: 'True',
        regularization: 'OLS (λ = 1e-3)',
        features_count: p,
      },
    },
  };
}

// Fast variance-reduction CART Decision Tree Regressor
interface InternalTreeNode {
  id: string;
  depth: number;
  samples: number;
  mse: number;
  value: number;
  featureIdx?: number;
  scaledThreshold?: number;
  left?: InternalTreeNode;
  right?: InternalTreeNode;
}

function buildCartTree(
  X: number[][],
  y: number[],
  indices: number[],
  depth: number,
  maxDepth: number,
  minSamplesSplit: number,
  minSamplesLeaf: number,
  featureSubsetSize: number | null,
  rng: () => number,
  importanceAccumulator: number[],
  nodeId: string
): InternalTreeNode {
  const n = indices.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += y[indices[i]];
  const mean = sum / (n || 1);

  let sqErr = 0;
  for (let i = 0; i < n; i++) {
    const d = y[indices[i]] - mean;
    sqErr += d * d;
  }
  const mse = sqErr / (n || 1);

  if (depth >= maxDepth || n < minSamplesSplit || mse < 1e4) {
    return { id: nodeId, depth, samples: n, mse, value: mean };
  }

  const numFeatures = X[0].length;
  let candidateFeatures: number[] = [];
  if (featureSubsetSize !== null && featureSubsetSize < numFeatures) {
    const pool = Array.from({ length: numFeatures }, (_, i) => i);
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    candidateFeatures = pool.slice(0, featureSubsetSize);
  } else {
    candidateFeatures = Array.from({ length: numFeatures }, (_, i) => i);
  }

  let bestGain = 0;
  let bestFeature = -1;
  let bestThreshold = 0;
  let bestLeft: number[] = [];
  let bestRight: number[] = [];

  for (const fIdx of candidateFeatures) {
    // Sort indices by feature value
    const sorted = [...indices].sort((a, b) => X[a][fIdx] - X[b][fIdx]);
    let leftSum = 0;
    let leftSqSum = 0;
    const totalSum = sum;
    let totalSqSum = 0;
    for (let i = 0; i < n; i++) {
      const val = y[sorted[i]];
      totalSqSum += val * val;
    }

    for (let i = 0; i < n - 1; i++) {
      const idx = sorted[i];
      const val = y[idx];
      leftSum += val;
      leftSqSum += val * val;

      const leftCount = i + 1;
      const rightCount = n - leftCount;

      if (X[sorted[i]][fIdx] === X[sorted[i + 1]][fIdx]) continue;
      if (leftCount < minSamplesLeaf || rightCount < minSamplesLeaf) continue;

      const rightSum = totalSum - leftSum;
      const rightSqSum = totalSqSum - leftSqSum;

      const leftSse = leftSqSum - (leftSum * leftSum) / leftCount;
      const rightSse = rightSqSum - (rightSum * rightSum) / rightCount;
      const gain = sqErr - (leftSse + rightSse);

      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = fIdx;
        bestThreshold = (X[sorted[i]][fIdx] + X[sorted[i + 1]][fIdx]) / 2;
      }
    }
  }

  if (bestFeature === -1 || bestGain <= 0) {
    return { id: nodeId, depth, samples: n, mse, value: mean };
  }

  for (const idx of indices) {
    if (X[idx][bestFeature] <= bestThreshold) bestLeft.push(idx);
    else bestRight.push(idx);
  }

  if (bestLeft.length < minSamplesLeaf || bestRight.length < minSamplesLeaf) {
    return { id: nodeId, depth, samples: n, mse, value: mean };
  }

  importanceAccumulator[bestFeature] += bestGain;

  return {
    id: nodeId,
    depth,
    samples: n,
    mse,
    value: mean,
    featureIdx: bestFeature,
    scaledThreshold: bestThreshold,
    left: buildCartTree(
      X,
      y,
      bestLeft,
      depth + 1,
      maxDepth,
      minSamplesSplit,
      minSamplesLeaf,
      featureSubsetSize,
      rng,
      importanceAccumulator,
      `${nodeId}L`
    ),
    right: buildCartTree(
      X,
      y,
      bestRight,
      depth + 1,
      maxDepth,
      minSamplesSplit,
      minSamplesLeaf,
      featureSubsetSize,
      rng,
      importanceAccumulator,
      `${nodeId}R`
    ),
  };
}

function predictCart(node: InternalTreeNode, x: number[]): number {
  if (
    node.featureIdx === undefined ||
    node.scaledThreshold === undefined ||
    !node.left ||
    !node.right
  ) {
    return node.value;
  }
  if (x[node.featureIdx] <= node.scaledThreshold) {
    return predictCart(node.left, x);
  }
  return predictCart(node.right, x);
}

function toExportTreeNode(
  node: InternalTreeNode,
  features: FeatureKey[],
  scaler: ScalerParams,
  scalingMethod: ScalingMethod
): TreeNode {
  const fKey =
    node.featureIdx !== undefined ? features[node.featureIdx] : undefined;
  const origThreshold =
    fKey && node.scaledThreshold !== undefined
      ? unscaleValue(node.scaledThreshold, fKey, scaler, scalingMethod)
      : undefined;

  return {
    id: node.id,
    depth: node.depth,
    samples: node.samples,
    mse: node.mse,
    value: node.value,
    feature: fKey,
    featureLabel: fKey ? featureLabel(fKey) : undefined,
    threshold: origThreshold,
    scaledThreshold: node.scaledThreshold,
    left: node.left
      ? toExportTreeNode(node.left, features, scaler, scalingMethod)
      : undefined,
    right: node.right
      ? toExportTreeNode(node.right, features, scaler, scalingMethod)
      : undefined,
  };
}

export interface TrainedTreeModel {
  evaluation: ModelEvaluation;
  predictVector: (x: number[]) => number;
}

export function trainDecisionTree(
  XTrain: number[][],
  yTrain: number[],
  XTest: number[][],
  yTest: number[],
  testRecords: CleanHousingRecord[],
  features: FeatureKey[],
  scaler: ScalerParams,
  scalingMethod: ScalingMethod,
  maxDepth: number,
  minSamplesSplit: number,
  randomState: number
): TrainedTreeModel {
  const rng = createRng(randomState + 7);
  const impAcc = Array(features.length).fill(0);
  const allIndices = XTrain.map((_, i) => i);

  const root = buildCartTree(
    XTrain,
    yTrain,
    allIndices,
    0,
    maxDepth,
    minSamplesSplit,
    3,
    null,
    rng,
    impAcc,
    'root'
  );

  const predictVector = (x: number[]) => predictCart(root, x);
  const trainPreds = XTrain.map(predictVector);
  const testPreds = XTest.map(predictVector);

  const totalImp = impAcc.reduce((a, b) => a + b, 0) || 1;
  const featureImportances = features
    .map((f, j) => ({
      feature: f,
      label: featureLabel(f),
      importance: impAcc[j] / totalImp,
    }))
    .sort((a, b) => b.importance - a.importance);

  return {
    predictVector,
    evaluation: {
      id: 'decision_tree',
      name: 'Decision Tree Regressor (CART)',
      shortName: 'Decision Tree',
      trainMetrics: computeMetrics(yTrain, trainPreds),
      testMetrics: computeMetrics(yTest, testPreds),
      predictionsTest: testPreds.map((pred, i) => ({
        actual: yTest[i],
        predicted: pred,
        residual: yTest[i] - pred,
        area: testRecords[i].area,
      })),
      featureImportances,
      treeRoot: toExportTreeNode(root, features, scaler, scalingMethod),
      hyperparameters: {
        criterion: 'squared_error (MSE)',
        max_depth: maxDepth,
        min_samples_split: minSamplesSplit,
        min_samples_leaf: 3,
      },
    },
  };
}

export interface TrainedForestModel {
  evaluation: ModelEvaluation;
  predictVector: (x: number[]) => number;
  predictWithSpread: (x: number[]) => { mean: number; std: number };
}

export function trainRandomForest(
  XTrain: number[][],
  yTrain: number[],
  XTest: number[][],
  yTest: number[],
  testRecords: CleanHousingRecord[],
  features: FeatureKey[],
  nEstimators: number,
  maxDepth: number,
  randomState: number
): TrainedForestModel {
  const rng = createRng(randomState + 101);
  const impAcc = Array(features.length).fill(0);
  const n = XTrain.length;
  const maxFeatures = features.length;

  const trees: InternalTreeNode[] = [];

  for (let t = 0; t < nEstimators; t++) {
    // Bootstrap sample
    const bootIndices: number[] = [];
    for (let i = 0; i < n; i++) {
      bootIndices.push(Math.floor(rng() * n));
    }
    const tree = buildCartTree(
      XTrain,
      yTrain,
      bootIndices,
      0,
      maxDepth,
      4,
      2,
      maxFeatures,
      rng,
      impAcc,
      `t${t}`
    );
    trees.push(tree);
  }

  const predictWithSpread = (x: number[]) => {
    const preds = trees.map((t) => predictCart(t, x));
    const mean = preds.reduce((a, b) => a + b, 0) / preds.length;
    const variance =
      preds.reduce((a, b) => a + (b - mean) * (b - mean), 0) / preds.length;
    return { mean, std: Math.sqrt(variance) };
  };

  const predictVector = (x: number[]) => predictWithSpread(x).mean;

  const trainPreds = XTrain.map(predictVector);
  const testPreds = XTest.map(predictVector);

  const totalImp = impAcc.reduce((a, b) => a + b, 0) || 1;
  const featureImportances = features
    .map((f, j) => ({
      feature: f,
      label: featureLabel(f),
      importance: impAcc[j] / totalImp,
    }))
    .sort((a, b) => b.importance - a.importance);

  return {
    predictVector,
    predictWithSpread,
    evaluation: {
      id: 'random_forest',
      name: 'Random Forest Regressor',
      shortName: 'Random Forest',
      trainMetrics: computeMetrics(yTrain, trainPreds),
      testMetrics: computeMetrics(yTest, testPreds),
      predictionsTest: testPreds.map((pred, i) => ({
        actual: yTest[i],
        predicted: pred,
        residual: yTest[i] - pred,
        area: testRecords[i].area,
      })),
      featureImportances,
      hyperparameters: {
        n_estimators: nEstimators,
        max_depth: maxDepth,
        max_features: maxFeatures,
        bootstrap: 'True',
      },
    },
  };
}
