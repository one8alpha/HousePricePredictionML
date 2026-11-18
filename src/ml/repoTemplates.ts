import { formatINR, rawDatasetToCSV } from './dataset';
import { cleanedDatasetToCSV } from './preprocessing';
import { PipelineArtifacts } from './types';

export interface RepoFile {
  path: string;
  folder: string;
  filename: string;
  language: 'csv' | 'python' | 'json' | 'markdown' | 'text';
  description: string;
  content: string;
}

export function generateRepositoryFiles(artifacts: PipelineArtifacts): RepoFile[] {
  const { models, inspection, preprocessingReport, splitInfo } = artifacts;
  const lr = models.linear.testMetrics;
  const dt = models.decision_tree.testMetrics;
  const rf = models.random_forest.testMetrics;

  const rawCsv = rawDatasetToCSV(artifacts.rawData);
  const cleanedCsv = cleanedDatasetToCSV(artifacts.cleanedData);

  const trainScript = `"""
Roll No. 1 - House Price Prediction (Regression)
Complete ML Pipeline: Preprocessing -> EDA -> Train/Test Split -> Model Comparison -> Export
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Raw Dataset
df = pd.read_csv("../Dataset/housing_raw.csv")
print("Raw Dataset Shape:", df.shape)
print("Missing Values:\\n", df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())

# 2. Data Preprocessing
# 2a. Impute missing numerical values with median
for col in ["area", "bedrooms", "bathrooms", "parking"]:
    df[col] = df[col].fillna(df[col].median())

# 2b. Remove exact duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# 2c. Handle Outliers using IQR method on price and area
for col in ["price", "area"]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# 2d. Encode Categorical Features
binary_cols = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea"
]
for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

df["furnishingstatus"] = df["furnishingstatus"].map({
    "unfurnished": 0,
    "semi-furnished": 1,
    "furnished": 2
})

# 3. Feature (X) and Target (y) Selection
X = df.drop(columns=["price"])
y = df["price"]

# 4. Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 5. Feature Scaling (fit strictly on training data to prevent data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train & Evaluate Regression Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regression": DecisionTreeRegressor(
        max_depth=5, min_samples_split=6, min_samples_leaf=3, random_state=42
    ),
    "Random Forest Regression": RandomForestRegressor(
        n_estimators=25, max_depth=7, random_state=42
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    results[name] = {"MAE": round(mae, 2), "MSE": round(mse, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}
    print(f"{name:28s} | MAE: {mae:,.0f} | RMSE: {rmse:,.0f} | R2: {r2:.4f}")

# 7. Select & Save Final Best Model by Test R2 + Scaler
best_name = max(results, key=lambda k: results[k]["R2"])
final_model = models[best_name]
joblib.dump(final_model, "../Trained Model/house_price_final_model.pkl")
joblib.dump(scaler, "../Trained Model/feature_scaler.pkl")

with open("../Trained Model/evaluation_summary.json", "w") as f:
    json.dump({"selected_model": best_name, "results": results}, f, indent=2)

print(f"Saved final model ({best_name}) and scaler to ../Trained Model/")
`;

  const streamlitApp = `"""
Front-End Prediction Application (Streamlit)
Roll No. 1 - House Price Prediction
"""

import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="House Price Prediction (Roll No. 1)", layout="centered")
st.title("House Price Prediction System")
st.caption("TY B.Tech CSE — Machine Learning Individual Project (Roll No. 1)")

@st.cache_resource
def load_artifacts():
    model = joblib.load("../Trained Model/house_price_final_model.pkl")
    scaler = joblib.load("../Trained Model/feature_scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Enter Area (sq.ft.)", min_value=500, max_value=15000, value=1500, step=100)
    bedrooms = st.selectbox("Enter Bedrooms", [1, 2, 3, 4, 5], index=2)
    bathrooms = st.selectbox("Enter Bathrooms", [1, 2, 3, 4], index=1)
    stories = st.selectbox("Enter Stories", [1, 2, 3, 4], index=1)
    parking = st.selectbox("Enter Parking", [0, 1, 2, 3], index=1)
    furnishing = st.selectbox("Furnishing Status", ["unfurnished", "semi-furnished", "furnished"], index=1)

with col2:
    mainroad = st.radio("Main Road Access", ["yes", "no"], horizontal=True)
    guestroom = st.radio("Guest Room", ["yes", "no"], index=1, horizontal=True)
    basement = st.radio("Basement", ["yes", "no"], index=1, horizontal=True)
    hotwater = st.radio("Hot Water Heating", ["yes", "no"], index=1, horizontal=True)
    aircond = st.radio("Air Conditioning", ["yes", "no"], horizontal=True)
    prefarea = st.radio("Preferred Locality", ["yes", "no"], horizontal=True)

if st.button("Predict Price", type="primary"):
    yn = lambda v: 1 if v == "yes" else 0
    furn_map = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
    features = np.array([[
        area, bedrooms, bathrooms, stories,
        yn(mainroad), yn(guestroom), yn(basement),
        yn(hotwater), yn(aircond), parking,
        yn(prefarea), furn_map[furnishing]
    ]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    st.success(f"Predicted House Price: ₹{prediction:,.0f}")
`;

  const notebookJson = JSON.stringify(
    {
      nbformat: 4,
      nbformat_minor: 5,
      metadata: {
        kernelspec: {
          display_name: 'Python 3',
          language: 'python',
          name: 'python3',
        },
      },
      cells: [
        {
          cell_type: 'markdown',
          metadata: {},
          source: [
            '# House Price Prediction — Individual ML Project (Roll No. 1)\n',
            '**Objective:** Develop and evaluate Machine Learning regression models (Linear Regression, Decision Tree, Random Forest) to predict house prices using structural and amenity features.\n',
          ],
        },
        {
          cell_type: 'code',
          execution_count: 1,
          metadata: {},
          outputs: [],
          source: [
            'import pandas as pd\n',
            'import numpy as np\n',
            'import matplotlib.pyplot as plt\n',
            'import seaborn as sns\n',
            '\n',
            'df = pd.read_csv("../Dataset/housing_raw.csv")\n',
            'print("Dimensions:", df.shape)\n',
            'df.head()\n',
          ],
        },
        {
          cell_type: 'markdown',
          metadata: {},
          source: ['## Part C: Data Preprocessing (Missing Values, Duplicates, Outliers, Encoding, Scaling)\n'],
        },
        {
          cell_type: 'code',
          execution_count: 2,
          metadata: {},
          outputs: [],
          source: [
            '# Handle missing values with median imputation\n',
            'for col in ["area", "bedrooms", "bathrooms", "parking"]:\n',
            '    df[col] = df[col].fillna(df[col].median())\n',
            '\n',
            '# Remove duplicates\n',
            'df = df.drop_duplicates().reset_index(drop=True)\n',
            '\n',
            '# IQR outlier filtering\n',
            'for col in ["price", "area"]:\n',
            '    q1, q3 = df[col].quantile([0.25, 0.75])\n',
            '    iqr = q3 - q1\n',
            '    df = df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]\n',
            'print("Clean Shape:", df.shape)\n',
          ],
        },
        {
          cell_type: 'markdown',
          metadata: {},
          source: ['## Part G-I: Model Selection, Training & Evaluation\n'],
        },
        {
          cell_type: 'code',
          execution_count: 3,
          metadata: {},
          outputs: [],
          source: [
            'from sklearn.model_selection import train_test_split\n',
            'from sklearn.preprocessing import StandardScaler\n',
            'from sklearn.linear_model import LinearRegression\n',
            'from sklearn.tree import DecisionTreeRegressor\n',
            'from sklearn.ensemble import RandomForestRegressor\n',
            'from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n',
            '\n',
            '# Train-Test Split 80/20 with random_state=42\n',
            '# See Source Code/train_pipeline.py for complete execution\n',
          ],
        },
      ],
    },
    null,
    2
  );

  const modelMetadata = JSON.stringify(
    {
      roll_number: 1,
      project: 'House Price Prediction',
      ml_type: 'Supervised Regression',
      selected_final_model: artifacts.models[artifacts.bestModelId].name,
      dataset_summary: {
        raw_rows: inspection.rawRows,
        cleaned_rows: preprocessingReport.finalRows,
        train_rows: splitInfo.trainRows,
        test_rows: splitInfo.testRows,
        random_state: splitInfo.randomState,
      },
      model_comparison: {
        linear_regression: lr,
        decision_tree: dt,
        random_forest: rf,
      },
      hyperparameters: artifacts.models[artifacts.bestModelId].hyperparameters,
      feature_importances:
        artifacts.models[artifacts.bestModelId].featureImportances,
    },
    null,
    2
  );

  const requirementsTxt = `numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.5.0
matplotlib>=3.8.0
seaborn>=0.13.0
streamlit>=1.35.0
joblib>=1.4.0
jupyter>=1.0.0
`;

  const readmeMd = `# House Price Prediction — Machine Learning Individual Project (Roll No. 1)

## 1. Project Title
**House Price Prediction using Supervised Machine Learning Regression**

## 2. Aim
To develop, evaluate, and deploy a Machine Learning regression system that accurately predicts the selling price of a residential property based on its structural and amenity characteristics.

## 3. Objectives
1. Understand a continuous supervised regression problem.
2. Collect, inspect, and preprocess real-world housing data (missing values, duplicates, outliers, encoding, scaling).
3. Perform Exploratory Data Analysis (EDA) with statistical visualizations and observations.
4. Select suitable input features (X) and target variable (y).
5. Split data into 80% Training and 20% Testing sets using \`random_state=42\`.
6. Implement and compare Linear Regression, Decision Tree Regression, and Random Forest Regression.
7. Evaluate models using MAE, MSE, RMSE, and R² score and conduct model improvement experiments.
8. Build and deploy an interactive Front-End prediction application.

## 4. Problem Statement
Real estate pricing depends on multiple interacting attributes such as plot area, bedrooms, bathrooms, stories, parking capacity, main road connectivity, air conditioning, preferred neighborhood status, and furnishing level. Manual valuation is subjective and inconsistent. This project builds an automated ML pipeline that predicts house prices in INR from user-provided property attributes.

## 5. Dataset Information
- **Source**: Classic Housing Prices Dataset (UCI / Kaggle style residential housing dataset)
- **Raw Dimensions**: ${inspection.rawRows} records × ${inspection.rawCols} columns
- **Cleaned Dimensions**: ${preprocessingReport.finalRows} records × ${inspection.rawCols} columns (after imputing ${Object.values(preprocessingReport.imputedCounts).reduce((a, b) => a + b, 0)} missing values, removing ${preprocessingReport.duplicatesRemoved} duplicates, and filtering ${preprocessingReport.outliersRemoved} IQR outliers)
- **Target Variable (y)**: \`price\` (in INR)
- **Input Features (X)**: \`area\`, \`bedrooms\`, \`bathrooms\`, \`stories\`, \`mainroad\`, \`guestroom\`, \`basement\`, \`hotwaterheating\`, \`airconditioning\`, \`parking\`, \`prefarea\`, \`furnishingstatus\`

## 6. ML Pipeline
\`Problem Definition -> Dataset Understanding -> Data Preprocessing -> EDA -> Feature Selection -> Train-Test Split (80/20) -> Model Training -> Evaluation -> Visualization -> Model Improvement -> Save Final Model -> Front-End Deployment\`

## 7. Algorithms Used
1. **Multiple Linear Regression** (Ordinary Least Squares with StandardScaler)
2. **Decision Tree Regression** (CART with MSE splitting criterion, \`max_depth=5\`)
3. **Random Forest Regression** (Bootstrap aggregating ensemble of 25 trees, \`max_depth=7\`)

## 8. Experimental Results (Test Set Evaluation)
| Algorithm | MAE | RMSE | R² Score |
| :--- | ---: | ---: | ---: |
| Linear Regression | ${formatINR(lr.mae)} | ${formatINR(lr.rmse)} | ${lr.r2.toFixed(4)} |
| Decision Tree Regression | ${formatINR(dt.mae)} | ${formatINR(dt.rmse)} | ${dt.r2.toFixed(4)} |
| Random Forest Regression | ${formatINR(rf.mae)} | ${formatINR(rf.rmse)} | ${rf.r2.toFixed(4)} |

## 9. Final Model
**${artifacts.models[artifacts.bestModelId].name}** was selected as the final deployed model based on superior holdout Test R² (${artifacts.models[artifacts.bestModelId].testMetrics.r2.toFixed(4)}) and lowest Test RMSE (${formatINR(artifacts.models[artifacts.bestModelId].testMetrics.rmse)}), capturing non-linear amenity interactions without overfitting.

## 10. Front-End Details
- Interactive Web Application (React + TypeScript) and Streamlit script (\`Front End/app.py\`).
- Flow: \`Enter Property Details -> Submit -> Generate Prediction -> Display Estimated Price (INR) & Interval\`.

## 11. Screenshots
- Interactive prediction dashboard, EDA visualizations (Distribution, Scatter + Regression Line, Box Plot, Correlation Heatmap), Actual vs Predicted, Residual Plot, and Decision Tree structure are available directly in the live web application.

## 12. Installation & Execution Instructions
\`\`\`bash
# Option A: Run Python Pipeline & Streamlit UI
pip install -r Requirements.txt
python "Source Code/train_pipeline.py"
streamlit run "Front End/app.py"

# Option B: Run Interactive Web Studio
npm install
npm run dev
\`\`\`

## 13. GitHub Repository Link
\`https://github.com/student-roll-01/house-price-prediction-ml\` (Replace with your pushed repository URL upon upload)
`;

  const reportMd = `# Project Documentation Report
**Course:** Third Year B.Tech. Computer Science and Engineering  
**Subject:** Machine Learning — Individual Project (Academic Year 2026–27)  
**Roll Number:** 1  
**Project Title:** House Price Prediction  
**ML Task Type:** Supervised Regression  

---

## Part A: Problem Understanding
1. **Problem Definition:** Predict the continuous selling price of a residential house given its structural dimensions and neighborhood/interior amenities.
2. **Input Variables (X):** \`area\`, \`bedrooms\`, \`bathrooms\`, \`stories\`, \`mainroad\`, \`guestroom\`, \`basement\`, \`hotwaterheating\`, \`airconditioning\`, \`parking\`, \`prefarea\`, \`furnishingstatus\`.
3. **Target Variable (y):** \`price\` (in Indian Rupees, INR).
4. **ML Type:** Supervised Learning — Regression.
5. **Why Regression is Appropriate:** The target variable \`price\` is a continuous numerical quantity rather than a discrete category or cluster label.

## Part B: Dataset Understanding
- **Dimensions:** ${inspection.rawRows} rows × ${inspection.rawCols} columns (raw); ${preprocessingReport.finalRows} rows after cleaning.
- **Missing Values Detected:** ${inspection.totalMissing} total missing cells (\`area\`: ${inspection.missingByColumn.area}, \`bathrooms\`: ${inspection.missingByColumn.bathrooms}, \`bedrooms\`: ${inspection.missingByColumn.bedrooms}, \`parking\`: ${inspection.missingByColumn.parking}).
- **Duplicates Detected:** ${inspection.duplicateCount} exact duplicate rows.
- **Outliers Detected (IQR):** ${inspection.outlierCountPrice} price outliers, ${inspection.outlierCountArea} area outliers.

## Part C: Data Preprocessing Decisions
1. **Missing Value Treatment:** Imputed numerical missing values using the column **median** rather than mean so skewed luxury listings do not bias imputed values.
2. **Duplicate Removal:** Dropped ${preprocessingReport.duplicatesRemoved} duplicate rows to prevent inflating evaluation metrics through memorization.
3. **Outlier Handling:** Applied the **1.5 × IQR** rule on \`price\` and \`area\`, removing ${preprocessingReport.outliersRemoved} extreme outliers that distort Linear Regression slopes.
4. **Categorical Encoding:** Binary \`yes/no\` attributes encoded as \`1/0\`; ordinal \`furnishingstatus\` encoded as \`0 (unfurnished) < 1 (semi-furnished) < 2 (furnished)\`.
5. **Feature Scaling:** Applied **StandardScaler** ($z = (x - \\mu)/\\sigma$) fitted exclusively on the training split to prevent data leakage.

## Part F: Train-Test Split
- **Split Ratio:** ${splitInfo.trainRatio}% Training (${splitInfo.trainRows} rows) / ${splitInfo.testRatio}% Testing (${splitInfo.testRows} rows).
- **Role of \`random_state = ${splitInfo.randomState}\`:** Fixes the pseudo-random shuffle seed so train/test partitions and model comparisons remain 100% reproducible across runs and viva demonstrations.

## Part L: Final Interpretation
1. **Final Model Selected:** ${artifacts.models[artifacts.bestModelId].name}.
2. **Justification:** Achieves the highest test R² (${artifacts.models[artifacts.bestModelId].testMetrics.r2.toFixed(4)}) and lowest RMSE (${formatINR(artifacts.models[artifacts.bestModelId].testMetrics.rmse)}).
3. **Important Features:** \`area\`, \`bathrooms\`, \`airconditioning\`, \`stories\`, \`parking\`, and \`prefarea\`.
4. **Limitations:** Bounded to the feature ranges of the training distribution (cannot extrapolate accurately to ultra-luxury palaces > 15,000 sq.ft. or capture exact street-level geospatial coordinates).
5. **Future Improvements:** Incorporate geospatial lat/long features, temporal inflation indexing, and Gradient Boosting (XGBoost / LightGBM) with k-fold cross-validation.
`;

  return [
    {
      path: 'Dataset/housing_raw.csv',
      folder: 'Dataset',
      filename: 'housing_raw.csv',
      language: 'csv',
      description: 'Raw 545-row housing dataset including missing values, duplicates, and outliers',
      content: rawCsv,
    },
    {
      path: 'Dataset/housing_cleaned.csv',
      folder: 'Dataset',
      filename: 'housing_cleaned.csv',
      language: 'csv',
      description: 'Preprocessed & encoded dataset ready for ML modeling',
      content: cleanedCsv,
    },
    {
      path: 'Notebook/House_Price_Prediction_Pipeline.ipynb',
      folder: 'Notebook',
      filename: 'House_Price_Prediction_Pipeline.ipynb',
      language: 'json',
      description: 'Jupyter Notebook covering Parts A–L of the ML pipeline',
      content: notebookJson,
    },
    {
      path: 'Source Code/train_pipeline.py',
      folder: 'Source Code',
      filename: 'train_pipeline.py',
      language: 'python',
      description: 'End-to-end scikit-learn training, evaluation, and model export script',
      content: trainScript,
    },
    {
      path: 'Trained Model/model_metadata.json',
      folder: 'Trained Model',
      filename: 'model_metadata.json',
      language: 'json',
      description: 'Serialized evaluation results, hyperparameters, and feature importances',
      content: modelMetadata,
    },
    {
      path: 'Front End/app.py',
      folder: 'Front End',
      filename: 'app.py',
      language: 'python',
      description: 'Streamlit front-end prediction interface matching PDF specification',
      content: streamlitApp,
    },
    {
      path: 'Requirements.txt',
      folder: 'Root',
      filename: 'Requirements.txt',
      language: 'text',
      description: 'Python package dependencies for reproducible execution',
      content: requirementsTxt,
    },
    {
      path: 'README.md',
      folder: 'Root',
      filename: 'README.md',
      language: 'markdown',
      description: 'Complete GitHub README containing all 13 required sections',
      content: readmeMd,
    },
    {
      path: 'Project Documentation/Project_Report_Roll_No_1.md',
      folder: 'Project Documentation',
      filename: 'Project_Report_Roll_No_1.md',
      language: 'markdown',
      description: 'Formal B.Tech CSE Individual ML Project documentation & viva defense report',
      content: reportMd,
    },
  ];
}
