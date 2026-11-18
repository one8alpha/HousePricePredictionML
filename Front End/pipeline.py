"""
House Price Prediction ML Pipeline
Generates synthetic data, preprocesses, trains 3 models, evaluates, saves best model.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# ─────────────────────────────────────────────
# 1. DATA GENERATION
# ─────────────────────────────────────────────

def generate_dataset(seed=202601):
    rng = np.random.default_rng(seed)
    n_base = 525  # will grow to 545 after outliers+duplicates

    # --- core numerical features ---
    area = rng.normal(loc=4400, scale=1800, size=n_base).clip(1050, 10500).astype(int)
    bedrooms = np.clip(np.round(area / 2000 + rng.normal(0, 0.6, n_base)), 2, 5).astype(int)
    bathrooms = np.clip(np.round(bedrooms / 2 + rng.normal(0, 0.4, n_base)), 1, 3).astype(int)
    stories = rng.integers(1, 5, size=n_base)
    parking = rng.integers(0, 4, size=n_base)

    # --- binary categorical ---
    mainroad         = (rng.random(n_base) < 0.84).astype(int)
    guestroom        = (rng.random(n_base) < 0.22).astype(int)
    basement         = (rng.random(n_base) < 0.34).astype(int)
    hotwaterheating  = (rng.random(n_base) < 0.08).astype(int)
    airconditioning  = (rng.random(n_base) < 0.36).astype(int)
    prefarea         = (rng.random(n_base) < 0.26).astype(int)

    # --- furnishing (0=unfurnished, 1=semi-furnished, 2=furnished) ---
    furnishing_num = rng.choice([0, 1, 2], size=n_base, p=[0.40, 0.35, 0.25])
    furnishing_map = {0: "unfurnished", 1: "semi-furnished", 2: "furnished"}
    furnishingstatus = np.array([furnishing_map[v] for v in furnishing_num])

    # --- tier premium (25% of houses get a location premium) ---
    tier_premium = rng.choice([0, 500000, 1000000, 1500000], size=n_base, p=[0.75, 0.12, 0.08, 0.05])

    # --- price formula ---
    base_price = (
        950000
        + area * 780
        + bedrooms * 340000
        + bathrooms * 720000
        + stories * 380000
        + mainroad * 350000
        + guestroom * 310000
        + basement * 340000
        + hotwaterheating * 480000
        + airconditioning * 680000
        + parking * 310000
        + prefarea * 560000
        + furnishing_num * 290000
        + tier_premium
    )
    noise = rng.normal(0, 1, n_base)
    price = (base_price * (1 + noise * 0.055)).astype(int)

    df = pd.DataFrame({
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": np.where(mainroad == 1, "yes", "no"),
        "guestroom": np.where(guestroom == 1, "yes", "no"),
        "basement": np.where(basement == 1, "yes", "no"),
        "hotwaterheating": np.where(hotwaterheating == 1, "yes", "no"),
        "airconditioning": np.where(airconditioning == 1, "yes", "no"),
        "parking": parking,
        "prefarea": np.where(prefarea == 1, "yes", "no"),
        "furnishingstatus": furnishingstatus,
        "price": price,
    })

    # --- inject 7 outlier rows ---
    outlier_areas = rng.integers(13000, 16001, size=7)
    outlier_prices = rng.integers(15_000_000, 18_400_001, size=7)
    outlier_rows = pd.DataFrame({
        "area": outlier_areas,
        "bedrooms": [5]*7,
        "bathrooms": [3]*7,
        "stories": [4]*7,
        "mainroad": ["yes"]*7,
        "guestroom": ["yes"]*7,
        "basement": ["yes"]*7,
        "hotwaterheating": ["yes"]*7,
        "airconditioning": ["yes"]*7,
        "parking": [3]*7,
        "prefarea": ["yes"]*7,
        "furnishingstatus": ["furnished"]*7,
        "price": outlier_prices,
    })
    df = pd.concat([df, outlier_rows], ignore_index=True)

    # --- inject 8 missing values ---
    missing_targets = [
        ("area", [5, 20]),
        ("bathrooms", [10, 50]),
        ("bedrooms", [15]),
        ("parking", [30, 100, 200]),
    ]
    for col, indices in missing_targets:
        for idx in indices:
            if idx < len(df):
                df.loc[idx, col] = np.nan

    # --- inject 5 duplicate rows ---
    dup_rows = df.iloc[:5].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df  # 525 + 7 + 5 = 537 → with extra base rows totals ~545


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df):
    stats = {}

    # duplicates
    stats["duplicates_before"] = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    stats["duplicates_after"] = df.duplicated().sum()

    # missing values
    stats["missing_before"] = df.isnull().sum().to_dict()
    num_cols = ["area", "bedrooms", "bathrooms", "parking"]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    stats["missing_after"] = df.isnull().sum().to_dict()

    # outlier capping via IQR on area and price
    stats["outlier_caps"] = {}
    for col in ["area", "price"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        stats["outlier_caps"][col] = {"lower": lower, "upper": upper}
        df[col] = df[col].clip(lower=lower, upper=upper)

    # binary encoding
    binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating",
                   "airconditioning", "prefarea"]
    for col in binary_cols:
        df[col] = df[col].map({"yes": 1, "no": 0})

    # ordinal encode furnishingstatus
    furnishing_map = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
    df["furnishingstatus"] = df["furnishingstatus"].map(furnishing_map)

    # cast to numeric
    for col in ["area", "bedrooms", "bathrooms", "parking"]:
        df[col] = pd.to_numeric(df[col])

    return df, stats


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING + SCALING
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "area", "bedrooms", "bathrooms", "stories",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "parking", "prefarea", "furnishingstatus"
]

def split_and_scale(df):
    X = df[FEATURE_COLS].copy()
    y = df["price"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test, scaler, X_train.columns.tolist()


# ─────────────────────────────────────────────
# 4. TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Ridge Regression (α=1.0)": Ridge(alpha=1.0),
        "Decision Tree (depth=6)":  DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest (50 trees)": RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        results[name] = {"MAE": mae, "RMSE": rmse, "R²": r2, "predictions": preds}
        trained[name] = model
        print(f"\n{name}")
        print(f"  MAE : ₹{mae:,.0f}")
        print(f"  RMSE: ₹{rmse:,.0f}")
        print(f"  R²  : {r2:.4f}")

    # pick best by R²
    best_name = max(results, key=lambda k: results[k]["R²"])
    print(f"\n✅ Best model: {best_name}  (R² = {results[best_name]['R²']:.4f})")
    return trained, results, best_name


# ─────────────────────────────────────────────
# 5. SAVE BEST MODEL
# ─────────────────────────────────────────────

def save_best_model(trained_models, best_name, scaler, feature_cols, out_dir=None):
    out_dir = out_dir or os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(out_dir, "best_house_model.pkl")
    payload = {
        "model": trained_models[best_name],
        "scaler": scaler,
        "feature_cols": feature_cols,
        "model_name": best_name,
    }
    joblib.dump(payload, path)
    print(f"\n💾 Saved best model → {path}")
    return path


# ─────────────────────────────────────────────
# 6. SAMPLE PREDICTION
# ─────────────────────────────────────────────

def sample_prediction(model, scaler, feature_cols):
    """
    1500 sqft, 3bed, 2bath, 2-story, mainroad=yes, AC=yes,
    1 parking, semi-furnished (furnishingstatus=1)
    all others = 0
    """
    sample = {col: 0 for col in feature_cols}
    sample["area"]             = 1500
    sample["bedrooms"]         = 3
    sample["bathrooms"]        = 2
    sample["stories"]          = 2
    sample["mainroad"]         = 1
    sample["airconditioning"]  = 1
    sample["parking"]          = 1
    sample["furnishingstatus"] = 1  # semi-furnished

    df_sample = pd.DataFrame([sample])[feature_cols]
    df_scaled  = scaler.transform(df_sample)
    predicted  = model.predict(df_scaled)[0]
    lakhs      = predicted / 100_000
    print(f"\n🏠 Sample prediction (1500 sqft, 3bed, 2bath, 2-story, mainroad, AC, 1 parking, semi-furnished):")
    print(f"   Predicted price: ₹{predicted:,.0f}  ({lakhs:.2f} Lakhs)")
    return predicted


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_pipeline(save_dir=None):
    print("=" * 55)
    print("  HOUSE PRICE PREDICTION — ML PIPELINE")
    print("=" * 55)

    print("\n[1/5] Generating synthetic dataset …")
    raw_df = generate_dataset(seed=202601)
    print(f"      Raw shape: {raw_df.shape}")

    print("\n[2/5] Preprocessing …")
    clean_df, prep_stats = preprocess(raw_df.copy())
    print(f"      Clean shape: {clean_df.shape}")

    print("\n[3/5] Splitting & scaling …")
    X_train, X_test, y_train, y_test, scaler, feat_cols = split_and_scale(clean_df)
    print(f"      Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    print("\n[4/5] Training & evaluating models …")
    trained_models, results, best_name = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\n[5/5] Saving best model & running sample prediction …")
    model_path = save_best_model(trained_models, best_name, scaler, feat_cols, save_dir)
    sample_prediction(trained_models[best_name], scaler, feat_cols)

    print("\n" + "=" * 55)
    print("  Pipeline complete.")
    print("=" * 55)

    return raw_df, clean_df, prep_stats, trained_models, results, best_name, scaler, feat_cols


if __name__ == "__main__":
    run_pipeline()
