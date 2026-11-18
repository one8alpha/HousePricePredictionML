"""
House Price Prediction — Streamlit App
Runs the full pipeline inline and presents results across 5 tabs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

# ── make pipeline importable from same directory ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (
    generate_dataset, preprocess, split_and_scale,
    train_and_evaluate, save_best_model, sample_prediction,
    FEATURE_COLS
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="House Price ML Pipeline", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# CACHED PIPELINE  (runs once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running ML pipeline …")
def load_pipeline():
    raw_df = generate_dataset(seed=202601)
    clean_df, prep_stats = preprocess(raw_df.copy())

    X_train, X_test, y_train, y_test, scaler, feat_cols = split_and_scale(clean_df)
    trained_models, results, best_name = train_and_evaluate(X_train, X_test, y_train, y_test)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_best_model(trained_models, best_name, scaler, feat_cols, save_dir)

    return raw_df, clean_df, prep_stats, trained_models, results, best_name, scaler, feat_cols, X_test, y_test


(raw_df, clean_df, prep_stats,
 trained_models, results, best_name,
 scaler, feat_cols, X_test, y_test) = load_pipeline()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🏠 House Price Prediction — ML Pipeline")
st.caption("Synthetic dataset · Ridge · Decision Tree · Random Forest")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset Overview",
    "🔍 EDA",
    "🛠️ Preprocessing",
    "📈 Model Evaluation",
    "🤖 Live Predictor",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Raw Dataset (first 20 rows)")
    st.dataframe(raw_df.head(20), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total rows", raw_df.shape[0])
    col2.metric("Total columns", raw_df.shape[1])
    col3.metric("Duplicate rows", int(raw_df.duplicated().sum()))

    st.subheader("Missing Value Counts")
    missing = raw_df.isnull().sum()
    missing_df = missing[missing > 0].rename("Missing Count").reset_index()
    missing_df.columns = ["Column", "Missing Count"]
    if missing_df.empty:
        st.info("No missing values found.")
    else:
        st.dataframe(missing_df, use_container_width=True)

    st.subheader("Data Types")
    dtype_df = raw_df.dtypes.rename("dtype").reset_index()
    dtype_df.columns = ["Column", "dtype"]
    st.dataframe(dtype_df, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Exploratory Data Analysis")

    # ── price distribution ────────────────────────────────────────────────────
    st.markdown("#### Price Distribution")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(clean_df["price"], bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Price (INR)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of House Prices (after preprocessing)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M"))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col_a, col_b = st.columns(2)

    # ── avg price by furnishing ───────────────────────────────────────────────
    with col_a:
        st.markdown("#### Avg Price by Furnishing Status")
        furn_map_rev = {0: "unfurnished", 1: "semi-furnished", 2: "furnished"}
        avg_furn = (
            clean_df.assign(furnishing=clean_df["furnishingstatus"].map(furn_map_rev))
            .groupby("furnishing")["price"].mean()
            .reindex(["unfurnished", "semi-furnished", "furnished"])
        )
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        bars = ax2.bar(avg_furn.index, avg_furn.values / 1e5, color=["#DD8452", "#55A868", "#4C72B0"])
        ax2.set_ylabel("Avg Price (Lakhs ₹)")
        ax2.set_title("Furnishing Status vs Avg Price")
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"₹{bar.get_height():.1f}L", ha="center", va="bottom", fontsize=9)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── avg price by bedrooms ─────────────────────────────────────────────────
    with col_b:
        st.markdown("#### Avg Price by Number of Bedrooms")
        avg_bed = clean_df.groupby("bedrooms")["price"].mean().sort_index()
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.bar(avg_bed.index.astype(str), avg_bed.values / 1e5, color="#C44E52")
        ax3.set_xlabel("Bedrooms")
        ax3.set_ylabel("Avg Price (Lakhs ₹)")
        ax3.set_title("Bedrooms vs Avg Price")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # ── correlation heatmap ───────────────────────────────────────────────────
    st.markdown("#### Correlation Heatmap (numerical features vs price)")
    num_df = clean_df[FEATURE_COLS + ["price"]].select_dtypes(include="number")
    corr   = num_df.corr()
    fig4, ax4 = plt.subplots(figsize=(10, 7))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax4, mask=mask)
    ax4.set_title("Feature Correlation Matrix")
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Preprocessing Steps")

    # duplicates
    st.markdown("#### Duplicate Rows")
    col1, col2 = st.columns(2)
    col1.metric("Before removal", prep_stats["duplicates_before"])
    col2.metric("After removal", prep_stats["duplicates_after"])

    # missing values
    st.markdown("#### Missing Values")
    before_mv = {k: v for k, v in prep_stats["missing_before"].items() if v > 0}
    after_mv  = {k: v for k, v in prep_stats["missing_after"].items() if v > 0}
    mv_df = pd.DataFrame({
        "Column": list(before_mv.keys()),
        "Missing Before": list(before_mv.values()),
        "Missing After": [after_mv.get(k, 0) for k in before_mv.keys()],
    })
    if mv_df.empty:
        st.info("No missing values detected.")
    else:
        st.dataframe(mv_df, use_container_width=True)

    # outlier caps
    st.markdown("#### Outlier Capping (IQR × 1.5 Winsorization)")
    cap_rows = []
    for col, bounds in prep_stats["outlier_caps"].items():
        cap_rows.append({
            "Feature": col,
            "Lower Cap": f"{bounds['lower']:,.0f}",
            "Upper Cap": f"{bounds['upper']:,.0f}",
        })
    st.dataframe(pd.DataFrame(cap_rows), use_container_width=True)

    # summary stats of cleaned df
    st.markdown("#### Cleaned Dataset — Summary Statistics")
    st.dataframe(clean_df.describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown("#### Cleaned Dataset (first 10 rows)")
    st.dataframe(clean_df.head(10), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Model Evaluation")

    # ── metrics table ─────────────────────────────────────────────────────────
    st.markdown("#### Metrics — MAE · RMSE · R²")
    metric_rows = []
    for name, res in results.items():
        metric_rows.append({
            "Model": name,
            "MAE (₹)": f"{res['MAE']:,.0f}",
            "RMSE (₹)": f"{res['RMSE']:,.0f}",
            "R²": f"{res['R²']:.4f}",
            "Best?": "✅" if name == best_name else "",
        })
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)

    col_left, col_right = st.columns(2)

    # ── feature importance — Random Forest ───────────────────────────────────
    with col_left:
        st.markdown("#### Feature Importance (Random Forest)")
        rf_key = [k for k in trained_models if "Forest" in k][0]
        rf_model = trained_models[rf_key]
        importances = pd.Series(rf_model.feature_importances_, index=feat_cols).sort_values()
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importances)))
        ax5.barh(importances.index, importances.values, color=colors)
        ax5.set_xlabel("Importance")
        ax5.set_title("Random Forest — Feature Importances")
        fig5.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

    # ── actual vs predicted ───────────────────────────────────────────────────
    with col_right:
        st.markdown(f"#### Actual vs Predicted — {best_name}")
        best_preds = results[best_name]["predictions"]
        fig6, ax6 = plt.subplots(figsize=(6, 5))
        ax6.scatter(y_test / 1e5, best_preds / 1e5, alpha=0.5, s=20, color="#4C72B0")
        lims = [
            min(y_test.min(), best_preds.min()) / 1e5,
            max(y_test.max(), best_preds.max()) / 1e5,
        ]
        ax6.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
        ax6.set_xlabel("Actual Price (Lakhs ₹)")
        ax6.set_ylabel("Predicted Price (Lakhs ₹)")
        ax6.set_title("Actual vs Predicted")
        ax6.legend()
        fig6.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — LIVE PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("🏠 Live House Price Predictor")
    st.caption(f"Using: **{best_name}**")

    with st.form("predictor_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            area         = st.number_input("Area (sq ft)", min_value=500, max_value=15000, value=1500, step=50)
            bedrooms     = st.slider("Bedrooms", 1, 5, 3)
            bathrooms    = st.slider("Bathrooms", 1, 4, 2)
            stories      = st.slider("Stories", 1, 4, 2)

        with c2:
            mainroad        = st.checkbox("Main Road access", value=True)
            guestroom       = st.checkbox("Guest Room")
            basement        = st.checkbox("Basement")
            hotwaterheating = st.checkbox("Hot Water Heating")
            airconditioning = st.checkbox("Air Conditioning", value=True)

        with c3:
            parking         = st.slider("Parking spots", 0, 3, 1)
            prefarea        = st.checkbox("Preferred Area")
            furnishingstatus = st.selectbox(
                "Furnishing Status",
                options=["unfurnished", "semi-furnished", "furnished"],
                index=1,
            )

        submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)

    if submitted:
        furn_encode = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
        input_dict = {
            "area":             area,
            "bedrooms":         bedrooms,
            "bathrooms":        bathrooms,
            "stories":          stories,
            "mainroad":         int(mainroad),
            "guestroom":        int(guestroom),
            "basement":         int(basement),
            "hotwaterheating":  int(hotwaterheating),
            "airconditioning":  int(airconditioning),
            "parking":          parking,
            "prefarea":         int(prefarea),
            "furnishingstatus": furn_encode[furnishingstatus],
        }
        input_df = pd.DataFrame([input_dict])[feat_cols]
        scaled   = scaler.transform(input_df)
        price    = trained_models[best_name].predict(scaled)[0]
        lakhs    = price / 1e5

        st.markdown("---")
        st.success(f"### Estimated Price: ₹{lakhs:.2f} Lakhs")

        metric_cols = st.columns(3)
        metric_cols[0].metric("Price (INR)", f"₹{price:,.0f}")
        metric_cols[1].metric("Price (Lakhs)", f"₹{lakhs:.2f} L")
        metric_cols[2].metric("Price (Crores)", f"₹{price/1e7:.3f} Cr")

        with st.expander("Input Summary"):
            st.json(input_dict)
