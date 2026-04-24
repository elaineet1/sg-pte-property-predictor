import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor

from preprocessing import (
    FEATURE_COLS, CURRENT_YEAR,
    SEGMENT_ORDER, PROPTYPE_ORDER, TENURE_ORDER,
    load_and_preprocess, build_encoders, apply_encoders,
    compute_segment_growth_rates, apply_target_encode,
    parse_floor, compute_lease_remaining,
)

st.set_page_config(page_title="SG Private Property Predictor", layout="wide")

MODEL_PATH   = "model.pkl"
ENCODER_PATH = "encoders.pkl"
MAX_PRED_YEARS = 5

# ── Session state defaults ─────────────────────────────────────────────────────
for k in ("model", "encoders", "df", "metrics", "growth_rates",
          "project_list", "project_meta", "trained", "mape_by_seg"):
    if k not in st.session_state:
        st.session_state[k] = None
if st.session_state.trained is None:
    st.session_state.trained = False

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1 · Upload Data")
    uploaded = st.file_uploader(
        "Upload 1–7 URA transaction CSV files",
        type="csv",
        accept_multiple_files=True,
    )
    st.caption("Only **Resale** transactions are used for training.")
    train_btn = st.button("Train Model", type="primary", disabled=not uploaded)

    if st.session_state.trained:
        st.success("Model ready ✓")
        df = st.session_state.df
        st.metric("Training rows", f"{len(df):,}")
        st.metric("Projects", f"{df['Project Name'].nunique():,}")
        st.metric("Years covered", f"{int(df['sale_year'].min())}–{int(df['sale_year'].max())}")

# ── Train ──────────────────────────────────────────────────────────────────────
if train_btn and uploaded:
    tmp_paths = []
    for f in uploaded:
        tmp = f"_tmp_{f.name}"
        with open(tmp, "wb") as out:
            out.write(f.read())
        tmp_paths.append(tmp)

    with st.spinner("Loading and cleaning data (Resale only)…"):
        df = load_and_preprocess(tmp_paths)
    for p in tmp_paths:
        os.remove(p)

    if df is None or df.empty:
        st.error("No valid Resale data found. Check your CSV files.")
        st.stop()

    with st.spinner("Building target encoders…"):
        train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
        encoders = build_encoders(train_df)
        train_enc = apply_encoders(train_df, encoders)
        test_enc  = apply_encoders(test_df,  encoders)

    with st.spinner("Training XGBoost model…"):
        X_train = train_enc[FEATURE_COLS]
        y_train = train_enc["price"]
        X_test  = test_enc[FEATURE_COLS]
        y_test  = test_enc["price"]

        model = XGBRegressor(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    metrics = {
        "r2":   r2_score(y_test, y_pred),
        "mae":  mean_absolute_error(y_test, y_pred),
        "mape": mean_absolute_percentage_error(y_test, y_pred) * 100,
    }

    # MAPE per segment (for uncertainty bands)
    test_enc2 = test_enc.copy()
    test_enc2["y_pred"] = y_pred
    test_enc2["y_true"] = y_test.values
    mape_by_seg = (
        test_enc2.groupby("Market Segment")
        .apply(lambda g: np.mean(np.abs((g["y_true"] - g["y_pred"]) / g["y_true"])) * 100)
        .to_dict()
    )

    growth_rates = compute_segment_growth_rates(df)

    # Project metadata for auto-fill
    project_meta = (
        df.groupby("Project Name")
        .agg(
            segment=("Market Segment",   lambda x: x.mode()[0]),
            proptype=("Property Type",   lambda x: x.mode()[0]),
            tenure=("tenure_label",      lambda x: x.mode()[0]),
            lease_start=("lease_start",  "median"),
            district=("district",        "median"),
            area_type=("Type of Area",   lambda x: x.mode()[0]),
        )
        .reset_index()
    )

    # Persist to session
    st.session_state.update({
        "model":        model,
        "encoders":     encoders,
        "df":           df,
        "metrics":      metrics,
        "mape_by_seg":  mape_by_seg,
        "growth_rates": growth_rates,
        "project_list": sorted(df["Project Name"].unique().tolist()),
        "project_meta": project_meta.set_index("Project Name").to_dict("index"),
        "trained":      True,
    })

    with open(MODEL_PATH,   "wb") as f: pickle.dump(model,    f)
    with open(ENCODER_PATH, "wb") as f: pickle.dump(encoders, f)
    st.rerun()

# ── Main content ───────────────────────────────────────────────────────────────
st.title("🏢 SG Private Property Resale Price Predictor")

if not st.session_state.trained:
    st.info("Upload your URA transaction CSV files in the sidebar and click **Train Model** to begin.")
    st.markdown("""
**Expected columns (standard URA export):**

| Column | Example |
|---|---|
| Project Name | MAYFAIR MODERN |
| Transacted Price ($) | 1,800,000 |
| Area (SQFT) | 807.3 |
| Sale Date | Apr-26 |
| Type of Sale | Resale |
| Property Type | Condominium |
| Tenure | 99 yrs lease commencing from 2018 |
| Market Segment | Rest of Central Region |
| Floor Level | 01 to 05 |
| Postal District | 21 |
| Type of Area | Strata |
    """)
    st.stop()

# ── Metrics row ────────────────────────────────────────────────────────────────
m = st.session_state.metrics
st.header("Model Performance")
c1, c2, c3 = st.columns(3)
c1.metric("R² Score", f"{m['r2']:.3f}",
          help="1.0 = perfect. Above 0.85 is strong for property data.")
c2.metric("Mean Absolute Error", f"S$ {m['mae']:,.0f}",
          help="Average dollar difference between predicted and actual price.")
c3.metric("MAPE", f"{m['mape']:.1f}%",
          help="Average % error. Below 10% is excellent.")

# ── Charts (collapsible) ───────────────────────────────────────────────────────
with st.expander("Data insights & feature importance", expanded=False):
    df   = st.session_state.df
    tab1, tab2, tab3 = st.tabs(["Sample data", "Price distribution", "Feature importance"])

    with tab1:
        st.dataframe(
            df[["Project Name", "Market Segment", "Property Type",
                "tenure_label", "area_sqft", "floor_mid", "sale_year", "price"]]
            .rename(columns={"price": "Transacted Price ($)", "tenure_label": "Tenure"})
            .head(100),
            use_container_width=True,
        )

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(df["price"] / 1e6, bins=60, color="#2563EB", edgecolor="white")
        axes[0].set_xlabel("Price (S$ M)")
        axes[0].set_title("Price Distribution")

        seg_med = df.groupby("Market Segment")["price"].median() / 1e6
        seg_med.sort_values().plot(kind="barh", ax=axes[1], color="#2563EB")
        axes[1].set_xlabel("Median Price (S$ M)")
        axes[1].set_title("Median by Segment")
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        model = st.session_state.model
        imp   = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
        labels = {
            "area_sqft": "Area (SQFT)", "floor_mid": "Floor Level",
            "lease_remaining": "Lease Remaining", "tenure_enc": "Tenure Type",
            "segment_enc": "Market Segment", "proptype_enc": "Property Type",
            "area_type_enc": "Strata vs Land", "sale_year": "Sale Year",
            "sale_month": "Sale Month", "district_enc": "Postal District",
            "project_enc": "Project Name",
        }
        imp.index = [labels.get(i, i) for i in imp.index]
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        imp.plot(kind="barh", ax=ax2, color="#2563EB")
        ax2.set_title("What drives the price?")
        plt.tight_layout()
        st.pyplot(fig2)

# ── Prediction form ────────────────────────────────────────────────────────────
st.header("Predict Resale Price")
st.markdown(f"Select a project or fill in details manually. Predictions capped at **{MAX_PRED_YEARS} years ahead** for reliability.")

project_list = st.session_state.project_list
project_meta = st.session_state.project_meta

use_project = st.toggle("Select by project name", value=True)

col1, col2, col3 = st.columns(3)

with col1:
    if use_project:
        selected_project = st.selectbox("Project Name", ["(manual entry)"] + project_list)
    else:
        selected_project = "(manual entry)"
        st.text_input("Project Name (for reference)", value="")

    # Auto-fill from project metadata
    meta = project_meta.get(selected_project, {}) if selected_project != "(manual entry)" else {}

    area_sqft = st.number_input("Area (SQFT)", min_value=100.0, max_value=30000.0,
                                 value=800.0, step=50.0)

    last_year = int(st.session_state.df["sale_year"].max())
    pred_year = st.number_input(
        "Target Year",
        min_value=last_year,
        max_value=last_year + MAX_PRED_YEARS,
        value=min(last_year + 1, last_year + MAX_PRED_YEARS),
        step=1,
        help=f"Predictions beyond {last_year + MAX_PRED_YEARS} are disabled for reliability."
    )
    pred_month = st.selectbox("Month", list(range(1, 13)),
                               format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"),
                               index=0)

with col2:
    seg_default = meta.get("segment", "Outside Central Region")
    seg_options = list(SEGMENT_ORDER.keys())
    seg_idx     = seg_options.index(seg_default) if seg_default in seg_options else 2
    market_seg  = st.selectbox("Market Segment", seg_options, index=seg_idx)

    pt_options  = list(PROPTYPE_ORDER.keys())
    pt_default  = meta.get("proptype", "Condominium")
    pt_idx      = pt_options.index(pt_default) if pt_default in pt_options else 4
    prop_type   = st.selectbox("Property Type", pt_options, index=pt_idx)

    floor_opts  = ["B1 to B5","01 to 05","06 to 10","11 to 15","16 to 20",
                   "21 to 25","26 to 30","31 to 35","36 to 40","41 to 45",
                   "46 to 50","51 to 55","56 to 60"]
    floor_sel   = st.selectbox("Floor Level", floor_opts, index=2)

with col3:
    ten_opts    = list(TENURE_ORDER.keys())
    ten_default = meta.get("tenure", "99yr")
    ten_idx     = ten_opts.index(ten_default) if ten_default in ten_opts else 2
    tenure_sel  = st.selectbox("Tenure", ten_opts, index=ten_idx)

    ls_default  = int(meta.get("lease_start", 2015)) if not pd.isna(meta.get("lease_start", np.nan)) else 2015
    lease_yr    = st.number_input("Lease Commencement Year", min_value=1900,
                                   max_value=CURRENT_YEAR, value=ls_default, step=1,
                                   help="Ignored for Freehold")

    dist_default = int(meta.get("district", 15)) if not pd.isna(meta.get("district", np.nan)) else 15
    district_no  = st.number_input("Postal District", min_value=1, max_value=28,
                                    value=dist_default, step=1)

predict_btn = st.button("Predict Price", type="primary")

# ── Prediction logic ───────────────────────────────────────────────────────────
if predict_btn:
    model    = st.session_state.model
    encoders = st.session_state.encoders
    growth   = st.session_state.growth_rates
    mape_seg = st.session_state.mape_by_seg

    proj_map, proj_global = encoders["project"]
    dist_map, dist_global = encoders["district"]

    proj_enc  = proj_map.get(selected_project, proj_global) if selected_project != "(manual entry)" else proj_global
    dist_enc  = dist_map.get(float(district_no), dist_global)
    floor_mid = parse_floor(floor_sel)
    lrem      = compute_lease_remaining(tenure_sel, float(lease_yr), float(pred_year))
    base_year = int(st.session_state.df["sale_year"].max())

    def build_row(year):
        lr = compute_lease_remaining(tenure_sel, float(lease_yr), float(year))
        return {
            "area_sqft":      area_sqft,
            "floor_mid":      floor_mid,
            "lease_remaining": lr,
            "tenure_enc":     TENURE_ORDER.get(tenure_sel, 0),
            "segment_enc":    SEGMENT_ORDER.get(market_seg, 0),
            "proptype_enc":   PROPTYPE_ORDER.get(prop_type, 0),
            "area_type_enc":  0.0,
            "sale_year":      float(base_year),   # model predicts at last known year
            "sale_month":     float(pred_month),
            "district_enc":   dist_enc,
            "project_enc":    proj_enc,
        }

    # Base prediction at last known year, then apply growth for future years
    base_row    = pd.DataFrame([build_row(base_year)])
    base_price  = float(model.predict(base_row)[0])
    annual_rate = growth.get(market_seg, 0.03)
    years_ahead = pred_year - base_year
    predicted   = base_price * ((1 + annual_rate) ** years_ahead)

    # Uncertainty band: MAPE widens 20% per year into future
    base_mape  = mape_seg.get(market_seg, m["mape"]) / 100
    total_mape = base_mape * (1 + 0.20 * years_ahead)
    lower      = predicted * (1 - total_mape)
    upper      = predicted * (1 + total_mape)
    psf        = predicted / area_sqft

    # ── Results ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"Predicted Resale Price for {pred_year}")

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Central Estimate",  f"S$ {predicted:,.0f}")
    rc2.metric("Lower Bound",       f"S$ {lower:,.0f}")
    rc3.metric("Upper Bound",       f"S$ {upper:,.0f}")
    rc4.metric("Price PSF",         f"S$ {psf:,.0f}")

    st.caption(
        f"Uncertainty band based on {base_mape*100:.1f}% model error on {market_seg}, "
        f"widening by 20% per year beyond {base_year}. "
        f"Annual appreciation rate used: **{annual_rate*100:.1f}%** ({market_seg})."
    )

    # ── Trend chart ─────────────────────────────────────────────────────────
    years   = list(range(base_year, pred_year + MAX_PRED_YEARS + 1))
    central, lowers, uppers = [], [], []

    for yr in years:
        yr_ahead = yr - base_year
        c = base_price * ((1 + annual_rate) ** yr_ahead)
        mape_yr = base_mape * (1 + 0.20 * yr_ahead)
        central.append(c / 1e6)
        lowers.append(c * (1 - mape_yr) / 1e6)
        uppers.append(c * (1 + mape_yr) / 1e6)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(years, lowers, uppers, alpha=0.2, color="#2563EB", label="Uncertainty band")
    ax.plot(years, central, marker="o", color="#2563EB", label="Central estimate", linewidth=2)
    ax.axvline(pred_year, color="red", linestyle="--", alpha=0.7, label=f"Target: {pred_year}")
    ax.axvline(base_year, color="gray", linestyle=":", alpha=0.5, label=f"Last data: {base_year}")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"S${x:.2f}M"))
    ax.set_xlabel("Year")
    ax.set_ylabel("Predicted Price")
    ax.set_title(
        f"Price Trend — {selected_project if selected_project != '(manual entry)' else prop_type} "
        f"| {area_sqft:.0f} sqft | {floor_sel} | {tenure_sel}"
    )
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.warning(
        f"⚠ Predictions beyond {base_year} are projections, not guarantees. "
        "SG property prices are sensitive to government cooling measures, interest rates, "
        "and macroeconomic conditions that no model can foresee."
    )
