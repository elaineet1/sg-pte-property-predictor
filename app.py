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
    TIERS, FEATURE_COLS, CURRENT_YEAR,
    SEGMENT_ORDER, PROPTYPE_ORDER, TENURE_ORDER, LANDED_TYPES,
    load_and_preprocess, build_encoders, apply_encoders,
    compute_growth_rates, apply_target_encode,
    parse_floor, compute_lease_remaining, classify_tier,
)

st.set_page_config(page_title="SG Private Property Predictor", layout="wide")

MAX_PRED_YEARS = 5
TIER_COLORS    = {"Normal": "#2563EB", "Luxury": "#7C3AED", "Ultra": "#B45309"}
TIER_LABELS    = {
    "Normal": "🏢 Normal  (OCR / RCR)",
    "Luxury": "💎 Luxury  (CCR)",
    "Ultra":  "🏡 Ultra-luxury  (Landed)",
}

# ── Session state defaults ─────────────────────────────────────────────────────
for k in ("models", "encoders", "metrics", "growth_rates", "mape_by_seg",
          "df", "tier_counts", "project_meta", "trained"):
    if k not in st.session_state:
        st.session_state[k] = None
if st.session_state.trained is None:
    st.session_state.trained = False


def xgb_params():
    return dict(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_alpha=0.05, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0,
    )


def train_tier(tier_df, tier):
    feats = FEATURE_COLS[tier]
    # district_enc and project_enc are added by apply_encoders — exclude from pre-encode dropna
    base_feats = [f for f in feats if f not in ("district_enc", "project_enc")]
    tier_df = tier_df.dropna(subset=base_feats + ["price"])

    if len(tier_df) < 50:
        return None, None, None, None

    train_df, test_df = train_test_split(tier_df, test_size=0.15, random_state=42)
    encoders   = build_encoders(train_df)
    train_enc  = apply_encoders(train_df, encoders)
    test_enc   = apply_encoders(test_df,  encoders)

    X_train, y_train = train_enc[feats], train_enc["price"]
    X_test,  y_test  = test_enc[feats],  test_enc["price"]

    model = XGBRegressor(**xgb_params())
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred  = model.predict(X_test)
    metrics = {
        "r2":   r2_score(y_test, y_pred),
        "mae":  mean_absolute_error(y_test, y_pred),
        "mape": mean_absolute_percentage_error(y_test, y_pred) * 100,
        "rows": len(tier_df),
    }

    # Per-segment MAPE for uncertainty bands
    tmp = test_enc.copy()
    tmp["y_pred"] = y_pred
    tmp["y_true"] = y_test.values
    mape_seg = (
        tmp.groupby("Market Segment")
        .apply(lambda g: np.mean(np.abs((g["y_true"] - g["y_pred"]) / g["y_true"])) * 100)
        .to_dict()
    )
    return model, encoders, metrics, mape_seg


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1 · Upload Data")
    uploaded = st.file_uploader(
        "Upload 1–7 URA transaction CSV files",
        type="csv", accept_multiple_files=True,
    )
    st.caption("Only **Resale** transactions are used. Data is split into 3 tiers automatically.")
    train_btn = st.button("Train Models", type="primary", disabled=not uploaded)

    if st.session_state.trained:
        st.success("Models ready ✓")
        counts = st.session_state.tier_counts
        for tier, label in TIER_LABELS.items():
            n = counts.get(tier, 0)
            st.metric(label, f"{n:,} rows")

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

    models, encoders_dict, metrics_dict, mape_dict = {}, {}, {}, {}
    tier_counts = df["tier"].value_counts().to_dict()

    for tier in TIERS:
        tier_df = df[df["tier"] == tier].copy()
        n = len(tier_df)
        with st.spinner(f"Training {TIER_LABELS[tier]} model on {n:,} rows…"):
            model, enc, met, mape_seg = train_tier(tier_df, tier)
        if model is None:
            st.warning(f"Not enough data for {TIER_LABELS[tier]} (only {n} rows). Skipping.")
            continue
        models[tier]       = model
        encoders_dict[tier] = enc
        metrics_dict[tier]  = met
        mape_dict[tier]     = mape_seg

    growth_rates = compute_growth_rates(df)

    # Project metadata for auto-fill
    project_meta = (
        df.groupby("Project Name")
        .agg(
            tier=("tier",          lambda x: x.mode()[0]),
            segment=("Market Segment",  lambda x: x.mode()[0]),
            proptype=("Property Type",  lambda x: x.mode()[0]),
            tenure=("tenure_label",     lambda x: x.mode()[0]),
            lease_start=("lease_start", "median"),
            district=("district",       "median"),
        )
        .reset_index()
    )

    st.session_state.update({
        "models":       models,
        "encoders":     encoders_dict,
        "metrics":      metrics_dict,
        "mape_by_seg":  mape_dict,
        "growth_rates": growth_rates,
        "df":           df,
        "tier_counts":  tier_counts,
        "project_meta": project_meta.set_index("Project Name").to_dict("index"),
        "trained":      True,
    })

    with open("models.pkl",   "wb") as f: pickle.dump(models,        f)
    with open("encoders.pkl", "wb") as f: pickle.dump(encoders_dict, f)
    st.rerun()

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("🏢 SG Private Property Resale Price Predictor")

if not st.session_state.trained:
    st.info("Upload your URA transaction CSV files in the sidebar and click **Train Models**.")
    st.markdown("""
**Expected columns (standard URA residential transaction export):**

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

# ── Model performance — one column per tier ────────────────────────────────────
st.header("Model Performance by Tier")

metrics_dict = st.session_state.metrics
cols = st.columns(3)
for i, (tier, label) in enumerate(TIER_LABELS.items()):
    with cols[i]:
        st.markdown(f"#### {label}")
        if tier not in metrics_dict:
            st.warning("Insufficient data")
            continue
        m = metrics_dict[tier]
        st.metric("R² Score", f"{m['r2']:.3f}")
        st.metric("MAE",      f"S$ {m['mae']:,.0f}")
        st.metric("MAPE",     f"{m['mape']:.1f}%")
        st.caption(f"{m['rows']:,} resale transactions")

# ── Data insights ──────────────────────────────────────────────────────────────
with st.expander("Data insights & feature importance", expanded=False):
    df = st.session_state.df
    tab1, tab2, tab3 = st.tabs(["Sample data", "Price by tier", "Feature importance"])

    with tab1:
        st.dataframe(
            df[["Project Name", "tier", "Market Segment", "Property Type",
                "tenure_label", "area_sqft", "floor_mid", "sale_year", "price"]]
            .rename(columns={"price": "Transacted Price ($)",
                             "tenure_label": "Tenure", "tier": "Tier"})
            .head(100),
            use_container_width=True,
        )

    with tab2:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for i, tier in enumerate(TIERS):
            sub = df[df["tier"] == tier]["price"] / 1e6
            color = TIER_COLORS[tier]
            axes[i].hist(sub, bins=40, color=color, edgecolor="white")
            axes[i].set_title(TIER_LABELS[tier])
            axes[i].set_xlabel("Price (S$ M)")
            axes[i].set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        models = st.session_state.models
        fig2, axes2 = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
        if len(models) == 1:
            axes2 = [axes2]
        for ax, (tier, model) in zip(axes2, models.items()):
            feats = FEATURE_COLS[tier]
            labels = {
                "area_sqft": "Area (SQFT)", "floor_mid": "Floor Level",
                "lease_remaining": "Lease Remaining", "tenure_enc": "Tenure",
                "segment_enc": "Market Segment", "proptype_enc": "Property Type",
                "area_type_enc": "Strata/Land", "sale_year": "Sale Year",
                "sale_month": "Sale Month", "district_enc": "District",
                "project_enc": "Project Name",
            }
            imp = pd.Series(model.feature_importances_, index=feats).sort_values()
            imp.index = [labels.get(i, i) for i in imp.index]
            imp.plot(kind="barh", ax=ax, color=TIER_COLORS[tier])
            ax.set_title(f"{TIER_LABELS[tier]}\nFeature Importance")
        plt.tight_layout()
        st.pyplot(fig2)

# ── Prediction ─────────────────────────────────────────────────────────────────
st.header("Predict Resale Price")

models       = st.session_state.models
project_meta = st.session_state.project_meta
df           = st.session_state.df

# Step 1: choose tier
selected_tier = st.radio(
    "Property Tier",
    options=list(TIERS.keys()),
    format_func=lambda t: TIER_LABELS[t],
    horizontal=True,
)

if selected_tier not in models:
    st.warning(f"No model available for {TIER_LABELS[selected_tier]} — insufficient training data.")
    st.stop()

# Projects belonging to this tier
tier_projects = sorted([p for p, m in project_meta.items() if m.get("tier") == selected_tier])

st.markdown(f"**{len(tier_projects):,} projects** available in this tier.")

col1, col2, col3 = st.columns(3)

with col1:
    use_project = st.toggle("Select by project name", value=True)
    if use_project and tier_projects:
        selected_project = st.selectbox("Project Name", ["(manual entry)"] + tier_projects)
    else:
        selected_project = "(manual entry)"
        st.text_input("Project Name (reference only)", "")

    meta = project_meta.get(selected_project, {}) if selected_project != "(manual entry)" else {}

    area_sqft = st.number_input("Area (SQFT)", min_value=100.0, max_value=50000.0,
                                 value=800.0, step=50.0)

    last_year = int(df["sale_year"].max())
    pred_year = st.number_input(
        "Target Year",
        min_value=last_year,
        max_value=last_year + MAX_PRED_YEARS,
        value=last_year + 1, step=1,
        help=f"Capped at {last_year + MAX_PRED_YEARS} years ahead for reliability.",
    )
    pred_month = st.selectbox(
        "Month", list(range(1, 13)),
        format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"),
    )

with col2:
    seg_options = list(SEGMENT_ORDER.keys())
    seg_default = meta.get("segment", seg_options[0])
    seg_idx     = seg_options.index(seg_default) if seg_default in seg_options else 0
    market_seg  = st.selectbox("Market Segment", seg_options, index=seg_idx)

    # Property type filtered to tier
    if selected_tier == "Ultra":
        pt_options = list(LANDED_TYPES)
    elif selected_tier == "Luxury":
        pt_options = ["Condominium", "Executive Condominium", "Apartment"]
    else:
        pt_options = ["Apartment", "Condominium", "Executive Condominium"]

    pt_default = meta.get("proptype", pt_options[0])
    pt_idx     = pt_options.index(pt_default) if pt_default in pt_options else 0
    prop_type  = st.selectbox("Property Type", pt_options, index=pt_idx)

    # Floor level only for non-landed tiers
    if selected_tier != "Ultra":
        floor_opts = ["B1 to B5","01 to 05","06 to 10","11 to 15","16 to 20",
                      "21 to 25","26 to 30","31 to 35","36 to 40","41 to 45",
                      "46 to 50","51 to 55","56 to 60"]
        floor_sel = st.selectbox("Floor Level", floor_opts, index=2)
    else:
        floor_sel = "01 to 05"
        st.info("Floor level not applicable for landed properties.")

with col3:
    ten_options = list(TENURE_ORDER.keys())
    ten_default = meta.get("tenure", "99yr")
    ten_idx     = ten_options.index(ten_default) if ten_default in ten_options else 2
    tenure_sel  = st.selectbox("Tenure", ten_options, index=ten_idx)

    ls_raw     = meta.get("lease_start", 2010)
    ls_default = int(ls_raw) if not pd.isna(ls_raw) else 2010
    lease_yr   = st.number_input("Lease Commencement Year", min_value=1900,
                                  max_value=CURRENT_YEAR, value=ls_default, step=1,
                                  help="Ignored for Freehold")

    dist_raw     = meta.get("district", 15)
    dist_default = int(dist_raw) if not pd.isna(dist_raw) else 15
    district_no  = st.number_input("Postal District", min_value=1, max_value=28,
                                    value=dist_default, step=1)

predict_btn = st.button("Predict Price", type="primary")

# ── Prediction logic ───────────────────────────────────────────────────────────
if predict_btn:
    model        = models[selected_tier]
    encoders     = st.session_state.encoders[selected_tier]
    growth       = st.session_state.growth_rates
    mape_seg_map = st.session_state.mape_by_seg.get(selected_tier, {})
    feats        = FEATURE_COLS[selected_tier]

    proj_map, proj_global = encoders["project"]
    dist_map, dist_global = encoders["district"]

    proj_enc  = proj_map.get(selected_project, proj_global) if selected_project != "(manual entry)" else proj_global
    dist_enc  = dist_map.get(float(district_no), dist_global)
    floor_mid = parse_floor(floor_sel)
    annual_rate = growth.get(market_seg, 0.03)
    years_ahead = pred_year - last_year

    def make_row(year):
        lr = compute_lease_remaining(tenure_sel, float(lease_yr), float(year))
        row = {
            "area_sqft":       area_sqft,
            "floor_mid":       floor_mid,
            "lease_remaining": lr,
            "tenure_enc":      float(TENURE_ORDER.get(tenure_sel, 0)),
            "segment_enc":     float(SEGMENT_ORDER.get(market_seg, 0)),
            "proptype_enc":    float(PROPTYPE_ORDER.get(prop_type, 0)),
            "area_type_enc":   1.0 if selected_tier == "Ultra" else 0.0,
            "sale_year":       float(last_year),
            "sale_month":      float(pred_month),
            "district_enc":    dist_enc,
            "project_enc":     proj_enc,
        }
        return {k: row[k] for k in feats}

    base_price  = float(model.predict(pd.DataFrame([make_row(last_year)]))[0])
    predicted   = base_price * ((1 + annual_rate) ** years_ahead)
    base_mape   = mape_seg_map.get(market_seg, metrics_dict[selected_tier]["mape"]) / 100
    total_mape  = base_mape * (1 + 0.20 * years_ahead)
    lower       = predicted * (1 - total_mape)
    upper       = predicted * (1 + total_mape)
    psf         = predicted / area_sqft
    color       = TIER_COLORS[selected_tier]

    # Results
    st.divider()
    st.subheader(f"{TIER_LABELS[selected_tier]} — Predicted Price for {pred_year}")
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Central Estimate", f"S$ {predicted:,.0f}")
    rc2.metric("Lower Bound",      f"S$ {lower:,.0f}")
    rc3.metric("Upper Bound",      f"S$ {upper:,.0f}")
    rc4.metric("Price PSF",        f"S$ {psf:,.0f}")

    st.caption(
        f"Base model MAPE: {base_mape*100:.1f}% · "
        f"Uncertainty widens 20% per year beyond {last_year} · "
        f"Growth rate ({market_seg}): {annual_rate*100:.1f}% p.a."
    )

    # Trend chart
    chart_years = list(range(last_year, pred_year + MAX_PRED_YEARS + 1))
    central, lowers, uppers = [], [], []
    for yr in chart_years:
        ya = yr - last_year
        c  = base_price * ((1 + annual_rate) ** ya)
        mp = base_mape * (1 + 0.20 * ya)
        central.append(c / 1e6)
        lowers.append(c * (1 - mp) / 1e6)
        uppers.append(c * (1 + mp) / 1e6)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(chart_years, lowers, uppers, alpha=0.18, color=color, label="Uncertainty band")
    ax.plot(chart_years, central, marker="o", color=color, linewidth=2, label="Central estimate")
    ax.axvline(pred_year,  color="red",  linestyle="--", alpha=0.7, label=f"Target: {pred_year}")
    ax.axvline(last_year,  color="gray", linestyle=":",  alpha=0.5, label=f"Last data: {last_year}")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"S${x:.2f}M"))
    ax.set_xlabel("Year")
    ax.set_ylabel("Predicted Price")
    proj_label = selected_project if selected_project != "(manual entry)" else prop_type
    ax.set_title(f"{proj_label} · {area_sqft:.0f} sqft"
                 + (f" · {floor_sel}" if selected_tier != "Ultra" else "")
                 + f" · {tenure_sel}")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.warning(
        f"⚠ Predictions beyond {last_year} are projections, not guarantees. "
        "SG property prices are highly sensitive to government cooling measures, "
        "interest rate changes, and macroeconomic events."
    )
