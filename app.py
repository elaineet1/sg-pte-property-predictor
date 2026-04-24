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
    parse_floor, compute_lease_remaining, classify_tier, get_region_cluster,
)

st.set_page_config(page_title="SG Private Property Predictor", layout="wide")

MAX_PRED_YEARS = 5

TIER_LABELS = {
    "OCR":   "🏢 Normal — OCR",
    "RCR":   "🏙️ Normal — RCR",
    "CCR":   "💎 Luxury — CCR",
    "Ultra": "🏡 Ultra-luxury — Landed",
}
TIER_COLORS = {
    "OCR":   "#2563EB",
    "RCR":   "#059669",
    "CCR":   "#7C3AED",
    "Ultra": "#B45309",
}
# Map tier → market segment for growth rate lookup
TIER_TO_SEGMENT = {
    "OCR":   "Outside Central Region",
    "RCR":   "Rest of Central Region",
    "CCR":   "Core Central Region",
    "Ultra": "Outside Central Region",
}

# ── Session state ──────────────────────────────────────────────────────────────
for k in ("models", "encoders", "metrics", "growth_rates", "mape_by_seg",
          "df", "tier_counts", "project_meta", "trained"):
    if k not in st.session_state:
        st.session_state[k] = None
if st.session_state.trained is None:
    st.session_state.trained = False


def xgb_params():
    return dict(
        n_estimators=600, max_depth=8, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_alpha=0.05, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0,
    )


def train_tier(tier_df, tier):
    feats      = FEATURE_COLS[tier]
    target_enc = {"district_enc", "project_enc", "region_cluster_enc"}
    base_feats = [f for f in feats if f not in target_enc]
    tier_df    = tier_df.dropna(subset=base_feats + ["price"])

    if len(tier_df) < 50:
        return None, None, None, None

    train_df, test_df = train_test_split(tier_df, test_size=0.15, random_state=42)
    encoders  = build_encoders(train_df)
    train_enc = apply_encoders(train_df, encoders)
    test_enc  = apply_encoders(test_df,  encoders)

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
    st.caption("Only **Resale** transactions are used. Auto-split into 4 tiers.")
    train_btn = st.button("Train Models", type="primary", disabled=not uploaded)

    if st.session_state.trained:
        st.success("Models ready ✓")
        counts = st.session_state.tier_counts or {}
        for tier, label in TIER_LABELS.items():
            st.metric(label, f"{counts.get(tier, 0):,} rows")

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

    for tier, full_label in TIERS.items():
        tier_df = df[df["tier"] == tier].copy()
        n = len(tier_df)
        with st.spinner(f"Training {TIER_LABELS[tier]} on {n:,} rows…"):
            model, enc, met, mape_seg = train_tier(tier_df, tier)
        if model is None:
            st.warning(f"Not enough data for {TIER_LABELS[tier]} ({n} rows). Skipping.")
            continue
        models[tier]        = model
        encoders_dict[tier] = enc
        metrics_dict[tier]  = met
        mape_dict[tier]     = mape_seg

    growth_rates = compute_growth_rates(df)

    project_meta = (
        df.groupby("Project Name")
        .agg(
            tier=("tier",            lambda x: x.mode()[0]),
            segment=("Market Segment", lambda x: x.mode()[0]),
            proptype=("Property Type",  lambda x: x.mode()[0]),
            tenure=("tenure_label",     lambda x: x.mode()[0]),
            lease_start=("lease_start", "median"),
            district=("district",       "median"),
            region=("region_cluster",   lambda x: x.mode()[0]),
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

# ── Metrics — 4 columns ────────────────────────────────────────────────────────
st.header("Model Performance by Tier")
metrics_dict = st.session_state.metrics
cols = st.columns(4)
for i, (tier, label) in enumerate(TIER_LABELS.items()):
    with cols[i]:
        color = TIER_COLORS[tier]
        st.markdown(f"<h4 style='color:{color}'>{label}</h4>", unsafe_allow_html=True)
        if tier not in metrics_dict:
            st.warning("Insufficient data")
            continue
        m = metrics_dict[tier]
        r2_color = "green" if m["r2"] > 0.7 else ("orange" if m["r2"] > 0.4 else "red")
        st.markdown(f"**R²:** <span style='color:{r2_color}'>{m['r2']:.3f}</span>",
                    unsafe_allow_html=True)
        st.metric("MAE",  f"S$ {m['mae']:,.0f}")
        st.metric("MAPE", f"{m['mape']:.1f}%")
        st.caption(f"{m['rows']:,} resale transactions")

# ── R² guide ──────────────────────────────────────────────────────────────────
with st.expander("How to read R² score", expanded=False):
    st.markdown("""
| R² | Meaning |
|---|---|
| > 0.85 | Excellent — model explains most price variation |
| 0.70 – 0.85 | Good — reliable for directional decisions |
| 0.50 – 0.70 | Moderate — use as one input among many |
| < 0.50 | Weak — treat predictions as rough estimates |
| < 0 | Model performs worse than predicting the average price |

**Why location split improves R²:** Each tier now trains on a homogeneous price band.
OCR properties cluster around S$700K–S$1.5M, RCR around S$1.2M–S$3M.
Narrower price bands → smaller variance → model errors are proportionally smaller → higher R².
    """)

# ── Data insights ──────────────────────────────────────────────────────────────
with st.expander("Data insights & feature importance", expanded=False):
    df   = st.session_state.df
    tab1, tab2, tab3 = st.tabs(["Sample data", "Price by tier & region", "Feature importance"])

    with tab1:
        st.dataframe(
            df[["Project Name", "tier", "region_cluster", "Property Type",
                "tenure_label", "area_sqft", "floor_mid", "sale_year", "price"]]
            .rename(columns={"price": "Transacted Price ($)", "tenure_label": "Tenure",
                             "tier": "Tier", "region_cluster": "Region"})
            .head(100),
            use_container_width=True,
        )

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # Price distribution by tier
        for tier, color in TIER_COLORS.items():
            sub = df[df["tier"] == tier]["price"] / 1e6
            if len(sub) > 0:
                axes[0].hist(sub, bins=40, alpha=0.6, color=color,
                             label=TIER_LABELS[tier], edgecolor="white")
        axes[0].set_xlabel("Price (S$ M)")
        axes[0].set_title("Price Distribution by Tier")
        axes[0].legend(fontsize=7)

        # Median price by region cluster
        region_med = df.groupby("region_cluster")["price"].median().sort_values() / 1e6
        region_med.plot(kind="barh", ax=axes[1], color="#2563EB")
        axes[1].set_xlabel("Median Price (S$ M)")
        axes[1].set_title("Median Price by Region Cluster")

        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        models_ss = st.session_state.models
        n_models  = len(models_ss)
        if n_models == 0:
            st.info("No models trained yet.")
        else:
            fig2, axes2 = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
            if n_models == 1:
                axes2 = [axes2]
            feat_labels = {
                "area_sqft": "Area (SQFT)", "floor_mid": "Floor Level",
                "lease_remaining": "Lease Remaining", "tenure_enc": "Tenure",
                "proptype_enc": "Property Type", "area_type_enc": "Strata/Land",
                "sale_year": "Sale Year", "sale_month": "Sale Month",
                "district_enc": "Postal District", "project_enc": "Project Name",
                "region_cluster_enc": "Region Cluster",
            }
            for ax, (tier, model) in zip(axes2, models_ss.items()):
                feats = FEATURE_COLS[tier]
                imp   = pd.Series(model.feature_importances_, index=feats).sort_values()
                imp.index = [feat_labels.get(i, i) for i in imp.index]
                imp.plot(kind="barh", ax=ax, color=TIER_COLORS[tier])
                ax.set_title(f"{TIER_LABELS[tier]}")
            plt.tight_layout()
            st.pyplot(fig2)

# ── Prediction ─────────────────────────────────────────────────────────────────
st.header("Predict Resale Price")

models       = st.session_state.models
project_meta = st.session_state.project_meta
df           = st.session_state.df

# Step 1 — select tier
selected_tier = st.radio(
    "Select property tier",
    options=list(TIERS.keys()),
    format_func=lambda t: TIER_LABELS[t],
    horizontal=True,
)

if selected_tier not in models:
    st.warning(f"No model for {TIER_LABELS[selected_tier]} — insufficient training data.")
    st.stop()

tier_projects = sorted([p for p, m in project_meta.items()
                        if m.get("tier") == selected_tier])
st.caption(f"{len(tier_projects):,} projects in this tier.")

col1, col2, col3 = st.columns(3)

with col1:
    use_project = st.toggle("Select by project name", value=True)
    if use_project and tier_projects:
        selected_project = st.selectbox("Project Name", ["(manual entry)"] + tier_projects)
    else:
        selected_project = "(manual entry)"

    meta = project_meta.get(selected_project, {}) if selected_project != "(manual entry)" else {}

    area_sqft  = st.number_input("Area (SQFT)", min_value=100.0, max_value=50000.0,
                                  value=800.0, step=50.0)
    last_year  = int(df["sale_year"].max())
    pred_year  = st.number_input("Target Year", min_value=last_year,
                                  max_value=last_year + MAX_PRED_YEARS,
                                  value=last_year + 1, step=1)
    pred_month = st.selectbox("Month", list(range(1, 13)),
                               format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"))

with col2:
    # Property type filtered to tier
    if selected_tier == "Ultra":
        pt_options = sorted(LANDED_TYPES)
    elif selected_tier == "CCR":
        pt_options = ["Condominium", "Apartment", "Executive Condominium"]
    else:
        pt_options = ["Apartment", "Condominium", "Executive Condominium"]

    pt_default = meta.get("proptype", pt_options[0])
    pt_idx     = pt_options.index(pt_default) if pt_default in pt_options else 0
    prop_type  = st.selectbox("Property Type", pt_options, index=pt_idx)

    if selected_tier != "Ultra":
        floor_opts = ["B1 to B5","01 to 05","06 to 10","11 to 15","16 to 20",
                      "21 to 25","26 to 30","31 to 35","36 to 40","41 to 45",
                      "46 to 50","51 to 55","56 to 60"]
        floor_sel = st.selectbox("Floor Level", floor_opts, index=2)
    else:
        floor_sel = "01 to 05"
        st.info("Floor level not applicable for landed properties.")

    dist_raw     = meta.get("district", 15)
    dist_default = int(dist_raw) if not pd.isna(dist_raw) else 15
    district_no  = st.number_input("Postal District (1–28)", min_value=1, max_value=28,
                                    value=dist_default, step=1)

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

predict_btn = st.button("Predict Price", type="primary")

# ── Prediction logic ───────────────────────────────────────────────────────────
if predict_btn:
    model        = models[selected_tier]
    encoders     = st.session_state.encoders[selected_tier]
    growth       = st.session_state.growth_rates
    mape_seg_map = st.session_state.mape_by_seg.get(selected_tier, {})
    feats        = FEATURE_COLS[selected_tier]
    segment      = TIER_TO_SEGMENT[selected_tier]
    color        = TIER_COLORS[selected_tier]

    proj_enc    = apply_target_encode(
        pd.Series([selected_project]), *encoders["project"]
    ).iloc[0] if selected_project != "(manual entry)" else encoders["project"][1]

    dist_enc    = apply_target_encode(
        pd.Series([float(district_no)]), *encoders["district"]
    ).iloc[0]

    region      = get_region_cluster(district_no)
    region_enc  = apply_target_encode(
        pd.Series([region]), *encoders["region_cluster"]
    ).iloc[0]

    floor_mid   = parse_floor(floor_sel)
    annual_rate = growth.get(segment, 0.03)
    years_ahead = pred_year - last_year

    def make_row(year):
        lr  = compute_lease_remaining(tenure_sel, float(lease_yr), float(year))
        row = {
            "area_sqft":          area_sqft,
            "floor_mid":          floor_mid,
            "lease_remaining":    lr,
            "tenure_enc":         float(TENURE_ORDER.get(tenure_sel, 0)),
            "proptype_enc":       float(PROPTYPE_ORDER.get(prop_type, 0)),
            "area_type_enc":      1.0 if selected_tier == "Ultra" else 0.0,
            "sale_year":          float(last_year),
            "sale_month":         float(pred_month),
            "district_enc":       dist_enc,
            "region_cluster_enc": region_enc,
            "project_enc":        proj_enc,
        }
        return {k: row[k] for k in feats}

    base_price  = float(model.predict(pd.DataFrame([make_row(last_year)]))[0])
    predicted   = base_price * ((1 + annual_rate) ** years_ahead)
    base_mape   = mape_seg_map.get(segment, metrics_dict[selected_tier]["mape"]) / 100
    total_mape  = base_mape * (1 + 0.20 * years_ahead)
    lower       = predicted * (1 - total_mape)
    upper       = predicted * (1 + total_mape)
    psf         = predicted / area_sqft

    st.divider()
    st.subheader(f"{TIER_LABELS[selected_tier]} — Predicted Price for {pred_year}")

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Central Estimate", f"S$ {predicted:,.0f}")
    rc2.metric("Lower Bound",      f"S$ {lower:,.0f}")
    rc3.metric("Upper Bound",      f"S$ {upper:,.0f}")
    rc4.metric("Price PSF",        f"S$ {psf:,.0f}")

    st.caption(
        f"Region: **{region}** · District: **{int(district_no)}** · "
        f"Base MAPE: {base_mape*100:.1f}% · "
        f"Growth rate ({segment}): {annual_rate*100:.1f}% p.a."
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
    ax.fill_between(chart_years, lowers, uppers, alpha=0.15, color=color,
                    label="Uncertainty band")
    ax.plot(chart_years, central, marker="o", color=color,
            linewidth=2, label="Central estimate")
    ax.axvline(pred_year, color="red",  linestyle="--", alpha=0.7, label=f"Target: {pred_year}")
    ax.axvline(last_year, color="gray", linestyle=":",  alpha=0.5, label=f"Last data: {last_year}")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"S${x:.2f}M"))
    ax.set_xlabel("Year")
    ax.set_ylabel("Predicted Price")
    proj_label = selected_project if selected_project != "(manual entry)" else prop_type
    ax.set_title(
        f"{proj_label} · {region} · {area_sqft:.0f} sqft"
        + (f" · {floor_sel}" if selected_tier != "Ultra" else "")
        + f" · {tenure_sel}"
    )
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.warning(
        "⚠ Predictions beyond the last data year are projections, not guarantees. "
        "SG property prices are sensitive to government cooling measures, "
        "interest rates, and macroeconomic conditions."
    )
