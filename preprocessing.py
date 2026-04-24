import pandas as pd
import numpy as np
import re
import os

CURRENT_YEAR = 2026

# ── Tier definitions ───────────────────────────────────────────────────────────
TIERS = {
    "Normal":      "Normal (OCR / RCR Condo & Apartment)",
    "Luxury":      "Luxury (CCR Condo)",
    "Ultra":       "Ultra-luxury (Landed)",
}

LANDED_TYPES = {"Terrace House", "Semi-Detached House", "Detached House"}

def classify_tier(property_type, market_segment):
    if property_type in LANDED_TYPES:
        return "Ultra"
    if market_segment == "Core Central Region":
        return "Luxury"
    return "Normal"

# Features used per tier (landed has no floor level)
FEATURE_COLS = {
    "Normal": ["area_sqft", "floor_mid", "lease_remaining", "tenure_enc",
               "segment_enc", "proptype_enc", "area_type_enc",
               "sale_year", "sale_month", "district_enc", "project_enc"],
    "Luxury": ["area_sqft", "floor_mid", "lease_remaining", "tenure_enc",
               "segment_enc", "proptype_enc", "area_type_enc",
               "sale_year", "sale_month", "district_enc", "project_enc"],
    "Ultra":  ["area_sqft", "lease_remaining", "tenure_enc",
               "segment_enc", "proptype_enc",
               "sale_year", "sale_month", "district_enc", "project_enc"],
}

# ── Parsers ────────────────────────────────────────────────────────────────────

def parse_price(val):
    try:
        return float(str(val).replace(",", "").replace("$", "").strip())
    except Exception:
        return np.nan


def parse_area(val):
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return np.nan


def parse_sale_date(val):
    """'Apr-26' → (2026, 4)"""
    try:
        dt = pd.to_datetime(str(val), format="%b-%y")
        return dt.year, dt.month
    except Exception:
        return np.nan, np.nan


def parse_tenure(val):
    """Returns (tenure_label, lease_start_year)"""
    val = str(val).strip()
    if val.lower() == "freehold":
        return "Freehold", np.nan
    m = re.match(r"(\d+)\s+yrs?\s+lease\s+commencing\s+from\s+(\d{4})", val, re.I)
    if m:
        yrs, start = int(m.group(1)), int(m.group(2))
        label = "999yr" if yrs >= 900 else "99yr"
        return label, float(start)
    m2 = re.match(r"(\d+)", val)
    if m2:
        return f"{m2.group(1)}yr", np.nan
    return "Other", np.nan


def parse_floor(val):
    """'06 to 10' → 8.0"""
    nums = re.findall(r"\d+", str(val))
    if len(nums) >= 2:
        return (int(nums[0]) + int(nums[1])) / 2.0
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def compute_lease_remaining(tenure_label, lease_start, ref_year):
    if tenure_label == "Freehold":
        return 999.0
    if tenure_label == "999yr":
        total = 999
    elif tenure_label == "99yr":
        total = 99
    else:
        m = re.match(r"(\d+)", str(tenure_label))
        total = int(m.group(1)) if m else 99
    if pd.isna(lease_start) or pd.isna(ref_year):
        return float(total) / 2
    return max(0.0, lease_start + total - ref_year)

# ── Encodings ──────────────────────────────────────────────────────────────────

SEGMENT_ORDER = {
    "Core Central Region":    3,
    "Rest of Central Region": 2,
    "Outside Central Region": 1,
}

PROPTYPE_ORDER = {
    "Detached House":      6,
    "Semi-Detached House": 5,
    "Terrace House":       4,
    "Executive Condominium": 3,
    "Condominium":         2,
    "Apartment":           1,
}

TENURE_ORDER = {
    "Freehold": 4,
    "999yr":    3,
    "99yr":     1,
    "Other":    0,
}


def smoothed_target_encode(train_df, col, target, smoothing=20):
    """
    Smoothed target encoding computed on training data only.
    Rare categories are pulled toward the global mean.
    """
    global_mean = train_df[target].mean()
    stats  = train_df.groupby(col)[target].agg(["mean", "count"])
    weight = stats["count"] / (stats["count"] + smoothing)
    encoded = weight * stats["mean"] + (1 - weight) * global_mean
    return encoded.to_dict(), global_mean


def apply_target_encode(series, mapping, global_mean):
    return series.map(mapping).fillna(global_mean)

# ── Main pipeline ──────────────────────────────────────────────────────────────

def load_and_preprocess(paths):
    """
    Load CSVs, filter to Resale only, engineer all features.
    Returns a single cleaned DataFrame with a 'tier' column.
    """
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"Warning: could not read {os.path.basename(p)}: {e}")
    if not frames:
        return None

    raw = pd.concat(frames, ignore_index=True)

    # Filter: Resale only
    raw = raw[raw["Type of Sale"].str.strip() == "Resale"].copy()

    # Parse target
    raw["price"] = raw["Transacted Price ($)"].apply(parse_price)

    # Parse numeric features
    raw["area_sqft"] = raw["Area (SQFT)"].apply(parse_area)

    # Parse sale date
    raw[["sale_year", "sale_month"]] = raw["Sale Date"].apply(
        lambda v: pd.Series(parse_sale_date(v))
    )

    # Parse tenure
    tenure_parsed      = raw["Tenure"].apply(parse_tenure)
    raw["tenure_label"] = tenure_parsed.apply(lambda x: x[0])
    raw["lease_start"]  = tenure_parsed.apply(lambda x: x[1])

    # Derived features
    raw["floor_mid"] = raw["Floor Level"].apply(parse_floor)

    raw["lease_remaining"] = raw.apply(
        lambda r: compute_lease_remaining(
            r["tenure_label"],
            r["lease_start"],
            r["sale_year"] if not pd.isna(r["sale_year"]) else CURRENT_YEAR,
        ),
        axis=1,
    )

    # Ordinal encodings
    raw["tenure_enc"]    = raw["tenure_label"].map(TENURE_ORDER).fillna(0)
    raw["segment_enc"]   = raw["Market Segment"].map(SEGMENT_ORDER).fillna(0)
    raw["proptype_enc"]  = raw["Property Type"].map(PROPTYPE_ORDER).fillna(0)
    raw["area_type_enc"] = (raw["Type of Area"].str.strip() == "Land").astype(float)
    raw["district"]      = pd.to_numeric(raw["Postal District"], errors="coerce")

    # Tier classification
    raw["tier"] = raw.apply(
        lambda r: classify_tier(r["Property Type"], r["Market Segment"]), axis=1
    )

    # Drop rows missing essentials
    clean = raw.dropna(subset=["price", "area_sqft", "sale_year", "district"])
    clean = clean[clean["price"] > 0].copy()
    return clean


def build_encoders(train_df):
    """Build target encoders from training data only (no leakage)."""
    proj_map, proj_global = smoothed_target_encode(train_df, "Project Name", "price", smoothing=20)
    dist_map, dist_global = smoothed_target_encode(train_df, "district",     "price", smoothing=10)
    return {
        "project":  (proj_map,  proj_global),
        "district": (dist_map,  dist_global),
    }


def apply_encoders(df, encoders):
    df = df.copy()
    proj_map,  proj_global = encoders["project"]
    dist_map,  dist_global = encoders["district"]
    df["project_enc"]  = apply_target_encode(df["Project Name"], proj_map,  proj_global)
    df["district_enc"] = apply_target_encode(df["district"],     dist_map,  dist_global)
    return df


def compute_growth_rates(df):
    """Annual YoY price appreciation rate per market segment."""
    yearly = (
        df.groupby(["Market Segment", "sale_year"])["price"]
        .median()
        .reset_index()
    )
    rates = {}
    for seg in yearly["Market Segment"].unique():
        sub = yearly[yearly["Market Segment"] == seg].sort_values("sale_year")
        if len(sub) < 2:
            rates[seg] = 0.03
            continue
        pct = sub["price"].pct_change().dropna().clip(-0.20, 0.30)
        rates[seg] = float(pct.mean())
    return rates
