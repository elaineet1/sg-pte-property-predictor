import pandas as pd
import numpy as np
import re
import os

CURRENT_YEAR = 2026

# ── Tier definitions ───────────────────────────────────────────────────────────
TIERS = {
    "OCR":   "Normal — Outside Central Region",
    "RCR":   "Normal — Rest of Central Region",
    "CCR":   "Luxury — Core Central Region",
    "Ultra": "Ultra-luxury — Landed",
}

LANDED_TYPES = {"Terrace House", "Semi-Detached House", "Detached House"}


def classify_tier(property_type, market_segment):
    if property_type in LANDED_TYPES:
        return "Ultra"
    if market_segment == "Core Central Region":
        return "CCR"
    if market_segment == "Rest of Central Region":
        return "RCR"
    return "OCR"


# ── District → broad region cluster ───────────────────────────────────────────
# Groups the 28 postal districts into 6 geographic regions.
# Within each tier, this gives the model a stronger location signal
# than raw district numbers (which are not ordinal).
DISTRICT_TO_REGION = {
    1: "City", 2: "City", 6: "City", 7: "City",
    8: "City", 9: "City", 10: "City", 11: "City",
    3: "South", 4: "South",
    5: "West", 21: "West", 22: "West", 23: "West", 24: "West",
    12: "Central", 13: "Central", 20: "Central",
    14: "East-Central", 15: "East-Central",
    16: "East", 17: "East", 18: "East",
    19: "North-East", 28: "North-East",
    25: "North", 26: "North", 27: "North",
}


def get_region_cluster(district):
    try:
        return DISTRICT_TO_REGION.get(int(district), "Other")
    except Exception:
        return "Other"


# ── Features per tier ──────────────────────────────────────────────────────────
# Ultra (landed) excludes floor_mid — not applicable for ground-level properties.
# All tiers include region_cluster_enc for stronger location signal.
FEATURE_COLS = {
    "OCR":   ["area_sqft", "floor_mid", "lease_remaining", "tenure_enc",
              "proptype_enc", "area_type_enc",
              "sale_year", "sale_month",
              "district_enc", "region_cluster_enc", "project_enc"],
    "RCR":   ["area_sqft", "floor_mid", "lease_remaining", "tenure_enc",
              "proptype_enc", "area_type_enc",
              "sale_year", "sale_month",
              "district_enc", "region_cluster_enc", "project_enc"],
    "CCR":   ["area_sqft", "floor_mid", "lease_remaining", "tenure_enc",
              "proptype_enc", "area_type_enc",
              "sale_year", "sale_month",
              "district_enc", "region_cluster_enc", "project_enc"],
    "Ultra": ["area_sqft", "lease_remaining", "tenure_enc",
              "proptype_enc",
              "sale_year", "sale_month",
              "district_enc", "region_cluster_enc", "project_enc"],
}

# Segment encoding kept for growth rate lookup (not a model feature after tier split)
SEGMENT_ORDER = {
    "Core Central Region":    3,
    "Rest of Central Region": 2,
    "Outside Central Region": 1,
}

PROPTYPE_ORDER = {
    "Detached House":        6,
    "Semi-Detached House":   5,
    "Terrace House":         4,
    "Executive Condominium": 3,
    "Condominium":           2,
    "Apartment":             1,
}

TENURE_ORDER = {
    "Freehold": 4,
    "999yr":    3,
    "99yr":     1,
    "Other":    0,
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
    try:
        dt = pd.to_datetime(str(val), format="%b-%y")
        return dt.year, dt.month
    except Exception:
        return np.nan, np.nan


def parse_tenure(val):
    val = str(val).strip()
    if val.lower() == "freehold":
        return "Freehold", np.nan
    m = re.match(r"(\d+)\s+yrs?\s+lease\s+commencing\s+from\s+(\d{4})", val, re.I)
    if m:
        yrs, start = int(m.group(1)), int(m.group(2))
        return ("999yr" if yrs >= 900 else "99yr"), float(start)
    m2 = re.match(r"(\d+)", val)
    if m2:
        return f"{m2.group(1)}yr", np.nan
    return "Other", np.nan


def parse_floor(val):
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

# ── Target encoding ────────────────────────────────────────────────────────────

def smoothed_target_encode(train_df, col, target, smoothing=20):
    """Smoothed mean encoding — computed on train only to prevent leakage."""
    global_mean = train_df[target].mean()
    stats  = train_df.groupby(col)[target].agg(["mean", "count"])
    weight = stats["count"] / (stats["count"] + smoothing)
    return (weight * stats["mean"] + (1 - weight) * global_mean).to_dict(), global_mean


def apply_target_encode(series, mapping, global_mean):
    return series.map(mapping).fillna(global_mean)

# ── Main pipeline ──────────────────────────────────────────────────────────────

def load_and_preprocess(paths):
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"Warning: could not read {os.path.basename(p)}: {e}")
    if not frames:
        return None

    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["Type of Sale"].str.strip() == "Resale"].copy()

    raw["price"]    = raw["Transacted Price ($)"].apply(parse_price)
    raw["area_sqft"] = raw["Area (SQFT)"].apply(parse_area)

    raw[["sale_year", "sale_month"]] = raw["Sale Date"].apply(
        lambda v: pd.Series(parse_sale_date(v))
    )

    tenure_parsed       = raw["Tenure"].apply(parse_tenure)
    raw["tenure_label"] = tenure_parsed.apply(lambda x: x[0])
    raw["lease_start"]  = tenure_parsed.apply(lambda x: x[1])

    raw["floor_mid"] = raw["Floor Level"].apply(parse_floor)

    raw["lease_remaining"] = raw.apply(
        lambda r: compute_lease_remaining(
            r["tenure_label"], r["lease_start"],
            r["sale_year"] if not pd.isna(r["sale_year"]) else CURRENT_YEAR,
        ), axis=1,
    )

    raw["tenure_enc"]    = raw["tenure_label"].map(TENURE_ORDER).fillna(0)
    raw["proptype_enc"]  = raw["Property Type"].map(PROPTYPE_ORDER).fillna(0)
    raw["area_type_enc"] = (raw["Type of Area"].str.strip() == "Land").astype(float)
    raw["district"]      = pd.to_numeric(raw["Postal District"], errors="coerce")
    raw["region_cluster"] = raw["district"].apply(get_region_cluster)

    raw["tier"] = raw.apply(
        lambda r: classify_tier(r["Property Type"], r["Market Segment"]), axis=1
    )

    clean = raw.dropna(subset=["price", "area_sqft", "sale_year", "district"])
    clean = clean[clean["price"] > 0].copy()
    return clean


def build_encoders(train_df):
    """All target encoders built from training data only."""
    proj_map,   proj_g   = smoothed_target_encode(train_df, "Project Name",    "price", 20)
    dist_map,   dist_g   = smoothed_target_encode(train_df, "district",         "price", 10)
    region_map, region_g = smoothed_target_encode(train_df, "region_cluster",   "price", 15)
    return {
        "project":        (proj_map,   proj_g),
        "district":       (dist_map,   dist_g),
        "region_cluster": (region_map, region_g),
    }


def apply_encoders(df, encoders):
    df = df.copy()
    df["project_enc"]       = apply_target_encode(df["Project Name"],   *encoders["project"])
    df["district_enc"]      = apply_target_encode(df["district"],        *encoders["district"])
    df["region_cluster_enc"] = apply_target_encode(df["region_cluster"], *encoders["region_cluster"])
    return df


def compute_growth_rates(df):
    """YoY median price appreciation per market segment."""
    yearly = (
        df.groupby(["Market Segment", "sale_year"])["price"]
        .median().reset_index()
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
