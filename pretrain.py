"""
pretrain.py — Run this locally to train models on your CSV files.
Saves model files that the app loads automatically on startup.

Usage:
    python pretrain.py --data_folder path/to/your/csv/folder

Example:
    python pretrain.py --data_folder C:/Users/elain/Downloads
"""

import argparse
import glob
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor

from preprocessing import (
    TIERS, FEATURE_COLS,
    load_and_preprocess, build_encoders, apply_encoders,
    compute_growth_rates,
)

MODELS_PATH   = "models.pkl"
ENCODERS_PATH = "encoders.pkl"
META_PATH     = "pretrained_meta.pkl"


def xgb_params():
    return dict(
        n_estimators=600, max_depth=8, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_alpha=0.05, reg_lambda=1.0,
        early_stopping_rounds=50,
        random_state=42, n_jobs=-1, verbosity=0,
    )


def train_tier(tier_df, tier):
    feats      = FEATURE_COLS[tier]
    target_enc = {"district_enc", "project_enc", "region_cluster_enc"}
    base_feats = [f for f in feats if f not in target_enc]
    tier_df    = tier_df.dropna(subset=base_feats + ["price"])

    if len(tier_df) < 50:
        print(f"  Skipping {tier} - only {len(tier_df)} rows (need 50+)")
        return None, None, None, None

    train_df, test_df = train_test_split(tier_df, test_size=0.15, random_state=42)
    encoders  = build_encoders(train_df)
    train_enc = apply_encoders(train_df, encoders)
    test_enc  = apply_encoders(test_df,  encoders)

    X_train = train_enc[feats]
    X_test  = test_enc[feats]

    # Log-transform price: compresses extreme values, improves R² for price regression
    y_train = np.log1p(train_enc["price"])
    y_test  = np.log1p(test_enc["price"])

    model = XGBRegressor(**xgb_params())
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=100)

    # Predict and reverse log-transform
    y_pred_log = model.predict(X_test)
    y_pred     = np.expm1(y_pred_log)
    y_true     = test_enc["price"].values

    metrics = {
        "r2":   r2_score(y_true, y_pred),
        "mae":  mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "rows": len(tier_df),
    }

    tmp = test_enc.copy()
    tmp["y_pred"] = y_pred
    tmp["y_true"] = y_true
    mape_seg = (
        tmp.groupby("Market Segment")
        .apply(lambda g: np.mean(np.abs((g["y_true"] - g["y_pred"]) / g["y_true"])) * 100)
        .to_dict()
    )
    return model, encoders, metrics, mape_seg


def main(data_folder):
    # Find all CSV files
    csv_paths = glob.glob(os.path.join(data_folder, "*.csv"))
    if not csv_paths:
        print(f"No CSV files found in: {data_folder}")
        return

    print(f"\nFound {len(csv_paths)} CSV files:")
    for p in csv_paths:
        print(f"  {os.path.basename(p)}")

    # Load and preprocess
    print("\nLoading and cleaning data (Resale only)...")
    df = load_and_preprocess(csv_paths)
    if df is None or df.empty:
        print("No valid Resale data found. Check your CSV files.")
        return

    print(f"Total resale transactions: {len(df):,}")
    print(f"Tier breakdown:\n{df['tier'].value_counts().to_string()}")

    # Train per tier
    models, encoders_dict, metrics_dict, mape_dict = {}, {}, {}, {}

    for tier in TIERS:
        tier_df = df[df["tier"] == tier].copy()
        n = len(tier_df)
        print(f"\nTraining {tier} ({n:,} rows)...")
        model, enc, met, mape_seg = train_tier(tier_df, tier)
        if model is None:
            continue
        models[tier]        = model
        encoders_dict[tier] = enc
        metrics_dict[tier]  = met
        mape_dict[tier]     = mape_seg
        print(f"  R²: {met['r2']:.3f}  MAE: S${met['mae']:,.0f}  MAPE: {met['mape']:.1f}%")

    # Compute growth rates and project metadata
    growth_rates = compute_growth_rates(df)

    project_meta = (
        df.groupby("Project Name")
        .agg(
            tier=("tier",              lambda x: x.mode()[0]),
            segment=("Market Segment", lambda x: x.mode()[0]),
            proptype=("Property Type", lambda x: x.mode()[0]),
            tenure=("tenure_label",    lambda x: x.mode()[0]),
            lease_start=("lease_start","median"),
            district=("district",      "median"),
            region=("region_cluster",  lambda x: x.mode()[0]),
        )
        .reset_index()
    )

    tier_counts = df["tier"].value_counts().to_dict()
    last_year   = int(df["sale_year"].max())

    # Save everything
    print(f"\nSaving models to {MODELS_PATH}...")
    with open(MODELS_PATH,   "wb") as f: pickle.dump(models,        f)
    with open(ENCODERS_PATH, "wb") as f: pickle.dump(encoders_dict, f)
    with open(META_PATH,     "wb") as f:
        pickle.dump({
            "metrics":      metrics_dict,
            "mape_by_seg":  mape_dict,
            "growth_rates": growth_rates,
            "project_meta": project_meta.set_index("Project Name").to_dict("index"),
            "tier_counts":  tier_counts,
            "last_year":    last_year,
        }, f)

    print("\nPre-training complete! Files saved:")
    print(f"  {MODELS_PATH}   ({os.path.getsize(MODELS_PATH)/1e6:.1f} MB)")
    print(f"  {ENCODERS_PATH} ({os.path.getsize(ENCODERS_PATH)/1e6:.1f} MB)")
    print(f"  {META_PATH}     ({os.path.getsize(META_PATH)/1e6:.1f} MB)")
    print("\nNext step: upload these 3 files to your Hugging Face Space.")
    print("The app will load them automatically — no training needed by users.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train SG property price models")
    parser.add_argument(
        "--data_folder",
        type=str,
        default=r"C:\Users\elain\Downloads",
        help="Folder containing URA transaction CSV files",
    )
    args = parser.parse_args()
    main(args.data_folder)
