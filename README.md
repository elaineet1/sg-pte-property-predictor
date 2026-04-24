# 🏢 SG Private Property Resale Price Predictor

A machine learning web app that predicts Singapore private property resale prices using XGBoost, trained on URA (Urban Redevelopment Authority) transaction data.

**Live demo:** [sg-pte-property-predictor.streamlit.app](https://sg-pte-property-predictor.streamlit.app)

---

## What This App Does

Upload your URA transaction CSV files, train a model in ~60 seconds, then predict what a private property unit is likely to resell for in any year up to 5 years ahead — with an uncertainty band showing the realistic price range.

---

## Features

- **4 separate ML models** — one per market tier for higher accuracy
- **Project name selector** — pick a project and all details auto-fill
- **Price trend chart** — central estimate + uncertainty band over time
- **Region-aware predictions** — uses Singapore's 6 geographic clusters
- **Honest uncertainty** — band widens the further into the future you predict
- **Model performance dashboard** — R², MAE, MAPE per tier

---

## The 4 Property Tiers

| Tier | Coverage | Typical Price Range |
|---|---|---|
| 🏢 Normal — OCR | Outside Central Region (Jurong, Punggol, Tampines, Woodlands) | S$700K – S$1.5M |
| 🏙️ Normal — RCR | Rest of Central Region (Queenstown, Bishan, Toa Payoh, Katong) | S$1.2M – S$3M |
| 💎 Luxury — CCR | Core Central Region (Orchard, Marina Bay, Sentosa, Bukit Timah) | S$2.5M – S$20M+ |
| 🏡 Ultra-luxury | Landed (Terrace, Semi-Detached, Detached — all regions) | S$3M – S$50M+ |

Each tier trains on its own homogeneous price band so the model learns tighter, more consistent pricing rules.

---

## Data Source

Data comes from **URA's Private Residential Property Transactions** portal — free to download.

**How to get the data:**
1. Go to [URA Property Market Information](https://eservice.ura.gov.sg/property-market-information/pmiResidentialTransactionSearch)
2. Select **Type of Sale → Resale**
3. Select a quarter (e.g. Q1 2020)
4. Click Search → Download CSV
5. Repeat for each quarter you want (recommend 2018–present for best results)

**Expected CSV columns:**

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

> Only **Resale** transactions are used for training. New Sale and Sub Sale rows are automatically excluded because developer pricing follows different dynamics from the open market.

---

## Feature Engineering

| Feature | Derived From | Why It Matters |
|---|---|---|
| `area_sqft` | Area (SQFT) | Strongest single price driver |
| `floor_mid` | Floor Level (`06 to 10` → `8`) | Higher floor = price premium |
| `lease_remaining` | Tenure + Sale Date | 99yr units depreciate as lease shortens |
| `tenure_enc` | Tenure string | Freehold > 999yr > 99yr premium |
| `proptype_enc` | Property Type | Detached > Semi-D > Terrace > Condo > Apt |
| `sale_year` | Sale Date | Captures market price trend over time |
| `sale_month` | Sale Date | Seasonal patterns in property market |
| `district_enc` | Postal District | Target-encoded — captures district price level |
| `region_cluster_enc` | Postal District → Region | Broader geographic signal (North/South/East/West) |
| `project_enc` | Project Name | Target-encoded — captures project brand premium |

**Features intentionally excluded:**
- `Unit Price ($ PSF)` — derived from price ÷ area, would let the model cheat
- `Area (SQM)` — duplicate of SQFT in different units
- `Nett Price` — mostly empty in URA data

---

## Model Architecture

```
Upload CSVs
    ↓
Filter: Resale transactions only
    ↓
Feature engineering (parse, derive, encode)
    ↓
Auto-classify into 4 tiers (OCR / RCR / CCR / Ultra)
    ↓
Per tier:
  ├── 80/15 train/test split
  ├── Build target encoders on train only (no leakage)
  ├── Train XGBoost (600 trees, depth 8, lr 0.04)
  └── Evaluate: R², MAE, MAPE on test set
    ↓
Predict:
  ├── XGBoost predicts base price at last known year
  ├── Apply segment-level annual growth rate for future years
  └── Uncertainty band = MAPE × (1 + 0.20 × years ahead)
```

**Why XGBoost?**
- Best-in-class for tabular/structured data
- Handles mixed numeric + categorical features natively
- Built-in feature importance
- Robust to outliers common in property transaction data

**Why target encoding for project name and district?**
- Thousands of unique project names — one-hot encoding would be impractical
- Target encoding replaces each project with its smoothed historical mean price
- Computed on training data only to prevent data leakage

---

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/elaineet1/sg-pte-property-predictor.git
cd sg-pte-property-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## How to Deploy (Streamlit Community Cloud — Free)

1. Fork this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **New app**
5. Fill in:
   - Repository: `your-username/sg-pte-property-predictor`
   - Branch: `main`
   - Main file: `app.py`
6. Click **Deploy**

Your app will be live at `https://your-username-sg-pte-property-predictor.streamlit.app`

> **Note:** The trained model is not stored between sessions on the free tier. Each new session requires re-uploading CSVs and retraining (~60 seconds).

---

## Project Structure

```
sg-pte-property-predictor/
├── app.py              # Streamlit UI — all screens, charts, predictions
├── preprocessing.py    # Data loading, feature engineering, encoders
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes CSV data and trained model binaries
└── README.md           # This file
```

---

## Accuracy & Limitations

### Current Model Performance (7 CSV files, ~72K transactions)

| Tier | R² | MAPE | Interpretation |
|---|---|---|---|
| OCR | 0.48 | 7.2% | Moderate — OCR covers large geographic area |
| RCR | 0.57 | 6.7% | Good — predictions within ~S$100K |
| CCR | 0.61 | 9.0% | Good — luxury pricing well captured |
| Ultra | — | — | Insufficient landed data in current files |

### How to Improve Accuracy

1. **Add more historical data** — download quarterly CSVs back to 2018 from URA (free). More years = model sees full market cycles
2. **Add MRT proximity** — distance to nearest MRT station is a top price driver in Singapore
3. **Add school proximity** — good schools within 1km add measurable premium

### Known Limitations

- Predictions beyond the last data year use a compound growth rate — **not a guarantee**
- Singapore property prices are heavily influenced by **government cooling measures** (ABSD, TDSR) that no model can predict
- **Ultra-luxury (Landed)** model requires more transaction data to train reliably
- The model predicts **market-rate resale prices** — individual unit condition, renovation quality, and negotiation are not captured

---

## Tech Stack

| Tool | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web app framework |
| [XGBoost](https://xgboost.readthedocs.io) | Machine learning model |
| [pandas](https://pandas.pydata.org) | Data processing |
| [scikit-learn](https://scikit-learn.org) | Train/test split, metrics |
| [matplotlib](https://matplotlib.org) | Charts |
| [URA Data](https://eservice.ura.gov.sg/property-market-information/pmiResidentialTransactionSearch) | Transaction data source |

---

## Disclaimer

This app is for **informational and educational purposes only**. Predictions are estimates based on historical transaction patterns and should not be used as the sole basis for any property purchase, sale, or financial decision. Always consult a licensed property agent and financial advisor before making property decisions.

---

*Built with Claude Code · Data from URA Singapore*
