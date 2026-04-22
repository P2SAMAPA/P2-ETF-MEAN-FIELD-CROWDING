# P2-ETF-MEAN-FIELD-CROWDING

**Advanced Mean‑Field Crowding Proxy – Momentum, Volume, Dynamic Macro Sensitivity & Predictive Validation for ETF Selection**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-MEAN-FIELD-CROWDING/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-MEAN-FIELD-CROWDING/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--mean--field--crowding--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-mean-field-crowding-results)

## Overview

`P2-ETF-MEAN-FIELD-CROWDING` estimates **crowding risk** for each ETF — a proxy for how many systematic market participants might be positioned similarly. The engine combines:

- **Momentum crowding**: Extreme recent returns relative to history.
- **Volume crowding**: Elevated trading volume versus long‑term average.
- **Dynamic Macro crowding**: Time‑varying sensitivity to the VIX (via Kalman filter), optionally weighted by volume.

The crowding score is then refined with **advanced analytical layers** to produce a robust, tradeable signal.

## ✨ Advanced Features (All Enabled)

| Feature | Description |
|---------|-------------|
| **1. Kalman‑Filtered Macro Sensitivity** | Tracks time‑varying beta of ETF returns to the VIX, replacing a static correlation. |
| **2. Cross‑Sectional Ranking** | Crowding scores are normalized to percentiles within each universe, making comparisons fair. |
| **3. Crowding Momentum** | Measures the rate of change of the crowding score; a sharp rise often precedes reversals. |
| **4. Volume‑Weighted Macro Sensitivity** | Higher trading volume amplifies the macro sensitivity component, capturing institutional flow. |
| **5. Regime‑Conditional Thresholds** | “High crowding” thresholds adapt to the current VIX level (stricter in calm markets, looser in stress). |
| **6. Return Decomposition** | `Adj Return = Pure Alpha – Crowding Penalty`. The penalty is isolated, showing exactly how much crowding detracts from expected return. |
| **7. Predictive Validation** | Rolling correlation between past crowding scores and future returns – a self‑assessment of the signal's historical reliability. |

## Universe Coverage

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |

Data source: [`P2SAMAPA/fi-etf-macro-signal-master-data`](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data)

## Methodology (Summary)

1. **Base Crowding Components** (bootstrap aggregated):
   - Momentum score (z‑score of 21‑day return)
   - Volume score (5‑day vs. 63‑day average)
   - Macro score (Kalman‑filtered VIX sensitivity)
2. **Advanced Refinements**:
   - Normalize scores via cross‑sectional rank.
   - Compute crowding momentum (difference vs. 21 days ago).
   - Adjust thresholds based on current VIX.
3. **Return Adjustment**:
Adj Return = Raw Return × (1 – Crowding Score) (for positive raw return)
Adj Return = Raw Return × (1 – 0.5 × Crowding Score) (for negative raw return)

text
4. **Decomposition**:
- **Pure Alpha** = Raw Return × (1 – Crowding Score)
- **Crowding Penalty** = Raw Return – Adj Return
5. **Predictive Validation**: Rolling correlation between lagged crowding score and forward 5‑day return.

## File Structure
P2-ETF-MEAN-FIELD-CROWDING/
├── config.py # Paths, universes, advanced feature toggles
├── data_manager.py # Data loading and preprocessing
├── crowding_model.py # Full crowding logic (Kalman, bootstrap, decomposition, validation)
├── trainer.py # Orchestration script
├── push_results.py # Upload results to Hugging Face
├── streamlit_app.py # Interactive dashboard with hero cards and detailed tables
├── us_calendar.py # U.S. market calendar utilities
├── requirements.txt # Python dependencies
├── .github/workflows/ # Scheduled GitHub Action
└── .streamlit/ # Streamlit theme

text

## Configuration (Key Parameters)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MOMENTUM_WINDOW` | 21 | Days for momentum calculation |
| `VOLUME_WINDOW` | 63 | Baseline for relative volume |
| `MACRO_CORR_WINDOW` | 126 | Window for macro sensitivity |
| `N_BOOTSTRAP` | 50 | Bootstrap samples for confidence intervals |
| `USE_KALMAN_MACRO` | True | Enable Kalman‑filtered VIX sensitivity |
| `USE_CROSS_SECTIONAL_RANK` | True | Normalize scores to percentiles |
| `USE_CROWDING_MOMENTUM` | True | Compute rate of change of crowding |
| `USE_VOLUME_WEIGHTED_MACRO` | True | Scale macro sensitivity by volume |
| `USE_REGIME_THRESHOLDS` | True | Adjust thresholds based on VIX level |
| `USE_RETURN_DECOMP` | True | Decompose adjusted return into alpha and penalty |
| `USE_PREDICTIVE_VALIDATION` | True | Compute historical reliability metric |
| `PREDICTIVE_LOOKFORWARD` | 5 | Days ahead for validation |
