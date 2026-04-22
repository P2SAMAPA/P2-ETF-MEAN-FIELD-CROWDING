# P2-ETF-MEAN-FIELD-CROWDING

**Mean‑Field Crowding Proxy – Momentum, Volume & Macro Sensitivity for ETF Selection**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-MEAN-FIELD-CROWDING/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-MEAN-FIELD-CROWDING/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--mean--field--crowding--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-mean-field-crowding-results)

## Overview

`P2-ETF-MEAN-FIELD-CROWDING` estimates **crowding risk** for each ETF — a proxy for how many systematic market participants might be positioned similarly. The engine combines three signals:

- **Momentum crowding**: Extreme recent returns relative to history.
- **Volume crowding**: Elevated trading volume versus long‑term average.
- **Macro crowding**: High correlation with the VIX (sensitivity to macro shocks).

Crowding scores are computed with **bootstrap confidence intervals**, then used to **penalize expected returns**. The final ranking selects ETFs with the highest **crowding‑adjusted return** — assets with positive return potential and lower downside risk from positioning unwinds.

## Universe Coverage

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |

Data source: [`P2SAMAPA/fi-etf-macro-signal-master-data`](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data)

## Methodology

1. **Momentum Crowding**: Absolute z‑score of 21‑day annualized return vs. historical distribution.
2. **Volume Crowding**: Ratio of recent (5‑day) volume to 63‑day average (capped at 3×).
3. **Macro Crowding**: Absolute correlation of 126‑day returns with VIX.
4. **Bootstrap Confidence**: 50 resamples produce 95% confidence intervals for the composite crowding score.
5. **Crowding‑Adjusted Return**:  
   `Adj Return = Raw Return × (1 - Crowding Score)`  
   (Negative raw returns receive a smaller penalty.)
6. **Ranking**: Top 3 ETFs per universe by adjusted return.

## File Structure
P2-ETF-MEAN-FIELD-CROWDING/
├── config.py # Paths, universes, crowding parameters
├── data_manager.py # Data loading and preprocessing
├── crowding_model.py # Crowding score & bootstrap logic
├── trainer.py # Main orchestration script
├── push_results.py # Upload results to Hugging Face
├── streamlit_app.py # Interactive dashboard
├── us_calendar.py # U.S. market calendar utilities
├── requirements.txt # Python dependencies
├── .github/workflows/ # Scheduled GitHub Action
└── .streamlit/ # Streamlit theme

text

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MOMENTUM_WINDOW` | 21 | Days for momentum calculation |
| `VOLUME_WINDOW` | 63 | Baseline for relative volume |
| `MACRO_CORR_WINDOW` | 126 | Window for VIX correlation |
| `MIN_OBSERVATIONS` | 252 | Minimum data required |
| `N_BOOTSTRAP` | 50 | Resamples for confidence intervals |
