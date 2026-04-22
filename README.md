# P2-ETF-MEAN-FIELD-CROWDING

**Mean‑Field Crowding Proxy – Momentum, Volume & Macro Sensitivity for ETF Selection**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-MEAN-FIELD-CROWDING/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-MEAN-FIELD-CROWDING/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--mean--field--crowding--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-mean-field-crowding-results)

## Overview

`P2-ETF-MEAN-FIELD-CROWDING` estimates crowding risk for each ETF based on:
- **Momentum crowding**: Absolute z‑score of recent returns.
- **Volume crowding**: Relative volume vs. moving average.
- **Macro crowding**: Correlation with VIX (sensitivity to macro shocks).

The engine computes a crowding score (0–1) and adjusts expected returns downward for crowded assets. The least crowded ETFs with positive return potential are selected as top picks.

## Universe
FI/Commodities, Equity Sectors, Combined (23 ETFs)
