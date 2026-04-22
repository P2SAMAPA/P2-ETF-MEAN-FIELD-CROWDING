"""
Configuration for P2-ETF-MEAN-FIELD-CROWDING engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-mean-field-crowding-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Macro Features ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- Crowding Parameters ---
MOMENTUM_WINDOW = 21
VOLUME_WINDOW = 63
MACRO_CORR_WINDOW = 126
MIN_OBSERVATIONS = 252
N_BOOTSTRAP = 50
RANDOM_SEED = 42

# --- Advanced Features ---
USE_KALMAN_MACRO = True            # Time-varying macro sensitivity via Kalman filter
USE_CROSS_SECTIONAL_RANK = True    # Normalize scores within universe
USE_CROWDING_MOMENTUM = True       # Rate of change of crowding score
USE_VOLUME_WEIGHTED_MACRO = True   # Volume × VIX correlation
USE_REGIME_THRESHOLDS = True       # Adjust high-crowding threshold based on VIX level
USE_RETURN_DECOMP = True           # Decompose expected return into alpha and crowding penalty
USE_PREDICTIVE_VALIDATION = True   # Correlation of past crowding with future returns

# --- Predictive Validation ---
PREDICTIVE_LOOKFORWARD = 5         # Days ahead for validation

# --- Shrinking Windows ---
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
