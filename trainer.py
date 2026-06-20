"""
Main training script for Mean-Field Crowding engine.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from crowding_model import CrowdingModel
import push_results

def run_crowding():
    print(f"=== P2-ETF-MEAN-FIELD-CROWDING Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    macro = data_manager.prepare_macro_features(df_master)

    model = CrowdingModel(
        momentum_window=config.MOMENTUM_WINDOW,
        volume_window=config.VOLUME_WINDOW,
        macro_corr_window=config.MACRO_CORR_WINDOW,
        n_bootstrap=config.N_BOOTSTRAP,
        use_kalman=config.USE_KALMAN_MACRO,
        use_cross_rank=config.USE_CROSS_SECTIONAL_RANK,
        use_momentum=config.USE_CROWDING_MOMENTUM,
        use_vol_weighted=config.USE_VOLUME_WEIGHTED_MACRO,
        use_regime=config.USE_REGIME_THRESHOLDS,
        use_decomp=config.USE_RETURN_DECOMP,
        use_predictive=config.USE_PREDICTIVE_VALIDATION,
        predictive_lookforward=config.PREDICTIVE_LOOKFORWARD
    )

    all_results = {}
    top_picks = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        volume = data_manager.prepare_volume_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        # Align data
        full_returns = returns
        full_volume = volume
        full_macro = macro.loc[full_returns.index].dropna()
        common_idx = full_returns.index.intersection(full_macro.index)
        full_returns = full_returns.loc[common_idx]
        full_volume = full_volume.loc[common_idx]
        full_macro = full_macro.loc[common_idx]

        # -----------------------------------------------------------------
        # FIX: Build a TRUE time-series crowding history for predictive validation
        # -----------------------------------------------------------------
        print("  Computing rolling crowding history for predictive validation...")
        crowding_history = pd.DataFrame(index=full_returns.index, columns=tickers, dtype=float)
        step = 5  # Calculate every 5 days to keep runtime reasonable
        min_start = config.MACRO_CORR_WINDOW + 21  # Need enough data for the model internally
        
        for i in range(min_start, len(full_returns), step):
            window_returns = full_returns.iloc[i - min_start : i]
            window_volume = full_volume.iloc[i - min_start : i]
            window_macro = full_macro.iloc[i - min_start : i]
            
            scores, _, _, _, _, _ = model.compute_crowding_score(
                window_returns, window_volume, window_macro
            )
            # Record the score at the current timestamp
            current_date = full_returns.index[i-1]
            for t in tickers:
                if t in scores.index:
                    crowding_history.at[current_date, t] = scores[t]
                    
        # Forward fill the gaps created by stepping
        crowding_history = crowding_history.ffill()

        # -----------------------------------------------------------------
        # Use recent window for CURRENT scores
        # -----------------------------------------------------------------
        recent_returns = full_returns.iloc[-config.MIN_OBSERVATIONS:]
        recent_volume = full_volume.iloc[-config.MIN_OBSERVATIONS:]
        recent_macro = full_macro.loc[recent_returns.index]
        
        crowding_scores, cis, crowd_mom, mom_raw, vol_raw, macro_raw = model.compute_crowding_score(
            recent_returns, recent_volume, recent_macro
        )
        expected_returns = model.compute_expected_return(recent_returns)
        adj_returns, alpha, penalty = model.compute_crowding_adjusted_return(expected_returns, crowding_scores)

        # Predictive validation using the TRUE rolling history
        predictive_valid = model.predictive_validation(full_returns, crowding_history)

        universe_results = {}
        for ticker in tickers:
            if ticker in adj_returns.index:
                universe_results[ticker] = {
                    "ticker": ticker,
                    "expected_return_raw": expected_returns.get(ticker, 0.0),
                    "crowding_score": crowding_scores.get(ticker, 0.5),
                    "crowding_ci_lower": cis.get(ticker, {}).get("lower", 0.5),
                    "crowding_ci_upper": cis.get(ticker, {}).get("upper", 0.5),
                    "crowding_momentum": crowd_mom.get(ticker, 0.0),
                    "momentum_raw": mom_raw.get(ticker, 0.5),
                    "volume_raw": vol_raw.get(ticker, 0.5),
                    "macro_raw": macro_raw.get(ticker, 0.5),
                    "expected_return_adj": adj_returns.get(ticker, 0.0),
                    "alpha": alpha.get(ticker, 0.0),
                    "crowding_penalty": penalty.get(ticker, 0.0),
                    "predictive_validity": predictive_valid.get(ticker, 0.0)
                }

        all_results[universe_name] = universe_results
        sorted_tickers = sorted(universe_results.items(),
                                key=lambda x: x[1]["expected_return_adj"], reverse=True)
        top_picks[universe_name] = [
            {k: v for k, v in d.items() if k != 'ticker'} | {"ticker": t}
            for t, d in sorted_tickers[:3]
        ]

    # Build config summary without sensitive keys
    config_summary = {}
    for k, v in config.__dict__.items():
        if not k.startswith("_") and k.isupper() and k not in ["HF_TOKEN"]:
            config_summary[k] = v

    output_payload = {
        "run_date": config.TODAY,
        "config": config_summary,
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_crowding()
