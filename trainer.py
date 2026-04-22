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

        # Use all available history for predictive validation
        full_returns = returns
        full_volume = volume
        full_macro = macro.loc[full_returns.index].dropna()
        common_idx = full_returns.index.intersection(full_macro.index)
        full_returns = full_returns.loc[common_idx]
        full_volume = full_volume.loc[common_idx]
        full_macro = full_macro.loc[common_idx]

        # Compute crowding scores on the full history
        crowding_scores_full, _, _, _, _, _ = model.compute_crowding_score(
            full_returns, full_volume, full_macro
        )
        # Create a DataFrame of historical crowding scores (one column per ticker)
        crowding_history = pd.DataFrame(index=full_returns.index)
        for t in tickers:
            if t in crowding_scores_full.index:
                crowding_history[t] = crowding_scores_full[t]

        # Use recent window for current scores
        recent_returns = full_returns.iloc[-config.MIN_OBSERVATIONS:]
        recent_volume = full_volume.iloc[-config.MIN_OBSERVATIONS:]
        recent_macro = full_macro.loc[recent_returns.index]
        crowding_scores, cis, crowd_mom, mom_raw, vol_raw, macro_raw = model.compute_crowding_score(
            recent_returns, recent_volume, recent_macro
        )
        expected_returns = model.compute_expected_return(recent_returns)
        adj_returns, alpha, penalty = model.compute_crowding_adjusted_return(expected_returns, crowding_scores)

        # Predictive validation using full history
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

    output_payload = {
        "run_date": config.TODAY,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_") and k.isupper()},
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_crowding()
