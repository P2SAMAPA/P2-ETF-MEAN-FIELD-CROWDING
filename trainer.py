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
        macro_corr_window=config.MACRO_CORR_WINDOW
    )

    all_results = {}
    top_picks = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        volume = data_manager.prepare_volume_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        recent_returns = returns.iloc[-config.MIN_OBSERVATIONS:]
        recent_volume = volume.iloc[-config.MIN_OBSERVATIONS:]
        recent_macro = macro.loc[recent_returns.index].dropna()
        common_idx = recent_returns.index.intersection(recent_macro.index)
        recent_returns = recent_returns.loc[common_idx]
        recent_volume = recent_volume.loc[common_idx]
        recent_macro = recent_macro.loc[common_idx]

        crowding_scores = model.compute_crowding_score(recent_returns, recent_volume, recent_macro)
        expected_returns = model.compute_expected_return(recent_returns)
        adj_returns = model.compute_crowding_adjusted_return(expected_returns, crowding_scores)

        universe_results = {}
        for ticker in tickers:
            if ticker in adj_returns.index:
                universe_results[ticker] = {
                    "ticker": ticker,
                    "expected_return_raw": expected_returns.get(ticker, 0.0),
                    "crowding_score": crowding_scores.get(ticker, 0.5),
                    "expected_return_adj": adj_returns.get(ticker, 0.0)
                }

        all_results[universe_name] = universe_results
        sorted_tickers = sorted(universe_results.items(),
                                key=lambda x: x[1]["expected_return_adj"], reverse=True)
        top_picks[universe_name] = [
            {"ticker": t, "expected_return_adj": d["expected_return_adj"],
             "crowding_score": d["crowding_score"]}
            for t, d in sorted_tickers[:3]
        ]

    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "momentum_window": config.MOMENTUM_WINDOW,
            "volume_window": config.VOLUME_WINDOW,
            "macro_corr_window": config.MACRO_CORR_WINDOW
        },
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_crowding()
