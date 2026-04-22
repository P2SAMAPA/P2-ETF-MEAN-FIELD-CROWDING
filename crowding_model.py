"""
Mean-Field Crowding Model: computes crowding scores based on momentum, volume, and macro correlation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

class CrowdingModel:
    def __init__(self, momentum_window=21, volume_window=63, macro_corr_window=126):
        self.momentum_window = momentum_window
        self.volume_window = volume_window
        self.macro_corr_window = macro_corr_window

    def compute_crowding_score(self, returns: pd.DataFrame, volume: pd.DataFrame, macro: pd.DataFrame) -> pd.Series:
        """
        Compute a crowding score (0-1) for each ETF.
        Higher score = more crowded (overbought, high volume, high macro sensitivity).
        """
        scores = {}
        common_idx = returns.index.intersection(macro.index)
        returns = returns.loc[common_idx]
        volume = volume.loc[common_idx]
        macro = macro.loc[common_idx]

        for ticker in returns.columns:
            if ticker not in returns.columns:
                continue

            # 1. Momentum crowding: absolute z-score of recent return vs history
            ret = returns[ticker]
            if len(ret) < self.momentum_window:
                continue
            recent_ret = ret.iloc[-self.momentum_window:].mean() * 252
            hist_ret = ret.iloc[:-self.momentum_window].mean() * 252 if len(ret) > self.momentum_window else recent_ret
            hist_std = ret.iloc[:-self.momentum_window].std() * np.sqrt(252) if len(ret) > self.momentum_window else 0.2
            if hist_std > 0:
                mom_z = abs(recent_ret - hist_ret) / hist_std
            else:
                mom_z = 0.0
            mom_score = 2 * stats.norm.cdf(mom_z) - 1  # map to 0-1

            # 2. Volume crowding: relative volume vs moving average
            vol = volume[ticker]
            if len(vol) >= self.volume_window:
                recent_vol = vol.iloc[-5:].mean()
                avg_vol = vol.iloc[-self.volume_window:].mean()
                vol_ratio = recent_vol / (avg_vol + 1e-6)
                vol_score = min(vol_ratio / 3.0, 1.0)  # cap at 3x average
            else:
                vol_score = 0.5

            # 3. Macro crowding: correlation with VIX (proxy for macro sensitivity)
            if len(ret) >= self.macro_corr_window and 'VIX' in macro.columns:
                common = ret.index.intersection(macro.index)
                if len(common) >= self.macro_corr_window:
                    corr = ret.loc[common[-self.macro_corr_window:]].corr(macro.loc[common[-self.macro_corr_window:], 'VIX'])
                    macro_score = abs(corr) if not np.isnan(corr) else 0.5
                else:
                    macro_score = 0.5
            else:
                macro_score = 0.5

            # Combine scores (equal weights)
            crowd_score = (mom_score + vol_score + macro_score) / 3.0
            scores[ticker] = crowd_score

        return pd.Series(scores)

    def compute_expected_return(self, returns: pd.DataFrame) -> pd.Series:
        """Simple expected return: recent 21-day annualized."""
        exp_ret = {}
        for ticker in returns.columns:
            ret = returns[ticker]
            if len(ret) >= 21:
                exp_ret[ticker] = ret.iloc[-21:].mean() * 252
            else:
                exp_ret[ticker] = 0.0
        return pd.Series(exp_ret)

    def compute_crowding_adjusted_return(self, expected_return: pd.Series, crowding_score: pd.Series) -> pd.Series:
        """
        Adjust expected return by crowding penalty.
        Formula: adj_return = expected_return * (1 - crowding_score) * (1 + 0.5 * (expected_return < 0))
        Negative returns get less penalty (crowding on short side is less risky).
        """
        adj = expected_return * (1 - crowding_score)
        # Slight boost for negative expected returns (contrarian)
        adj = adj.where(expected_return >= 0, expected_return * (1 - 0.5 * crowding_score))
        return adj
