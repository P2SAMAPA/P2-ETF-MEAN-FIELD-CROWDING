"""
Mean-Field Crowding Model: computes crowding scores with bootstrap confidence intervals.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

class CrowdingModel:
    def __init__(self, momentum_window=21, volume_window=63, macro_corr_window=126, n_bootstrap=50):
        self.momentum_window = momentum_window
        self.volume_window = volume_window
        self.macro_corr_window = macro_corr_window
        self.n_bootstrap = n_bootstrap

    def compute_crowding_score(self, returns: pd.DataFrame, volume: pd.DataFrame, macro: pd.DataFrame) -> pd.Series:
        """
        Compute a crowding score (0-1) for each ETF with bootstrap confidence intervals.
        """
        scores = {}
        cis = {}
        common_idx = returns.index.intersection(macro.index)
        returns = returns.loc[common_idx]
        volume = volume.loc[common_idx]
        macro = macro.loc[common_idx]

        for ticker in returns.columns:
            if ticker not in returns.columns:
                continue

            ret = returns[ticker]
            vol = volume[ticker]
            if len(ret) < self.macro_corr_window:
                continue

            # Bootstrap to get confidence intervals
            boot_scores = []
            for _ in range(self.n_bootstrap):
                # Resample with replacement
                idx = resample(range(len(ret)), n_samples=len(ret), random_state=np.random.randint(10000))
                ret_boot = ret.iloc[idx].values
                vol_boot = vol.iloc[idx].values
                macro_boot = macro.iloc[idx].values

                # Recompute scores on bootstrap sample
                mom_score = self._momentum_score(ret_boot)
                vol_score = self._volume_score(vol_boot)
                macro_score = self._macro_score(ret_boot, macro_boot)
                boot_scores.append((mom_score + vol_score + macro_score) / 3.0)

            crowd_score = np.mean(boot_scores)
            ci_lower = np.percentile(boot_scores, 2.5)
            ci_upper = np.percentile(boot_scores, 97.5)

            scores[ticker] = crowd_score
            cis[ticker] = {"lower": ci_lower, "upper": ci_upper}

        return pd.Series(scores), cis

    def _momentum_score(self, ret: np.ndarray) -> float:
        if len(ret) < self.momentum_window:
            return 0.5
        recent = ret[-self.momentum_window:].mean() * 252
        hist = ret[:-self.momentum_window].mean() * 252 if len(ret) > self.momentum_window else recent
        hist_std = ret[:-self.momentum_window].std() * np.sqrt(252) if len(ret) > self.momentum_window else 0.2
        if hist_std > 0:
            mom_z = abs(recent - hist) / hist_std
        else:
            mom_z = 0.0
        return 2 * stats.norm.cdf(mom_z) - 1

    def _volume_score(self, vol: np.ndarray) -> float:
        if len(vol) < self.volume_window:
            return 0.5
        recent = vol[-5:].mean()
        avg = vol[-self.volume_window:].mean()
        ratio = recent / (avg + 1e-6)
        return min(ratio / 3.0, 1.0)

    def _macro_score(self, ret: np.ndarray, macro: np.ndarray) -> float:
        if len(ret) < self.macro_corr_window or macro.shape[1] == 0:
            return 0.5
        # Use VIX column (assumed first macro column)
        vix = macro[:, 0]
        corr = np.corrcoef(ret[-self.macro_corr_window:], vix[-self.macro_corr_window:])[0, 1]
        return abs(corr) if not np.isnan(corr) else 0.5

    def compute_expected_return(self, returns: pd.DataFrame) -> pd.Series:
        exp_ret = {}
        for ticker in returns.columns:
            ret = returns[ticker]
            if len(ret) >= 21:
                exp_ret[ticker] = ret.iloc[-21:].mean() * 252
            else:
                exp_ret[ticker] = 0.0
        return pd.Series(exp_ret)

    def compute_crowding_adjusted_return(self, expected_return: pd.Series, crowding_score: pd.Series) -> pd.Series:
        adj = expected_return * (1 - crowding_score)
        adj = adj.where(expected_return >= 0, expected_return * (1 - 0.5 * crowding_score))
        return adj
