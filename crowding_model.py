"""
Mean-Field Crowding Model with advanced features.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import resample

class CrowdingModel:
    def __init__(self, momentum_window=21, volume_window=63, macro_corr_window=126,
                 n_bootstrap=50, use_kalman=True, use_cross_rank=True, use_momentum=True,
                 use_vol_weighted=True, use_regime=True, use_decomp=True, use_predictive=True,
                 predictive_lookforward=5):
        self.momentum_window = momentum_window
        self.volume_window = volume_window
        self.macro_corr_window = macro_corr_window
        self.n_bootstrap = n_bootstrap
        self.use_kalman = use_kalman
        self.use_cross_rank = use_cross_rank
        self.use_momentum = use_momentum
        self.use_vol_weighted = use_vol_weighted
        self.use_regime = use_regime
        self.use_decomp = use_decomp
        self.use_predictive = use_predictive
        self.predictive_lookforward = predictive_lookforward

    # --------------------------------------------------------------------------
    # 1. Dynamic macro sensitivity via rolling linear regression
    # --------------------------------------------------------------------------
    def _dynamic_macro_sensitivity(self, ret: np.ndarray, macro: np.ndarray) -> float:
        if len(ret) < 50 or macro.shape[1] == 0:
            return 0.5
        vix = macro[:, 0]
        window = min(50, len(ret) // 2)
        if window < 10:
            return 0.5
        recent_ret = ret[-window:]
        recent_vix = vix[-window:]
        beta = np.cov(recent_ret, recent_vix)[0, 1] / (np.var(recent_vix) + 1e-6)
        return abs(beta)

    # --------------------------------------------------------------------------
    # 2. Base crowding components
    # --------------------------------------------------------------------------
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

    def _macro_score(self, ret: np.ndarray, macro: np.ndarray, vol: np.ndarray = None) -> float:
        if len(ret) < self.macro_corr_window or macro.shape[1] == 0:
            return 0.5
        if self.use_kalman:
            base = self._dynamic_macro_sensitivity(ret[-self.macro_corr_window:], macro[-self.macro_corr_window:])
        else:
            vix = macro[:, 0]
            corr = np.corrcoef(ret[-self.macro_corr_window:], vix[-self.macro_corr_window:])[0, 1]
            base = abs(corr) if not np.isnan(corr) else 0.5
        if self.use_vol_weighted and vol is not None:
            vol_ratio = self._volume_score(vol)
            base = base * (0.5 + 0.5 * vol_ratio)
        return base

    # --------------------------------------------------------------------------
    # 3. Main crowding score with bootstrapping
    # --------------------------------------------------------------------------
    def compute_crowding_score(self, returns: pd.DataFrame, volume: pd.DataFrame,
                               macro: pd.DataFrame) -> tuple:
        scores = {}
        cis = {}
        crowding_momentum = {}
        macro_raw = {}
        volume_raw = {}
        momentum_raw = {}
        common_idx = returns.index.intersection(macro.index)
        returns = returns.loc[common_idx]
        volume = volume.loc[common_idx]
        macro = macro.loc[common_idx]

        for ticker in returns.columns:
            if ticker not in returns.columns:
                continue
            ret = returns[ticker].values
            vol = volume[ticker].values if ticker in volume.columns else np.ones_like(ret)
            if len(ret) < self.macro_corr_window:
                continue

            boot_scores = []
            boot_mom = []
            boot_vol = []
            boot_macro = []
            for _ in range(self.n_bootstrap):
                idx = resample(range(len(ret)), n_samples=len(ret), random_state=np.random.randint(10000))
                ret_boot = ret[idx]
                vol_boot = vol[idx]
                macro_boot = macro.iloc[idx].values

                mom = self._momentum_score(ret_boot)
                vol_score = self._volume_score(vol_boot)
                macro_score = self._macro_score(ret_boot, macro_boot, vol_boot)
                boot_mom.append(mom)
                boot_vol.append(vol_score)
                boot_macro.append(macro_score)
                boot_scores.append((mom + vol_score + macro_score) / 3.0)

            crowd_score = np.mean(boot_scores)
            ci_lower = np.percentile(boot_scores, 2.5)
            ci_upper = np.percentile(boot_scores, 97.5)

            scores[ticker] = crowd_score
            cis[ticker] = {"lower": ci_lower, "upper": ci_upper}
            momentum_raw[ticker] = np.mean(boot_mom)
            volume_raw[ticker] = np.mean(boot_vol)
            macro_raw[ticker] = np.mean(boot_macro)

            # Crowding momentum
            if self.use_momentum and len(ret) >= self.momentum_window + 21:
                past_ret = ret[:-21]
                past_vol = vol[:-21] if len(vol) > 21 else vol
                past_macro = macro.iloc[:-21].values
                past_scores = []
                for _ in range(self.n_bootstrap // 2):
                    idx = resample(range(len(past_ret)), n_samples=len(past_ret), random_state=np.random.randint(10000))
                    ret_boot = past_ret[idx]
                    vol_boot = past_vol[idx]
                    macro_boot = past_macro[idx]
                    mom = self._momentum_score(ret_boot)
                    vol_score = self._volume_score(vol_boot)
                    macro_score = self._macro_score(ret_boot, macro_boot, vol_boot)
                    past_scores.append((mom + vol_score + macro_score) / 3.0)
                past_crowd = np.mean(past_scores) if past_scores else crowd_score
                crowding_momentum[ticker] = crowd_score - past_crowd
            else:
                crowding_momentum[ticker] = 0.0

        # Cross-sectional ranking
        if self.use_cross_rank:
            score_series = pd.Series(scores)
            rank_pct = score_series.rank(pct=True)
            for t in scores:
                scores[t] = rank_pct[t]

        # Regime-conditional adjustment
        if self.use_regime:
            vix_level = macro['VIX'].iloc[-1] if 'VIX' in macro.columns else 20
            if vix_level > 30:
                for t in scores:
                    scores[t] = min(scores[t] * 1.2, 1.0)
            elif vix_level < 15:
                for t in scores:
                    scores[t] = scores[t] * 0.8

        return pd.Series(scores), cis, pd.Series(crowding_momentum), pd.Series(momentum_raw), pd.Series(volume_raw), pd.Series(macro_raw)

    # --------------------------------------------------------------------------
    # 4. Expected return and decomposition
    # --------------------------------------------------------------------------
    def compute_expected_return(self, returns: pd.DataFrame) -> pd.Series:
        exp_ret = {}
        for ticker in returns.columns:
            ret = returns[ticker]
            if len(ret) >= 21:
                exp_ret[ticker] = ret.iloc[-21:].mean() * 252
            else:
                exp_ret[ticker] = 0.0
        return pd.Series(exp_ret)

    def compute_crowding_adjusted_return(self, expected_return: pd.Series,
                                         crowding_score: pd.Series) -> tuple:
        adj = expected_return * (1 - crowding_score)
        adj = adj.where(expected_return >= 0, expected_return * (1 - 0.5 * crowding_score))
        alpha = expected_return * (1 - crowding_score)
        penalty = expected_return - adj
        return adj, alpha, penalty

    # --------------------------------------------------------------------------
    # 5. Predictive validation (fixed)
    # --------------------------------------------------------------------------
    def predictive_validation(self, returns: pd.DataFrame, 
                              crowding_history: pd.DataFrame) -> pd.Series:
        """
        Correlation between historical crowding scores and future returns.
        crowding_history: DataFrame with dates as index and tickers as columns.
        """
        if not self.use_predictive:
            return pd.Series(index=crowding_history.columns, data=0.0)
        
        valid = {}
        common_idx = returns.index.intersection(crowding_history.index)
        returns = returns.loc[common_idx]
        crowding_history = crowding_history.loc[common_idx]

        for ticker in returns.columns:
            if ticker not in crowding_history.columns:
                continue
            ret = returns[ticker]
            crowd = crowding_history[ticker]
            if len(ret) < self.macro_corr_window + self.predictive_lookforward:
                valid[ticker] = 0.0
                continue
            fwd_ret = ret.shift(-self.predictive_lookforward).rolling(self.macro_corr_window).mean()
            corr = crowd.rolling(self.macro_corr_window).corr(fwd_ret).iloc[-1]
            valid[ticker] = corr if not np.isnan(corr) else 0.0
        return pd.Series(valid)
