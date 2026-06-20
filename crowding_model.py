"""
Mean-Field Crowding Model v2.0
- Removed fatal time-shuffling bootstrap
- Added momentum exhaustion detection
- Added cross-sectional crowding (universe-level)
- Added volatility compression signal
- Adaptive regime conditioning
- Fixed asymmetric unwind penalty
"""

import numpy as np
import pandas as pd
from scipy import stats


class CrowdingModel:
    def __init__(self, momentum_window=21, volume_window=63, macro_corr_window=126,
                 n_bootstrap=50, use_kalman=True, use_cross_rank=True, use_momentum=True,
                 use_vol_weighted=True, use_regime=True, use_decomp=True, use_predictive=True,
                 predictive_lookforward=5):
        # Backward compat: accept old params, n_bootstrap is ignored (was broken)
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
    # 1. Cross-Sectional Crowding (Universe Level)
    # --------------------------------------------------------------------------
    def _cross_sectional_crowding(self, returns: pd.DataFrame, window: int = 63) -> float:
        """
        Average pairwise correlation of recent returns.
        High = everything moving together = crowded universe.
        """
        if len(returns) < window + 5:
            return 0.5
        
        recent = returns.iloc[-window:]
        corr_matrix = recent.corr()
        n = len(corr_matrix)
        # Upper triangle average (exclude diagonal)
        avg_corr = (corr_matrix.values.sum() - n) / (n * (n - 1))
        # Map [-1, 1] -> [0, 1]
        return np.clip((avg_corr + 1) / 2, 0, 1)

    # --------------------------------------------------------------------------
    # 2. Momentum Exhaustion (Deceleration)
    # --------------------------------------------------------------------------
    def _momentum_exhaustion(self, ret: np.ndarray) -> float:
        """
        Compare recent momentum to prior momentum.
        Returns 0-1 where 1 = fully exhausted (decelerating or reversing).
        """
        w = self.momentum_window
        if len(ret) < w * 3:
            return 0.5
        
        recent_mom = np.mean(ret[-w:]) * 252
        prior_mom = np.mean(ret[-w * 2:-w]) * 252
        
        # Check if momentum reversed direction
        same_direction = (recent_mom > 0 and prior_mom > 0) or (recent_mom < 0 and prior_mom < 0)
        
        if not same_direction:
            return 0.8  # Reversal = high exhaustion
        
        if abs(prior_mom) > 0.01:
            ratio = abs(recent_mom) / abs(prior_mom)
            # ratio < 1 = decelerating, ratio > 1 = accelerating
            exhaustion = 1.0 - np.clip(ratio, 0, 2) / 2.0
        else:
            exhaustion = 0.5
        
        return exhaustion

    # --------------------------------------------------------------------------
    # 3. Volatility Compression
    # --------------------------------------------------------------------------
    def _volatility_compression(self, ret: np.ndarray) -> float:
        """
        Short-term vol vs medium-term vol.
        Compression often precedes crowded unwind breakouts.
        """
        if len(ret) < self.volume_window:
            return 0.5
        
        short_vol = np.std(ret[-21:]) * np.sqrt(252)
        medium_vol = np.std(ret[-self.volume_window:]) * np.sqrt(252)
        
        if medium_vol > 0.001:
            ratio = short_vol / medium_vol
            # Low ratio = compression = higher score
            compression = 1.0 - np.clip(ratio, 0, 2) / 2.0
        else:
            compression = 0.5
        
        return compression

    # --------------------------------------------------------------------------
    # 4. Individual Ticker Components
    # --------------------------------------------------------------------------
    def _momentum_score(self, ret: np.ndarray) -> float:
        """
        Z-score of recent return vs ROLLING history (not all history).
        Fixed: uses fixed lookback for baseline, preventing ancient data anchoring.
        """
        w = self.momentum_window
        hist_window = min(252, len(ret) - w)
        if hist_window < w:
            return 0.5
        
        recent = np.mean(ret[-w:]) * 252
        hist = ret[-(w + hist_window):-w]
        hist_mean = np.mean(hist) * 252
        hist_std = np.std(hist) * np.sqrt(252)
        
        if hist_std > 0.001:
            mom_z = (recent - hist_mean) / hist_std
        else:
            mom_z = 0.0
        
        # Map to [-1, 1]
        return 2 * stats.norm.cdf(mom_z) - 1

    def _volume_score(self, vol: np.ndarray) -> float:
        if len(vol) < self.volume_window:
            return 0.5
        recent = vol[-5:].mean()
        avg = vol[-self.volume_window:].mean()
        ratio = recent / (avg + 1e-6)
        return min(ratio / 3.0, 1.0)

    def _macro_score(self, ret: np.ndarray, macro: np.ndarray, vol: np.ndarray = None) -> float:
        """Dynamic macro sensitivity over proper rolling window."""
        w = self.macro_corr_window
        if len(ret) < w or macro.shape[1] == 0:
            return 0.5
        
        vix = macro[:, 0]
        recent_ret = ret[-w:]
        recent_vix = vix[-w:]
        
        beta = np.cov(recent_ret, recent_vix)[0, 1] / (np.var(recent_vix) + 1e-6)
        base = np.clip(abs(beta), 0, 1)
        
        if self.use_vol_weighted and vol is not None:
            vol_ratio = self._volume_score(vol)
            base = base * (0.5 + 0.5 * vol_ratio)
        
        return base

    # --------------------------------------------------------------------------
    # 5. Main Crowding Score (No broken bootstrap)
    # --------------------------------------------------------------------------
    def compute_crowding_score(self, returns: pd.DataFrame, volume: pd.DataFrame,
                               macro: pd.DataFrame) -> tuple:
        scores = {}
        cis = {}
        crowding_momentum = {}
        macro_raw = {}
        volume_raw = {}
        momentum_raw = {}

        common_idx = returns.index.intersection(macro.index).intersection(volume.index)
        returns = returns.loc[common_idx]
        volume = volume.loc[common_idx]
        macro = macro.loc[common_idx]

        # Universe-level crowding
        xs_crowding = self._cross_sectional_crowding(returns)

        for ticker in returns.columns:
            ret = returns[ticker].values
            vol = volume[ticker].values if ticker in volume.columns else np.ones_like(ret)
            if len(ret) < self.macro_corr_window:
                continue

            # Compute components deterministically
            mom = self._momentum_score(ret)
            vol_score = self._volume_score(vol)
            macro_score = self._macro_score(ret, macro.values, vol)
            exhaustion = self._momentum_exhaustion(ret)
            vol_comp = self._volatility_compression(ret)

            # Structural composition
            crowd_score = (
                0.25 * abs(mom) +        # Extreme momentum
                0.20 * macro_score +      # High macro sensitivity
                0.25 * exhaustion +       # Momentum decelerating
                0.15 * vol_comp +         # Volatility compression
                0.15 * xs_crowding        # Everything moving together
            )

            # Analytical CI approximation (replaces broken bootstrap)
            ci_width = 0.1 * crowd_score + 0.02
            ci_lower = max(0, crowd_score - ci_width)
            ci_upper = min(1, crowd_score + ci_width)

            scores[ticker] = crowd_score
            cis[ticker] = {"lower": ci_lower, "upper": ci_upper}
            momentum_raw[ticker] = mom
            volume_raw[ticker] = vol_score
            macro_raw[ticker] = macro_score

            # Crowding momentum (change over 21 days)
            if self.use_momentum and len(ret) >= self.macro_corr_window + 21:
                past_ret = ret[:-21]
                past_vol = vol[:-21]
                past_macro = macro.values[:-21]
                past_returns_df = returns.iloc[:-21]

                past_mom = self._momentum_score(past_ret)
                past_macro_score = self._macro_score(past_ret, past_macro, past_vol)
                past_exhaustion = self._momentum_exhaustion(past_ret)
                past_vol_comp = self._volatility_compression(past_ret)
                past_xs = self._cross_sectional_crowding(past_returns_df)

                past_crowd = (
                    0.25 * abs(past_mom) +
                    0.20 * past_macro_score +
                    0.25 * past_exhaustion +
                    0.15 * past_vol_comp +
                    0.15 * past_xs
                )
                crowding_momentum[ticker] = crowd_score - past_crowd
            else:
                crowding_momentum[ticker] = 0.0

        # Cross-sectional ranking
        if self.use_cross_rank:
            score_series = pd.Series(scores)
            rank_pct = score_series.rank(pct=True)
            for t in scores:
                scores[t] = rank_pct[t]

        # Adaptive regime conditioning (percentile-based, not static thresholds)
        if self.use_regime and 'VIX' in macro.columns:
            vix_series = macro['VIX']
            vix_current = vix_series.iloc[-1]
            vix_lookback = vix_series.iloc[-252:] if len(vix_series) >= 252 else vix_series
            vix_median = vix_lookback.median()
            
            # How elevated is VIX relative to its recent norm?
            vix_percentile = (vix_current - vix_median) / (vix_median + 1e-6)
            
            for t in scores:
                if vix_percentile > 0.5:
                    scores[t] = min(scores[t] * (1 + 0.3 * vix_percentile), 1.0)
                else:
                    scores[t] = scores[t] * (1 - 0.2 * abs(vix_percentile))

        return (pd.Series(scores), cis, pd.Series(crowding_momentum),
                pd.Series(momentum_raw), pd.Series(volume_raw), pd.Series(macro_raw))

    # --------------------------------------------------------------------------
    # 6. Expected Return and Decomposition
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
        """
        Fixed: Crowded trade going WRONG gets extra penalty (unwind risk).
        Original penalized longs more than shorts, which is backwards.
        """
        adj = {}
        alpha = {}
        penalty = {}

        for ticker in expected_return.index:
            exp = expected_return.get(ticker, 0.0)
            crowd = np.clip(crowding_score.get(ticker, 0.0), 0, 1)

            base_penalty = 0.5 * crowd

            if exp < 0:
                # Crowded AND losing = unwind in progress, extra penalty
                unwind_penalty = 0.3 * crowd
                total_penalty = base_penalty + unwind_penalty
            else:
                total_penalty = base_penalty

            adjusted = exp * (1 - total_penalty)
            adj[ticker] = adjusted
            alpha[ticker] = adjusted
            penalty[ticker] = exp - adjusted

        return pd.Series(adj), pd.Series(alpha), pd.Series(penalty)

    # --------------------------------------------------------------------------
    # 7. Predictive Validation
    # --------------------------------------------------------------------------
    def predictive_validation(self, returns: pd.DataFrame,
                              crowding_history: pd.DataFrame) -> pd.Series:
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
