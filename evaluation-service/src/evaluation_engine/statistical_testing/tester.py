"""
Statistical Tester - Advanced statistical hypothesis testing
"""

import asyncio
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind, ttest_rel, wilcoxon

from ..models import StatisticalTestType

logger = logging.getLogger(__name__)


class StatisticalTester:
    """Advanced statistical hypothesis testing"""

    def __init__(self):
        self.test_methods = {
            StatisticalTestType.T_TEST: self._t_test,
            StatisticalTestType.WELCH_T_TEST: self._welch_t_test,
            StatisticalTestType.MANN_WHITNEY_U: self._mann_whitney_u_test,
            StatisticalTestType.KOLMOGOROV_SMIRNOV: self._kolmogorov_smirnov_test,
            StatisticalTestType.CHI_SQUARE: self._chi_square_test,
            StatisticalTestType.FISHER_EXACT: self._fisher_exact_test,
            StatisticalTestType.WILCOXON: self._wilcoxon_test,
        }

    async def initialize(self):
        """Initialize the statistical tester"""
        logger.info("Initializing statistical tester...")
        # No specific initialization needed
        logger.info("Statistical tester initialized successfully")

    async def cleanup(self):
        """Cleanup statistical tester resources"""
        logger.info("Cleaning up statistical tester...")
        # No specific cleanup needed
        logger.info("Statistical tester cleanup completed")

    async def two_sample_test(
        self,
        data1: List[float],
        data2: List[float],
        test_type: StatisticalTestType = StatisticalTestType.WELCH_T_TEST,
        alternative: str = "two-sided",
    ) -> Dict[str, Any]:
        """Perform two-sample statistical test"""

        logger.info(f"Performing {test_type.value} test")

        if test_type not in self.test_methods:
            raise ValueError(f"Unsupported test type: {test_type}")

        # Convert to numpy arrays
        data1 = np.array(data1)
        data2 = np.array(data2)

        # Check for empty data
        if len(data1) == 0 or len(data2) == 0:
            return {
                "test_type": test_type.value,
                "statistic": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "effect_size": 0.0,
                "confidence_interval": {"lower": 0, "upper": 0},
                "n1": len(data1),
                "n2": len(data2),
                "mean1": 0.0,
                "mean2": 0.0,
            }

        # Perform the test
        result = await self.test_methods[test_type](data1, data2, alternative)

        # Calculate effect size
        effect_size = await self.calculate_effect_size(data1, data2)

        # Calculate confidence interval
        confidence_interval = await self.calculate_confidence_interval(data1, data2)

        # Add additional information
        result.update(
            {
                "effect_size": effect_size,
                "confidence_interval": confidence_interval,
                "n1": len(data1),
                "n2": len(data2),
                "mean1": np.mean(data1),
                "mean2": np.mean(data2),
                "std1": np.std(data1, ddof=1),
                "std2": np.std(data2, ddof=1),
            }
        )

        return result

    async def _t_test(
        self, data1: np.ndarray, data2: np.ndarray, alternative: str
    ) -> Dict[str, Any]:
        """Independent t-test"""

        statistic, p_value = ttest_ind(data1, data2)

        return {
            "test_type": "t_test",
            "statistic": statistic,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "alternative": alternative,
        }

    async def _welch_t_test(
        self, data1: np.ndarray, data2: np.ndarray, alternative: str
    ) -> Dict[str, Any]:
        """Welch's t-test (unequal variances)"""

        statistic, p_value = ttest_ind(data1, data2, equal_var=False)

        return {
            "test_type": "welch_t_test",
            "statistic": statistic,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "alternative": alternative,
        }

    async def _mann_whitney_u_test(
        self, data1: np.ndarray, data2: np.ndarray, alternative: str
    ) -> Dict[str, Any]:
        """Mann-Whitney U test (non-parametric)"""

        statistic, p_value = mannwhitneyu(data1, data2, alternative=alternative)

        return {
            "test_type": "mann_whitney_u",
            "statistic": statistic,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "alternative": alternative,
        }

    async def _kolmogorov_smirnov_test(
        self, data1: np.ndarray, data2: np.ndarray, alternative: str
    ) -> Dict[str, Any]:
        """Kolmogorov-Smirnov test"""

        statistic, p_value = stats.ks_2samp(data1, data2)

        return {
            "test_type": "kolmogorov_smirnov",
            "statistic": statistic,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "alternative": alternative,
        }

    async def _chi_square_test(
        self, data1: np.ndarray, data2: np.ndarray, alternative: str
    ) -> Dict[str, Any]:
        """Chi-square test of independence"""

        # Create contingency table
        # This is a simplified implementation - in practice, you'd have categorical data
        observed = np.array(
            [
                [np.sum(data1 > np.median(data1)), np.sum(data1 <= np.median(data1))],
                [np.sum(data2 > np.median(data2)), np.sum(data2 <= np.median(data2))],
            ]
        )

        statistic, p_value, dof, expected = chi2_contingency(observed)

        return {
            "test_type": "chi_square",
            "statistic": statistic,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "degrees_of_freedom": dof,
            "alternative": alternative,
        }

    async def _fisher_exact_test(
        self, data1: np.ndarray, data2: np.ndarray, alternative: str
    ) -> Dict[str, Any]:
        """Fisher's exact test"""

        # Create 2x2 contingency table
        # This is a simplified implementation
        table = np.array(
            [
                [np.sum(data1 > np.median(data1)), np.sum(data1 <= np.median(data1))],
                [np.sum(data2 > np.median(data2)), np.sum(data2 <= np.median(data2))],
            ]
        )

        odds_ratio, p_value = stats.fisher_exact(table, alternative=alternative)

        return {
            "test_type": "fisher_exact",
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "alternative": alternative,
        }

    async def _wilcoxon_test(
        self, data1: np.ndarray, data2: np.ndarray, alternative: str
    ) -> Dict[str, Any]:
        """Wilcoxon signed-rank test (paired)"""

        # For paired data, we need equal length arrays
        if len(data1) != len(data2):
            raise ValueError("Wilcoxon test requires paired data with equal length")

        statistic, p_value = wilcoxon(data1, data2, alternative=alternative)

        return {
            "test_type": "wilcoxon",
            "statistic": statistic,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "alternative": alternative,
        }

    async def calculate_effect_size(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""

        if len(data1) == 0 or len(data2) == 0:
            return 0.0

        # Calculate means
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)

        # Calculate pooled standard deviation
        var1 = np.var(data1, ddof=1)
        var2 = np.var(data2, ddof=1)
        n1 = len(data1)
        n2 = len(data2)

        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = math.sqrt(pooled_var)

        if pooled_std == 0:
            return 0.0

        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std

        return cohens_d

    async def calculate_confidence_interval(
        self, data1: np.ndarray, data2: np.ndarray, confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Calculate confidence interval for difference in means"""

        if len(data1) == 0 or len(data2) == 0:
            return {"lower": 0, "upper": 0, "level": confidence_level}

        # Calculate difference in means
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)
        diff = mean1 - mean2

        # Calculate standard error
        se1 = np.std(data1, ddof=1) / math.sqrt(len(data1))
        se2 = np.std(data2, ddof=1) / math.sqrt(len(data2))
        se_diff = math.sqrt(se1**2 + se2**2)

        # Calculate degrees of freedom (Welch-Satterthwaite)
        var1 = np.var(data1, ddof=1)
        var2 = np.var(data2, ddof=1)
        n1 = len(data1)
        n2 = len(data2)

        df = (se1**2 + se2**2) ** 2 / (se1**4 / (n1 - 1) + se2**4 / (n2 - 1))

        # Calculate critical value
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, df)

        # Calculate margin of error
        margin_error = t_critical * se_diff

        return {
            "lower": diff - margin_error,
            "upper": diff + margin_error,
            "level": confidence_level,
            "difference": diff,
            "standard_error": se_diff,
        }

    async def multiple_comparison_correction(
        self, p_values: List[float], method: str = "bonferroni"
    ) -> Dict[str, Any]:
        """Apply multiple comparison correction"""

        p_values = np.array(p_values)
        n_tests = len(p_values)

        if method == "bonferroni":
            corrected_p = p_values * n_tests
            corrected_p = np.minimum(corrected_p, 1.0)  # Cap at 1.0

        elif method == "holm":
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            corrected_p = np.zeros_like(p_values)

            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = p_values[idx] * (n_tests - i)
                if i > 0:
                    corrected_p[idx] = max(corrected_p[idx], corrected_p[sorted_indices[i - 1]])

            corrected_p = np.minimum(corrected_p, 1.0)

        elif method == "fdr":
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_values)
            corrected_p = np.zeros_like(p_values)

            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = p_values[idx] * n_tests / (i + 1)

            # Ensure monotonicity
            for i in range(len(corrected_p) - 2, -1, -1):
                corrected_p[i] = min(corrected_p[i], corrected_p[i + 1])

            corrected_p = np.minimum(corrected_p, 1.0)

        else:
            raise ValueError(f"Unsupported correction method: {method}")

        return {
            "method": method,
            "original_p_values": p_values.tolist(),
            "corrected_p_values": corrected_p.tolist(),
            "n_tests": n_tests,
            "significant_after_correction": np.sum(corrected_p < 0.05),
        }

    async def power_analysis(
        self,
        effect_size: float,
        sample_size: int,
        alpha: float = 0.05,
        test_type: str = "two_sample",
    ) -> Dict[str, Any]:
        """Calculate statistical power"""

        if test_type == "two_sample":
            # Two-sample t-test power
            ncp = effect_size * math.sqrt(sample_size / 2)  # Non-centrality parameter
            critical_value = stats.norm.ppf(1 - alpha / 2)

            power = 1 - stats.norm.cdf(critical_value - ncp) + stats.norm.cdf(-critical_value - ncp)

        elif test_type == "one_sample":
            # One-sample t-test power
            ncp = effect_size * math.sqrt(sample_size)
            critical_value = stats.norm.ppf(1 - alpha / 2)

            power = 1 - stats.norm.cdf(critical_value - ncp) + stats.norm.cdf(-critical_value - ncp)

        else:
            raise ValueError(f"Unsupported test type: {test_type}")

        return {
            "effect_size": effect_size,
            "sample_size": sample_size,
            "alpha": alpha,
            "power": max(0, min(1, power)),
            "test_type": test_type,
        }

    async def bootstrap_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """Bootstrap hypothesis test"""

        if len(data1) == 0 or len(data2) == 0:
            return {
                "test_type": "bootstrap",
                "p_value": 1.0,
                "is_significant": False,
                "confidence_interval": {"lower": 0, "upper": 0},
            }

        # Calculate observed difference
        observed_diff = np.mean(data1) - np.mean(data2)

        # Bootstrap samples
        bootstrap_diffs = []
        combined_data = np.concatenate([data1, data2])
        n1 = len(data1)
        n2 = len(data2)

        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(
                combined_data, size=len(combined_data), replace=True
            )
            bootstrap_data1 = bootstrap_sample[:n1]
            bootstrap_data2 = bootstrap_sample[n1:]

            bootstrap_diff = np.mean(bootstrap_data1) - np.mean(bootstrap_data2)
            bootstrap_diffs.append(bootstrap_diff)

        bootstrap_diffs = np.array(bootstrap_diffs)

        # Calculate p-value (two-tailed)
        p_value = 2 * min(
            np.mean(bootstrap_diffs >= observed_diff), np.mean(bootstrap_diffs <= observed_diff)
        )

        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_diffs, lower_percentile)
        ci_upper = np.percentile(bootstrap_diffs, upper_percentile)

        return {
            "test_type": "bootstrap",
            "observed_difference": observed_diff,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "confidence_interval": {
                "lower": ci_lower,
                "upper": ci_upper,
                "level": confidence_level,
            },
            "bootstrap_differences": bootstrap_diffs.tolist(),
            "n_bootstrap": n_bootstrap,
        }
