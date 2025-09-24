"""
Power Analysis Module
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats
from scipy.stats import norm

logger = logging.getLogger(__name__)


class PowerAnalyzer:
    """Statistical power analysis for experiments"""

    def __init__(self):
        self.power_methods = {
            "t_test": self._t_test_power,
            "z_test": self._z_test_power,
            "chi_square": self._chi_square_power,
            "anova": self._anova_power,
        }

    async def initialize(self):
        """Initialize the power analyzer"""
        logger.info("Initializing power analyzer...")
        # No specific initialization needed
        logger.info("Power analyzer initialized successfully")

    async def cleanup(self):
        """Cleanup power analyzer resources"""
        logger.info("Cleaning up power analyzer...")
        # No specific cleanup needed
        logger.info("Power analyzer cleanup completed")

    async def calculate_sample_size(
        self,
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.8,
        test_type: str = "t_test",
        baseline_conversion_rate: float = 0.5,
    ) -> int:
        """Calculate required sample size for statistical test"""

        logger.info(f"Calculating sample size for effect size {effect_size}")

        if test_type not in self.power_methods:
            raise ValueError(f"Unsupported test type: {test_type}")

        # Calculate sample size using specified method
        sample_size = await self.power_methods[test_type](
            effect_size, alpha, power, baseline_conversion_rate
        )

        return max(sample_size, 100)  # Minimum sample size

    async def _t_test_power(
        self, effect_size: float, alpha: float, power: float, baseline_rate: float
    ) -> int:
        """Calculate sample size for t-test"""

        # Non-centrality parameter
        ncp = effect_size * np.sqrt(2)  # For two-sample t-test

        # Critical value
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)

        # Sample size calculation
        n_per_group = ((z_alpha + z_beta) / effect_size) ** 2
        total_n = int(2 * n_per_group)

        return total_n

    async def _z_test_power(
        self, effect_size: float, alpha: float, power: float, baseline_rate: float
    ) -> int:
        """Calculate sample size for z-test (proportions)"""

        # For proportions, effect size is difference in proportions
        p1 = baseline_rate
        p2 = baseline_rate + effect_size

        # Pooled proportion
        p_pool = (p1 + p2) / 2

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / 2 + 1 / 2))

        # Critical values
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)

        # Sample size calculation
        n_per_group = ((z_alpha + z_beta) * se / effect_size) ** 2
        total_n = int(2 * n_per_group)

        return total_n

    async def _chi_square_power(
        self, effect_size: float, alpha: float, power: float, baseline_rate: float
    ) -> int:
        """Calculate sample size for chi-square test"""

        # For chi-square, effect size is Cramér's V
        # This is a simplified calculation
        n_per_group = int((norm.ppf(1 - alpha / 2) + norm.ppf(power)) ** 2 / effect_size**2)
        total_n = 2 * n_per_group

        return total_n

    async def _anova_power(
        self, effect_size: float, alpha: float, power: float, baseline_rate: float
    ) -> int:
        """Calculate sample size for ANOVA"""

        # For ANOVA, effect size is Cohen's f
        # This is a simplified calculation
        n_per_group = int((norm.ppf(1 - alpha / 2) + norm.ppf(power)) ** 2 / effect_size**2)
        total_n = 3 * n_per_group  # Assuming 3 groups

        return total_n

    async def calculate_power(
        self, effect_size: float, sample_size: int, alpha: float = 0.05, test_type: str = "t_test"
    ) -> float:
        """Calculate statistical power for given parameters"""

        logger.info(f"Calculating power for effect size {effect_size}, sample size {sample_size}")

        if test_type == "t_test":
            # Two-sample t-test power
            ncp = effect_size * np.sqrt(sample_size / 2)
            critical_value = norm.ppf(1 - alpha / 2)
            power = 1 - norm.cdf(critical_value - ncp) + norm.cdf(-critical_value - ncp)

        elif test_type == "z_test":
            # Z-test power (proportions)
            ncp = effect_size * np.sqrt(sample_size / 2)
            critical_value = norm.ppf(1 - alpha / 2)
            power = 1 - norm.cdf(critical_value - ncp) + norm.cdf(-critical_value - ncp)

        else:
            # Default power calculation
            power = 0.8

        return max(0, min(1, power))

    async def calculate_effect_size(
        self, sample_size: int, alpha: float = 0.05, power: float = 0.8, test_type: str = "t_test"
    ) -> float:
        """Calculate minimum detectable effect size"""

        logger.info(f"Calculating effect size for sample size {sample_size}")

        if test_type == "t_test":
            # Two-sample t-test effect size
            z_alpha = norm.ppf(1 - alpha / 2)
            z_beta = norm.ppf(power)
            effect_size = (z_alpha + z_beta) / np.sqrt(sample_size / 2)

        elif test_type == "z_test":
            # Z-test effect size (proportions)
            z_alpha = norm.ppf(1 - alpha / 2)
            z_beta = norm.ppf(power)
            effect_size = (z_alpha + z_beta) / np.sqrt(sample_size / 2)

        else:
            # Default effect size
            effect_size = 0.5

        return effect_size

    async def power_analysis_report(
        self, effect_size: float, sample_size: int, alpha: float = 0.05, test_type: str = "t_test"
    ) -> Dict[str, Any]:
        """Generate comprehensive power analysis report"""

        logger.info("Generating power analysis report")

        # Calculate power
        power = await self.calculate_power(effect_size, sample_size, alpha, test_type)

        # Calculate minimum detectable effect
        min_effect = await self.calculate_effect_size(sample_size, alpha, 0.8, test_type)

        # Calculate required sample size for 80% power
        required_n = await self.calculate_sample_size(effect_size, alpha, 0.8, test_type)

        # Power curve data
        power_curve = []
        for n in range(100, min(sample_size * 2, 10000), 100):
            p = await self.calculate_power(effect_size, n, alpha, test_type)
            power_curve.append({"sample_size": n, "power": p})

        return {
            "effect_size": effect_size,
            "sample_size": sample_size,
            "alpha": alpha,
            "power": power,
            "minimum_detectable_effect": min_effect,
            "required_sample_size_80_power": required_n,
            "power_curve": power_curve,
            "test_type": test_type,
            "recommendations": self._generate_power_recommendations(power, sample_size, required_n),
        }

    def _generate_power_recommendations(
        self, power: float, sample_size: int, required_n: int
    ) -> List[str]:
        """Generate power analysis recommendations"""

        recommendations = []

        if power < 0.8:
            recommendations.append(f"Current power ({power:.3f}) is below recommended 0.8")
            recommendations.append(f"Consider increasing sample size to {required_n} for 80% power")

        if sample_size < 1000:
            recommendations.append(
                "Sample size is relatively small - consider increasing for more reliable results"
            )

        if power > 0.95:
            recommendations.append(
                "Power is very high - consider reducing sample size for efficiency"
            )

        if not recommendations:
            recommendations.append("Power analysis looks good - no changes needed")

        return recommendations
