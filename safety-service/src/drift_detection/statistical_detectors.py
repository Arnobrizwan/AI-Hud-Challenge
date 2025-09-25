"""
Statistical Drift Detectors
Various statistical tests for detecting data drift
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from safety_engine.config import get_drift_config
from safety_engine.models import StatisticalTestResult
from scipy import stats
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

logger = logging.getLogger(__name__)


class BaseStatisticalDetector:
    """Base class for statistical drift detectors"""

    def __init__(self):
        self.config = get_drift_config()
        self.is_initialized = False

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the detector"""
        self.is_initialized = True

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup resources"""
        pass

    async def test(
        self, reference_data: pd.Series, current_data: pd.Series
    ) -> StatisticalTestResult:
        """Perform statistical test"""
        raise NotImplementedError


class KolmogorovSmirnovDetector(BaseStatisticalDetector):
    """Kolmogorov-Smirnov test for drift detection"""

    def __init__(self):
        super().__init__()
        self.alpha = 0.05

    async def test(
        self, reference_data: pd.Series, current_data: pd.Series
    ) -> StatisticalTestResult:
        """Perform KS test"""
        try:
            if not pd.api.types.is_numeric_dtype(reference_data):
                # For non-numeric data, return non-significant result
                return StatisticalTestResult(
                    test_name="ks_test",
                    p_value=1.0,
                    statistic=0.0,
                    is_significant=False,
                    drift_score=0.0,
                    threshold=0.05,
                )

            # Perform KS test
            statistic, p_value = ks_2samp(reference_data, current_data)

            # Determine significance
            is_significant = p_value < self.alpha

            # Calculate drift score (higher statistic = more drift)
            drift_score = min(statistic, 1.0)

            return StatisticalTestResult(
                test_name="ks_test",
                p_value=p_value,
                statistic=statistic,
                is_significant=is_significant,
                drift_score=drift_score,
            )

        except Exception as e:
            logger.error(f"KS test failed: {str(e)}")
            return StatisticalTestResult(
                test_name="ks_test",
                p_value=1.0,
                statistic=0.0,
                is_significant=False,
                drift_score=0.0,
            )


class ChiSquareDetector(BaseStatisticalDetector):
    """Chi-square test for categorical drift detection"""

    def __init__(self):
        super().__init__()
        self.alpha = 0.05

    async def test(
        self, reference_data: pd.Series, current_data: pd.Series
    ) -> StatisticalTestResult:
        """Perform chi-square test"""
        try:
            # Create contingency table
            ref_counts = reference_data.value_counts()
            curr_counts = current_data.value_counts()

            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_counts = ref_counts.reindex(all_categories, fill_value=0)
            curr_counts = curr_counts.reindex(all_categories, fill_value=0)

            # Create contingency table
            contingency_table = np.array(
                [ref_counts.values, curr_counts.values])

            # Perform chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            # Determine significance
            is_significant = p_value < self.alpha

            # Calculate drift score
            drift_score = min(chi2 / (chi2 + dof), 1.0) if chi2 > 0 else 0.0

            return StatisticalTestResult(
                test_name="chi_square",
                p_value=p_value,
                statistic=chi2,
                is_significant=is_significant,
                drift_score=drift_score,
            )

        except Exception as e:
            logger.error(f"Chi-square test failed: {str(e)}")
            return StatisticalTestResult(
                test_name="chi_square",
                p_value=1.0,
                statistic=0.0,
                is_significant=False,
                drift_score=0.0,
            )


class PopulationStabilityIndexDetector(BaseStatisticalDetector):
    """Population Stability Index for drift detection"""

    def __init__(self):
        super().__init__()
        self.psi_threshold = 0.2

    async def test(
        self, reference_data: pd.Series, current_data: pd.Series
    ) -> StatisticalTestResult:
        """Calculate PSI"""
        try:
            psi_score = self.calculate_psi(reference_data, current_data)

            # Determine significance
            is_significant = psi_score > self.psi_threshold

            # Normalize drift score
            # PSI > 1.0 is very high drift
            drift_score = min(psi_score / 1.0, 1.0)

            return StatisticalTestResult(
                test_name="psi",
                # Convert to p-value-like metric
                p_value=1.0 - min(psi_score, 1.0),
                statistic=psi_score,
                is_significant=is_significant,
                drift_score=drift_score,
            )

        except Exception as e:
            logger.error(f"PSI calculation failed: {str(e)}")
            return StatisticalTestResult(
                test_name="psi",
                p_value=1.0,
                statistic=0.0,
                is_significant=False,
                drift_score=0.0)

    def calculate_psi(self, reference_data: pd.Series,
                      current_data: pd.Series) -> float:
        """Calculate Population Stability Index"""
        try:
            if pd.api.types.is_numeric_dtype(reference_data):
                # For numeric data, create bins
                bins = self.create_bins(reference_data, current_data)
                ref_bins = pd.cut(
                    reference_data,
                    bins=bins,
                    include_lowest=True)
                curr_bins = pd.cut(
                    current_data, bins=bins, include_lowest=True)
            else:
                # For categorical data, use categories directly
                ref_bins = reference_data
                curr_bins = current_data

            # Calculate proportions
            ref_props = ref_bins.value_counts(normalize=True)
            curr_props = curr_bins.value_counts(normalize=True)

            # Align categories
            all_categories = set(ref_props.index) | set(curr_props.index)
            ref_props = ref_props.reindex(
                all_categories, fill_value=1e-6)  # Avoid log(0)
            curr_props = curr_props.reindex(all_categories, fill_value=1e-6)

            # Calculate PSI
            psi = np.sum((curr_props - ref_props) *
                         np.log(curr_props / ref_props))

            return max(psi, 0.0)  # PSI should be non-negative

        except Exception as e:
            logger.error(f"PSI calculation error: {str(e)}")
            return 0.0

    def create_bins(
        self, ref_data: pd.Series, curr_data: pd.Series, n_bins: int = 10
    ) -> np.ndarray:
        """Create bins for numeric data"""
        try:
            # Use quantile-based binning
            all_data = pd.concat([ref_data, curr_data])
            bins = np.quantile(all_data, np.linspace(0, 1, n_bins + 1))
            bins[0] = -np.inf  # Ensure all data is included
            bins[-1] = np.inf
            return np.unique(bins)
        except Exception as e:
            logger.error(f"Bin creation failed: {str(e)}")
            return np.array([-np.inf, np.inf])


class WassersteinDetector(BaseStatisticalDetector):
    """Wasserstein distance for drift detection"""

    def __init__(self):
        super().__init__()
        self.distance_threshold = 0.1

    async def test(
        self, reference_data: pd.Series, current_data: pd.Series
    ) -> StatisticalTestResult:
        """Calculate Wasserstein distance"""
        try:
            if not pd.api.types.is_numeric_dtype(reference_data):
                # For non-numeric data, return non-significant result
                return StatisticalTestResult(
                    test_name="wasserstein",
                    p_value=1.0,
                    statistic=0.0,
                    is_significant=False,
                    drift_score=0.0,
                )

            # Calculate Wasserstein distance
            distance = wasserstein_distance(reference_data, current_data)

            # Determine significance
            is_significant = distance > self.distance_threshold

            # Normalize drift score
            drift_score = min(distance / 1.0, 1.0)  # Normalize to [0, 1]

            return StatisticalTestResult(
                test_name="wasserstein",
                # Convert to p-value-like metric
                p_value=1.0 - min(distance, 1.0),
                statistic=distance,
                is_significant=is_significant,
                drift_score=drift_score,
            )

        except Exception as e:
            logger.error(f"Wasserstein distance calculation failed: {str(e)}")
            return StatisticalTestResult(
                test_name="wasserstein",
                p_value=1.0,
                statistic=0.0,
                is_significant=False,
                drift_score=0.0,
            )
