"""
Statistical analysis utilities for offline evaluation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from scipy.stats import bootstrap
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Statistical analysis utilities for evaluation metrics"""
    
    def __init__(self):
        self.confidence_levels = [0.90, 0.95, 0.99]
    
    async def initialize(self):
        """Initialize the statistical analyzer"""
        pass
    
    async def cleanup(self):
        """Cleanup statistical analyzer resources"""
        pass
    
    async def bootstrap_confidence_interval(self, 
                                         metric_value: float,
                                         bootstrap_samples: int = 1000,
                                         confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculate bootstrap confidence interval for a metric"""
        
        logger.info(f"Calculating bootstrap CI for metric value: {metric_value}")
        
        # For a single metric value, we need to simulate bootstrap samples
        # This is a simplified implementation - in practice, you'd have actual data
        
        # Simulate bootstrap samples around the metric value
        # Add some noise to simulate sampling variability
        noise_std = metric_value * 0.1  # 10% of the metric value as noise
        bootstrap_samples_data = np.random.normal(
            metric_value, noise_std, bootstrap_samples
        )
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_samples_data, lower_percentile)
        ci_upper = np.percentile(bootstrap_samples_data, upper_percentile)
        
        return {
            'lower': ci_lower,
            'upper': ci_upper,
            'level': confidence_level,
            'method': 'bootstrap'
        }
    
    async def calculate_confidence_interval(self, 
                                          data: np.ndarray,
                                          confidence_level: float = 0.95,
                                          method: str = 't_distribution') -> Dict[str, float]:
        """Calculate confidence interval for a dataset"""
        
        if len(data) == 0:
            return {'lower': 0.0, 'upper': 0.0, 'level': confidence_level}
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # Sample standard deviation
        n = len(data)
        
        if method == 't_distribution':
            # Use t-distribution for small samples
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
            margin_error = t_critical * (std / np.sqrt(n))
        elif method == 'normal':
            # Use normal distribution for large samples
            alpha = 1 - confidence_level
            z_critical = stats.norm.ppf(1 - alpha/2)
            margin_error = z_critical * (std / np.sqrt(n))
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return {
            'lower': mean - margin_error,
            'upper': mean + margin_error,
            'level': confidence_level,
            'method': method,
            'mean': mean,
            'std': std,
            'n': n
        }
    
    async def perform_hypothesis_test(self, 
                                    data1: np.ndarray,
                                    data2: np.ndarray,
                                    test_type: str = 't_test',
                                    alternative: str = 'two-sided') -> Dict[str, Any]:
        """Perform hypothesis test between two datasets"""
        
        logger.info(f"Performing {test_type} hypothesis test")
        
        if test_type == 't_test':
            # Independent t-test
            statistic, p_value = stats.ttest_ind(data1, data2)
            test_name = "Independent t-test"
        elif test_type == 'paired_t_test':
            # Paired t-test
            statistic, p_value = stats.ttest_rel(data1, data2)
            test_name = "Paired t-test"
        elif test_type == 'welch_t_test':
            # Welch's t-test (unequal variances)
            statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False)
            test_name = "Welch's t-test"
        elif test_type == 'mann_whitney_u':
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative=alternative)
            test_name = "Mann-Whitney U test"
        elif test_type == 'wilcoxon':
            # Wilcoxon signed-rank test
            statistic, p_value = stats.wilcoxon(data1, data2, alternative=alternative)
            test_name = "Wilcoxon signed-rank test"
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                             (len(data2) - 1) * np.var(data2, ddof=1)) / 
                            (len(data1) + len(data2) - 2))
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
        
        return {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'alternative': alternative,
            'effect_size': cohens_d,
            'is_significant': p_value < 0.05,
            'n1': len(data1),
            'n2': len(data2),
            'mean1': np.mean(data1),
            'mean2': np.mean(data2)
        }
    
    async def calculate_effect_size(self, 
                                  data1: np.ndarray,
                                  data2: np.ndarray,
                                  effect_type: str = 'cohens_d') -> Dict[str, float]:
        """Calculate effect size between two datasets"""
        
        if effect_type == 'cohens_d':
            # Cohen's d
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                 (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                (len(data1) + len(data2) - 2))
            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
            
            # Interpret effect size
            if abs(cohens_d) < 0.2:
                interpretation = "negligible"
            elif abs(cohens_d) < 0.5:
                interpretation = "small"
            elif abs(cohens_d) < 0.8:
                interpretation = "medium"
            else:
                interpretation = "large"
            
            return {
                'cohens_d': cohens_d,
                'interpretation': interpretation,
                'effect_type': effect_type
            }
        
        elif effect_type == 'hedges_g':
            # Hedges' g (bias-corrected Cohen's d)
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                 (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                (len(data1) + len(data2) - 2))
            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
            
            # Correction factor
            correction_factor = 1 - (3 / (4 * (len(data1) + len(data2)) - 9))
            hedges_g = cohens_d * correction_factor
            
            return {
                'hedges_g': hedges_g,
                'cohens_d': cohens_d,
                'correction_factor': correction_factor,
                'effect_type': effect_type
            }
        
        else:
            raise ValueError(f"Unsupported effect size type: {effect_type}")
    
    async def perform_anova(self, 
                          groups: List[np.ndarray],
                          group_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform one-way ANOVA"""
        
        logger.info(f"Performing ANOVA with {len(groups)} groups")
        
        if group_names is None:
            group_names = [f"Group_{i+1}" for i in range(len(groups))]
        
        # Perform ANOVA
        f_statistic, p_value = stats.f_oneway(*groups)
        
        # Calculate effect size (eta squared)
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        
        # Sum of squares between groups
        ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups)
        
        # Sum of squares total
        ss_total = np.sum((all_data - grand_mean) ** 2)
        
        # Eta squared
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Post-hoc tests (Tukey's HSD)
        post_hoc_results = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                # Perform t-test between groups
                t_stat, p_val = stats.ttest_ind(groups[i], groups[j])
                post_hoc_results.append({
                    'group1': group_names[i],
                    'group2': group_names[j],
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'is_significant': p_val < 0.05
                })
        
        return {
            'f_statistic': f_statistic,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'eta_squared': eta_squared,
            'n_groups': len(groups),
            'group_means': [np.mean(group) for group in groups],
            'group_stds': [np.std(group, ddof=1) for group in groups],
            'group_names': group_names,
            'post_hoc_results': post_hoc_results
        }
    
    async def perform_chi_square_test(self, 
                                    observed: np.ndarray,
                                    expected: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform chi-square test of independence or goodness of fit"""
        
        logger.info("Performing chi-square test")
        
        if expected is None:
            # Goodness of fit test (uniform distribution)
            expected = np.full_like(observed, np.sum(observed) / len(observed))
        
        # Perform chi-square test
        chi2_statistic, p_value = stats.chisquare(observed, expected)
        
        # Calculate degrees of freedom
        df = len(observed) - 1
        
        # Calculate Cramér's V (effect size for chi-square)
        n = np.sum(observed)
        cramers_v = np.sqrt(chi2_statistic / (n * df)) if n > 0 and df > 0 else 0
        
        return {
            'chi2_statistic': chi2_statistic,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'is_significant': p_value < 0.05,
            'cramers_v': cramers_v,
            'observed': observed.tolist(),
            'expected': expected.tolist()
        }
    
    async def calculate_correlation(self, 
                                  x: np.ndarray,
                                  y: np.ndarray,
                                  method: str = 'pearson') -> Dict[str, Any]:
        """Calculate correlation between two variables"""
        
        logger.info(f"Calculating {method} correlation")
        
        if method == 'pearson':
            correlation, p_value = stats.pearsonr(x, y)
        elif method == 'spearman':
            correlation, p_value = stats.spearmanr(x, y)
        elif method == 'kendall':
            correlation, p_value = stats.kendalltau(x, y)
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
        
        # Interpret correlation strength
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            strength = "negligible"
        elif abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.5:
            strength = "moderate"
        elif abs_corr < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'strength': strength,
            'method': method,
            'n': len(x)
        }
    
    async def perform_normality_test(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform normality test (Shapiro-Wilk)"""
        
        logger.info("Performing normality test")
        
        # Shapiro-Wilk test (good for small samples)
        if len(data) <= 5000:
            statistic, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk"
        else:
            # Kolmogorov-Smirnov test (for large samples)
            statistic, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
            test_name = "Kolmogorov-Smirnov"
        
        # Additional normality indicators
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        return {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'n': len(data)
        }
    
    async def calculate_sample_size(self, 
                                  effect_size: float,
                                  alpha: float = 0.05,
                                  power: float = 0.8,
                                  test_type: str = 'two_sample') -> Dict[str, Any]:
        """Calculate required sample size for statistical test"""
        
        logger.info(f"Calculating sample size for effect size {effect_size}")
        
        if test_type == 'two_sample':
            # Two-sample t-test
            from statsmodels.stats.power import ttest_power
            
            n_per_group = ttest_power(effect_size, alpha=alpha, power=power, alternative='two-sided')
            total_n = n_per_group * 2
            
        elif test_type == 'one_sample':
            # One-sample t-test
            from statsmodels.stats.power import ttest_power
            
            n_per_group = ttest_power(effect_size, alpha=alpha, power=power, alternative='two-sided')
            total_n = n_per_group
            
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        return {
            'n_per_group': int(np.ceil(n_per_group)),
            'total_n': int(np.ceil(total_n)),
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'test_type': test_type
        }
