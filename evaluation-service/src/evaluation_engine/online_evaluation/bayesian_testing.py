"""
Bayesian Testing Framework
"""

import asyncio
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from scipy import stats
from scipy.stats import beta, gamma, norm

logger = logging.getLogger(__name__)


class BayesianTestingFramework:
    """Bayesian testing framework for A/B tests"""

    def __init__(self):
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.bayesian_methods = {
            "beta_binomial": self._beta_binomial_test,
            "normal_normal": self._normal_normal_test,
            "gamma_poisson": self._gamma_poisson_test,
        }

    async def initialize(self):
        """Initialize the Bayesian testing framework"""
        logger.info("Initializing Bayesian testing framework...")
        # No specific initialization needed
        logger.info("Bayesian testing framework initialized successfully")

    async def cleanup(self):
        """Cleanup Bayesian testing framework resources"""
        logger.info("Cleaning up Bayesian testing framework...")
        self.active_tests.clear()
        logger.info("Bayesian testing framework cleanup completed")

    async def create_bayesian_test(self, test_config: Dict[str, Any]) -> str:
        """Create a new Bayesian test"""

        test_id = str(uuid4())
        method = test_config.get("method", "beta_binomial")

        if method not in self.bayesian_methods:
            raise ValueError(f"Unsupported Bayesian method: {method}")

        test = {
            "id": test_id,
            "method": method,
            "parameters": test_config.get("parameters", {}),
            "prior_parameters": test_config.get("prior_parameters", {}),
            "control_data": [],
            "treatment_data": [],
            "control_posterior": None,
            "treatment_posterior": None,
            "created_at": datetime.utcnow(),
            "status": "active",
        }

        self.active_tests[test_id] = test

        logger.info(f"Created Bayesian test {test_id} with method {method}")
        return test_id

    async def add_data(
        self, test_id: str, control_value: float, treatment_value: float
    ) -> Dict[str, Any]:
        """Add data point to Bayesian test"""

        if test_id not in self.active_tests:
            raise ValueError(f"Bayesian test {test_id} not found")

        test = self.active_tests[test_id]

        if test["status"] != "active":
            return {"status": "stopped", "reason": "Test already stopped"}

        # Add data
        test["control_data"].append(control_value)
        test["treatment_data"].append(treatment_value)

        # Update posteriors
        await self._update_posteriors(test)

        # Calculate probability of treatment being better
        prob_better = await self._calculate_probability_better(test)

        return {
            "test_id": test_id,
            "control_samples": len(test["control_data"]),
            "treatment_samples": len(test["treatment_data"]),
            "probability_treatment_better": prob_better,
            "status": test["status"],
        }

    async def analyze_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze Bayesian test results"""

        if test_id not in self.active_tests:
            raise ValueError(f"Bayesian test {test_id} not found")

        test = self.active_tests[test_id]

        # Update posteriors
        await self._update_posteriors(test)

        # Calculate key metrics
        prob_better = await self._calculate_probability_better(test)
        expected_loss = await self._calculate_expected_loss(test)
        credible_interval = await self._calculate_credible_interval(test)

        # Determine if test should stop
        stopping_decision = await self._check_stopping_conditions(test, prob_better, expected_loss)

        return {
            "test_id": test_id,
            "method": test["method"],
            "status": test["status"],
            "control_samples": len(test["control_data"]),
            "treatment_samples": len(test["treatment_data"]),
            "probability_treatment_better": prob_better,
            "expected_loss": expected_loss,
            "credible_interval": credible_interval,
            "stopping_decision": stopping_decision,
            "analysis_timestamp": datetime.utcnow(),
        }

    async def _update_posteriors(self, test: Dict[str, Any]):
        """Update posterior distributions"""

        method = test["method"]

        if method not in self.bayesian_methods:
            raise ValueError(f"Unsupported method: {method}")

        # Update posteriors using the specified method
        await self.bayesian_methods[method](test)

    async def _beta_binomial_test(self, test: Dict[str, Any]):
        """Beta-Binomial Bayesian test for conversion rates"""

        control_data = np.array(test["control_data"])
        treatment_data = np.array(test["treatment_data"])

        # Prior parameters (Beta distribution)
        prior_alpha = test["prior_parameters"].get("alpha", 1.0)
        prior_beta = test["prior_parameters"].get("beta", 1.0)

        # Control posterior: Beta(alpha + successes, beta + failures)
        control_successes = np.sum(control_data)
        control_failures = len(control_data) - control_successes
        test["control_posterior"] = {
            "alpha": prior_alpha + control_successes,
            "beta": prior_beta + control_failures,
            "mean": (prior_alpha + control_successes)
            / (prior_alpha + prior_beta + len(control_data)),
        }

        # Treatment posterior: Beta(alpha + successes, beta + failures)
        treatment_successes = np.sum(treatment_data)
        treatment_failures = len(treatment_data) - treatment_successes
        test["treatment_posterior"] = {
            "alpha": prior_alpha + treatment_successes,
            "beta": prior_beta + treatment_failures,
            "mean": (prior_alpha + treatment_successes)
            / (prior_alpha + prior_beta + len(treatment_data)),
        }

    async def _normal_normal_test(self, test: Dict[str, Any]):
        """Normal-Normal Bayesian test for continuous metrics"""

        control_data = np.array(test["control_data"])
        treatment_data = np.array(test["treatment_data"])

        # Prior parameters (Normal distribution)
        prior_mean = test["prior_parameters"].get("mean", 0.0)
        prior_precision = test["prior_parameters"].get("precision", 0.01)  # 1/variance

        # Data precision (assuming known variance)
        data_precision = test["prior_parameters"].get("data_precision", 1.0)

        # Control posterior: Normal(mean, precision)
        if len(control_data) > 0:
            control_mean = np.mean(control_data)
            control_precision = prior_precision + len(control_data) * data_precision
            control_posterior_mean = (
                prior_precision * prior_mean + len(control_data) * data_precision * control_mean
            ) / control_precision

            test["control_posterior"] = {
                "mean": control_posterior_mean,
                "precision": control_precision,
                "variance": 1.0 / control_precision,
            }
        else:
            test["control_posterior"] = {
                "mean": prior_mean,
                "precision": prior_precision,
                "variance": 1.0 / prior_precision,
            }

        # Treatment posterior: Normal(mean, precision)
        if len(treatment_data) > 0:
            treatment_mean = np.mean(treatment_data)
            treatment_precision = prior_precision + len(treatment_data) * data_precision
            treatment_posterior_mean = (
                prior_precision * prior_mean + len(treatment_data) * data_precision * treatment_mean
            ) / treatment_precision

            test["treatment_posterior"] = {
                "mean": treatment_posterior_mean,
                "precision": treatment_precision,
                "variance": 1.0 / treatment_precision,
            }
        else:
            test["treatment_posterior"] = {
                "mean": prior_mean,
                "precision": prior_precision,
                "variance": 1.0 / prior_precision,
            }

    async def _gamma_poisson_test(self, test: Dict[str, Any]):
        """Gamma-Poisson Bayesian test for count data"""

        control_data = np.array(test["control_data"])
        treatment_data = np.array(test["treatment_data"])

        # Prior parameters (Gamma distribution)
        prior_shape = test["prior_parameters"].get("shape", 1.0)
        prior_rate = test["prior_parameters"].get("rate", 1.0)

        # Control posterior: Gamma(shape + sum, rate + n)
        control_sum = np.sum(control_data)
        test["control_posterior"] = {
            "shape": prior_shape + control_sum,
            "rate": prior_rate + len(control_data),
            "mean": (prior_shape + control_sum) / (prior_rate + len(control_data)),
        }

        # Treatment posterior: Gamma(shape + sum, rate + n)
        treatment_sum = np.sum(treatment_data)
        test["treatment_posterior"] = {
            "shape": prior_shape + treatment_sum,
            "rate": prior_rate + len(treatment_data),
            "mean": (prior_shape + treatment_sum) / (prior_rate + len(treatment_data)),
        }

    async def _calculate_probability_better(self, test: Dict[str, Any]) -> float:
        """Calculate probability that treatment is better than control"""

        method = test["method"]

        if method == "beta_binomial":
            return self._beta_binomial_probability_better(test)
        elif method == "normal_normal":
            return self._normal_normal_probability_better(test)
        elif method == "gamma_poisson":
            return self._gamma_poisson_probability_better(test)
        else:
            return 0.5  # Default to 50% if unknown method

    def _beta_binomial_probability_better(self, test: Dict[str, Any]) -> float:
        """Calculate probability for Beta-Binomial test"""

        control_posterior = test["control_posterior"]
        treatment_posterior = test["treatment_posterior"]

        # Sample from posterior distributions
        n_samples = 10000
        control_samples = beta.rvs(
            control_posterior["alpha"], control_posterior["beta"], size=n_samples
        )
        treatment_samples = beta.rvs(
            treatment_posterior["alpha"], treatment_posterior["beta"], size=n_samples
        )

        # Calculate probability that treatment > control
        return np.mean(treatment_samples > control_samples)

    def _normal_normal_probability_better(self, test: Dict[str, Any]) -> float:
        """Calculate probability for Normal-Normal test"""

        control_posterior = test["control_posterior"]
        treatment_posterior = test["treatment_posterior"]

        # Calculate difference distribution
        diff_mean = treatment_posterior["mean"] - control_posterior["mean"]
        diff_variance = treatment_posterior["variance"] + control_posterior["variance"]
        diff_std = math.sqrt(diff_variance)

        # Probability that difference > 0
        return 1 - norm.cdf(0, diff_mean, diff_std)

    def _gamma_poisson_probability_better(self, test: Dict[str, Any]) -> float:
        """Calculate probability for Gamma-Poisson test"""

        control_posterior = test["control_posterior"]
        treatment_posterior = test["treatment_posterior"]

        # Sample from posterior distributions
        n_samples = 10000
        control_samples = gamma.rvs(
            control_posterior["shape"], scale=1.0 / control_posterior["rate"], size=n_samples
        )
        treatment_samples = gamma.rvs(
            treatment_posterior["shape"], scale=1.0 / treatment_posterior["rate"], size=n_samples
        )

        # Calculate probability that treatment > control
        return np.mean(treatment_samples > control_samples)

    async def _calculate_expected_loss(self, test: Dict[str, Any]) -> float:
        """Calculate expected loss of choosing treatment"""

        method = test["method"]

        if method == "beta_binomial":
            return self._beta_binomial_expected_loss(test)
        elif method == "normal_normal":
            return self._normal_normal_expected_loss(test)
        elif method == "gamma_poisson":
            return self._gamma_poisson_expected_loss(test)
        else:
            return 0.0

    def _beta_binomial_expected_loss(self, test: Dict[str, Any]) -> float:
        """Calculate expected loss for Beta-Binomial test"""

        control_posterior = test["control_posterior"]
        treatment_posterior = test["treatment_posterior"]

        # Sample from posterior distributions
        n_samples = 10000
        control_samples = beta.rvs(
            control_posterior["alpha"], control_posterior["beta"], size=n_samples
        )
        treatment_samples = beta.rvs(
            treatment_posterior["alpha"], treatment_posterior["beta"], size=n_samples
        )

        # Expected loss = E[max(0, control - treatment)]
        loss_samples = np.maximum(0, control_samples - treatment_samples)
        return np.mean(loss_samples)

    def _normal_normal_expected_loss(self, test: Dict[str, Any]) -> float:
        """Calculate expected loss for Normal-Normal test"""

        control_posterior = test["control_posterior"]
        treatment_posterior = test["treatment_posterior"]

        # Calculate difference distribution
        diff_mean = treatment_posterior["mean"] - control_posterior["mean"]
        diff_variance = treatment_posterior["variance"] + control_posterior["variance"]
        diff_std = math.sqrt(diff_variance)

        # Expected loss = E[max(0, -difference)]
        # This is the expected value of the truncated normal distribution
        if diff_mean >= 0:
            return 0.0
        else:
            # Expected value of max(0, -X) where X ~ N(diff_mean, diff_std^2)
            return -diff_mean * norm.cdf(diff_mean / diff_std) + diff_std * norm.pdf(
                diff_mean / diff_std
            )

    def _gamma_poisson_expected_loss(self, test: Dict[str, Any]) -> float:
        """Calculate expected loss for Gamma-Poisson test"""

        control_posterior = test["control_posterior"]
        treatment_posterior = test["treatment_posterior"]

        # Sample from posterior distributions
        n_samples = 10000
        control_samples = gamma.rvs(
            control_posterior["shape"], scale=1.0 / control_posterior["rate"], size=n_samples
        )
        treatment_samples = gamma.rvs(
            treatment_posterior["shape"], scale=1.0 / treatment_posterior["rate"], size=n_samples
        )

        # Expected loss = E[max(0, control - treatment)]
        loss_samples = np.maximum(0, control_samples - treatment_samples)
        return np.mean(loss_samples)

    async def _calculate_credible_interval(
        self, test: Dict[str, Any], level: float = 0.95
    ) -> Dict[str, Any]:
        """Calculate credible interval for treatment effect"""

        method = test["method"]

        if method == "beta_binomial":
            return self._beta_binomial_credible_interval(test, level)
        elif method == "normal_normal":
            return self._normal_normal_credible_interval(test, level)
        elif method == "gamma_poisson":
            return self._gamma_poisson_credible_interval(test, level)
        else:
            return {"lower": 0, "upper": 0}

    def _beta_binomial_credible_interval(
        self, test: Dict[str, Any], level: float
    ) -> Dict[str, Any]:
        """Calculate credible interval for Beta-Binomial test"""

        control_posterior = test["control_posterior"]
        treatment_posterior = test["treatment_posterior"]

        # Sample from posterior distributions
        n_samples = 10000
        control_samples = beta.rvs(
            control_posterior["alpha"], control_posterior["beta"], size=n_samples
        )
        treatment_samples = beta.rvs(
            treatment_posterior["alpha"], treatment_posterior["beta"], size=n_samples
        )

        # Calculate difference
        diff_samples = treatment_samples - control_samples

        # Calculate credible interval
        alpha = 1 - level
        lower = np.percentile(diff_samples, (alpha / 2) * 100)
        upper = np.percentile(diff_samples, (1 - alpha / 2) * 100)

        return {"lower": lower, "upper": upper, "level": level}

    def _normal_normal_credible_interval(
        self, test: Dict[str, Any], level: float
    ) -> Dict[str, Any]:
        """Calculate credible interval for Normal-Normal test"""

        control_posterior = test["control_posterior"]
        treatment_posterior = test["treatment_posterior"]

        # Calculate difference distribution
        diff_mean = treatment_posterior["mean"] - control_posterior["mean"]
        diff_variance = treatment_posterior["variance"] + control_posterior["variance"]
        diff_std = math.sqrt(diff_variance)

        # Calculate credible interval
        alpha = 1 - level
        z_critical = norm.ppf(1 - alpha / 2)

        lower = diff_mean - z_critical * diff_std
        upper = diff_mean + z_critical * diff_std

        return {"lower": lower, "upper": upper, "level": level}

    def _gamma_poisson_credible_interval(
        self, test: Dict[str, Any], level: float
    ) -> Dict[str, Any]:
        """Calculate credible interval for Gamma-Poisson test"""

        control_posterior = test["control_posterior"]
        treatment_posterior = test["treatment_posterior"]

        # Sample from posterior distributions
        n_samples = 10000
        control_samples = gamma.rvs(
            control_posterior["shape"], scale=1.0 / control_posterior["rate"], size=n_samples
        )
        treatment_samples = gamma.rvs(
            treatment_posterior["shape"], scale=1.0 / treatment_posterior["rate"], size=n_samples
        )

        # Calculate difference
        diff_samples = treatment_samples - control_samples

        # Calculate credible interval
        alpha = 1 - level
        lower = np.percentile(diff_samples, (alpha / 2) * 100)
        upper = np.percentile(diff_samples, (1 - alpha / 2) * 100)

        return {"lower": lower, "upper": upper, "level": level}

    async def _check_stopping_conditions(
        self, test: Dict[str, Any], prob_better: float, expected_loss: float
    ) -> Dict[str, Any]:
        """Check if test should stop"""

        # Stop if probability is very high or very low
        if prob_better >= 0.95:
            return {
                "should_stop": True,
                "reason": "high_confidence",
                "decision": "choose_treatment",
            }
        elif prob_better <= 0.05:
            return {"should_stop": True, "reason": "high_confidence", "decision": "choose_control"}

        # Stop if expected loss is very small
        if expected_loss < 0.01:  # 1% threshold
            return {
                "should_stop": True,
                "reason": "small_expected_loss",
                "decision": "choose_treatment" if prob_better > 0.5 else "choose_control",
            }

        return {"should_stop": False, "reason": "continue", "decision": "continue"}

    async def get_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get Bayesian test by ID"""
        return self.active_tests.get(test_id)

    async def list_tests(self) -> List[Dict[str, Any]]:
        """List all active Bayesian tests"""
        return list(self.active_tests.values())

    async def stop_test(self, test_id: str) -> bool:
        """Stop a Bayesian test"""
        if test_id not in self.active_tests:
            return False

        self.active_tests[test_id]["status"] = "stopped"
        return True
