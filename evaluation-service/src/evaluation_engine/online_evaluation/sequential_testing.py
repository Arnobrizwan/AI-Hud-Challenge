"""
Sequential Testing Framework
"""

import asyncio
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class SequentialTestingFramework:
    """Sequential testing framework for early stopping"""

    def __init__(self):
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.sequential_methods = {
            "wald_sequential": self._wald_sequential_test,
            "group_sequential": self._group_sequential_test,
            "alpha_spending": self._alpha_spending_test,
        }

    async def initialize(self):
        """Initialize the sequential testing framework"""
        logger.info("Initializing sequential testing framework...")
        # No specific initialization needed
        logger.info("Sequential testing framework initialized successfully")

    async def cleanup(self):
        """Cleanup sequential testing framework resources"""
        logger.info("Cleaning up sequential testing framework...")
        self.active_tests.clear()
        logger.info("Sequential testing framework cleanup completed")

    async def create_sequential_test(self, test_config: Dict[str, Any]) -> str:
        """Create a new sequential test"""

        test_id = str(uuid4())
        method = test_config.get("method", "wald_sequential")

        if method not in self.sequential_methods:
            raise ValueError(f"Unsupported sequential method: {method}")

        test = {
            "id": test_id,
            "method": method,
            "parameters": test_config.get("parameters", {}),
            "alpha": test_config.get("alpha", 0.05),
            "power": test_config.get("power", 0.8),
            "effect_size": test_config.get("effect_size", 0.5),
            "max_samples": test_config.get("max_samples", 10000),
            "interim_analyses": [],
            "current_sample_size": 0,
            "control_data": [],
            "treatment_data": [],
            "created_at": datetime.utcnow(),
            "status": "active",
            "stopping_decision": None,
        }

        self.active_tests[test_id] = test

        logger.info(f"Created sequential test {test_id} with method {method}")
        return test_id

    async def add_data(
        self, test_id: str, control_value: float, treatment_value: float
    ) -> Dict[str, Any]:
        """Add data point to sequential test"""

        if test_id not in self.active_tests:
            raise ValueError(f"Sequential test {test_id} not found")

        test = self.active_tests[test_id]

        if test["status"] != "active":
            return {"status": "stopped", "reason": "Test already stopped"}

        # Add data
        test["control_data"].append(control_value)
        test["treatment_data"].append(treatment_value)
        test["current_sample_size"] += 1

        # Perform interim analysis
        interim_result = await self._perform_interim_analysis(test)
        test["interim_analyses"].append(interim_result)

        # Check stopping conditions
        stopping_decision = await self._check_stopping_conditions(test, interim_result)

        if stopping_decision["should_stop"]:
            test["status"] = "stopped"
            test["stopping_decision"] = stopping_decision

        return {
            "test_id": test_id,
            "current_sample_size": test["current_sample_size"],
            "interim_result": interim_result,
            "stopping_decision": stopping_decision,
            "status": test["status"],
        }

    async def analyze_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze sequential test results"""

        if test_id not in self.active_tests:
            raise ValueError(f"Sequential test {test_id} not found")

        test = self.active_tests[test_id]

        # Calculate final statistics
        control_data = np.array(test["control_data"])
        treatment_data = np.array(test["treatment_data"])

        if len(control_data) == 0 or len(treatment_data) == 0:
            return {
                "test_id": test_id,
                "status": "insufficient_data",
                "message": "Insufficient data for analysis",
            }

        # Basic statistics
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        effect_size = (treatment_mean - control_mean) / np.sqrt(
            (np.var(control_data) + np.var(treatment_data)) / 2
        )

        # Statistical test
        t_stat, p_value = stats.ttest_ind(treatment_data, control_data)

        # Power analysis
        power = self._calculate_power(effect_size, len(control_data), test["alpha"])

        return {
            "test_id": test_id,
            "method": test["method"],
            "status": test["status"],
            "stopping_decision": test["stopping_decision"],
            "final_statistics": {
                "control_mean": control_mean,
                "treatment_mean": treatment_mean,
                "effect_size": effect_size,
                "t_statistic": t_stat,
                "p_value": p_value,
                "power": power,
                "sample_size": len(control_data),
            },
            "interim_analyses": test["interim_analyses"],
            "analysis_timestamp": datetime.utcnow(),
        }

    async def _perform_interim_analysis(self, test: Dict[str, Any]) -> Dict[str, Any]:
        """Perform interim analysis"""

        method = test["method"]

        if method not in self.sequential_methods:
            raise ValueError(f"Unsupported method: {method}")

        return await self.sequential_methods[method](test)

    async def _wald_sequential_test(self, test: Dict[str, Any]) -> Dict[str, Any]:
        """Wald's Sequential Probability Ratio Test"""

        control_data = np.array(test["control_data"])
        treatment_data = np.array(test["treatment_data"])

        if len(control_data) < 2 or len(treatment_data) < 2:
            return {
                "sample_size": len(control_data),
                "test_statistic": 0,
                "p_value": 1.0,
                "stopping_boundary": None,
            }

        # Calculate test statistic
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)

        # Pooled variance
        pooled_var = (np.var(control_data, ddof=1) + np.var(treatment_data, ddof=1)) / 2

        # Wald statistic
        n = len(control_data)
        wald_stat = (treatment_mean - control_mean) / math.sqrt(2 * pooled_var / n)

        # Stopping boundaries
        alpha = test["alpha"]
        beta = 1 - test["power"]

        # Approximate boundaries
        a = math.log(beta / (1 - alpha))
        b = math.log((1 - beta) / alpha)

        # Check stopping conditions
        should_stop = wald_stat <= a or wald_stat >= b

        return {
            "sample_size": n,
            "test_statistic": wald_stat,
            "p_value": 2 * (1 - stats.norm.cdf(abs(wald_stat))),
            "stopping_boundary": {"lower": a, "upper": b},
            "should_stop": should_stop,
            "decision": "stop" if should_stop else "continue",
        }

    async def _group_sequential_test(self, test: Dict[str, Any]) -> Dict[str, Any]:
        """Group Sequential Test (O'Brien-Fleming)"""

        control_data = np.array(test["control_data"])
        treatment_data = np.array(test["treatment_data"])

        if len(control_data) < 2 or len(treatment_data) < 2:
            return {
                "sample_size": len(control_data),
                "test_statistic": 0,
                "p_value": 1.0,
                "stopping_boundary": None,
            }

        # Calculate test statistic
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        pooled_std = math.sqrt((np.var(control_data, ddof=1) + np.var(treatment_data, ddof=1)) / 2)

        n = len(control_data)
        t_stat = (treatment_mean - control_mean) / (pooled_std * math.sqrt(2 / n))

        # O'Brien-Fleming boundaries
        alpha = test["alpha"]
        max_analyses = test.get("max_analyses", 5)

        # Calculate information fraction
        information_fraction = n / test["max_samples"]

        # O'Brien-Fleming critical values
        if information_fraction <= 0.2:
            critical_value = 4.56  # Very conservative early
        elif information_fraction <= 0.4:
            critical_value = 3.23
        elif information_fraction <= 0.6:
            critical_value = 2.63
        elif information_fraction <= 0.8:
            critical_value = 2.28
        else:
            critical_value = 1.96  # Final analysis

        should_stop = abs(t_stat) >= critical_value

        return {
            "sample_size": n,
            "test_statistic": t_stat,
            "p_value": 2 * (1 - stats.norm.cdf(abs(t_stat))),
            "stopping_boundary": {
                "critical_value": critical_value,
                "information_fraction": information_fraction,
            },
            "should_stop": should_stop,
            "decision": "stop" if should_stop else "continue",
        }

    async def _alpha_spending_test(self, test: Dict[str, Any]) -> Dict[str, Any]:
        """Alpha Spending Function Test"""

        control_data = np.array(test["control_data"])
        treatment_data = np.array(test["treatment_data"])

        if len(control_data) < 2 or len(treatment_data) < 2:
            return {
                "sample_size": len(control_data),
                "test_statistic": 0,
                "p_value": 1.0,
                "stopping_boundary": None,
            }

        # Calculate test statistic
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        pooled_std = math.sqrt((np.var(control_data, ddof=1) + np.var(treatment_data, ddof=1)) / 2)

        n = len(control_data)
        t_stat = (treatment_mean - control_mean) / (pooled_std * math.sqrt(2 / n))

        # Alpha spending function (Pocock)
        alpha = test["alpha"]
        information_fraction = n / test["max_samples"]

        # Pocock alpha spending
        spent_alpha = alpha * information_fraction

        # Critical value for spent alpha
        critical_value = stats.norm.ppf(1 - spent_alpha / 2)

        should_stop = abs(t_stat) >= critical_value

        return {
            "sample_size": n,
            "test_statistic": t_stat,
            "p_value": 2 * (1 - stats.norm.cdf(abs(t_stat))),
            "stopping_boundary": {
                "critical_value": critical_value,
                "spent_alpha": spent_alpha,
                "information_fraction": information_fraction,
            },
            "should_stop": should_stop,
            "decision": "stop" if should_stop else "continue",
        }

    async def _check_stopping_conditions(
        self, test: Dict[str, Any], interim_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if test should stop"""

        # Check if interim analysis suggests stopping
        if interim_result.get("should_stop", False):
            return {
                "should_stop": True,
                "reason": "interim_analysis",
                "decision": interim_result.get("decision", "stop"),
            }

        # Check if maximum sample size reached
        if test["current_sample_size"] >= test["max_samples"]:
            return {"should_stop": True, "reason": "max_samples_reached", "decision": "stop"}

        return {"should_stop": False, "reason": "continue", "decision": "continue"}

    def _calculate_power(self, effect_size: float, sample_size: int, alpha: float) -> float:
        """Calculate statistical power"""

        # Approximate power calculation for two-sample t-test
        ncp = effect_size * math.sqrt(sample_size / 2)  # Non-centrality parameter
        critical_value = stats.norm.ppf(1 - alpha / 2)

        power = 1 - stats.norm.cdf(critical_value - ncp) + stats.norm.cdf(-critical_value - ncp)

        return max(0, min(1, power))

    async def get_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get sequential test by ID"""
        return self.active_tests.get(test_id)

    async def list_tests(self) -> List[Dict[str, Any]]:
        """List all active sequential tests"""
        return list(self.active_tests.values())

    async def stop_test(self, test_id: str) -> bool:
        """Stop a sequential test"""
        if test_id not in self.active_tests:
            return False

        self.active_tests[test_id]["status"] = "stopped"
        return True
