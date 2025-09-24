"""
MLflow Client - Integration with MLflow for experiment tracking
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.tensorflow

from .config import get_mlflow_config

logger = logging.getLogger(__name__)


class MLflowClient:
    """MLflow client for experiment tracking and model registry"""

    def __init__(self):
        self.tracking_uri = None
        self.registry_uri = None
        self.experiment_name = None
        self.active_run = None

    async def initialize(self):
        """Initialize MLflow client"""
        logger.info("Initializing MLflow client...")

        try:
            config = get_mlflow_config()
            self.tracking_uri = config["tracking_uri"]
            self.registry_uri = config["registry_uri"]
            self.experiment_name = config["experiment_name"]

            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)

            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(
                    f"Created MLflow experiment: {self.experiment_name} (ID: {experiment_id})"
                )
            else:
                logger.info(f"Using existing MLflow experiment: {self.experiment_name}")

            logger.info("MLflow client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MLflow client: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup MLflow client resources"""
        logger.info("Cleaning up MLflow client...")

        if self.active_run:
            mlflow.end_run()
            self.active_run = None

        logger.info("MLflow client cleanup completed")

    async def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> str:
        """Start a new MLflow run"""

        try:
            with mlflow.start_run(run_name=run_name) as run:
                self.active_run = run

                # Add tags
                if tags:
                    mlflow.set_tags(tags)

                # Add default tags
                mlflow.set_tag("evaluation_type", "comprehensive")
                mlflow.set_tag("timestamp", datetime.utcnow().isoformat())

                logger.info(f"Started MLflow run: {run.info.run_id}")
                return run.info.run_id

        except Exception as e:
            logger.error(f"Failed to start MLflow run: {str(e)}")
            raise

    async def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run"""

        if self.active_run:
            mlflow.end_run(status=status)
            self.active_run = None
            logger.info("Ended MLflow run")

    async def log_evaluation_results(self, results: Dict[str, Any]):
        """Log evaluation results to MLflow"""

        try:
            if not self.active_run:
                await self.start_run("evaluation_run")

            # Log metrics
            if "offline_results" in results:
                await self._log_offline_results(results["offline_results"])

            if "online_results" in results:
                await self._log_online_results(results["online_results"])

            if "business_impact" in results:
                await self._log_business_impact(results["business_impact"])

            if "drift_analysis" in results:
                await self._log_drift_analysis(results["drift_analysis"])

            # Log parameters
            mlflow.log_params(
                {
                    "evaluation_id": results.get("evaluation_id", "unknown"),
                    "evaluation_type": "comprehensive",
                    "timestamp": results.get("started_at", datetime.utcnow()).isoformat(),
                }
            )

            # Log tags
            mlflow.set_tags(
                {
                    "evaluation_status": results.get("status", "unknown"),
                    "evaluation_id": results.get("evaluation_id", "unknown"),
                }
            )

            logger.info("Logged evaluation results to MLflow")

        except Exception as e:
            logger.error(f"Failed to log evaluation results to MLflow: {str(e)}")
            raise

    async def _log_offline_results(self, offline_results: Dict[str, Any]):
        """Log offline evaluation results"""

        if "models" in offline_results:
            for model_result in offline_results["models"]:
                if "metrics" in model_result:
                    # Log model metrics
                    for metric_type, metrics in model_result["metrics"].items():
                        if isinstance(metrics, dict):
                            for metric_name, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    mlflow.log_metric(f"offline_{metric_type}_{metric_name}", value)
                        elif isinstance(metrics, (int, float)):
                            mlflow.log_metric(f"offline_{metric_type}", metrics)

    async def _log_online_results(self, online_results: Dict[str, Any]):
        """Log online evaluation results"""

        if "experiments" in online_results:
            for experiment in online_results["experiments"]:
                if "analysis" in experiment:
                    analysis = experiment["analysis"]
                    if "statistical_results" in analysis:
                        stats = analysis["statistical_results"]
                        if "variant_results" in stats:
                            for variant, result in stats["variant_results"].items():
                                if "primary_metric_result" in result:
                                    primary = result["primary_metric_result"]
                                    if "p_value" in primary:
                                        mlflow.log_metric(
                                            f"online_{variant}_p_value", primary["p_value"]
                                        )
                                    if "test_statistic" in primary:
                                        mlflow.log_metric(
                                            f"online_{variant}_test_statistic",
                                            primary["test_statistic"],
                                        )

    async def _log_business_impact(self, business_impact: Dict[str, Any]):
        """Log business impact results"""

        if "overall_roi" in business_impact:
            mlflow.log_metric("business_roi", business_impact["overall_roi"])

        if "metric_impacts" in business_impact:
            for metric, impact in business_impact["metric_impacts"].items():
                if "revenue_change" in impact:
                    mlflow.log_metric(f"business_{metric}_revenue_change", impact["revenue_change"])
                if "engagement_change" in impact:
                    mlflow.log_metric(
                        f"business_{metric}_engagement_change", impact["engagement_change"]
                    )

    async def _log_drift_analysis(self, drift_analysis: Dict[str, Any]):
        """Log drift analysis results"""

        if "drift_severity" in drift_analysis:
            mlflow.log_metric("drift_severity", drift_analysis["drift_severity"])

        if "drift_results" in drift_analysis:
            for result_name, result in drift_analysis["drift_results"].items():
                if isinstance(result, dict) and "overall_drift_score" in result:
                    mlflow.log_metric(f"drift_{result_name}_score", result["overall_drift_score"])

    async def log_model(self, model, model_name: str, model_version: str = "1.0.0"):
        """Log model to MLflow"""

        try:
            if not self.active_run:
                await self.start_run(f"model_logging_{model_name}")

            # Log model based on type
            if hasattr(model, "predict"):
                # Sklearn model
                mlflow.sklearn.log_model(model, "model")
            elif hasattr(model, "forward"):
                # PyTorch model
                mlflow.pytorch.log_model(model, "model")
            else:
                # Generic model
                mlflow.log_artifact(model, "model")

            # Log model metadata
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "model_version": model_version,
                    "model_type": type(model).__name__,
                }
            )

            logger.info(f"Logged model {model_name} to MLflow")

        except Exception as e:
            logger.error(f"Failed to log model to MLflow: {str(e)}")
            raise

    async def get_experiment_runs(self, experiment_name: str = None) -> List[Dict[str, Any]]:
        """Get experiment runs"""

        try:
            if experiment_name:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if not experiment:
                    return []
                experiment_id = experiment.experiment_id
            else:
                experiment_id = None

            runs = mlflow.search_runs(experiment_ids=[experiment_id] if experiment_id else None)

            return runs.to_dict("records")

        except Exception as e:
            logger.error(f"Failed to get experiment runs: {str(e)}")
            return []

    async def get_run_metrics(self, run_id: str) -> Dict[str, Any]:
        """Get metrics for a specific run"""

        try:
            run = mlflow.get_run(run_id)
            return run.data.metrics

        except Exception as e:
            logger.error(f"Failed to get run metrics: {str(e)}")
            return {}

    async def get_run_params(self, run_id: str) -> Dict[str, Any]:
        """Get parameters for a specific run"""

        try:
            run = mlflow.get_run(run_id)
            return run.data.params

        except Exception as e:
            logger.error(f"Failed to get run parameters: {str(e)}")
            return {}

    async def get_run_tags(self, run_id: str) -> Dict[str, Any]:
        """Get tags for a specific run"""

        try:
            run = mlflow.get_run(run_id)
            return run.data.tags

        except Exception as e:
            logger.error(f"Failed to get run tags: {str(e)}")
            return {}
