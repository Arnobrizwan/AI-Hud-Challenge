"""
Core Evaluation Engine - Main orchestration system
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .business_impact import BusinessImpactAnalyzer
from .causal_inference import CausalInferenceAnalyzer
from .drift_detection import ModelDriftDetector
from .mlflow_client import MLflowClient
from .models import EvaluationConfig, EvaluationResults, EvaluationStatus, EvaluationType, ModelType
from .offline_evaluation import OfflineEvaluator
from .online_evaluation import OnlineEvaluator
from .statistical_testing import StatisticalTester

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Comprehensive evaluation system for ML pipelines"""

    def __init__(self):
        self.offline_evaluator = OfflineEvaluator()
        self.online_evaluator = OnlineEvaluator()
        self.ab_tester = None  # Will be initialized in online_evaluator
        self.business_impact_analyzer = BusinessImpactAnalyzer()
        self.drift_detector = ModelDriftDetector()
        self.causal_analyzer = CausalInferenceAnalyzer()
        self.mlflow_client = MLflowClient()
        self.statistical_tester = StatisticalTester()

        # Evaluation tracking
        self.active_evaluations: Dict[str, EvaluationResults] = {}
        self.evaluation_history: List[EvaluationResults] = []

    async def initialize(self) -> Dict[str, Any]:
    """Initialize the evaluation engine"""
        try:
            logger.info("Initializing evaluation engine...")

            # Initialize components
            await self.offline_evaluator.initialize()
            await self.online_evaluator.initialize()
            await self.business_impact_analyzer.initialize()
            await self.drift_detector.initialize()
            await self.causal_analyzer.initialize()
            await self.mlflow_client.initialize()
            await self.statistical_tester.initialize()

            # Get A/B tester from online evaluator
            self.ab_tester = self.online_evaluator.ab_tester

            logger.info("Evaluation engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize evaluation engine: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
    """Cleanup evaluation engine resources"""
        try:
            logger.info("Cleaning up evaluation engine...")

            # Cancel active evaluations
            for evaluation_id in list(self.active_evaluations.keys()):
    await self.cancel_evaluation(evaluation_id)

            # Cleanup components
            await self.offline_evaluator.cleanup()
            await self.online_evaluator.cleanup()
            await self.business_impact_analyzer.cleanup()
            await self.drift_detector.cleanup()
            await self.causal_analyzer.cleanup()
            await self.mlflow_client.cleanup()
            await self.statistical_tester.cleanup()

            logger.info("Evaluation engine cleanup completed")

        except Exception as e:
            logger.error(f"Error during evaluation engine cleanup: {str(e)}")

    async def run_comprehensive_evaluation(
        self, evaluation_config: EvaluationConfig
    ) -> EvaluationResults:
        """Run complete evaluation suite"""

        evaluation_id = str(uuid4())

        results = EvaluationResults(
            evaluation_id=evaluation_id,
            config=evaluation_config,
            status=EvaluationStatus.RUNNING,
            started_at=datetime.utcnow(),
            created_by=evaluation_config.created_by,
        )

        # Track active evaluation
        self.active_evaluations[evaluation_id] = results

        try:
            logger.info(f"Starting comprehensive evaluation {evaluation_id}")

            # Offline evaluation
            if evaluation_config.include_offline:
                logger.info("Running offline evaluation...")
                offline_results = await self.offline_evaluator.evaluate(
                    evaluation_config.models, evaluation_config.datasets, evaluation_config.metrics
                )
                results.offline_results = offline_results
                logger.info("Offline evaluation completed")

            # Online evaluation
            if evaluation_config.include_online:
                logger.info("Running online evaluation...")
                online_results = await self.online_evaluator.evaluate(
                    evaluation_config.online_experiments, evaluation_config.evaluation_period
                )
                results.online_results = online_results
                logger.info("Online evaluation completed")

            # Business impact analysis
            if evaluation_config.include_business_impact:
                logger.info("Running business impact analysis...")
                business_impact = await self.business_impact_analyzer.analyze(
                    evaluation_config.business_metrics, evaluation_config.evaluation_period
                )
                results.business_impact = business_impact
                logger.info("Business impact analysis completed")

            # Model drift analysis
            if evaluation_config.include_drift_analysis:
                logger.info("Running drift analysis...")
                drift_results = await self.drift_detector.analyze_drift(
                    evaluation_config.models, evaluation_config.drift_config
                )
                results.drift_analysis = drift_results
                logger.info("Drift analysis completed")

            # Causal impact analysis
            if evaluation_config.include_causal_analysis:
                logger.info("Running causal analysis...")
                causal_results = await self.causal_analyzer.analyze_causal_impact(
                    evaluation_config.causal_config
                )
                results.causal_analysis = causal_results
                logger.info("Causal analysis completed")

            # Generate recommendations
            logger.info("Generating recommendations...")
            results.recommendations = await self.generate_recommendations(results)

            # Mark as completed
            results.status = EvaluationStatus.COMPLETED
            results.completed_at = datetime.utcnow()

            # Calculate duration
            if results.started_at and results.completed_at:
                results.duration_seconds = (
                    results.completed_at - results.started_at
                ).total_seconds()

            # Store results in MLflow
            await self.log_evaluation_results(results)

            # Move to history
            self.evaluation_history.append(results)
            if evaluation_id in self.active_evaluations:
                del self.active_evaluations[evaluation_id]

            logger.info(
                f"Comprehensive evaluation {evaluation_id} completed successfully")

        except Exception as e:
            logger.error(
                f"Error in comprehensive evaluation {evaluation_id}: {str(e)}")
            results.status = EvaluationStatus.FAILED
            results.error_message = str(e)
            results.completed_at = datetime.utcnow()

            # Move to history even if failed
            self.evaluation_history.append(results)
            if evaluation_id in self.active_evaluations:
                del self.active_evaluations[evaluation_id]

            raise

        return results

    async def run_offline_evaluation(
        self, models: List[Dict[str, Any]], datasets: List[Dict[str, Any]], metrics: Dict[str, Any]
    ) -> EvaluationResults:
        """Run offline evaluation only"""

        evaluation_id = str(uuid4())

        results = EvaluationResults(
            evaluation_id=evaluation_id,
            config=EvaluationConfig(
                include_offline=True,
                include_online=False,
                include_business_impact=False,
                include_drift_analysis=False,
                include_causal_analysis=False,
                models=models,
                datasets=datasets,
                metrics=metrics,
            ),
            status=EvaluationStatus.RUNNING,
            started_at=datetime.utcnow(),
        )

        try:
            logger.info(f"Starting offline evaluation {evaluation_id}")

            offline_results = await self.offline_evaluator.evaluate(models, datasets, metrics)
            results.offline_results = offline_results

            results.status = EvaluationStatus.COMPLETED
            results.completed_at = datetime.utcnow()

            if results.started_at and results.completed_at:
                results.duration_seconds = (
                    results.completed_at - results.started_at
                ).total_seconds()

            await self.log_evaluation_results(results)

            logger.info(
                f"Offline evaluation {evaluation_id} completed successfully")

        except Exception as e:
            logger.error(
                f"Error in offline evaluation {evaluation_id}: {str(e)}")
            results.status = EvaluationStatus.FAILED
            results.error_message = str(e)
            results.completed_at = datetime.utcnow()
            raise

        return results

    async def run_online_evaluation(
        self, experiments: List[Dict[str, Any]], evaluation_period: Dict[str, Any]
    ) -> EvaluationResults:
        """Run online evaluation only"""

        evaluation_id = str(uuid4())

        results = EvaluationResults(
            evaluation_id=evaluation_id,
            config=EvaluationConfig(
                include_offline=False,
                include_online=True,
                include_business_impact=False,
                include_drift_analysis=False,
                include_causal_analysis=False,
                online_experiments=experiments,
                evaluation_period=evaluation_period,
            ),
            status=EvaluationStatus.RUNNING,
            started_at=datetime.utcnow(),
        )

        try:
            logger.info(f"Starting online evaluation {evaluation_id}")

            online_results = await self.online_evaluator.evaluate(experiments, evaluation_period)
            results.online_results = online_results

            results.status = EvaluationStatus.COMPLETED
            results.completed_at = datetime.utcnow()

            if results.started_at and results.completed_at:
                results.duration_seconds = (
                    results.completed_at - results.started_at
                ).total_seconds()

            await self.log_evaluation_results(results)

            logger.info(
                f"Online evaluation {evaluation_id} completed successfully")

        except Exception as e:
            logger.error(
                f"Error in online evaluation {evaluation_id}: {str(e)}")
            results.status = EvaluationStatus.FAILED
            results.error_message = str(e)
            results.completed_at = datetime.utcnow()
            raise

        return results

    async def cancel_evaluation(self, evaluation_id: str) -> bool:
        """Cancel a running evaluation"""

        if evaluation_id not in self.active_evaluations:
            return False

        evaluation = self.active_evaluations[evaluation_id]
        evaluation.status = EvaluationStatus.CANCELLED
        evaluation.completed_at = datetime.utcnow()

        # Move to history
        self.evaluation_history.append(evaluation)
        del self.active_evaluations[evaluation_id]

        logger.info(f"Evaluation {evaluation_id} cancelled")
        return True

    async def get_evaluation_status(
            self, evaluation_id: str) -> Optional[EvaluationResults]:
        """Get evaluation status"""

        # Check active evaluations
        if evaluation_id in self.active_evaluations:
            return self.active_evaluations[evaluation_id]

        # Check history
        for evaluation in self.evaluation_history:
            if evaluation.evaluation_id == evaluation_id:
                return evaluation

        return None

    async def list_evaluations(
            self,
            status: Optional[EvaluationStatus] = None,
            limit: int = 100,
            offset: int = 0) -> List[EvaluationResults]:
        """List evaluations with optional filtering"""

        all_evaluations = list(
            self.active_evaluations.values()) + self.evaluation_history

        if status:
            all_evaluations = [
                e for e in all_evaluations if e.status == status]

        # Sort by started_at descending
        all_evaluations.sort(
            key=lambda x: x.started_at or datetime.min,
            reverse=True)

        return all_evaluations[offset: offset + limit]

    async def generate_recommendations(
            self, results: EvaluationResults) -> List[Dict[str, Any]]:
        """Generate recommendations based on evaluation results"""

        recommendations = []

        # Offline evaluation recommendations
        if results.offline_results:
            offline_recs = await self._generate_offline_recommendations(results.offline_results)
            recommendations.extend(offline_recs)

        # Online evaluation recommendations
        if results.online_results:
            online_recs = await self._generate_online_recommendations(results.online_results)
            recommendations.extend(online_recs)

        # Business impact recommendations
        if results.business_impact:
            business_recs = await self._generate_business_recommendations(results.business_impact)
            recommendations.extend(business_recs)

        # Drift analysis recommendations
        if results.drift_analysis:
            drift_recs = await self._generate_drift_recommendations(results.drift_analysis)
            recommendations.extend(drift_recs)

        return recommendations

    async def _generate_offline_recommendations(
            self, offline_results) -> List[Dict[str, Any]]:
        """Generate recommendations from offline evaluation results"""
        recommendations = []

        # Add model performance recommendations
        for model_result in offline_results.model_results:
            if model_result.overall_score < 0.7:
                recommendations.append(
                    {
                        "type": "model_performance",
                        "priority": "high",
                        "title": f"Low performance for {model_result.model_name}",
                        "description": f"Model {model_result.model_name} has overall score {model_result.overall_score:.3f}",
                        "action": "Consider retraining or hyperparameter tuning",
                    })

        return recommendations

    async def _generate_online_recommendations(
            self, online_results) -> List[Dict[str, Any]]:
        """Generate recommendations from online evaluation results"""
        recommendations = []

        # Add A/B test recommendations
        for experiment in online_results.experiments:
            if experiment.status == "completed":
                if experiment.winner_variant:
                    recommendations.append(
                        {
                            "type": "ab_test_winner",
                            "priority": "medium",
                            "title": f"Winner found for experiment {experiment.experiment_id}",
                            "description": f"Variant {experiment.winner_variant} won with {experiment.winner_confidence:.3f} confidence",
                            "action": "Consider deploying winning variant",
                        })

        return recommendations

    async def _generate_business_recommendations(
            self, business_impact) -> List[Dict[str, Any]]:
        """Generate recommendations from business impact analysis"""
        recommendations = []

        # Add ROI recommendations
        if business_impact.overall_roi < 1.0:
            recommendations.append(
                {
                    "type": "low_roi",
                    "priority": "high",
                    "title": "Low ROI detected",
                    "description": f"Overall ROI is {business_impact.overall_roi:.3f}",
                    "action": "Review model performance and business metrics",
                })

        return recommendations

    async def _generate_drift_recommendations(
            self, drift_analysis) -> List[Dict[str, Any]]:
        """Generate recommendations from drift analysis"""
        recommendations = []

        # Add drift recommendations
        if drift_analysis.drift_severity > 0.7:
            recommendations.append(
                {
                    "type": "model_drift",
                    "priority": "high",
                    "title": "Significant model drift detected",
                    "description": f"Drift severity: {drift_analysis.drift_severity:.3f}",
                    "action": "Retrain model or investigate data changes",
                })

        return recommendations

    async def log_evaluation_results(self, results: EvaluationResults) -> Dict[str, Any]:
    """Log evaluation results to MLflow"""
        try:
    await self.mlflow_client.log_evaluation_results(results)
            logger.info(
                f"Logged evaluation results {results.evaluation_id} to MLflow")
        except Exception as e:
            logger.error(
                f"Failed to log evaluation results to MLflow: {str(e)}")
            # Don't raise - logging failure shouldn't break evaluation
