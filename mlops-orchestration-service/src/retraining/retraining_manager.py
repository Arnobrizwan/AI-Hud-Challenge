"""
Automated Retraining Manager - Manages automated model retraining based on triggers
Supports performance degradation, data drift, scheduled, and data volume triggers
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from src.config.settings import Settings
from src.deployment.deployment_manager import ModelDeploymentManager
from src.feature_store.feature_store_manager import FeatureStoreManager
from src.models.retraining_models import (
    DataDriftTrigger,
    DataVolumeTrigger,
    PerformanceTrigger,
    RetrainingResult,
    RetrainingStatus,
    RetrainingTrigger,
    RetrainingTriggerConfig,
    ScheduledTrigger,
    TriggerStatus,
)
from src.monitoring.monitoring_service import ModelMonitoringService
from src.registry.model_registry import ModelRegistry
from src.training.training_orchestrator import ModelTrainingOrchestrator
from src.utils.exceptions import RetrainingError, ValidationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class AutomatedRetrainingManager:
    """Manage automated model retraining based on triggers"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.trigger_monitor = RetrainingTriggerMonitor()
        self.data_monitor = DataQualityMonitor()
        self.performance_monitor = ModelPerformanceMonitor()
        self.training_orchestrator = ModelTrainingOrchestrator(settings)
        self.model_registry = ModelRegistry(settings)
        self.monitoring_service = ModelMonitoringService(settings)
        self.feature_store_manager = FeatureStoreManager(settings)
        self.deployment_manager = ModelDeploymentManager(settings)
        self.scheduler = RetrainingScheduler()

        # Retraining state
        self._active_triggers: Dict[str, RetrainingTrigger] = {}
        self._retraining_jobs: Dict[str, RetrainingResult] = {}
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}

    async def initialize(self) -> None:
        """Initialize the retraining manager"""
        try:
            logger.info("Initializing Automated Retraining Manager...")

            await self.trigger_monitor.initialize()
            await self.data_monitor.initialize()
            await self.performance_monitor.initialize()
            await self.training_orchestrator.initialize()
            await self.model_registry.initialize()
            await self.monitoring_service.initialize()
            await self.feature_store_manager.initialize()
            await self.deployment_manager.initialize()
            await self.scheduler.initialize()

            logger.info("Automated Retraining Manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Automated Retraining Manager: {str(e)}")
            raise

    async def setup_retraining_triggers(self, model_name: str, trigger_config: RetrainingTriggerConfig) -> None:
        """Set up automated retraining triggers"""

        try:
            logger.info(f"Setting up retraining triggers for model: {model_name}")

            triggers = []

            # Performance degradation trigger
            if trigger_config.performance_threshold:
                performance_trigger = PerformanceTrigger(
                    id=str(uuid4()),
                    model_name=model_name,
                    metric=trigger_config.performance_metric,
                    threshold=trigger_config.performance_threshold,
                    evaluation_window=trigger_config.evaluation_window,
                    status=TriggerStatus.ACTIVE,
                    created_at=datetime.utcnow(),
                )
                triggers.append(performance_trigger)

            # Data drift trigger
            if trigger_config.data_drift_threshold:
                drift_trigger = DataDriftTrigger(
                    id=str(uuid4()),
                    model_name=model_name,
                    drift_threshold=trigger_config.data_drift_threshold,
                    monitoring_features=trigger_config.drift_monitoring_features,
                    status=TriggerStatus.ACTIVE,
                    created_at=datetime.utcnow(),
                )
                triggers.append(drift_trigger)

            # Time-based trigger
            if trigger_config.retraining_schedule:
                scheduled_trigger = ScheduledTrigger(
                    id=str(uuid4()),
                    model_name=model_name,
                    schedule=trigger_config.retraining_schedule,
                    status=TriggerStatus.ACTIVE,
                    created_at=datetime.utcnow(),
                )
                triggers.append(scheduled_trigger)

            # Data volume trigger
            if trigger_config.new_data_threshold:
                data_volume_trigger = DataVolumeTrigger(
                    id=str(uuid4()),
                    model_name=model_name,
                    threshold=trigger_config.new_data_threshold,
                    status=TriggerStatus.ACTIVE,
                    created_at=datetime.utcnow(),
                )
                triggers.append(data_volume_trigger)

            # Register triggers
            for trigger in triggers:
                self._active_triggers[trigger.id] = trigger
                await self.trigger_monitor.register_trigger(trigger)

            # Start monitoring tasks
            await self._start_trigger_monitoring(model_name, triggers)

            logger.info(f"Set up {len(triggers)} retraining triggers for {model_name}")

        except Exception as e:
            logger.error(f"Failed to setup retraining triggers: {str(e)}")
            raise RetrainingError(f"Trigger setup failed: {str(e)}")

    async def start_trigger_monitoring(self) -> None:
        """Start monitoring all active triggers"""

        try:
            logger.info("Starting trigger monitoring...")

            # Start monitoring task for each model
            models = set(trigger.model_name for trigger in self._active_triggers.values())

            for model_name in models:
                model_triggers = [
                    trigger for trigger in self._active_triggers.values() if trigger.model_name == model_name
                ]

                task = asyncio.create_task(self._monitor_model_triggers(model_name, model_triggers))
                self._monitoring_tasks[model_name] = task

            logger.info(f"Started monitoring for {len(models)} models")

        except Exception as e:
            logger.error(f"Failed to start trigger monitoring: {str(e)}")
            raise RetrainingError(f"Trigger monitoring startup failed: {str(e)}")

    async def stop_trigger_monitoring(self) -> None:
        """Stop monitoring all triggers"""

        try:
            logger.info("Stopping trigger monitoring...")

            # Cancel all monitoring tasks
            for task in self._monitoring_tasks.values():
                task.cancel()

            # Wait for tasks to complete
            await asyncio.gather(*self._monitoring_tasks.values(), return_exceptions=True)

            self._monitoring_tasks.clear()

            logger.info("Trigger monitoring stopped")

        except Exception as e:
            logger.error(f"Failed to stop trigger monitoring: {str(e)}")

    async def check_retraining_triggers(self) -> None:
        """Check all active retraining triggers"""

        try:
            for trigger in self._active_triggers.values():
                if trigger.status != TriggerStatus.ACTIVE:
                    continue

                trigger_fired = await self._evaluate_trigger(trigger)

                if trigger_fired:
                    await self._initiate_retraining(trigger)

        except Exception as e:
            logger.error(f"Failed to check retraining triggers: {str(e)}")

    async def _monitor_model_triggers(self, model_name: str, triggers: List[RetrainingTrigger]) -> None:
        """Monitor triggers for a specific model"""

        while True:
            try:
                for trigger in triggers:
                    if trigger.status != TriggerStatus.ACTIVE:
                        continue

                    trigger_fired = await self._evaluate_trigger(trigger)

                    if trigger_fired:
                        await self._initiate_retraining(trigger)

                # Wait before next check
                await asyncio.sleep(self.settings.retraining_check_interval)

            except asyncio.CancelledError:
                logger.info(f"Trigger monitoring cancelled for model: {model_name}")
                break
            except Exception as e:
                logger.error(f"Error in trigger monitoring for {model_name}: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _evaluate_trigger(self, trigger: RetrainingTrigger) -> bool:
        """Evaluate if a trigger should fire"""

        try:
            if isinstance(trigger, PerformanceTrigger):
                return await self._evaluate_performance_trigger(trigger)
            elif isinstance(trigger, DataDriftTrigger):
                return await self._evaluate_drift_trigger(trigger)
            elif isinstance(trigger, ScheduledTrigger):
                return await self._evaluate_scheduled_trigger(trigger)
            elif isinstance(trigger, DataVolumeTrigger):
                return await self._evaluate_data_volume_trigger(trigger)
            else:
                logger.warning(f"Unknown trigger type: {type(trigger)}")
                return False

        except Exception as e:
            logger.error(f"Failed to evaluate trigger {trigger.id}: {str(e)}")
            return False

    async def _evaluate_performance_trigger(self, trigger: PerformanceTrigger) -> bool:
        """Evaluate performance degradation trigger"""

        try:
            # Get recent performance metrics
            metrics = await self.performance_monitor.get_recent_metrics(
                model_name=trigger.model_name,
                metric=trigger.metric,
                window_minutes=trigger.evaluation_window,
            )

            if not metrics:
                return False

            # Calculate average performance
            avg_performance = sum(metrics) / len(metrics)

            # Check if performance is below threshold
            if avg_performance < trigger.threshold:
                logger.info(
                    f"Performance trigger fired for {trigger.model_name}: " f"{avg_performance} < {trigger.threshold}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to evaluate performance trigger: {str(e)}")
            return False

    async def _evaluate_drift_trigger(self, trigger: DataDriftTrigger) -> bool:
        """Evaluate data drift trigger"""

        try:
            # Get drift scores for monitoring features
            drift_scores = await self.data_monitor.calculate_drift_scores(
                model_name=trigger.model_name, features=trigger.monitoring_features
            )

            if not drift_scores:
                return False

            # Check if any feature exceeds drift threshold
            max_drift = max(drift_scores.values())

            if max_drift > trigger.drift_threshold:
                logger.info(
                    f"Data drift trigger fired for {trigger.model_name}: "
                    f"max drift {max_drift} > {trigger.drift_threshold}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to evaluate drift trigger: {str(e)}")
            return False

    async def _evaluate_scheduled_trigger(self, trigger: ScheduledTrigger) -> bool:
        """Evaluate scheduled trigger"""

        try:
            # Check if it's time for scheduled retraining
            return await self.scheduler.is_time_for_retraining(trigger)

        except Exception as e:
            logger.error(f"Failed to evaluate scheduled trigger: {str(e)}")
            return False

    async def _evaluate_data_volume_trigger(self, trigger: DataVolumeTrigger) -> bool:
        """Evaluate data volume trigger"""

        try:
            # Get new data volume since last retraining
            new_data_volume = await self.data_monitor.get_new_data_volume(model_name=trigger.model_name)

            if new_data_volume >= trigger.threshold:
                logger.info(
                    f"Data volume trigger fired for {trigger.model_name}: " f"{new_data_volume} >= {trigger.threshold}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to evaluate data volume trigger: {str(e)}")
            return False

    async def _initiate_retraining(self, trigger: RetrainingTrigger) -> None:
        """Initiate automated retraining process"""

        try:
            logger.info(f"Initiating retraining for {trigger.model_name} due to trigger: {trigger.type}")

            # Check if retraining is already in progress
            active_retraining = [
                job
                for job in self._retraining_jobs.values()
                if job.model_name == trigger.model_name and job.status == RetrainingStatus.RUNNING
            ]

            if active_retraining:
                logger.info(f"Retraining already in progress for {trigger.model_name}")
                return

            # Get current model configuration
            current_model = await self.model_registry.get_current_model(trigger.model_name)
            if not current_model:
                logger.error(f"No current model found for {trigger.model_name}")
                return

            # Prepare retraining configuration
            retraining_config = await self._prepare_retraining_config(current_model, trigger)

            # Create retraining result
            retraining_result = RetrainingResult(
                id=str(uuid4()),
                model_name=trigger.model_name,
                trigger_id=trigger.id,
                trigger_type=trigger.type,
                status=RetrainingStatus.RUNNING,
                started_at=datetime.utcnow(),
                config=retraining_config,
            )

            self._retraining_jobs[retraining_result.id] = retraining_result

            # Submit retraining job
            training_result = await self.training_orchestrator.execute_training_pipeline(retraining_config)

            retraining_result.training_result = training_result

            # Evaluate retrained model
            if training_result.registered_model_version:
                # Trigger A/B test to compare with current model
                await self._initiate_model_comparison(current_model, training_result.registered_model_version)

                retraining_result.status = RetrainingStatus.COMPLETED
            else:
                retraining_result.status = RetrainingStatus.FAILED
                retraining_result.error_message = "Model did not meet quality threshold"

            retraining_result.completed_at = datetime.utcnow()
            retraining_result.duration = retraining_result.completed_at - retraining_result.started_at

            logger.info(f"Retraining completed for {trigger.model_name}: {retraining_result.status}")

        except Exception as e:
            logger.error(f"Retraining initiation failed: {str(e)}")
            if retraining_result.id in self._retraining_jobs:
                self._retraining_jobs[retraining_result.id].status = RetrainingStatus.FAILED
                self._retraining_jobs[retraining_result.id].error_message = str(e)

    async def _prepare_retraining_config(self, current_model: Dict[str, Any], trigger: RetrainingTrigger) -> Any:
        """Prepare retraining configuration based on current model and trigger"""

        # This would create a TrainingConfig based on the current model
        # and any trigger-specific modifications
        return {
            "model_name": current_model["name"],
            "model_class": current_model["model_class"],
            "model_params": current_model["params"],
            "data_sources": current_model["data_sources"],
            "hyperparameter_tuning": True,  # Enable for retraining
            "quality_threshold": current_model.get("quality_threshold", 0.8),
        }

    async def _initiate_model_comparison(
        self, current_model: Dict[str, Any], new_model_version: Dict[str, Any]
    ) -> None:
        """Initiate A/B test comparison between current and new model"""

        try:
            logger.info(f"Initiating A/B test for {current_model['name']}")

            # Create A/B test configuration
            ab_test_config = {
                "model_name": current_model["name"],
                "control_version": current_model["version"],
                "treatment_version": new_model_version["version"],
                "control_traffic_percentage": 50,
                "treatment_traffic_percentage": 50,
                "test_duration_days": 7,
            }

            # Deploy A/B test
            deployment_config = {
                "model_name": current_model["name"],
                "model_version": new_model_version["version"],
                "strategy": "ab_test",
                "ab_test_config": ab_test_config,
            }

            await self.deployment_manager.deploy_model(deployment_config)

            logger.info(f"A/B test initiated for {current_model['name']}")

        except Exception as e:
            logger.error(f"A/B test initiation failed: {str(e)}")

    async def get_retraining_status(self, model_name: str) -> List[RetrainingResult]:
        """Get retraining status for a model"""

        return [job for job in self._retraining_jobs.values() if job.model_name == model_name]

    async def get_trigger_status(self, model_name: str) -> List[RetrainingTrigger]:
        """Get trigger status for a model"""

        return [trigger for trigger in self._active_triggers.values() if trigger.model_name == model_name]

    async def disable_trigger(self, trigger_id: str) -> bool:
        """Disable a retraining trigger"""

        try:
            trigger = self._active_triggers.get(trigger_id)
            if not trigger:
                return False

            trigger.status = TriggerStatus.DISABLED
            await self.trigger_monitor.update_trigger_status(trigger_id, TriggerStatus.DISABLED)

            logger.info(f"Trigger disabled: {trigger_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to disable trigger: {str(e)}")
            return False

    async def enable_trigger(self, trigger_id: str) -> bool:
        """Enable a retraining trigger"""

        try:
            trigger = self._active_triggers.get(trigger_id)
            if not trigger:
                return False

            trigger.status = TriggerStatus.ACTIVE
            await self.trigger_monitor.update_trigger_status(trigger_id, TriggerStatus.ACTIVE)

            logger.info(f"Trigger enabled: {trigger_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to enable trigger: {str(e)}")
            return False


# Supporting classes
class RetrainingTriggerMonitor:
    def __init__(self):
        self.triggers = {}

    async def initialize(self) -> Dict[str, Any]:
        pass

    async def register_trigger(self, trigger: RetrainingTrigger) -> Dict[str, Any]:
        """Register a retraining trigger"""
        self.triggers[trigger.id] = trigger

    async def update_trigger_status(self, trigger_id: str, status: TriggerStatus) -> Dict[str, Any]:
        """Update trigger status"""
        if trigger_id in self.triggers:
            self.triggers[trigger_id].status = status


class DataQualityMonitor:
    def __init__(self):
        pass

    async def initialize(self) -> Dict[str, Any]:
        pass

    async def calculate_drift_scores(self, model_name: str, features: List[str]) -> Dict[str, float]:
        """Calculate drift scores for features"""
        # Implement drift detection logic
        return {feature: 0.1 for feature in features}

    async def get_new_data_volume(self, model_name: str) -> int:
        """Get new data volume since last retraining"""
        # Implement data volume calculation
        return 1000


class ModelPerformanceMonitor:
    def __init__(self):
        pass

    async def initialize(self) -> Dict[str, Any]:
        pass

    async def get_recent_metrics(self, model_name: str, metric: str, window_minutes: int) -> List[float]:
        """Get recent performance metrics"""
        # Implement metrics retrieval
        return [0.85, 0.87, 0.83, 0.86, 0.84]


class RetrainingScheduler:
    def __init__(self):
        pass

    async def initialize(self) -> Dict[str, Any]:
        pass

    async def is_time_for_retraining(self, trigger: ScheduledTrigger) -> bool:
        """Check if it's time for scheduled retraining"""
        # Implement scheduling logic
        return False
