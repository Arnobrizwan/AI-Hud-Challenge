"""
Model Training Orchestrator - Manages model training with hyperparameter optimization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import joblib
import numpy as np
import optuna
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config.settings import Settings
from src.feature_store.feature_store_manager import FeatureStoreManager
from src.infrastructure.resource_manager import ResourceManager
from src.models.training_models import (
    ExperimentConfig,
    HyperparameterSpace,
    ModelEvaluationResult,
    ModelTrainingResult,
    TrainingConfig,
    TrainingData,
    TrainingResult,
)
from src.monitoring.monitoring_service import ModelMonitoringService
from src.orchestration.vertex_ai_client import VertexAIPipelineClient
from src.registry.model_registry import ModelRegistry
from src.utils.exceptions import TrainingError, ValidationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelTrainingOrchestrator:
    """Orchestrate model training with hyperparameter optimization"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.vertex_ai = VertexAIPipelineClient(settings)
        self.model_registry = ModelRegistry(settings)
        self.monitoring_service = ModelMonitoringService(settings)
        self.feature_store_manager = FeatureStoreManager(settings)
        self.resource_manager = ResourceManager(settings)

        # Training state
        self._active_training_jobs: Dict[str, TrainingResult] = {}
        self._hyperparameter_trials: Dict[str, List[Dict]] = {}

    async def initialize(self) -> None:
        """Initialize the training orchestrator"""
        try:
            logger.info("Initializing Model Training Orchestrator...")

            await self.vertex_ai.initialize()
            await self.model_registry.initialize()
            await self.monitoring_service.initialize()
            await self.feature_store_manager.initialize()
            await self.resource_manager.initialize()

            logger.info("Model Training Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Model Training Orchestrator: {str(e)}")
            raise

    async def execute_training_pipeline(self, training_config: TrainingConfig) -> TrainingResult:
        """Execute complete model training pipeline"""

        try:
            logger.info(f"Starting training pipeline for model: {training_config.model_name}")

            # Create training result
            training_result = TrainingResult(
                id=str(uuid4()),
                model_name=training_config.model_name,
                config=training_config,
                status="running",
                started_at=datetime.utcnow(),
            )

            self._active_training_jobs[training_result.id] = training_result

            # Create MLflow experiment
            experiment = await self._create_experiment(training_config)
            training_result.experiment_id = experiment.id

            # Prepare training data
            training_data = await self.prepare_training_data(training_config)
            training_result.training_data_info = {
                "train_size": len(training_data.train_set),
                "validation_size": len(training_data.validation_set),
                "test_size": len(training_data.test_set),
                "feature_count": training_data.feature_count,
            }

            # Execute hyperparameter tuning if configured
            if training_config.enable_hyperparameter_tuning:
                logger.info("Starting hyperparameter optimization...")
                best_params = await self._optimize_hyperparameters(training_config, training_data, experiment.id)
                training_config.model_params.update(best_params)
                training_result.best_hyperparameters = best_params

            # Execute training with best parameters
            model_result = await self._train_model(training_config, training_data, experiment.id)
            training_result.model_result = model_result

            # Evaluate trained model
            evaluation_result = await self._evaluate_trained_model(model_result.model, training_data, training_config)
            training_result.evaluation_result = evaluation_result

            # Register model if meets criteria
            if evaluation_result.meets_quality_threshold(training_config.quality_threshold):
                model_version = await self._register_model(
                    model_result, evaluation_result, experiment.id, training_config
                )
                training_result.registered_model_version = model_version
                training_result.status = "completed"
            else:
                training_result.status = "failed"
                training_result.error_message = "Model did not meet quality threshold"

            training_result.completed_at = datetime.utcnow()
            training_result.duration = training_result.completed_at - training_result.started_at

            # Cleanup
            del self._active_training_jobs[training_result.id]

            logger.info(f"Training pipeline completed: {training_result.id}")
            return training_result

        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            if training_result.id in self._active_training_jobs:
                self._active_training_jobs[training_result.id].status = "failed"
                self._active_training_jobs[training_result.id].error_message = str(e)
            raise TrainingError(f"Training pipeline failed: {str(e)}")

    async def prepare_training_data(self, config: TrainingConfig) -> TrainingData:
        """Prepare training data from various sources"""

        try:
            logger.info("Preparing training data...")

            # Get data from feature store
            if config.feature_store_config:
                feature_data = await self.feature_store_manager.get_training_features(config.feature_store_config)
            else:
                # Load data from configured sources
                feature_data = await self._load_training_data(config.data_sources)

            # Split data
            train_size = config.train_split
            val_size = config.validation_split
            test_size = 1 - train_size - val_size

            train_data, val_data, test_data = self._split_data(feature_data, train_size, val_size, test_size)

            return TrainingData(
                train_set=train_data,
                validation_set=val_data,
                test_set=test_data,
                # Assuming last column is target
                feature_count=feature_data.shape[1] - 1,
                feature_names=list(feature_data.columns[:-1]),
                target_name=feature_data.columns[-1],
            )

        except Exception as e:
            logger.error(f"Failed to prepare training data: {str(e)}")
            raise TrainingError(f"Data preparation failed: {str(e)}")

    async def _create_experiment(self, config: TrainingConfig) -> Dict[str, Any]:
    """Create MLflow experiment for training"""
        experiment_name = f"{config.model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        experiment = {
            "id": str(uuid4()),
            "name": experiment_name,
            "description": config.description or f"Training experiment for {config.model_name}",
            "created_at": datetime.utcnow(),
            "tags": {
                "model_name": config.model_name,
                "experiment_type": "training",
                "hyperparameter_tuning": str(config.enable_hyperparameter_tuning),
            },
        }

        # Register in MLflow
        await self.model_registry.create_experiment(experiment)

        return experiment

    async def _optimize_hyperparameters(
        self, config: TrainingConfig, training_data: TrainingData, experiment_id: str
    ) -> Dict[str, Any]:
    """Optimize hyperparameters using Optuna"""

        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in config.hyperparameter_space.items():
                if param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, param_config["choices"])
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step", 1),
                    )
                elif param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step", None),
                    )

            # Create model with sampled parameters
            model_class = config.model_class
            model = model_class(**params)

            # Cross-validation
            cv_scores = cross_val_score(
                model,
                training_data.train_set.features,
                training_data.train_set.labels,
                cv=config.cv_folds,
                scoring=config.optimization_metric,
            )

            return cv_scores.mean()

        # Create study
        study = optuna.create_study(
            direction=(
                "maximize" if config.optimization_metric in ["accuracy", "f1", "precision", "recall"] else "minimize"
            ),
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        # Optimize
        study.optimize(objective, n_trials=config.hyperparameter_trials, timeout=config.hyperparameter_timeout)

        # Store trials
        self._hyperparameter_trials[experiment_id] = [
            {
                "trial_number": trial.number,
                "params": trial.params,
                "value": trial.value,
                "state": trial.state.name,
            }
            for trial in study.trials
        ]

        logger.info(f"Hyperparameter optimization completed. Best value: {study.best_value}")
        return study.best_params

    async def _train_model(
        self, config: TrainingConfig, training_data: TrainingData, experiment_id: str
    ) -> ModelTrainingResult:
        """Execute model training with monitoring"""

        try:
            logger.info("Starting model training...")

            # Initialize model
            model_class = config.model_class
            model = model_class(**config.model_params)

            # Set up training callbacks
            callbacks = await self._setup_training_callbacks(experiment_id, config)

            # Execute training
            training_start = datetime.utcnow()

            if hasattr(model, "fit"):
                # Scikit-learn style training
                trained_model = model.fit(training_data.train_set.features, training_data.train_set.labels)

                # Get training history if available
                training_history = getattr(trained_model, "history", {})

            else:
                # Custom training loop
                trained_model = await self._custom_training_loop(model, training_data, config, callbacks)
                training_history = trained_model.get_training_history()

            training_duration = datetime.utcnow() - training_start

            # Save model
            model_path = f"models/{experiment_id}/model.pkl"
            joblib.dump(trained_model, model_path)

            # Log training metrics
            await self._log_training_metrics(
                experiment_id,
                {
                    "training_duration_seconds": training_duration.total_seconds(),
                    "final_train_loss": (training_history.get("loss", [0])[-1] if training_history else 0),
                    "final_val_loss": (training_history.get("val_loss", [0])[-1] if training_history else 0),
                    "epochs": len(training_history.get("loss", [])) if training_history else 1,
                },
            )

            return ModelTrainingResult(
                model=trained_model,
                model_path=model_path,
                training_duration=training_duration,
                training_history=training_history,
                experiment_id=experiment_id,
            )

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise TrainingError(f"Model training failed: {str(e)}")

    async def _evaluate_trained_model(
        self, model: Any, training_data: TrainingData, config: TrainingConfig
    ) -> ModelEvaluationResult:
        """Evaluate trained model performance"""

        try:
            logger.info("Evaluating trained model...")

            # Make predictions
            train_predictions = model.predict(training_data.train_set.features)
            val_predictions = model.predict(training_data.validation_set.features)
            test_predictions = model.predict(training_data.test_set.features)

            # Calculate metrics
            metrics = {}

            for split_name, predictions, labels in [
                ("train", train_predictions, training_data.train_set.labels),
                ("validation", val_predictions, training_data.validation_set.labels),
                ("test", test_predictions, training_data.test_set.labels),
            ]:
                if config.task_type == "classification":
                    metrics[f"{split_name}_accuracy"] = accuracy_score(labels, predictions)
                    metrics[f"{split_name}_precision"] = precision_score(labels, predictions, average="weighted")
                    metrics[f"{split_name}_recall"] = recall_score(labels, predictions, average="weighted")
                    metrics[f"{split_name}_f1"] = f1_score(labels, predictions, average="weighted")
                else:
                    # Regression metrics
                    mse = np.mean((labels - predictions) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(labels - predictions))
                    r2 = 1 - (np.sum((labels - predictions) ** 2) / np.sum((labels - np.mean(labels)) ** 2))

                    metrics[f"{split_name}_mse"] = mse
                    metrics[f"{split_name}_rmse"] = rmse
                    metrics[f"{split_name}_mae"] = mae
                    metrics[f"{split_name}_r2"] = r2

            # Determine primary metric
            primary_metric = config.primary_metric
            primary_value = metrics.get(f"validation_{primary_metric}", 0)

            # Check if meets quality threshold
            meets_threshold = primary_value >= config.quality_threshold

            return ModelEvaluationResult(
                metrics=metrics,
                primary_metric=primary_metric,
                primary_value=primary_value,
                meets_threshold=meets_threshold,
                quality_threshold=config.quality_threshold,
                evaluation_timestamp=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise TrainingError(f"Model evaluation failed: {str(e)}")

    async def _register_model(
        self,
        model_result: ModelTrainingResult,
        evaluation_result: ModelEvaluationResult,
        experiment_id: str,
        config: TrainingConfig,
    ) -> Dict[str, Any]:
    """Register trained model in model registry"""
        try:
            logger.info("Registering trained model...")

            model_version = {
                "id": str(uuid4()),
                "model_name": config.model_name,
                "version": f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "model_path": model_result.model_path,
                "experiment_id": experiment_id,
                "metrics": evaluation_result.metrics,
                "primary_metric": evaluation_result.primary_metric,
                "primary_value": evaluation_result.primary_value,
                "created_at": datetime.utcnow(),
                "status": "staging",
            }

            # Register in MLflow
            await self.model_registry.register_model_version(model_version)

            logger.info(f"Model registered: {model_version['id']}")
            return model_version

        except Exception as e:
            logger.error(f"Model registration failed: {str(e)}")
            raise TrainingError(f"Model registration failed: {str(e)}")

    async def _setup_training_callbacks(self, experiment_id: str, config: TrainingConfig) -> List[Any]:
        """Set up training callbacks for monitoring"""

        callbacks = []

        # MLflow callback
        callbacks.append(MLflowCallback(experiment_id))

        # Early stopping callback
        if config.early_stopping_patience:
            callbacks.append(
                EarlyStoppingCallback(patience=config.early_stopping_patience, monitor=config.early_stopping_monitor)
            )

        # Model checkpoint callback
        callbacks.append(
            ModelCheckpointCallback(
                filepath=f"checkpoints/{experiment_id}/model_{{epoch:02d}}.pkl",
                save_best_only=True,
                monitor=config.early_stopping_monitor,
            )
        )

        return callbacks

    async def _custom_training_loop(
        self, model: Any, training_data: TrainingData, config: TrainingConfig, callbacks: List[Any]
    ) -> Any:
        """Custom training loop for non-scikit-learn models"""

        # This would be implemented based on the specific model framework
        # For now, return the model as-is
        return model

    async def _log_training_metrics(self, experiment_id: str, metrics: Dict[str, float]) -> None:
        """Log training metrics to MLflow"""

        await self.model_registry.log_metrics(experiment_id, metrics)

    def _split_data(
        self, data: pd.DataFrame, train_size: float, val_size: float, test_size: float
    ) -> Tuple[Any, Any, Any]:
        """Split data into train/validation/test sets"""

        # Simple random split
        n_samples = len(data)
        train_end = int(n_samples * train_size)
        val_end = int(n_samples * (train_size + val_size))

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        return train_data, val_data, test_data

    async def _load_training_data(self, data_sources: List[Dict[str, Any]]) -> pd.DataFrame:
        """Load training data from configured sources"""

        # This would implement loading from various data sources
        # For now, return a dummy DataFrame
        return pd.DataFrame(
            {
                "feature_1": np.random.randn(1000),
                "feature_2": np.random.randn(1000),
                "target": np.random.randint(0, 2, 1000),
            }
        )


# Callback classes
class MLflowCallback:
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        # Log metrics to MLflow
        pass


class EarlyStoppingCallback:
    def __init__(self, patience: int, monitor: str):
        self.patience = patience
        self.monitor = monitor
        self.best_score = None
        self.wait = 0

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        current_score = logs.get(self.monitor, 0)
        if self.best_score is None or current_score > self.best_score:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True  # Stop training
        return False


class ModelCheckpointCallback:
    def __init__(self, filepath: str, save_best_only: bool, monitor: str):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.best_score = None

    def on_epoch_end(self, epoch: int, logs: Dict[str, float], model: Any):
        current_score = logs.get(self.monitor, 0)
        if not self.save_best_only or self.best_score is None or current_score > self.best_score:
            self.best_score = current_score
            # Save model
            joblib.dump(model, self.filepath.format(epoch=epoch))
