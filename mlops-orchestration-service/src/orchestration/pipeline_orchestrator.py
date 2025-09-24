"""
MLOps Pipeline Orchestrator - Core orchestration service
Manages complete ML lifecycle including training, deployment, monitoring, and automated retraining
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import Settings
from src.deployment.deployment_manager import ModelDeploymentManager
from src.feature_store.feature_store_manager import FeatureStoreManager
from src.infrastructure.resource_manager import ResourceManager
from src.models.pipeline_models import (
    ComponentType,
    MLPipeline,
    MLPipelineConfig,
    PipelineComponent,
    PipelineExecution,
    PipelineStatus,
    PipelineType,
)
from src.monitoring.monitoring_service import ModelMonitoringService
from src.orchestration.airflow_client import AirflowClient
from src.orchestration.kubeflow_client import KubeflowClient
from src.orchestration.vertex_ai_client import VertexAIPipelineClient
from src.registry.model_registry import ModelRegistry
from src.utils.exceptions import PipelineError, ValidationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MLOpsPipelineOrchestrator:
    """Complete MLOps pipeline orchestration and management"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.airflow_client = AirflowClient(settings)
        self.vertex_ai_client = VertexAIPipelineClient(settings)
        self.kubeflow_client = KubeflowClient(settings)
        self.model_registry = ModelRegistry(settings)
        self.deployment_manager = ModelDeploymentManager(settings)
        self.monitoring_service = ModelMonitoringService(settings)
        self.feature_store_manager = FeatureStoreManager(settings)
        self.resource_manager = ResourceManager(settings)

        # Pipeline registry
        self._pipelines: Dict[str, MLPipeline] = {}
        self._executions: Dict[str, PipelineExecution] = {}

    async def initialize(self) -> None:
        """Initialize the orchestrator and all clients"""
        try:
            logger.info("Initializing MLOps Pipeline Orchestrator...")

            # Initialize clients
            await self.airflow_client.initialize()
            await self.vertex_ai_client.initialize()
            await self.kubeflow_client.initialize()
            await self.model_registry.initialize()
            await self.deployment_manager.initialize()
            await self.monitoring_service.initialize()
            await self.feature_store_manager.initialize()
            await self.resource_manager.initialize()

            # Load existing pipelines
            await self._load_existing_pipelines()

            logger.info("MLOps Pipeline Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MLOps Pipeline Orchestrator: {str(e)}")
            raise

    async def create_ml_pipeline(self, pipeline_config: MLPipelineConfig) -> MLPipeline:
        """Create complete ML pipeline with orchestration"""

        try:
            logger.info(f"Creating ML pipeline: {pipeline_config.name}")

            # Validate pipeline configuration
            await self._validate_pipeline_config(pipeline_config)

            # Create pipeline instance
            pipeline = MLPipeline(
                id=str(uuid4()),
                name=pipeline_config.name,
                description=pipeline_config.description,
                pipeline_type=pipeline_config.pipeline_type,
                orchestrator=pipeline_config.orchestrator,
                created_at=datetime.utcnow(),
                config=pipeline_config,
                status=PipelineStatus.CREATED,
            )

            # Create pipeline components
            components = await self.create_pipeline_components(pipeline_config)
            pipeline.components = components

            # Create orchestration workflow
            if pipeline_config.orchestrator == "airflow":
                workflow_id = await self._create_airflow_dag(pipeline, components)
                pipeline.airflow_dag_id = workflow_id
            elif pipeline_config.orchestrator == "vertex_ai":
                workflow_id = await self._create_vertex_ai_pipeline(pipeline, components)
                pipeline.vertex_pipeline_id = workflow_id
            elif pipeline_config.orchestrator == "kubeflow":
                workflow_id = await self._create_kubeflow_pipeline(pipeline, components)
                pipeline.kubeflow_pipeline_id = workflow_id

            # Register pipeline in system
            await self._register_pipeline(pipeline)

            # Set up monitoring
            if pipeline_config.monitoring_enabled:
                await self.monitoring_service.setup_pipeline_monitoring(pipeline)

            pipeline.status = PipelineStatus.READY
            logger.info(f"ML pipeline created successfully: {pipeline.id}")

            return pipeline

        except Exception as e:
            logger.error(f"Failed to create ML pipeline: {str(e)}")
            raise PipelineError(f"Pipeline creation failed: {str(e)}")

    async def create_pipeline_components(self, config: MLPipelineConfig) -> List[PipelineComponent]:
        """Create individual pipeline components"""

        components = []

        try:
            # Data validation component
            if config.include_data_validation:
                data_validation = await self._create_data_validation_component(config)
                components.append(data_validation)

            # Feature engineering component
            if config.include_feature_engineering:
                feature_engineering = await self._create_feature_engineering_component(config)
                components.append(feature_engineering)

            # Model training component
            if config.include_training:
                model_training = await self._create_training_component(config)
                components.append(model_training)

            # Model validation component
            if config.include_validation:
                model_validation = await self._create_validation_component(config)
                components.append(model_validation)

            # Model deployment component
            if config.include_deployment:
                model_deployment = await self._create_deployment_component(config)
                components.append(model_deployment)

            # Model monitoring component
            if config.include_monitoring:
                model_monitoring = await self._create_monitoring_component(config)
                components.append(model_monitoring)

            logger.info(f"Created {len(components)} pipeline components")
            return components

        except Exception as e:
            logger.error(f"Failed to create pipeline components: {str(e)}")
            raise PipelineError(f"Component creation failed: {str(e)}")

    async def trigger_pipeline_execution(
        self, pipeline_id: str, execution_params: Dict[str, Any]
    ) -> PipelineExecution:
        """Trigger pipeline execution with parameters"""

        try:
            logger.info(f"Triggering pipeline execution: {pipeline_id}")

            # Get pipeline
            pipeline = await self.get_pipeline(pipeline_id)
            if not pipeline:
                raise PipelineError(f"Pipeline not found: {pipeline_id}")

            # Check resource availability
            if not await self.resource_manager.check_resource_availability(pipeline):
                raise PipelineError("Insufficient resources for pipeline execution")

            # Create execution record
            execution = PipelineExecution(
                id=str(uuid4()),
                pipeline_id=pipeline_id,
                execution_params=execution_params,
                status=PipelineStatus.RUNNING,
                started_at=datetime.utcnow(),
                triggered_by=execution_params.get("triggered_by", "manual"),
            )

            # Submit to orchestrator
            if pipeline.orchestrator == "airflow":
                run_id = await self.airflow_client.trigger_dag_run(
                    dag_id=pipeline.airflow_dag_id, run_id=execution.id, conf=execution_params
                )
                execution.external_run_id = run_id

            elif pipeline.orchestrator == "vertex_ai":
                job_id = await self.vertex_ai_client.submit_pipeline_job(
                    pipeline_spec=pipeline.vertex_pipeline_id, parameters=execution_params
                )
                execution.external_run_id = job_id

            elif pipeline.orchestrator == "kubeflow":
                run_id = await self.kubeflow_client.submit_pipeline_run(
                    pipeline_id=pipeline.kubeflow_pipeline_id, parameters=execution_params
                )
                execution.external_run_id = run_id

            # Store execution record
            await self._store_pipeline_execution(execution)

            # Reserve resources
            await self.resource_manager.reserve_resources(pipeline, execution)

            logger.info(f"Pipeline execution triggered: {execution.id}")
            return execution

        except Exception as e:
            logger.error(f"Failed to trigger pipeline execution: {str(e)}")
            raise PipelineError(f"Pipeline execution failed: {str(e)}")

    async def get_pipeline(self, pipeline_id: str) -> Optional[MLPipeline]:
        """Get pipeline by ID"""
        return self._pipelines.get(pipeline_id)

    async def get_pipeline_execution(self, execution_id: str) -> Optional[PipelineExecution]:
        """Get pipeline execution by ID"""
        return self._executions.get(execution_id)

    async def list_pipelines(
        self, pipeline_type: Optional[PipelineType] = None, status: Optional[PipelineStatus] = None
    ) -> List[MLPipeline]:
        """List pipelines with optional filtering"""

        pipelines = list(self._pipelines.values())

        if pipeline_type:
            pipelines = [p for p in pipelines if p.pipeline_type == pipeline_type]

        if status:
            pipelines = [p for p in pipelines if p.status == status]

        return pipelines

    async def update_pipeline_status(self, pipeline_id: str, status: PipelineStatus) -> None:
        """Update pipeline status"""

        pipeline = await self.get_pipeline(pipeline_id)
        if pipeline:
            pipeline.status = status
            pipeline.updated_at = datetime.utcnow()
            logger.info(f"Pipeline status updated: {pipeline_id} -> {status}")

    async def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete pipeline and cleanup resources"""

        try:
            pipeline = await self.get_pipeline(pipeline_id)
            if not pipeline:
                return False

            # Stop all running executions
            running_executions = [
                exec
                for exec in self._executions.values()
                if exec.pipeline_id == pipeline_id and exec.status == PipelineStatus.RUNNING
            ]

            for execution in running_executions:
                await self.stop_pipeline_execution(execution.id)

            # Cleanup orchestration resources
            if pipeline.orchestrator == "airflow" and pipeline.airflow_dag_id:
                await self.airflow_client.delete_dag(pipeline.airflow_dag_id)
            elif pipeline.orchestrator == "vertex_ai" and pipeline.vertex_pipeline_id:
                await self.vertex_ai_client.delete_pipeline(pipeline.vertex_pipeline_id)
            elif pipeline.orchestrator == "kubeflow" and pipeline.kubeflow_pipeline_id:
                await self.kubeflow_client.delete_pipeline(pipeline.kubeflow_pipeline_id)

            # Remove from registry
            del self._pipelines[pipeline_id]

            logger.info(f"Pipeline deleted: {pipeline_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete pipeline: {str(e)}")
            return False

    async def stop_pipeline_execution(self, execution_id: str) -> bool:
        """Stop running pipeline execution"""

        try:
            execution = await self.get_pipeline_execution(execution_id)
            if not execution or execution.status != PipelineStatus.RUNNING:
                return False

            # Stop execution in orchestrator
            pipeline = await self.get_pipeline(execution.pipeline_id)

            if pipeline.orchestrator == "airflow":
                await self.airflow_client.stop_dag_run(
                    dag_id=pipeline.airflow_dag_id, run_id=execution.external_run_id
                )
            elif pipeline.orchestrator == "vertex_ai":
                await self.vertex_ai_client.cancel_pipeline_job(execution.external_run_id)
            elif pipeline.orchestrator == "kubeflow":
                await self.kubeflow_client.stop_pipeline_run(execution.external_run_id)

            # Update execution status
            execution.status = PipelineStatus.STOPPED
            execution.stopped_at = datetime.utcnow()

            # Release resources
            await self.resource_manager.release_resources(execution)

            logger.info(f"Pipeline execution stopped: {execution_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to stop pipeline execution: {str(e)}")
            return False

    async def _validate_pipeline_config(self, config: MLPipelineConfig) -> None:
        """Validate pipeline configuration"""

        if not config.name:
            raise ValidationError("Pipeline name is required")

        if not config.pipeline_type:
            raise ValidationError("Pipeline type is required")

        if config.orchestrator not in ["airflow", "vertex_ai", "kubeflow"]:
            raise ValidationError(f"Invalid orchestrator: {config.orchestrator}")

        # Validate component dependencies
        if config.include_training and not config.include_data_validation:
            logger.warning("Training enabled without data validation - this may cause issues")

        if config.include_deployment and not config.include_training:
            raise ValidationError("Deployment requires training to be enabled")

    async def _create_data_validation_component(
        self, config: MLPipelineConfig
    ) -> PipelineComponent:
        """Create data validation component"""

        return PipelineComponent(
            id=str(uuid4()),
            name="data_validation",
            component_type=ComponentType.DATA_VALIDATION,
            config={
                "data_sources": config.data_sources,
                "validation_rules": config.validation_rules,
                "quality_threshold": config.data_quality_threshold,
            },
            dependencies=[],
            timeout=config.data_validation_timeout,
        )

    async def _create_feature_engineering_component(
        self, config: MLPipelineConfig
    ) -> PipelineComponent:
        """Create feature engineering component"""

        return PipelineComponent(
            id=str(uuid4()),
            name="feature_engineering",
            component_type=ComponentType.FEATURE_ENGINEERING,
            config={
                "feature_definitions": config.feature_definitions,
                "transformation_pipeline": config.transformation_pipeline,
                "feature_store_config": config.feature_store_config,
            },
            dependencies=["data_validation"],
            timeout=config.feature_engineering_timeout,
        )

    async def _create_training_component(self, config: MLPipelineConfig) -> PipelineComponent:
        """Create model training component"""

        return PipelineComponent(
            id=str(uuid4()),
            name="model_training",
            component_type=ComponentType.MODEL_TRAINING,
            config={
                "model_class": config.model_class,
                "model_params": config.model_params,
                "training_data_config": config.training_data_config,
                "hyperparameter_tuning": config.hyperparameter_tuning,
                "experiment_config": config.experiment_config,
            },
            dependencies=["feature_engineering"],
            timeout=config.training_timeout,
        )

    async def _create_validation_component(self, config: MLPipelineConfig) -> PipelineComponent:
        """Create model validation component"""

        return PipelineComponent(
            id=str(uuid4()),
            name="model_validation",
            component_type=ComponentType.MODEL_VALIDATION,
            config={
                "validation_metrics": config.validation_metrics,
                "quality_threshold": config.model_quality_threshold,
                "validation_data_config": config.validation_data_config,
            },
            dependencies=["model_training"],
            timeout=config.validation_timeout,
        )

    async def _create_deployment_component(self, config: MLPipelineConfig) -> PipelineComponent:
        """Create model deployment component"""

        return PipelineComponent(
            id=str(uuid4()),
            name="model_deployment",
            component_type=ComponentType.MODEL_DEPLOYMENT,
            config={
                "deployment_strategy": config.deployment_strategy,
                "deployment_config": config.deployment_config,
                "environment_config": config.environment_config,
            },
            dependencies=["model_validation"],
            timeout=config.deployment_timeout,
        )

    async def _create_monitoring_component(self, config: MLPipelineConfig) -> PipelineComponent:
        """Create model monitoring component"""

        return PipelineComponent(
            id=str(uuid4()),
            name="model_monitoring",
            component_type=ComponentType.MODEL_MONITORING,
            config={
                "monitoring_metrics": config.monitoring_metrics,
                "alerting_config": config.alerting_config,
                "monitoring_dashboard": config.monitoring_dashboard,
            },
            dependencies=["model_deployment"],
            timeout=config.monitoring_timeout,
        )

    async def _create_airflow_dag(
        self, pipeline: MLPipeline, components: List[PipelineComponent]
    ) -> str:
        """Create Airflow DAG for pipeline"""

        dag_id = f"ml_pipeline_{pipeline.name.lower().replace(' ', '_')}"

        # Create DAG using Airflow client
        dag_config = {
            "pipeline_id": pipeline.id,
            "components": [comp.dict() for comp in components],
            "schedule_interval": pipeline.config.schedule_interval,
            "max_retries": pipeline.config.max_retries,
            "retry_delay": pipeline.config.retry_delay,
        }

        await self.airflow_client.create_dag(dag_id, dag_config)
        return dag_id

    async def _create_vertex_ai_pipeline(
        self, pipeline: MLPipeline, components: List[PipelineComponent]
    ) -> str:
        """Create Vertex AI pipeline for pipeline"""

        pipeline_spec = {
            "pipeline_id": pipeline.id,
            "components": [comp.dict() for comp in components],
            "parameters": pipeline.config.parameters,
            "outputs": pipeline.config.outputs,
        }

        pipeline_id = await self.vertex_ai_client.create_pipeline(pipeline_spec)
        return pipeline_id

    async def _create_kubeflow_pipeline(
        self, pipeline: MLPipeline, components: List[PipelineComponent]
    ) -> str:
        """Create Kubeflow pipeline for pipeline"""

        pipeline_spec = {
            "pipeline_id": pipeline.id,
            "components": [comp.dict() for comp in components],
            "parameters": pipeline.config.parameters,
            "outputs": pipeline.config.outputs,
        }

        pipeline_id = await self.kubeflow_client.create_pipeline(pipeline_spec)
        return pipeline_id

    async def _register_pipeline(self, pipeline: MLPipeline) -> None:
        """Register pipeline in system"""

        self._pipelines[pipeline.id] = pipeline
        await self.model_registry.register_pipeline(pipeline)

    async def _store_pipeline_execution(self, execution: PipelineExecution) -> None:
        """Store pipeline execution record"""

        self._executions[execution.id] = execution
        await self.model_registry.store_pipeline_execution(execution)

    async def _load_existing_pipelines(self) -> None:
        """Load existing pipelines from registry"""

        try:
            pipelines = await self.model_registry.get_all_pipelines()
            for pipeline in pipelines:
                self._pipelines[pipeline.id] = pipeline

            logger.info(f"Loaded {len(pipelines)} existing pipelines")

        except Exception as e:
            logger.warning(f"Failed to load existing pipelines: {str(e)}")
