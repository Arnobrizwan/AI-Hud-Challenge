"""
AI Pipeline Manager - Core orchestration system
Manages the entire ML pipeline lifecycle with modular components
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class PipelineStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class ComponentType(Enum):
    DATA_PROCESSOR = "data_processor"
    MODEL_TRAINER = "model_trainer"
    EVALUATOR = "evaluator"
    DEPLOYER = "deployer"
    MONITOR = "monitor"

@dataclass
class PipelineComponent:
    id: str
    name: str
    component_type: ComponentType
    config: Dict[str, Any]
    status: PipelineStatus = PipelineStatus.IDLE
    last_run: Optional[datetime] = None
    metrics: Dict[str, Any] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class PipelineExecution:
    id: str
    pipeline_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    components_executed: List[str] = None
    metrics: Dict[str, Any] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.components_executed is None:
            self.components_executed = []
        if self.metrics is None:
            self.metrics = {}

class PipelineManager:
    """Manages AI/ML pipeline execution and monitoring"""
    
    def __init__(self):
        self.pipelines: Dict[str, List[PipelineComponent]] = {}
        self.executions: Dict[str, PipelineExecution] = {}
        self.component_registry: Dict[str, Any] = {}
        self.running_executions: Dict[str, asyncio.Task] = {}
        
    async def create_pipeline(self, name: str, description: str = "") -> str:
        """Create a new pipeline"""
        pipeline_id = str(uuid.uuid4())
        self.pipelines[pipeline_id] = []
        
        logger.info(f"Created pipeline: {name} (ID: {pipeline_id})")
        return pipeline_id
    
    async def add_component(
        self, 
        pipeline_id: str, 
        name: str, 
        component_type: ComponentType,
        config: Dict[str, Any],
        dependencies: List[str] = None
    ) -> str:
        """Add a component to a pipeline"""
        component_id = str(uuid.uuid4())
        component = PipelineComponent(
            id=component_id,
            name=name,
            component_type=component_type,
            config=config,
            dependencies=dependencies or []
        )
        
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
            
        self.pipelines[pipeline_id].append(component)
        logger.info(f"Added component {name} to pipeline {pipeline_id}")
        return component_id
    
    async def execute_pipeline(self, pipeline_id: str) -> str:
        """Execute a pipeline"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        execution_id = str(uuid.uuid4())
        execution = PipelineExecution(
            id=execution_id,
            pipeline_id=pipeline_id,
            status=PipelineStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        
        self.executions[execution_id] = execution
        
        # Start pipeline execution asynchronously
        task = asyncio.create_task(self._execute_pipeline_async(execution_id))
        self.running_executions[execution_id] = task
        
        logger.info(f"Started pipeline execution: {execution_id}")
        return execution_id
    
    async def _execute_pipeline_async(self, execution_id: str):
        """Execute pipeline components in order"""
        execution = self.executions[execution_id]
        pipeline_id = execution.pipeline_id
        components = self.pipelines[pipeline_id]
        
        try:
            # Sort components by dependencies
            sorted_components = self._sort_components_by_dependencies(components)
            
            for component in sorted_components:
                logger.info(f"Executing component: {component.name}")
                component.status = PipelineStatus.RUNNING
                
                # Execute component
                result = await self._execute_component(component)
                component.metrics.update(result.get('metrics', {}))
                component.status = PipelineStatus.COMPLETED
                component.last_run = datetime.utcnow()
                
                execution.components_executed.append(component.id)
                execution.metrics[component.name] = result.get('metrics', {})
                
                logger.info(f"Completed component: {component.name}")
            
            execution.status = PipelineStatus.COMPLETED
            execution.end_time = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            execution.status = PipelineStatus.FAILED
            execution.error_message = str(e)
            execution.end_time = datetime.utcnow()
        
        finally:
            # Clean up running task
            if execution_id in self.running_executions:
                del self.running_executions[execution_id]
    
    def _sort_components_by_dependencies(self, components: List[PipelineComponent]) -> List[PipelineComponent]:
        """Sort components by their dependencies"""
        sorted_components = []
        remaining = components.copy()
        
        while remaining:
            # Find components with no unmet dependencies
            ready = []
            for component in remaining:
                if not component.dependencies or all(
                    dep in [c.id for c in sorted_components] 
                    for dep in component.dependencies
                ):
                    ready.append(component)
            
            if not ready:
                # Circular dependency or missing dependency
                logger.warning("Circular dependency detected, adding remaining components")
                sorted_components.extend(remaining)
                break
                
            sorted_components.extend(ready)
            for component in ready:
                remaining.remove(component)
        
        return sorted_components
    
    async def _execute_component(self, component: PipelineComponent) -> Dict[str, Any]:
        """Execute a single component"""
        component_type = component.component_type
        
        if component_type == ComponentType.DATA_PROCESSOR:
            return await self._execute_data_processor(component)
        elif component_type == ComponentType.MODEL_TRAINER:
            return await self._execute_model_trainer(component)
        elif component_type == ComponentType.EVALUATOR:
            return await self._execute_evaluator(component)
        elif component_type == ComponentType.DEPLOYER:
            return await self._execute_deployer(component)
        elif component_type == ComponentType.MONITOR:
            return await self._execute_monitor(component)
        else:
            raise ValueError(f"Unknown component type: {component_type}")
    
    async def _execute_data_processor(self, component: PipelineComponent) -> Dict[str, Any]:
        """Execute data processing component"""
        # Simulate data processing
        await asyncio.sleep(1)
        
        return {
            "metrics": {
                "records_processed": 1000,
                "processing_time": 1.0,
                "data_quality_score": 0.95
            }
        }
    
    async def _execute_model_trainer(self, component: PipelineComponent) -> Dict[str, Any]:
        """Execute model training component"""
        # Simulate model training
        await asyncio.sleep(2)
        
        return {
            "metrics": {
                "training_accuracy": 0.92,
                "validation_accuracy": 0.89,
                "training_time": 2.0,
                "model_size_mb": 45.2
            }
        }
    
    async def _execute_evaluator(self, component: PipelineComponent) -> Dict[str, Any]:
        """Execute model evaluation component"""
        # Simulate evaluation
        await asyncio.sleep(1)
        
        return {
            "metrics": {
                "test_accuracy": 0.88,
                "f1_score": 0.87,
                "precision": 0.89,
                "recall": 0.85
            }
        }
    
    async def _execute_deployer(self, component: PipelineComponent) -> Dict[str, Any]:
        """Execute model deployment component"""
        # Simulate deployment
        await asyncio.sleep(0.5)
        
        return {
            "metrics": {
                "deployment_time": 0.5,
                "deployment_status": "success",
                "model_version": "v1.2.3"
            }
        }
    
    async def _execute_monitor(self, component: PipelineComponent) -> Dict[str, Any]:
        """Execute monitoring component"""
        # Simulate monitoring
        await asyncio.sleep(0.5)
        
        return {
            "metrics": {
                "monitoring_interval": 60,
                "alerts_triggered": 0,
                "system_health": "healthy"
            }
        }
    
    async def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get pipeline status and metrics"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        components = self.pipelines[pipeline_id]
        recent_executions = [
            exec for exec in self.executions.values() 
            if exec.pipeline_id == pipeline_id
        ]
        
        return {
            "pipeline_id": pipeline_id,
            "components": [asdict(comp) for comp in components],
            "recent_executions": [asdict(exec) for exec in recent_executions[-5:]],
            "total_executions": len(recent_executions),
            "last_execution": recent_executions[-1] if recent_executions else None
        }
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get pipeline analytics"""
        total_pipelines = len(self.pipelines)
        total_executions = len(self.executions)
        running_executions = len(self.running_executions)
        
        # Calculate success rate
        completed_executions = [
            exec for exec in self.executions.values() 
            if exec.status == PipelineStatus.COMPLETED
        ]
        success_rate = len(completed_executions) / total_executions if total_executions > 0 else 0
        
        return {
            "total_pipelines": total_pipelines,
            "total_executions": total_executions,
            "running_executions": running_executions,
            "success_rate": success_rate,
            "avg_execution_time": self._calculate_avg_execution_time(),
            "component_usage": self._get_component_usage_stats()
        }
    
    def _calculate_avg_execution_time(self) -> float:
        """Calculate average execution time"""
        completed_executions = [
            exec for exec in self.executions.values() 
            if exec.status == PipelineStatus.COMPLETED and exec.end_time
        ]
        
        if not completed_executions:
            return 0.0
        
        total_time = sum(
            (exec.end_time - exec.start_time).total_seconds() 
            for exec in completed_executions
        )
        return total_time / len(completed_executions)
    
    def _get_component_usage_stats(self) -> Dict[str, int]:
        """Get component usage statistics"""
        usage = {}
        for components in self.pipelines.values():
            for component in components:
                comp_type = component.component_type.value
                usage[comp_type] = usage.get(comp_type, 0) + 1
        return usage
