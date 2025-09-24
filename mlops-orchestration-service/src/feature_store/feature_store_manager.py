"""
Feature Store Manager - Manages feature store operations and serving
Integrates with Vertex AI Feature Store for production-grade feature management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.config.settings import Settings
from src.models.feature_models import (
    FeatureConfig,
    FeatureDefinition,
    FeatureLineage,
    FeatureQualityMetrics,
    FeatureServingRequest,
    FeatureSet,
    FeatureTransformation,
    FeatureValidationResult,
    FeatureVector,
)
from src.monitoring.monitoring_service import ModelMonitoringService
from src.orchestration.vertex_ai_client import VertexAIPipelineClient
from src.registry.model_registry import ModelRegistry
from src.utils.exceptions import FeatureStoreError, ValidationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class FeatureStoreManager:
    """Manage feature store operations and serving"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.vertex_ai_feature_store = VertexAIPipelineClient(settings)
        self.feature_validator = FeatureValidator()
        self.serving_client = FeatureServingClient()
        self.lineage_tracker = LineageTracker()
        self.model_registry = ModelRegistry(settings)
        self.monitoring_service = ModelMonitoringService(settings)

        # Feature store state
        self._feature_sets: Dict[str, FeatureSet] = {}
        self._feature_cache: Dict[str, Any] = {}
        self._serving_requests: Dict[str, FeatureServingRequest] = {}

    async def initialize(self) -> None:
        """Initialize the feature store manager"""
        try:
            logger.info("Initializing Feature Store Manager...")

            await self.vertex_ai_feature_store.initialize()
            await self.feature_validator.initialize()
            await self.serving_client.initialize()
            await self.lineage_tracker.initialize()
            await self.model_registry.initialize()
            await self.monitoring_service.initialize()

            # Load existing feature sets
            await self._load_existing_feature_sets()

            logger.info("Feature Store Manager initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize Feature Store Manager: {str(e)}")
            raise

    async def register_feature_set(
            self, feature_config: FeatureConfig) -> FeatureSet:
        """Register new feature set in feature store"""

        try:
            logger.info(f"Registering feature set: {feature_config.name}")

            # Validate feature definitions
            validation_result = await self.feature_validator.validate_features(
                feature_config.feature_definitions
            )

            if not validation_result.is_valid:
                raise FeatureStoreError(
                    f"Feature validation failed: {validation_result.errors}")

            # Create feature set
            feature_set = FeatureSet(
                id=str(uuid4()),
                name=feature_config.name,
                description=feature_config.description,
                features=feature_config.feature_definitions,
                source_config=feature_config.source_config,
                update_schedule=feature_config.update_schedule,
                created_at=datetime.utcnow(),
                status="active",
            )

            # Register in Vertex AI Feature Store
            vertex_feature_set = await self.vertex_ai_feature_store.create_feature_set(
                feature_set_id=feature_set.id, feature_set_config=feature_config
            )

            feature_set.vertex_feature_set_name = vertex_feature_set.name

            # Set up feature ingestion pipeline
            ingestion_pipeline = await self._create_feature_ingestion_pipeline(feature_set)
            feature_set.ingestion_pipeline_id = ingestion_pipeline.id

            # Track lineage
            await self.lineage_tracker.track_feature_set_creation(feature_set)

            # Store feature set
            self._feature_sets[feature_set.id] = feature_set
            await self._store_feature_set(feature_set)

            logger.info(
                f"Feature set registered successfully: {feature_set.id}")
            return feature_set

        except Exception as e:
            logger.error(f"Failed to register feature set: {str(e)}")
            raise FeatureStoreError(
                f"Feature set registration failed: {str(e)}")

    async def serve_features(
            self,
            serving_request: FeatureServingRequest) -> FeatureVector:
        """Serve features for online inference"""

        try:
            logger.info(
                f"Serving features for entities: {len(serving_request.entity_ids)}")

            # Validate serving request
            if not serving_request.entity_ids:
                raise ValidationError(
                    "Entity IDs are required for feature serving")

            # Check cache first
            cache_key = self._generate_cache_key(serving_request)
            if cache_key in self._feature_cache:
                cached_features = self._feature_cache[cache_key]
                if self._is_cache_valid(
                        cached_features,
                        serving_request.cache_ttl):
                    logger.info("Returning cached features")
                    return cached_features

            # Fetch features from store
            feature_values = await self.serving_client.get_online_features(
                feature_store_name=serving_request.feature_store_name,
                feature_set_names=serving_request.feature_set_names,
                entity_ids=serving_request.entity_ids,
            )

            # Apply feature transformations if configured
            if serving_request.transformations:
                feature_values = await self._apply_feature_transformations(
                    feature_values, serving_request.transformations
                )

            # Create feature vector
            feature_vector = FeatureVector(
                entity_ids=serving_request.entity_ids,
                feature_values=feature_values,
                served_at=datetime.utcnow(),
                feature_set_names=serving_request.feature_set_names,
            )

            # Cache features
            self._feature_cache[cache_key] = feature_vector

            # Log feature serving for monitoring
            await self._log_feature_serving(serving_request, feature_vector)

            logger.info(
                f"Features served successfully for {len(serving_request.entity_ids)} entities"
            )
            return feature_vector

        except Exception as e:
            logger.error(f"Failed to serve features: {str(e)}")
            raise FeatureStoreError(f"Feature serving failed: {str(e)}")

    async def get_training_features(
            self, feature_config: Dict[str, Any]) -> pd.DataFrame:
        """Get features for training data preparation"""

        try:
            logger.info("Fetching training features...")

            # Get feature sets
            feature_sets = feature_config.get("feature_sets", [])
            entity_ids = feature_config.get("entity_ids", [])
            start_time = feature_config.get("start_time")
            end_time = feature_config.get("end_time")

            # Fetch features from all specified feature sets
            all_features = []
            for feature_set_name in feature_sets:
                features = await self.serving_client.get_historical_features(
                    feature_set_name=feature_set_name,
                    entity_ids=entity_ids,
                    start_time=start_time,
                    end_time=end_time,
                )
                all_features.append(features)

            # Combine features
            if all_features:
                combined_features = pd.concat(all_features, axis=1)
            else:
                combined_features = pd.DataFrame()

            # Apply feature engineering if specified
            if feature_config.get("feature_engineering"):
                combined_features = await self._apply_feature_engineering(
                    combined_features, feature_config["feature_engineering"]
                )

            logger.info(
                f"Retrieved {len(combined_features)} training samples with {len(combined_features.columns)} features"
            )
            return combined_features

        except Exception as e:
            logger.error(f"Failed to get training features: {str(e)}")
            raise FeatureStoreError(
                f"Training feature retrieval failed: {str(e)}")

    async def update_feature_set(
        self, feature_set_id: str, update_config: Dict[str, Any]
    ) -> FeatureSet:
        """Update existing feature set"""

        try:
            feature_set = self._feature_sets.get(feature_set_id)
            if not feature_set:
                raise FeatureStoreError(
                    f"Feature set not found: {feature_set_id}")

            logger.info(f"Updating feature set: {feature_set_id}")

            # Update feature definitions if provided
            if "feature_definitions" in update_config:
                validation_result = await self.feature_validator.validate_features(
                    update_config["feature_definitions"]
                )
                if not validation_result.is_valid:
                    raise FeatureStoreError(
                        f"Feature validation failed: {validation_result.errors}"
                    )

                feature_set.features = update_config["feature_definitions"]

            # Update other properties
            if "description" in update_config:
                feature_set.description = update_config["description"]

            if "update_schedule" in update_config:
                feature_set.update_schedule = update_config["update_schedule"]

            # Update in Vertex AI
            await self.vertex_ai_feature_store.update_feature_set(
                feature_set_id=feature_set_id, update_config=update_config
            )

            # Update ingestion pipeline if needed
            if "source_config" in update_config:
    await self._update_ingestion_pipeline(feature_set, update_config["source_config"])

            feature_set.updated_at = datetime.utcnow()

            # Store updated feature set
            await self._store_feature_set(feature_set)

            logger.info(f"Feature set updated successfully: {feature_set_id}")
            return feature_set

        except Exception as e:
            logger.error(f"Failed to update feature set: {str(e)}")
            raise FeatureStoreError(f"Feature set update failed: {str(e)}")

    async def delete_feature_set(self, feature_set_id: str) -> bool:
        """Delete feature set and cleanup resources"""

        try:
            feature_set = self._feature_sets.get(feature_set_id)
            if not feature_set:
                return False

            logger.info(f"Deleting feature set: {feature_set_id}")

            # Stop ingestion pipeline
            if feature_set.ingestion_pipeline_id:
    await self._stop_ingestion_pipeline(feature_set.ingestion_pipeline_id)

            # Delete from Vertex AI
            await self.vertex_ai_feature_store.delete_feature_set(feature_set_id)

            # Cleanup cache
            self._cleanup_feature_cache(feature_set_id)

            # Remove from registry
            del self._feature_sets[feature_set_id]
            await self._delete_feature_set(feature_set_id)

            logger.info(f"Feature set deleted successfully: {feature_set_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete feature set: {str(e)}")
            return False

    async def get_feature_quality_metrics(
            self, feature_set_id: str) -> FeatureQualityMetrics:
        """Get feature quality metrics"""

        try:
            feature_set = self._feature_sets.get(feature_set_id)
            if not feature_set:
                raise FeatureStoreError(
                    f"Feature set not found: {feature_set_id}")

            # Calculate quality metrics
            metrics = await self._calculate_feature_quality_metrics(feature_set)

            return FeatureQualityMetrics(
                feature_set_id=feature_set_id,
                metrics=metrics,
                calculated_at=datetime.utcnow())

        except Exception as e:
            logger.error(f"Failed to get feature quality metrics: {str(e)}")
            raise FeatureStoreError(
                f"Quality metrics calculation failed: {str(e)}")

    async def get_feature_lineage(self, feature_name: str) -> FeatureLineage:
        """Get feature lineage information"""

        try:
            lineage = await self.lineage_tracker.get_feature_lineage(feature_name)
            return lineage

        except Exception as e:
            logger.error(f"Failed to get feature lineage: {str(e)}")
            raise FeatureStoreError(
                f"Feature lineage retrieval failed: {str(e)}")

    async def list_feature_sets(
            self, status: Optional[str] = None) -> List[FeatureSet]:
        """List feature sets with optional filtering"""

        feature_sets = list(self._feature_sets.values())

        if status:
            feature_sets = [fs for fs in feature_sets if fs.status == status]

        return feature_sets

    async def _create_feature_ingestion_pipeline(
            self, feature_set: FeatureSet) -> Dict[str, Any]:
    """Create feature ingestion pipeline"""
        pipeline_config = {
            "feature_set_id": feature_set.id,
            "source_config": feature_set.source_config,
            "update_schedule": feature_set.update_schedule,
            "feature_definitions": feature_set.features,
        }

        # Create pipeline in orchestrator
        pipeline = await self.vertex_ai_feature_store.create_ingestion_pipeline(pipeline_config)

        return pipeline

    async def _apply_feature_transformations(
        self, feature_values: Dict[str, Any], transformations: List[FeatureTransformation]
    ) -> Dict[str, Any]:
    """Apply feature transformations"""
        transformed_values = feature_values.copy()

        for transformation in transformations:
            if transformation.type == "normalization":
                transformed_values = await self._apply_normalization(
                    transformed_values, transformation.config
                )
            elif transformation.type == "scaling":
                transformed_values = await self._apply_scaling(
                    transformed_values, transformation.config
                )
            elif transformation.type == "encoding":
                transformed_values = await self._apply_encoding(
                    transformed_values, transformation.config
                )

        return transformed_values

    async def _apply_normalization(
        self, feature_values: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
    """Apply normalization transformation"""
        method = config.get("method", "z_score")

        for feature_name, values in feature_values.items():
            if isinstance(values, (list, np.ndarray)):
                if method == "z_score":
                    mean = np.mean(values)
                    std = np.std(values)
                    feature_values[feature_name] = [
                        (x - mean) / std for x in values]
                elif method == "min_max":
                    min_val = np.min(values)
                    max_val = np.max(values)
                    feature_values[feature_name] = [
                        (x - min_val) / (max_val - min_val) for x in values
                    ]

        return feature_values

    async def _apply_scaling(
        self, feature_values: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
    """Apply scaling transformation"""
        scale_factor = config.get("scale_factor", 1.0)

        for feature_name, values in feature_values.items():
            if isinstance(values, (list, np.ndarray)):
                feature_values[feature_name] = [
                    x * scale_factor for x in values]

        return feature_values

    async def _apply_encoding(
        self, feature_values: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
    """Apply encoding transformation"""
        encoding_type = config.get("type", "one_hot")

        if encoding_type == "one_hot":
            # Implement one-hot encoding
            pass
        elif encoding_type == "label":
            # Implement label encoding
            pass

        return feature_values

    async def _apply_feature_engineering(
        self, features: pd.DataFrame, engineering_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply feature engineering transformations"""

        # Implement feature engineering logic
        # This would include operations like:
        # - Creating new features from existing ones
        # - Aggregating features
        # - Feature selection
        # - Dimensionality reduction

        return features

    def _generate_cache_key(
            self,
            serving_request: FeatureServingRequest) -> str:
        """Generate cache key for feature serving request"""

        key_parts = [
            serving_request.feature_store_name,
            ",".join(sorted(serving_request.feature_set_names)),
            ",".join(sorted(serving_request.entity_ids)),
        ]

        return "|".join(key_parts)

    def _is_cache_valid(
            self,
            cached_features: FeatureVector,
            cache_ttl: int) -> bool:
        """Check if cached features are still valid"""

        if not cached_features:
            return False

        age_seconds = (
            datetime.utcnow() -
            cached_features.served_at).total_seconds()
        return age_seconds < cache_ttl

    async def _log_feature_serving(self,
                                   serving_request: FeatureServingRequest,
                                   feature_vector: FeatureVector) -> None:
        """Log feature serving for monitoring"""

        serving_metrics = {
            "entity_count": len(serving_request.entity_ids),
            "feature_set_count": len(serving_request.feature_set_names),
            "serving_latency_ms": (datetime.utcnow() - serving_request.request_time).total_seconds()
            * 1000,
            "cache_hit": False,  # Would be determined by cache lookup
        }

        await self.monitoring_service.log_feature_serving_metrics(serving_metrics)

    async def _calculate_feature_quality_metrics(
            self, feature_set: FeatureSet) -> Dict[str, Any]:
    """Calculate feature quality metrics"""
        # This would implement actual quality metrics calculation
        # For now, return dummy metrics
        return {
            "completeness": 0.95,
            "accuracy": 0.98,
            "consistency": 0.92,
            "timeliness": 0.90,
            "validity": 0.96,
        }

    async def _load_existing_feature_sets(self) -> None:
        """Load existing feature sets from registry"""

        try:
            feature_sets = await self.model_registry.get_all_feature_sets()
            for feature_set in feature_sets:
                self._feature_sets[feature_set.id] = feature_set

            logger.info(f"Loaded {len(feature_sets)} existing feature sets")

        except Exception as e:
            logger.warning(f"Failed to load existing feature sets: {str(e)}")

    async def _store_feature_set(self, feature_set: FeatureSet) -> None:
        """Store feature set in registry"""

        await self.model_registry.store_feature_set(feature_set)

    async def _delete_feature_set(self, feature_set_id: str) -> None:
        """Delete feature set from registry"""

        await self.model_registry.delete_feature_set(feature_set_id)

    def _cleanup_feature_cache(self, feature_set_id: str) -> None:
        """Cleanup feature cache for deleted feature set"""

        keys_to_remove = [
            key for key in self._feature_cache.keys() if feature_set_id in key]

        for key in keys_to_remove:
            del self._feature_cache[key]


# Supporting classes
class FeatureValidator:
    def __init__(self):
        self.validation_rules = {}

    async def initialize(self) -> Dict[str, Any]:
        pass

    async def validate_features(
        self, feature_definitions: List[FeatureDefinition]
    ) -> FeatureValidationResult:
        """Validate feature definitions"""

        errors = []

        for feature in feature_definitions:
            if not feature.name:
                errors.append(f"Feature name is required")

            if not feature.data_type:
                errors.append(
                    f"Data type is required for feature {feature.name}")

            if feature.is_required and not feature.default_value:
                errors.append(
                    f"Default value is required for required feature {feature.name}")

        return FeatureValidationResult(
            is_valid=len(errors) == 0, errors=errors)


class FeatureServingClient:
    def __init__(self):
        self.vertex_ai_client = None

    async def initialize(self) -> Dict[str, Any]:
        pass

    async def get_online_features(self,
                                  feature_store_name: str,
                                  feature_set_names: List[str],
                                  entity_ids: List[str]) -> Dict[str, Any]:
    """Get features for online serving"""
        # Implement actual feature serving logic
        # This would connect to Vertex AI Feature Store
        return {"dummy_features": "dummy_values"}

    async def get_historical_features(
        self,
        feature_set_name: str,
        entity_ids: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Get features for training data"""

        # Implement actual historical feature retrieval
        return pd.DataFrame()


class LineageTracker:
    def __init__(self):
        self.lineage_data = {}

    async def initialize(self) -> Dict[str, Any]:
        pass

    async def track_feature_set_creation(self, feature_set: FeatureSet) -> Dict[str, Any]:
        """Track feature set creation in lineage"""
        pass

    async def get_feature_lineage(self, feature_name: str) -> FeatureLineage:
        """Get feature lineage information"""

        return FeatureLineage(
            feature_name=feature_name,
            source_datasets=[],
            transformations=[],
            dependencies=[],
            created_at=datetime.utcnow(),
        )
