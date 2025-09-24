"""
Features API endpoints - REST API for feature store management
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel

from src.feature_store.feature_store_manager import FeatureStoreManager
from src.models.feature_models import (
    FeatureConfig,
    FeatureDefinition,
    FeatureServingRequest,
    FeatureSet,
    FeatureType,
    FeatureVector,
)
from src.utils.exceptions import FeatureStoreError, ValidationError

router = APIRouter()


# Request/Response models
class CreateFeatureSetRequest(BaseModel):
    config: FeatureConfig


class CreateFeatureSetResponse(BaseModel):
    feature_set_id: str
    status: str
    message: str


class FeatureSetListResponse(BaseModel):
    feature_sets: List[FeatureSet]
    total: int
    page: int
    page_size: int


class ServeFeaturesRequest(BaseModel):
    request: FeatureServingRequest


class ServeFeaturesResponse(BaseModel):
    feature_vector: FeatureVector
    served_at: datetime


@router.post("/feature-sets", response_model=CreateFeatureSetResponse)
async def create_feature_set(
    request: CreateFeatureSetRequest,
    background_tasks: BackgroundTasks,
    feature_store_manager: FeatureStoreManager = Depends(),
):
    """Create a new feature set"""

    try:
        feature_set = await feature_store_manager.register_feature_set(request.config)

        return CreateFeatureSetResponse(
            feature_set_id=feature_set.id,
            status=feature_set.status.value,
            message=f"Feature set '{feature_set.name}' created successfully",
        )

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FeatureStoreError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/feature-sets/{feature_set_id}", response_model=FeatureSet)
async def get_feature_set(
    feature_set_id: str, feature_store_manager: FeatureStoreManager = Depends()
):
    """Get feature set by ID"""

    try:
        feature_sets = await feature_store_manager.list_feature_sets()
        feature_set = next((fs for fs in feature_sets if fs.id == feature_set_id), None)

        if not feature_set:
            raise HTTPException(status_code=404, detail="Feature set not found")

        return feature_set

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feature set: {str(e)}")


@router.get("/feature-sets", response_model=FeatureSetListResponse)
async def list_feature_sets(
    status: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size"),
    feature_store_manager: FeatureStoreManager = Depends(),
):
    """List feature sets with optional filtering"""

    try:
        feature_sets = await feature_store_manager.list_feature_sets(status=status)

        # Pagination
        total = len(feature_sets)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_feature_sets = feature_sets[start_idx:end_idx]

        return FeatureSetListResponse(
            feature_sets=paginated_feature_sets, total=total, page=page, page_size=page_size
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list feature sets: {str(e)}")


@router.post("/serve", response_model=ServeFeaturesResponse)
async def serve_features(
    request: ServeFeaturesRequest, feature_store_manager: FeatureStoreManager = Depends()
):
    """Serve features for online inference"""

    try:
        feature_vector = await feature_store_manager.serve_features(request.request)

        return ServeFeaturesResponse(
            feature_vector=feature_vector, served_at=feature_vector.served_at
        )

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FeatureStoreError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to serve features: {str(e)}")


@router.get("/feature-sets/{feature_set_id}/quality")
async def get_feature_quality_metrics(
    feature_set_id: str, feature_store_manager: FeatureStoreManager = Depends()
):
    """Get feature quality metrics"""

    try:
        quality_metrics = await feature_store_manager.get_feature_quality_metrics(feature_set_id)

        return quality_metrics

    except FeatureStoreError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quality metrics: {str(e)}")


@router.get("/feature-sets/{feature_set_id}/lineage")
async def get_feature_lineage(
    feature_set_id: str,
    feature_name: str = Query(..., description="Feature name to get lineage for"),
    feature_store_manager: FeatureStoreManager = Depends(),
):
    """Get feature lineage information"""

    try:
        lineage = await feature_store_manager.get_feature_lineage(feature_name)

        return lineage

    except FeatureStoreError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feature lineage: {str(e)}")


@router.put("/feature-sets/{feature_set_id}")
async def update_feature_set(
    feature_set_id: str,
    update_config: Dict[str, Any],
    feature_store_manager: FeatureStoreManager = Depends(),
):
    """Update feature set"""

    try:
        updated_feature_set = await feature_store_manager.update_feature_set(
            feature_set_id, update_config
        )

        return {
            "feature_set_id": feature_set_id,
            "status": "updated",
            "message": f"Feature set '{updated_feature_set.name}' updated successfully",
        }

    except FeatureStoreError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update feature set: {str(e)}")


@router.delete("/feature-sets/{feature_set_id}")
async def delete_feature_set(
    feature_set_id: str, feature_store_manager: FeatureStoreManager = Depends()
):
    """Delete feature set"""

    try:
        success = await feature_store_manager.delete_feature_set(feature_set_id)

        if not success:
            raise HTTPException(status_code=404, detail="Feature set not found")

        return {
            "feature_set_id": feature_set_id,
            "status": "deleted",
            "message": f"Feature set '{feature_set_id}' deleted successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete feature set: {str(e)}")


@router.get("/feature-sets/{feature_set_id}/statistics")
async def get_feature_statistics(
    feature_set_id: str, feature_store_manager: FeatureStoreManager = Depends()
):
    """Get feature statistics"""

    try:
        # This would integrate with actual statistics calculation
        statistics = {
            "feature_set_id": feature_set_id,
            "total_features": 25,
            "numerical_features": 15,
            "categorical_features": 8,
            "text_features": 2,
            "completeness": 0.95,
            "last_updated": datetime.utcnow().isoformat(),
        }

        return statistics

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feature statistics: {str(e)}")


@router.post("/feature-sets/{feature_set_id}/ingest")
async def trigger_feature_ingestion(
    feature_set_id: str,
    data_source: Dict[str, Any],
    feature_store_manager: FeatureStoreManager = Depends(),
):
    """Trigger feature ingestion"""

    try:
        # This would trigger actual feature ingestion
        ingestion_job = {
            "job_id": f"ingestion_{feature_set_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "feature_set_id": feature_set_id,
            "status": "started",
            "started_at": datetime.utcnow().isoformat(),
        }

        return ingestion_job

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to trigger feature ingestion: {str(e)}"
        )


@router.get("/feature-sets/{feature_set_id}/ingestion/{job_id}")
async def get_ingestion_status(
    feature_set_id: str, job_id: str, feature_store_manager: FeatureStoreManager = Depends()
):
    """Get feature ingestion job status"""

    try:
        # This would get actual ingestion job status
        job_status = {
            "job_id": job_id,
            "feature_set_id": feature_set_id,
            "status": "completed",
            "records_processed": 10000,
            "records_failed": 50,
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
        }

        return job_status

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get ingestion status: {str(e)}")
