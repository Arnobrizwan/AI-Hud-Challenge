"""
Incidents API routes
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel

from monitoring.incidents import (
    Incident,
    IncidentManager,
    IncidentSeverity,
    IncidentStatus,
    IncidentType,
)
from monitoring.observability_engine import observability_engine

logger = logging.getLogger(__name__)

router = APIRouter()


class IncidentRequest(BaseModel):
    """Incident creation request"""

    title: str
    description: str
    severity: str
    incident_type: str
    affected_services: List[str] = []
    affected_users: Optional[int] = None
    tags: List[str] = []
    metadata: Dict[str, Any] = {}


class IncidentResponse(BaseModel):
    """Incident response"""

    id: str
    title: str
    description: str
    status: str
    severity: str
    incident_type: str
    created_at: str
    created_by: str
    assigned_to: Optional[str] = None
    resolved_at: Optional[str] = None
    closed_at: Optional[str] = None
    affected_services: List[str]
    affected_users: Optional[int] = None
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    tags: List[str]
    metadata: Dict[str, Any]
    timeline: List[Dict[str, Any]]


class IncidentUpdateRequest(BaseModel):
    """Incident update request"""

    status: Optional[str] = None
    assigned_to: Optional[str] = None
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    notes: Optional[str] = None


class PostMortemRequest(BaseModel):
    """Post-mortem creation request"""

    title: str
    summary: str
    root_cause: str
    impact: str
    resolution: str
    lessons_learned: List[str] = []
    action_items: List[Dict[str, Any]] = []


class PostMortemResponse(BaseModel):
    """Post-mortem response"""

    id: str
    incident_id: str
    title: str
    summary: str
    root_cause: str
    impact: str
    resolution: str
    lessons_learned: List[str]
    action_items: List[Dict[str, Any]]
    created_at: str
    created_by: str
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    published: bool


@router.post("/", response_model=IncidentResponse)
async def create_incident(incident_request: IncidentRequest):
    """Create new incident"""

    try:
        if not observability_engine or not observability_engine.incident_manager:
            raise HTTPException(status_code=503, detail="Incident manager not available")

        # Create incident data
        incident_data = {
            "title": incident_request.title,
            "description": incident_request.description,
            "severity": incident_request.severity,
            "incident_type": incident_request.incident_type,
            "affected_services": incident_request.affected_services,
            "affected_users": incident_request.affected_users,
            "tags": incident_request.tags,
            "metadata": incident_request.metadata,
        }

        incident = await observability_engine.incident_manager.create_incident(incident_data)

        return IncidentResponse(
            id=incident.id,
            title=incident.title,
            description=incident.description,
            status=incident.status.value,
            severity=incident.severity.value,
            incident_type=incident.incident_type.value,
            created_at=incident.created_at.isoformat(),
            created_by=incident.created_by,
            assigned_to=incident.assigned_to,
            resolved_at=incident.resolved_at.isoformat() if incident.resolved_at else None,
            closed_at=incident.closed_at.isoformat() if incident.closed_at else None,
            affected_services=incident.affected_services,
            affected_users=incident.affected_users,
            root_cause=incident.root_cause,
            resolution=incident.resolution,
            tags=incident.tags,
            metadata=incident.metadata,
            timeline=incident.timeline,
        )

    except Exception as e:
        logger.error(f"Failed to create incident: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create incident: {str(e)}")


@router.post("/emergency")
async def create_emergency_incident(
    incident_type: str = Body(..., embed=True),
    severity: str = Body(..., embed=True),
    description: str = Body(..., embed=True),
):
    """Create emergency incident with immediate response"""

    try:
        if not observability_engine or not observability_engine.incident_manager:
            raise HTTPException(status_code=503, detail="Incident manager not available")

        # Create emergency incident
        incident = await observability_engine.incident_manager.create_emergency_incident(
            incident_type=incident_type, severity=severity, description=description
        )

        return {
            "message": "Emergency incident created and response triggered",
            "incident_id": incident.id,
            "title": incident.title,
            "severity": incident.severity.value,
            "status": incident.status.value,
            "created_at": incident.created_at.isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to create emergency incident: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create emergency incident: {str(e)}"
        )


@router.get("/", response_model=List[IncidentResponse])
async def get_incidents(
    status: Optional[str] = Query(None, description="Filter by incident status"),
    severity: Optional[str] = Query(None, description="Filter by incident severity"),
    incident_type: Optional[str] = Query(None, description="Filter by incident type"),
    service: Optional[str] = Query(None, description="Filter by affected service"),
    limit: int = Query(100, description="Maximum number of incidents to return"),
):
    """Get incidents with optional filtering"""

    try:
        if not observability_engine or not observability_engine.incident_manager:
            raise HTTPException(status_code=503, detail="Incident manager not available")

        # Get all incidents
        incident_manager = observability_engine.incident_manager
        all_incidents = list(incident_manager.incidents.values())

        # Apply filters
        filtered_incidents = all_incidents

        if status:
            filtered_incidents = [i for i in filtered_incidents if i.status.value == status]

        if severity:
            filtered_incidents = [i for i in filtered_incidents if i.severity.value == severity]

        if incident_type:
            filtered_incidents = [
                i for i in filtered_incidents if i.incident_type.value == incident_type
            ]

        if service:
            filtered_incidents = [i for i in filtered_incidents if service in i.affected_services]

        # Limit results
        filtered_incidents = filtered_incidents[:limit]

        # Convert to response format
        incidents = []
        for incident in filtered_incidents:
            incidents.append(
                IncidentResponse(
                    id=incident.id,
                    title=incident.title,
                    description=incident.description,
                    status=incident.status.value,
                    severity=incident.severity.value,
                    incident_type=incident.incident_type.value,
                    created_at=incident.created_at.isoformat(),
                    created_by=incident.created_by,
                    assigned_to=incident.assigned_to,
                    resolved_at=incident.resolved_at.isoformat() if incident.resolved_at else None,
                    closed_at=incident.closed_at.isoformat() if incident.closed_at else None,
                    affected_services=incident.affected_services,
                    affected_users=incident.affected_users,
                    root_cause=incident.root_cause,
                    resolution=incident.resolution,
                    tags=incident.tags,
                    metadata=incident.metadata,
                    timeline=incident.timeline,
                )
            )

        return incidents

    except Exception as e:
        logger.error(f"Failed to get incidents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get incidents: {str(e)}")


@router.get("/active", response_model=List[IncidentResponse])
async def get_active_incidents():
    """Get active incidents"""

    try:
        if not observability_engine or not observability_engine.incident_manager:
            raise HTTPException(status_code=503, detail="Incident manager not available")

        # Get active incidents
        active_incidents = await observability_engine.incident_manager.get_active_incidents()

        # Convert to response format
        incidents = []
        for incident in active_incidents:
            incidents.append(
                IncidentResponse(
                    id=incident.id,
                    title=incident.title,
                    description=incident.description,
                    status=incident.status.value,
                    severity=incident.severity.value,
                    incident_type=incident.incident_type.value,
                    created_at=incident.created_at.isoformat(),
                    created_by=incident.created_by,
                    assigned_to=incident.assigned_to,
                    resolved_at=incident.resolved_at.isoformat() if incident.resolved_at else None,
                    closed_at=incident.closed_at.isoformat() if incident.closed_at else None,
                    affected_services=incident.affected_services,
                    affected_users=incident.affected_users,
                    root_cause=incident.root_cause,
                    resolution=incident.resolution,
                    tags=incident.tags,
                    metadata=incident.metadata,
                    timeline=incident.timeline,
                )
            )

        return incidents

    except Exception as e:
        logger.error(f"Failed to get active incidents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get active incidents: {str(e)}")


@router.get("/{incident_id}", response_model=IncidentResponse)
async def get_incident(incident_id: str):
    """Get specific incident by ID"""

    try:
        if not observability_engine or not observability_engine.incident_manager:
            raise HTTPException(status_code=503, detail="Incident manager not available")

        # Get incident
        incident = await observability_engine.incident_manager.get_incident(incident_id)

        if not incident:
            raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

        return IncidentResponse(
            id=incident.id,
            title=incident.title,
            description=incident.description,
            status=incident.status.value,
            severity=incident.severity.value,
            incident_type=incident.incident_type.value,
            created_at=incident.created_at.isoformat(),
            created_by=incident.created_by,
            assigned_to=incident.assigned_to,
            resolved_at=incident.resolved_at.isoformat() if incident.resolved_at else None,
            closed_at=incident.closed_at.isoformat() if incident.closed_at else None,
            affected_services=incident.affected_services,
            affected_users=incident.affected_users,
            root_cause=incident.root_cause,
            resolution=incident.resolution,
            tags=incident.tags,
            metadata=incident.metadata,
            timeline=incident.timeline,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get incident {incident_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get incident {incident_id}: {str(e)}"
        )


@router.put("/{incident_id}", response_model=IncidentResponse)
async def update_incident(
    incident_id: str, update_request: IncidentUpdateRequest, user: str = Body(..., embed=True)
):
    """Update incident"""

    try:
        if not observability_engine or not observability_engine.incident_manager:
            raise HTTPException(status_code=503, detail="Incident manager not available")

        # Get incident
        incident = await observability_engine.incident_manager.get_incident(incident_id)

        if not incident:
            raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

        # Update incident fields
        if update_request.status:
            await observability_engine.incident_manager.update_incident_status(
                incident_id, IncidentStatus(update_request.status), user, update_request.notes
            )

        if update_request.assigned_to:
            incident.assigned_to = update_request.assigned_to
            incident.timeline.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "assigned",
                    "user": user,
                    "description": f"Incident assigned to {update_request.assigned_to}",
                }
            )

        if update_request.root_cause:
            incident.root_cause = update_request.root_cause
            incident.timeline.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "root_cause_updated",
                    "user": user,
                    "description": f"Root cause updated: {update_request.root_cause}",
                }
            )

        if update_request.resolution:
            incident.resolution = update_request.resolution
            incident.timeline.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "resolution_updated",
                    "user": user,
                    "description": f"Resolution updated: {update_request.resolution}",
                }
            )

        return IncidentResponse(
            id=incident.id,
            title=incident.title,
            description=incident.description,
            status=incident.status.value,
            severity=incident.severity.value,
            incident_type=incident.incident_type.value,
            created_at=incident.created_at.isoformat(),
            created_by=incident.created_by,
            assigned_to=incident.assigned_to,
            resolved_at=incident.resolved_at.isoformat() if incident.resolved_at else None,
            closed_at=incident.closed_at.isoformat() if incident.closed_at else None,
            affected_services=incident.affected_services,
            affected_users=incident.affected_users,
            root_cause=incident.root_cause,
            resolution=incident.resolution,
            tags=incident.tags,
            metadata=incident.metadata,
            timeline=incident.timeline,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update incident {incident_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update incident {incident_id}: {str(e)}"
        )


@router.get("/{incident_id}/timeline")
async def get_incident_timeline(incident_id: str):
    """Get incident timeline"""

    try:
        if not observability_engine or not observability_engine.incident_manager:
            raise HTTPException(status_code=503, detail="Incident manager not available")

        # Get incident
        incident = await observability_engine.incident_manager.get_incident(incident_id)

        if not incident:
            raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

        return {
            "incident_id": incident_id,
            "timeline": incident.timeline,
            "total_events": len(incident.timeline),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get timeline for incident {incident_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get timeline for incident {incident_id}: {str(e)}"
        )


@router.post("/{incident_id}/post-mortem", response_model=PostMortemResponse)
async def create_post_mortem(incident_id: str, post_mortem_request: PostMortemRequest):
    """Create post-mortem for incident"""

    try:
        if not observability_engine or not observability_engine.incident_manager:
            raise HTTPException(status_code=503, detail="Incident manager not available")

        # Create post-mortem
        post_mortem = await observability_engine.incident_manager.create_post_mortem(incident_id)

        # Update post-mortem with provided data
        post_mortem.title = post_mortem_request.title
        post_mortem.summary = post_mortem_request.summary
        post_mortem.root_cause = post_mortem_request.root_cause
        post_mortem.impact = post_mortem_request.impact
        post_mortem.resolution = post_mortem_request.resolution
        post_mortem.lessons_learned = post_mortem_request.lessons_learned
        post_mortem.action_items = post_mortem_request.action_items

        return PostMortemResponse(
            id=post_mortem.id,
            incident_id=post_mortem.incident_id,
            title=post_mortem.title,
            summary=post_mortem.summary,
            root_cause=post_mortem.root_cause,
            impact=post_mortem.impact,
            resolution=post_mortem.resolution,
            lessons_learned=post_mortem.lessons_learned,
            action_items=post_mortem.action_items,
            created_at=post_mortem.created_at.isoformat(),
            created_by=post_mortem.created_by,
            reviewed_by=post_mortem.reviewed_by,
            reviewed_at=post_mortem.reviewed_at.isoformat() if post_mortem.reviewed_at else None,
            published=post_mortem.published,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create post-mortem for incident {incident_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create post-mortem for incident {incident_id}: {str(e)}",
        )


@router.get("/{incident_id}/post-mortem", response_model=PostMortemResponse)
async def get_post_mortem(incident_id: str):
    """Get post-mortem for incident"""

    try:
        if not observability_engine or not observability_engine.incident_manager:
            raise HTTPException(status_code=503, detail="Incident manager not available")

        # Find post-mortem for incident
        post_mortems = observability_engine.incident_manager.post_mortems
        post_mortem = next(
            (pm for pm in post_mortems.values() if pm.incident_id == incident_id), None
        )

        if not post_mortem:
            raise HTTPException(
                status_code=404, detail=f"Post-mortem for incident {incident_id} not found"
            )

        return PostMortemResponse(
            id=post_mortem.id,
            incident_id=post_mortem.incident_id,
            title=post_mortem.title,
            summary=post_mortem.summary,
            root_cause=post_mortem.root_cause,
            impact=post_mortem.impact,
            resolution=post_mortem.resolution,
            lessons_learned=post_mortem.lessons_learned,
            action_items=post_mortem.action_items,
            created_at=post_mortem.created_at.isoformat(),
            created_by=post_mortem.created_by,
            reviewed_by=post_mortem.reviewed_by,
            reviewed_at=post_mortem.reviewed_at.isoformat() if post_mortem.reviewed_at else None,
            published=post_mortem.published,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get post-mortem for incident {incident_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get post-mortem for incident {incident_id}: {str(e)}",
        )


@router.get("/stats")
async def get_incident_statistics(
    time_window: int = Query(2592000, description="Time window in seconds (default: 30 days)")
):
    """Get incident statistics"""

    try:
        if not observability_engine or not observability_engine.incident_manager:
            raise HTTPException(status_code=503, detail="Incident manager not available")

        # Get incident metrics
        time_window_delta = timedelta(seconds=time_window)
        metrics = await observability_engine.incident_manager.get_incident_metrics(
            time_window_delta
        )

        return {
            "incident_metrics": metrics,
            "time_window_seconds": time_window,
            "generated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get incident statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get incident statistics: {str(e)}")


@router.get("/search")
async def search_incidents(
    query: str = Query(..., description="Search query"),
    limit: int = Query(20, description="Maximum number of results"),
):
    """Search incidents by title or description"""

    try:
        if not observability_engine or not observability_engine.incident_manager:
            raise HTTPException(status_code=503, detail="Incident manager not available")

        # Search incidents
        incidents = await observability_engine.incident_manager.search_incidents(query)

        # Limit results
        incidents = incidents[:limit]

        # Convert to response format
        incident_responses = []
        for incident in incidents:
            incident_responses.append(
                IncidentResponse(
                    id=incident.id,
                    title=incident.title,
                    description=incident.description,
                    status=incident.status.value,
                    severity=incident.severity.value,
                    incident_type=incident.incident_type.value,
                    created_at=incident.created_at.isoformat(),
                    created_by=incident.created_by,
                    assigned_to=incident.assigned_to,
                    resolved_at=incident.resolved_at.isoformat() if incident.resolved_at else None,
                    closed_at=incident.closed_at.isoformat() if incident.closed_at else None,
                    affected_services=incident.affected_services,
                    affected_users=incident.affected_users,
                    root_cause=incident.root_cause,
                    resolution=incident.resolution,
                    tags=incident.tags,
                    metadata=incident.metadata,
                    timeline=incident.timeline,
                )
            )

        return {
            "query": query,
            "results": incident_responses,
            "total_results": len(incident_responses),
        }

    except Exception as e:
        logger.error(f"Failed to search incidents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search incidents: {str(e)}")
