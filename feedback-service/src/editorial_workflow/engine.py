"""
Editorial workflow management with approval chains
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import structlog
from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.database import ReviewResult as ReviewResultDB
from ..models.database import ReviewTask as ReviewTaskDB
from ..models.database import User
from ..models.schemas import (
    ReviewDecision,
    ReviewResult,
    ReviewResultCreate,
    ReviewTask,
    ReviewTaskCreate,
    TaskPriority,
    TaskStatus,
)
from ..notification.service import NotificationService
from ..rbac.engine import RoleBasedAccessControl
from ..realtime.websocket_manager import WebSocketManager

logger = structlog.get_logger(__name__)


class EditorialWorkflowEngine:
    """Manage editorial workflows and human oversight"""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.task_manager = TaskManager(db_session)
        self.approval_engine = ApprovalEngine(db_session)
        self.notification_service = NotificationService()
        self.rbac = RoleBasedAccessControl(db_session)
        self.websocket_manager = WebSocketManager()

    async def create_review_task(
        self,
        content_id: UUID,
        task_type: str,
        priority: str = "normal",
        created_by: Optional[UUID] = None,
    ) -> ReviewTask:
        """Create editorial review task"""

        try:
            # Determine appropriate reviewers
            eligible_reviewers = await self.rbac.get_eligible_reviewers(task_type, content_id)

            if not eligible_reviewers:
                raise ValueError(
                    f"No eligible reviewers found for task type: {task_type}")

            # Create task with metadata
            task = ReviewTaskDB(
                id=uuid4(),
                content_id=content_id,
                task_type=task_type,
                priority=TaskPriority(priority),
                assigned_to=await self.select_reviewer(eligible_reviewers, task_type),
                created_by=created_by,
                due_date=self.calculate_due_date(priority, task_type),
                status=TaskStatus.ASSIGNED,
                created_at=datetime.utcnow(),
            )

            # Store task
            self.db.add(task)
            await self.db.commit()
            await self.db.refresh(task)

            # Notify assigned reviewer
            await self.notification_service.notify_reviewer(task)

            # Add to real-time dashboard
            await self.websocket_manager.broadcast_task_update(task)

            logger.info(
                "Review task created",
                task_id=str(task.id),
                assigned_to=str(task.assigned_to),
                task_type=task_type,
            )

            return self._db_to_schema(task)

        except Exception as e:
            logger.error(
                "Error creating review task",
                error=str(e),
                content_id=str(content_id))
            raise

    async def process_review_completion(
        self, task_id: UUID, review_result: ReviewResultCreate
    ) -> Dict[str, Any]:
    """Process completed editorial review"""
        try:
            # Get task
            task = await self.get_task(task_id)
            if not task:
                raise ValueError(f"Task not found: {task_id}")

            # Validate reviewer permissions
            if not await self.rbac.can_complete_task(review_result.reviewer_id, task):
                raise PermissionError(
                    "Insufficient permissions to complete task")

            # Process review decision
            workflow_actions = []

            if review_result.decision == ReviewDecision.APPROVE:
                workflow_actions = await self.approve_content(task, review_result)
            elif review_result.decision == ReviewDecision.REJECT:
                workflow_actions = await self.reject_content(task, review_result)
            elif review_result.decision == ReviewDecision.REQUEST_CHANGES:
                workflow_actions = await self.request_changes(task, review_result)
            elif review_result.decision == ReviewDecision.ESCALATE:
                workflow_actions = await self.escalate_task(task, review_result)

            # Store review result
            result_db = ReviewResultDB(
                id=uuid4(),
                task_id=task_id,
                reviewer_id=review_result.reviewer_id,
                decision=review_result.decision,
                comments=review_result.comments,
                changes_requested=review_result.changes_requested,
                metadata=review_result.metadata or {},
                created_at=datetime.utcnow(),
            )

            self.db.add(result_db)

            # Update task status
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()

            await self.db.commit()

            # Trigger downstream workflows
            await self.trigger_post_review_workflows(task, review_result)

            # Notify stakeholders
            await self.notify_review_completion(task, review_result)

            logger.info(
                "Review completed",
                task_id=str(task_id),
                decision=review_result.decision.value)

            return {
                "task_id": str(task_id),
                "decision": review_result.decision.value,
                "workflow_actions": workflow_actions,
                "completed_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(
                "Error processing review completion",
                error=str(e),
                task_id=str(task_id))
            raise

    async def get_task(self, task_id: UUID) -> Optional[ReviewTaskDB]:
        """Get review task by ID"""

        result = await self.db.execute(select(ReviewTaskDB).where(ReviewTaskDB.id == task_id))
        return result.scalar_one_or_none()

    async def get_tasks_for_user(
        self, user_id: UUID, status: Optional[TaskStatus] = None
    ) -> List[ReviewTask]:
        """Get review tasks for a specific user"""

        query = select(ReviewTaskDB).where(ReviewTaskDB.assigned_to == user_id)

        if status:
            query = query.where(ReviewTaskDB.status == status)

        result = await self.db.execute(query)
        tasks = result.scalars().all()

        return [self._db_to_schema(task) for task in tasks]

    async def get_overdue_tasks(self) -> List[ReviewTask]:
        """Get overdue review tasks"""

        now = datetime.utcnow()
        result = await self.db.execute(
            select(ReviewTaskDB).where(
                and_(
                    ReviewTaskDB.due_date < now,
                    ReviewTaskDB.status.in_([TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]),
                )
            )
        )

        tasks = result.scalars().all()
        return [self._db_to_schema(task) for task in tasks]

    async def approve_content(
        self, task: ReviewTaskDB, review_result: ReviewResultCreate
    ) -> List[str]:
        """Approve content and trigger downstream actions"""

        actions = ["content_approved"]

        # Mark content as approved
        # This would update content status in the content service
        logger.info("Content approved", content_id=str(task.content_id))

        # Trigger content publication workflow
        actions.append("publication_triggered")

        # Update content metrics
        actions.append("metrics_updated")

        return actions

    async def reject_content(
        self, task: ReviewTaskDB, review_result: ReviewResultCreate
    ) -> List[str]:
        """Reject content and trigger appropriate actions"""

        actions = ["content_rejected"]

        # Mark content as rejected
        logger.info("Content rejected", content_id=str(task.content_id))

        # Notify content creator
        actions.append("creator_notified")

        # Archive or remove content
        actions.append("content_archived")

        return actions

    async def request_changes(
        self, task: ReviewTaskDB, review_result: ReviewResultCreate
    ) -> List[str]:
        """Request changes to content"""

        actions = ["changes_requested"]

        # Create new task for content creator
        actions.append("revision_task_created")

        # Notify content creator
        actions.append("creator_notified")

        return actions

    async def escalate_task(
        self, task: ReviewTaskDB, review_result: ReviewResultCreate
    ) -> List[str]:
        """Escalate task to higher authority"""

        actions = ["task_escalated"]

        # Find appropriate escalation reviewer
        escalated_reviewer = await self.find_escalation_reviewer(task)

        if escalated_reviewer:
            # Reassign task
            task.assigned_to = escalated_reviewer
            task.priority = TaskPriority.URGENT
            actions.append("task_reassigned")

        return actions

    async def select_reviewer(
            self,
            eligible_reviewers: List[User],
            task_type: str) -> UUID:
        """Select appropriate reviewer from eligible candidates"""

        # Simple round-robin selection
        # In production, this would consider workload, expertise, etc.

        if not eligible_reviewers:
            raise ValueError("No eligible reviewers available")

        # Get current workload for each reviewer
        workloads = {}
        for reviewer in eligible_reviewers:
            workload = await self.get_reviewer_workload(reviewer.id)
            workloads[reviewer.id] = workload

        # Select reviewer with lowest workload
        selected_reviewer = min(workloads.items(), key=lambda x: x[1])[0]

        return selected_reviewer

    async def get_reviewer_workload(self, reviewer_id: UUID) -> int:
        """Get current workload for a reviewer"""

        result = await self.db.execute(
            select(ReviewTaskDB).where(
                and_(
                    ReviewTaskDB.assigned_to == reviewer_id,
                    ReviewTaskDB.status.in_([TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]),
                )
            )
        )

        return len(result.scalars().all())

    def calculate_due_date(self, priority: str, task_type: str) -> datetime:
        """Calculate due date based on priority and task type"""

        base_hours = {
            "low": 72,  # 3 days
            "normal": 24,  # 1 day
            "high": 8,  # 8 hours
            "urgent": 2,  # 2 hours
        }

        hours = base_hours.get(priority, 24)

        # Adjust based on task type
        if task_type in ["fact_check", "legal_review"]:
            hours += 24  # Add extra time for complex tasks

        return datetime.utcnow() + timedelta(hours=hours)

    async def find_escalation_reviewer(
            self, task: ReviewTaskDB) -> Optional[UUID]:
        """Find appropriate reviewer for escalated task"""

        # This would implement escalation logic
        # For now, return None (no escalation)
        return None

    async def trigger_post_review_workflows(
        self, task: ReviewTaskDB, review_result: ReviewResultCreate
    ) -> None:
        """Trigger downstream workflows after review completion"""

        # This would trigger various downstream processes
        # based on the review decision and content type

        logger.info(
            "Post-review workflows triggered",
            task_id=str(task.id),
            decision=review_result.decision.value,
        )

    async def notify_review_completion(
        self, task: ReviewTaskDB, review_result: ReviewResultCreate
    ) -> None:
        """Notify stakeholders about review completion"""

        # Send WebSocket notification
        await self.websocket_manager.broadcast_review_completion(task, review_result)

        # Send other notifications
        logger.info(
            "Review completion notifications sent",
            task_id=str(
                task.id))

    def _db_to_schema(self, task: ReviewTaskDB) -> ReviewTask:
        """Convert database model to schema"""

        return ReviewTask(
            id=task.id,
            content_id=task.content_id,
            task_type=task.task_type,
            priority=task.priority,
            assigned_to=task.assigned_to,
            created_by=task.created_by,
            status=task.status,
            due_date=task.due_date,
            created_at=task.created_at,
            updated_at=task.updated_at,
            completed_at=task.completed_at,
        )


class TaskManager:
    """Manage review tasks"""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def create_task(self, task: ReviewTaskDB) -> ReviewTaskDB:
        """Create a new review task"""

        self.db.add(task)
        await self.db.commit()
        await self.db.refresh(task)

        return task

    async def update_task(self, task: ReviewTaskDB) -> ReviewTaskDB:
        """Update an existing review task"""

        await self.db.commit()
        await self.db.refresh(task)

        return task


class ApprovalEngine:
    """Handle approval workflows"""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def process_approval_chain(
        self, content_id: UUID, approval_chain: List[str]
    ) -> Dict[str, Any]:
    """Process multi-level approval chain"""
        # This would implement complex approval workflows
        # For now, return simple success
        return {"status": "approved", "approval_chain": approval_chain}
