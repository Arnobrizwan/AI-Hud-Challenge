"""
Crowdsourcing manager for external feedback collection
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import structlog
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.database import Campaign as CampaignDB
from ..models.database import CampaignTask as CampaignTaskDB
from ..models.schemas import Campaign, CampaignCreate, CampaignSubmission, CampaignTask

logger = structlog.get_logger(__name__)


class CrowdsourcingManager:
    """Manage crowdsourced feedback and annotations"""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.task_distributor = TaskDistributor()
        self.quality_controller = CrowdsourceQualityController()
        self.consensus_builder = ConsensusBuilder()

    async def create_campaign(self, campaign: CampaignCreate, created_by: UUID) -> Campaign:
        """Create crowdsourcing campaign"""

        try:
            campaign_db = CampaignDB(
                id=uuid4(),
                name=campaign.name,
                description=campaign.description,
                task_type=campaign.task_type,
                target_annotations=campaign.target_annotations,
                reward_per_task=campaign.reward_per_task,
                quality_threshold=campaign.quality_threshold,
                status="draft",
                created_by=created_by,
                created_at=datetime.utcnow(),
            )

            self.db.add(campaign_db)
            await self.db.commit()
            await self.db.refresh(campaign_db)

            logger.info("Campaign created", campaign_id=str(campaign_db.id))

            return self._db_to_schema(campaign_db)

        except Exception as e:
            logger.error("Error creating campaign", error=str(e))
            raise

    async def create_campaign_tasks(
        self, campaign_id: UUID, content_ids: List[UUID]
    ) -> List[CampaignTask]:
        """Create tasks for a campaign"""

        try:
            tasks = []

            for content_id in content_ids:
                task = CampaignTaskDB(
                    id=uuid4(),
                    campaign_id=campaign_id,
                    content_id=content_id,
                    task_data={},  # Would be populated with task-specific data
                    status="pending",
                    created_at=datetime.utcnow(),
                )

                tasks.append(task)
                self.db.add(task)

            await self.db.commit()

            logger.info(
                "Campaign tasks created", campaign_id=str(campaign_id), task_count=len(tasks)
            )

            return [self._task_to_schema(task) for task in tasks]

        except Exception as e:
            logger.error("Error creating campaign tasks", error=str(e))
            raise

    async def submit_campaign_response(
        self, task_id: UUID, worker_id: str, submission_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Submit response to campaign task"""

        try:
            # Validate submission
            validation_result = await self.validate_submission(submission_data)
            if not validation_result["is_valid"]:
                raise ValueError(validation_result["error_message"])

            # Create submission
            submission = CampaignSubmission(
                id=uuid4(),
                task_id=task_id,
                worker_id=worker_id,
                submission_data=submission_data,
                quality_score=None,  # Will be calculated
                created_at=datetime.utcnow(),
            )

            self.db.add(submission)
            await self.db.commit()

            # Calculate quality score
            quality_score = await self.quality_controller.score_submission(submission_data)
            submission.quality_score = quality_score

            await self.db.commit()

            logger.info(
                "Campaign response submitted",
                task_id=str(task_id),
                worker_id=worker_id,
                quality_score=quality_score,
            )

            return {
                "submission_id": str(submission.id),
                "quality_score": quality_score,
                "status": "submitted",
            }

        except Exception as e:
            logger.error("Error submitting campaign response", error=str(e))
            raise

    async def process_campaign_results(self, campaign_id: UUID) -> Dict[str, Any]:
        """Process and validate campaign results"""

        try:
            # Get campaign
            result = await self.db.execute(select(CampaignDB).where(CampaignDB.id == campaign_id))
            campaign = result.scalar_one_or_none()

            if not campaign:
                raise ValueError("Campaign not found")

            # Get all submissions
            result = await self.db.execute(
                select(CampaignSubmission)
                .join(CampaignTaskDB)
                .where(CampaignTaskDB.campaign_id == campaign_id)
            )
            submissions = result.scalars().all()

            # Quality control filtering
            validated_submissions = []
            for submission in submissions:
                if submission.quality_score >= campaign.quality_threshold:
                    validated_submissions.append(submission)

            # Build consensus
            consensus_results = await self.consensus_builder.build_consensus(validated_submissions)

            # Calculate metrics
            metrics = {
                "total_submissions": len(submissions),
                "validated_submissions": len(validated_submissions),
                "consensus_results": len(consensus_results),
                "average_quality_score": (
                    sum(s.quality_score for s in validated_submissions) / len(validated_submissions)
                    if validated_submissions
                    else 0
                ),
                "completion_rate": (
                    len(validated_submissions) / len(submissions) if submissions else 0
                ),
            }

            logger.info("Campaign results processed", campaign_id=str(campaign_id), metrics=metrics)

            return {
                "campaign_id": str(campaign_id),
                "consensus_results": consensus_results,
                "metrics": metrics,
                "processed_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Error processing campaign results", error=str(e))
            raise

    async def validate_submission(self, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate crowdsourced submission"""

        try:
            # Basic validation
            if not submission_data:
                return {"is_valid": False, "error_message": "Submission data is required"}

            # Check for required fields
            required_fields = ["response", "confidence"]
            for field in required_fields:
                if field not in submission_data:
                    return {"is_valid": False, "error_message": f"Field '{field}' is required"}

            # Validate confidence score
            confidence = submission_data.get("confidence", 0)
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                return {"is_valid": False, "error_message": "Confidence must be between 0 and 1"}

            return {"is_valid": True, "error_message": None}

        except Exception as e:
            logger.error("Error validating submission", error=str(e))
            return {"is_valid": False, "error_message": "Validation error"}

    def _db_to_schema(self, campaign: CampaignDB) -> Campaign:
        """Convert database model to schema"""

        return Campaign(
            id=str(campaign.id),
            name=campaign.name,
            description=campaign.description,
            task_type=campaign.task_type,
            target_annotations=campaign.target_annotations,
            reward_per_task=campaign.reward_per_task,
            quality_threshold=campaign.quality_threshold,
            status=campaign.status,
            created_by=str(campaign.created_by) if campaign.created_by else None,
            created_at=campaign.created_at,
        )

    def _task_to_schema(self, task: CampaignTaskDB) -> CampaignTask:
        """Convert database model to schema"""

        return CampaignTask(
            id=str(task.id),
            campaign_id=str(task.campaign_id),
            content_id=str(task.content_id),
            task_data=task.task_data,
            status=task.status,
            created_at=task.created_at,
        )


class TaskDistributor:
    """Distribute tasks to crowd workers"""

    def distribute_tasks(self, tasks: List[Any], worker_criteria: Dict[str, Any]) -> None:
        """Distribute tasks to appropriate workers"""

        # This would implement task distribution logic
        logger.info("Tasks distributed", count=len(tasks))


class CrowdsourceQualityController:
    """Control quality of crowdsourced submissions"""

    async def score_submission(self, submission_data: Dict[str, Any]) -> float:
        """Score quality of crowdsourced submission"""

        try:
            scores = []

            # Check completeness
            if submission_data.get("response"):
                scores.append(1.0)
            else:
                scores.append(0.0)

            # Check confidence score
            confidence = submission_data.get("confidence", 0)
            if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
                scores.append(confidence)
            else:
                scores.append(0.0)

            # Check response length (if applicable)
            response = submission_data.get("response", "")
            if isinstance(response, str) and len(response.strip()) > 0:
                scores.append(1.0)
            else:
                scores.append(0.0)

            return sum(scores) / len(scores) if scores else 0.0

        except Exception as e:
            logger.error("Error scoring submission", error=str(e))
            return 0.0


class ConsensusBuilder:
    """Build consensus from multiple submissions"""

    async def build_consensus(self, submissions: List[Any]) -> List[Dict[str, Any]]:
        """Build consensus from multiple submissions"""

        try:
            if not submissions:
                return []

            # Group submissions by task
            task_submissions = {}
            for submission in submissions:
                task_id = str(submission.task_id)
                if task_id not in task_submissions:
                    task_submissions[task_id] = []
                task_submissions[task_id].append(submission)

            consensus_results = []

            for task_id, task_subs in task_submissions.items():
                if len(task_subs) >= 2:  # Need at least 2 submissions for consensus
                    consensus = self._calculate_consensus(task_subs)
                    consensus_results.append(
                        {
                            "task_id": task_id,
                            "consensus": consensus,
                            "submission_count": len(task_subs),
                        }
                    )

            return consensus_results

        except Exception as e:
            logger.error("Error building consensus", error=str(e))
            return []

    def _calculate_consensus(self, submissions: List[Any]) -> Dict[str, Any]:
        """Calculate consensus from submissions"""

        # Simple majority voting
        responses = [sub.submission_data.get("response") for sub in submissions]

        # Count responses
        response_counts = {}
        for response in responses:
            response_counts[response] = response_counts.get(response, 0) + 1

        # Find most common response
        consensus_response = max(response_counts.items(), key=lambda x: x[1])[0]

        return {
            "response": consensus_response,
            "confidence": len([r for r in responses if r == consensus_response]) / len(responses),
        }
