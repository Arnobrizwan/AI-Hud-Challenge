"""
Interactive annotation tools for human oversight
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from ..models.schemas import (
    AnnotationTask, AnnotationTaskCreate, Annotation, AnnotationCreate,
    TaskStatus, AnnotationType, QualityLevel
)
from ..models.database import (
    AnnotationTask as AnnotationTaskDB, Annotation as AnnotationDB,
    AnnotationAssignment, User, ContentItem
)
from ..quality_assurance.controller import QualityController
from ..notification.service import NotificationService

logger = structlog.get_logger(__name__)

class AnnotationManager:
    """Manage annotation tasks and interfaces"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.quality_controller = QualityController()
        self.inter_annotator = InterAnnotatorAgreement(db_session)
        self.notification_service = NotificationService()
        
    async def create_annotation_task(self, content_id: UUID, annotation_type: AnnotationType,
                                   annotator_ids: List[UUID], guidelines: Optional[str] = None,
                                   deadline: Optional[datetime] = None) -> AnnotationTask:
        """Create annotation task for human labelers"""
        
        try:
            # Set default deadline if not provided
            if not deadline:
                deadline = datetime.utcnow() + timedelta(hours=24)
            
            # Create task
            task = AnnotationTaskDB(
                id=uuid4(),
                content_id=content_id,
                annotation_type=annotation_type,
                guidelines=guidelines or await self.get_annotation_guidelines(annotation_type),
                deadline=deadline,
                status=TaskStatus.PENDING,
                created_at=datetime.utcnow()
            )
            
            self.db.add(task)
            await self.db.commit()
            await self.db.refresh(task)
            
            # Create assignments for annotators
            assignments = []
            for annotator_id in annotator_ids:
                assignment = AnnotationAssignment(
                    id=uuid4(),
                    task_id=task.id,
                    annotator_id=annotator_id,
                    assigned_at=datetime.utcnow()
                )
                assignments.append(assignment)
                self.db.add(assignment)
            
            await self.db.commit()
            
            # Notify annotators
            for annotator_id in annotator_ids:
                await self.notify_annotator(annotator_id, task)
            
            logger.info("Annotation task created", 
                       task_id=str(task.id), 
                       annotators=len(annotator_ids),
                       annotation_type=annotation_type.value)
            
            return self._db_to_schema(task)
            
        except Exception as e:
            logger.error("Error creating annotation task", error=str(e), content_id=str(content_id))
            raise
    
    async def submit_annotation(self, task_id: UUID, annotator_id: UUID,
                              annotation_data: Dict[str, Any], 
                              confidence_score: Optional[float] = None,
                              time_spent_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Process submitted annotation"""
        
        try:
            # Validate annotation
            validation_result = await self.validate_annotation(annotation_data, task_id)
            if not validation_result["is_valid"]:
                raise ValueError(validation_result["error_message"])
            
            # Store annotation
            annotation = AnnotationDB(
                id=uuid4(),
                task_id=task_id,
                annotator_id=annotator_id,
                annotation_data=annotation_data,
                confidence_score=confidence_score,
                time_spent_seconds=time_spent_seconds,
                created_at=datetime.utcnow()
            )
            
            self.db.add(annotation)
            
            # Update assignment completion
            await self.update_assignment_completion(task_id, annotator_id)
            
            await self.db.commit()
            
            # Check if task is complete
            task = await self.get_task(task_id)
            if await self.is_annotation_task_complete(task):
                await self.finalize_annotation_task(task)
            
            # Calculate inter-annotator agreement if multiple annotations
            agreement_score = await self.inter_annotator.calculate_agreement(task_id)
            if agreement_score is not None:
                await self.store_agreement_metrics(task_id, agreement_score)
            
            # Calculate quality score
            quality_score = await self.quality_controller.score_annotation(annotation_data)
            
            logger.info("Annotation submitted", 
                       task_id=str(task_id), 
                       annotator_id=str(annotator_id),
                       quality_score=quality_score)
            
            return {
                "task_id": str(task_id),
                "annotator_id": str(annotator_id),
                "annotation_id": str(annotation.id),
                "quality_score": quality_score,
                "agreement_score": agreement_score
            }
            
        except Exception as e:
            logger.error("Error submitting annotation", error=str(e), task_id=str(task_id))
            raise
    
    async def get_task(self, task_id: UUID) -> Optional[AnnotationTaskDB]:
        """Get annotation task by ID"""
        
        result = await self.db.execute(
            select(AnnotationTaskDB).where(AnnotationTaskDB.id == task_id)
        )
        return result.scalar_one_or_none()
    
    async def get_tasks_for_annotator(self, annotator_id: UUID, 
                                    status: Optional[TaskStatus] = None) -> List[AnnotationTask]:
        """Get annotation tasks for a specific annotator"""
        
        query = select(AnnotationTaskDB).join(AnnotationAssignment).where(
            AnnotationAssignment.annotator_id == annotator_id
        )
        
        if status:
            query = query.where(AnnotationTaskDB.status == status)
        
        result = await self.db.execute(query)
        tasks = result.scalars().all()
        
        return [self._db_to_schema(task) for task in tasks]
    
    async def get_annotation_guidelines(self, annotation_type: AnnotationType) -> str:
        """Get annotation guidelines for specific type"""
        
        guidelines = {
            AnnotationType.SENTIMENT: """
            Sentiment Annotation Guidelines:
            1. Read the entire text carefully
            2. Determine the overall emotional tone
            3. Classify as: positive, negative, or neutral
            4. Consider context and implied meaning
            5. If mixed sentiment, choose the dominant one
            """,
            AnnotationType.TOPIC: """
            Topic Annotation Guidelines:
            1. Identify the main subject matter
            2. Choose from predefined topic categories
            3. Select the most specific relevant topic
            4. If multiple topics, choose the primary one
            5. Use "other" only if no category fits
            """,
            AnnotationType.QUALITY: """
            Quality Annotation Guidelines:
            1. Assess content clarity and coherence
            2. Check for factual accuracy (if verifiable)
            3. Evaluate writing quality and structure
            4. Consider relevance and usefulness
            5. Rate on scale of 1-5 (1=very poor, 5=excellent)
            """,
            AnnotationType.BIAS: """
            Bias Annotation Guidelines:
            1. Look for subjective language and opinions
            2. Check for unfair representation of groups
            3. Identify loaded or emotional language
            4. Consider cultural and social context
            5. Mark as: unbiased, slightly biased, or heavily biased
            """,
            AnnotationType.FACTUAL: """
            Factual Annotation Guidelines:
            1. Verify claims against known facts
            2. Check for logical consistency
            3. Identify unsupported assertions
            4. Note any misleading information
            5. Rate factual accuracy on scale of 1-5
            """,
            AnnotationType.COMPLETENESS: """
            Completeness Annotation Guidelines:
            1. Assess if topic is fully covered
            2. Check for missing important information
            3. Evaluate depth of analysis
            4. Consider if conclusion is supported
            5. Rate completeness on scale of 1-5
            """
        }
        
        return guidelines.get(annotation_type, "Please provide accurate and consistent annotations.")
    
    async def validate_annotation(self, annotation_data: Dict[str, Any], 
                                task_id: UUID) -> Dict[str, Any]:
        """Validate annotation data"""
        
        try:
            # Get task to understand requirements
            task = await self.get_task(task_id)
            if not task:
                return {"is_valid": False, "error_message": "Task not found"}
            
            # Basic validation based on annotation type
            if task.annotation_type == AnnotationType.SENTIMENT:
                if "sentiment" not in annotation_data:
                    return {"is_valid": False, "error_message": "Sentiment field required"}
                
                valid_sentiments = ["positive", "negative", "neutral"]
                if annotation_data["sentiment"] not in valid_sentiments:
                    return {"is_valid": False, "error_message": "Invalid sentiment value"}
            
            elif task.annotation_type == AnnotationType.QUALITY:
                if "quality_score" not in annotation_data:
                    return {"is_valid": False, "error_message": "Quality score required"}
                
                score = annotation_data["quality_score"]
                if not isinstance(score, (int, float)) or score < 1 or score > 5:
                    return {"is_valid": False, "error_message": "Quality score must be 1-5"}
            
            # Add more validation rules as needed
            
            return {"is_valid": True, "error_message": None}
            
        except Exception as e:
            logger.error("Error validating annotation", error=str(e))
            return {"is_valid": False, "error_message": "Validation error"}
    
    async def update_assignment_completion(self, task_id: UUID, annotator_id: UUID) -> None:
        """Update assignment completion status"""
        
        result = await self.db.execute(
            select(AnnotationAssignment).where(
                and_(
                    AnnotationAssignment.task_id == task_id,
                    AnnotationAssignment.annotator_id == annotator_id
                )
            )
        )
        
        assignment = result.scalar_one_or_none()
        if assignment:
            assignment.completed_at = datetime.utcnow()
            await self.db.commit()
    
    async def is_annotation_task_complete(self, task: AnnotationTaskDB) -> bool:
        """Check if annotation task is complete"""
        
        # Get all assignments for this task
        result = await self.db.execute(
            select(AnnotationAssignment).where(AnnotationAssignment.task_id == task.id)
        )
        assignments = result.scalars().all()
        
        # Check if all assignments are completed
        return all(assignment.completed_at is not None for assignment in assignments)
    
    async def finalize_annotation_task(self, task: AnnotationTaskDB) -> None:
        """Finalize completed annotation task"""
        
        # Update task status
        task.status = TaskStatus.COMPLETED
        
        # Get all annotations for this task
        result = await self.db.execute(
            select(AnnotationDB).where(AnnotationDB.task_id == task.id)
        )
        annotations = result.scalars().all()
        
        # Calculate consensus annotation
        consensus = await self.calculate_consensus_annotation(annotations, task.annotation_type)
        
        # Store consensus (this would be implemented based on your needs)
        logger.info("Annotation task finalized", 
                   task_id=str(task.id), 
                   annotation_count=len(annotations))
        
        await self.db.commit()
    
    async def calculate_consensus_annotation(self, annotations: List[AnnotationDB], 
                                           annotation_type: AnnotationType) -> Dict[str, Any]:
        """Calculate consensus from multiple annotations"""
        
        if not annotations:
            return {}
        
        if len(annotations) == 1:
            return annotations[0].annotation_data
        
        # Simple majority voting for categorical annotations
        if annotation_type == AnnotationType.SENTIMENT:
            sentiments = [ann.annotation_data.get("sentiment") for ann in annotations]
            consensus_sentiment = max(set(sentiments), key=sentiments.count)
            return {"sentiment": consensus_sentiment}
        
        # Average for numerical annotations
        elif annotation_type == AnnotationType.QUALITY:
            scores = [ann.annotation_data.get("quality_score", 0) for ann in annotations]
            avg_score = sum(scores) / len(scores)
            return {"quality_score": round(avg_score, 2)}
        
        # Default to first annotation
        return annotations[0].annotation_data
    
    async def store_agreement_metrics(self, task_id: UUID, agreement_score: float) -> None:
        """Store inter-annotator agreement metrics"""
        
        # This would store agreement metrics in a dedicated table
        logger.info("Agreement metrics stored", 
                   task_id=str(task_id), 
                   agreement_score=agreement_score)
    
    async def notify_annotator(self, annotator_id: UUID, task: AnnotationTaskDB) -> None:
        """Notify annotator about new task"""
        
        # This would send notification via email, WebSocket, etc.
        logger.info("Annotator notified", 
                   annotator_id=str(annotator_id), 
                   task_id=str(task.id))
    
    def _db_to_schema(self, task: AnnotationTaskDB) -> AnnotationTask:
        """Convert database model to schema"""
        
        return AnnotationTask(
            id=task.id,
            content_id=task.content_id,
            annotation_type=task.annotation_type,
            guidelines=task.guidelines,
            deadline=task.deadline,
            status=task.status,
            created_at=task.created_at,
            updated_at=task.updated_at
        )

class InterAnnotatorAgreement:
    """Calculate inter-annotator agreement metrics"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def calculate_agreement(self, task_id: UUID) -> Optional[float]:
        """Calculate inter-annotator agreement for a task"""
        
        try:
            # Get all annotations for this task
            result = await self.db.execute(
                select(AnnotationDB).where(AnnotationDB.task_id == task_id)
            )
            annotations = result.scalars().all()
            
            if len(annotations) < 2:
                return None
            
            # Calculate agreement based on annotation type
            # This is a simplified implementation
            agreements = []
            
            for i in range(len(annotations)):
                for j in range(i + 1, len(annotations)):
                    agreement = self._calculate_pairwise_agreement(
                        annotations[i].annotation_data,
                        annotations[j].annotation_data
                    )
                    agreements.append(agreement)
            
            return sum(agreements) / len(agreements) if agreements else None
            
        except Exception as e:
            logger.error("Error calculating agreement", error=str(e), task_id=str(task_id))
            return None
    
    def _calculate_pairwise_agreement(self, annotation1: Dict[str, Any], 
                                    annotation2: Dict[str, Any]) -> float:
        """Calculate agreement between two annotations"""
        
        # Simple exact match for categorical data
        if "sentiment" in annotation1 and "sentiment" in annotation2:
            return 1.0 if annotation1["sentiment"] == annotation2["sentiment"] else 0.0
        
        # Numerical agreement for scores
        if "quality_score" in annotation1 and "quality_score" in annotation2:
            score1 = annotation1["quality_score"]
            score2 = annotation2["quality_score"]
            # Consider agreement if within 0.5 points
            return 1.0 if abs(score1 - score2) <= 0.5 else 0.0
        
        # Default to no agreement
        return 0.0
