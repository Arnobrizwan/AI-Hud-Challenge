"""
Role-based access control engine
"""

import logging
from typing import List, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.database import User

logger = logging.getLogger(__name__)

class RoleBasedAccessControl:
    """Role-based access control for editorial workflows"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.role_permissions = {
            "admin": ["*"],  # All permissions
            "editor": ["review", "approve", "reject", "assign"],
            "annotator": ["annotate", "submit"],
            "viewer": ["view"]
        }
    
    async def get_eligible_reviewers(self, task_type: str, content_id: UUID) -> List[User]:
        """Get eligible reviewers for a task type"""
        
        try:
            # Get users with appropriate roles
            result = await self.db.execute(
                select(User).where(
                    User.role.in_(["admin", "editor"])
                ).where(User.is_active == True)
            )
            
            users = result.scalars().all()
            
            # Filter by task type requirements
            eligible_users = []
            for user in users:
                if self.can_handle_task_type(user.role, task_type):
                    eligible_users.append(user)
            
            return eligible_users
            
        except Exception as e:
            logger.error("Error getting eligible reviewers", error=str(e))
            return []
    
    async def can_complete_task(self, user_id: UUID, task: Any) -> bool:
        """Check if user can complete a task"""
        
        try:
            # Get user
            result = await self.db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user or not user.is_active:
                return False
            
            # Check if user is assigned to task
            if hasattr(task, 'assigned_to') and task.assigned_to != user_id:
                return False
            
            # Check role permissions
            return self.has_permission(user.role, "review")
            
        except Exception as e:
            logger.error("Error checking task completion permission", error=str(e))
            return False
    
    def can_handle_task_type(self, role: str, task_type: str) -> bool:
        """Check if role can handle task type"""
        
        # Define task type requirements
        task_requirements = {
            "content_review": ["admin", "editor"],
            "fact_check": ["admin", "editor"],
            "quality_review": ["admin", "editor"],
            "annotation": ["admin", "editor", "annotator"],
            "moderation": ["admin", "editor"]
        }
        
        required_roles = task_requirements.get(task_type, ["admin"])
        return role in required_roles
    
    def has_permission(self, role: str, permission: str) -> bool:
        """Check if role has specific permission"""
        
        permissions = self.role_permissions.get(role, [])
        return "*" in permissions or permission in permissions
