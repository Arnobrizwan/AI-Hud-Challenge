"""
WebSocket manager for real-time feedback processing
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog
from fastapi import WebSocket, WebSocketDisconnect

logger = structlog.get_logger(__name__)


class WebSocketManager:
    """Manage WebSocket connections and real-time communication"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.cleanup_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket, user_id: str) -> None:
        """Accept WebSocket connection and register user"""

        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.connection_metadata[user_id] = {
            "connected_at": datetime.utcnow(),
            "last_ping": datetime.utcnow(),
            "message_count": 0,
        }

        logger.info("WebSocket connected", user_id=user_id)

    def disconnect(self, user_id: str) -> None:
        """Disconnect user and clean up"""

        if user_id in self.active_connections:
            del self.active_connections[user_id]
            del self.connection_metadata[user_id]
            logger.info("WebSocket disconnected", user_id=user_id)

    async def send_personal_message(
            self, message: Dict[str, Any], user_id: str) -> bool:
        """Send message to specific user"""

        try:
            if user_id in self.active_connections:
                websocket = self.active_connections[user_id]
                await websocket.send_text(json.dumps(message))

                # Update metadata
                self.connection_metadata[user_id]["message_count"] += 1
                self.connection_metadata[user_id]["last_ping"] = datetime.utcnow(
                )

                return True
            else:
                logger.warning("User not connected", user_id=user_id)
                return False

        except Exception as e:
            logger.error(
                "Error sending personal message",
                error=str(e),
                user_id=user_id)
            return False

    async def broadcast_message(
        self, message: Dict[str, Any], exclude_user: Optional[str] = None
    ) -> int:
        """Broadcast message to all connected users"""

        sent_count = 0

        for user_id, websocket in self.active_connections.items():
            if exclude_user and user_id == exclude_user:
                continue

            try:
    await websocket.send_text(json.dumps(message))
                sent_count += 1

                # Update metadata
                self.connection_metadata[user_id]["message_count"] += 1
                self.connection_metadata[user_id]["last_ping"] = datetime.utcnow(
                )

            except Exception as e:
                logger.error(
                    "Error broadcasting message",
                    error=str(e),
                    user_id=user_id)
                # Remove failed connection
                self.disconnect(user_id)

        logger.info(
            "Message broadcasted",
            sent_count=sent_count,
            total_connections=len(self.active_connections),
        )
        return sent_count

    async def process_realtime_feedback(
        self, user_id: str, feedback_data: Dict[str, Any]
    ) -> Dict[str, Any]:
    """Process real-time feedback and return result"""
        try:
            # Basic validation
            if not feedback_data.get("content_id"):
                return {"status": "error", "message": "Content ID is required"}

            # Process feedback (simplified)
            feedback_id = f"rt_{datetime.utcnow().timestamp()}"

            # Simulate processing
            await asyncio.sleep(0.1)  # Simulate processing time

            result = {
                "status": "processed",
                "feedback_id": feedback_id,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
            }

            # Broadcast to relevant stakeholders
            await self.broadcast_feedback_update(feedback_data)

            return result

        except Exception as e:
            logger.error(
                "Error processing real-time feedback",
                error=str(e),
                user_id=user_id)
            return {"status": "error", "message": str(e)}

    async def broadcast_feedback_update(
            self, feedback_data: Dict[str, Any]) -> None:
        """Broadcast feedback update to relevant stakeholders"""

        message = {
            "type": "feedback_update",
            "data": feedback_data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self.broadcast_message(message)

    async def broadcast_task_update(self, task_data: Any) -> None:
        """Broadcast task update to relevant stakeholders"""

        message = {
            "type": "task_update",
            "data": {
                "task_id": str(
                    task_data.id),
                "status": (
                    task_data.status.value if hasattr(
                        task_data.status,
                        "value") else str(
                        task_data.status)),
                "assigned_to": str(
                    task_data.assigned_to) if task_data.assigned_to else None,
                "task_type": task_data.task_type,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self.broadcast_message(message)

    async def broadcast_review_completion(
            self, task_data: Any, review_result: Any) -> None:
        """Broadcast review completion notification"""

        message = {
            "type": "review_completion",
            "data": {
                "task_id": str(task_data.id),
                "decision": (
                    review_result.decision.value
                    if hasattr(review_result.decision, "value")
                    else str(review_result.decision)
                ),
                "reviewer_id": str(review_result.reviewer_id),
                "content_id": str(task_data.content_id),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self.broadcast_message(message)

    async def send_alert(
        self, alert_data: Dict[str, Any], target_users: Optional[List[str]] = None
    ) -> int:
        """Send alert to specific users or all users"""

        message = {"type": "alert", "data": alert_data,
                   "timestamp": datetime.utcnow().isoformat()}

        if target_users:
            sent_count = 0
            for user_id in target_users:
                if await self.send_personal_message(message, user_id):
                    sent_count += 1
            return sent_count
        else:
            return await self.broadcast_message(message)

    async def start_cleanup_task(self) -> None:
        """Start background cleanup task for stale connections"""

        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(
                self._cleanup_stale_connections())

    async def _cleanup_stale_connections(self) -> None:
        """Clean up stale WebSocket connections"""

        while True:
            try:
    await asyncio.sleep(300)  # Check every 5 minutes

                current_time = datetime.utcnow()
                stale_users = []

                for user_id, metadata in self.connection_metadata.items():
                    # Consider connection stale if no ping in 10 minutes
                    if (current_time -
                            metadata["last_ping"]).total_seconds() > 600:
                        stale_users.append(user_id)

                # Remove stale connections
                for user_id in stale_users:
                    self.disconnect(user_id)

                if stale_users:
                    logger.info(
                        "Cleaned up stale connections",
                        count=len(stale_users))

            except Exception as e:
                logger.error("Error in cleanup task", error=str(e))

    async def cleanup(self) -> None:
        """Clean up all connections and tasks"""

        # Cancel cleanup task
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
    await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for user_id, websocket in self.active_connections.items():
            try:
    await websocket.close()
            except Exception as e:
                logger.error(
                    "Error closing WebSocket",
                    error=str(e),
                    user_id=user_id)

        self.active_connections.clear()
        self.connection_metadata.clear()

        logger.info("WebSocket manager cleaned up")

    def get_connection_stats(self) -> Dict[str, Any]:
    """Get connection statistics"""
        return {
            "active_connections": len(self.active_connections),
            "total_messages_sent": sum(
                metadata["message_count"] for metadata in self.connection_metadata.values()
            ),
            "connections": [
                {
                    "user_id": user_id,
                    "connected_at": metadata["connected_at"].isoformat(),
                    "last_ping": metadata["last_ping"].isoformat(),
                    "message_count": metadata["message_count"],
                }
                for user_id, metadata in self.connection_metadata.items()
            ],
        }
