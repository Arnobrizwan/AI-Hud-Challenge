"""
WebSocket Manager for Real-Time Connections
Handles WebSocket connections, messaging, and auto-reconnection
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


@dataclass
class WebSocketConnection:
    """WebSocket connection representation"""

    id: str
    websocket: WebSocket
    user_id: str
    connected_at: datetime
    last_heartbeat: datetime
    last_message_sent: Optional[datetime] = None
    subscriptions: Set[str] = field(default_factory=set)
    groups: Set[str] = field(default_factory=set)
    is_active: bool = True


@dataclass
class HeartbeatConfig:
    """Heartbeat configuration"""

    interval: int = 30  # seconds
    timeout: int = 60  # seconds
    max_missed: int = 3


class HeartbeatManager:
    """Manages WebSocket heartbeats and connection health"""

    def __init__(self, config: HeartbeatConfig = None):
        self.config = config or HeartbeatConfig()
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}

    async def start_monitoring(self, connection: WebSocketConnection) -> Dict[str, Any]:
        """Start heartbeat monitoring for a connection"""
        task = asyncio.create_task(self._monitor_connection(connection))
        self.monitoring_tasks[connection.id] = task

    async def stop_monitoring(self, connection_id: str) -> Dict[str, Any]:
        """Stop heartbeat monitoring for a connection"""
        if connection_id in self.monitoring_tasks:
            task = self.monitoring_tasks[connection_id]
            task.cancel()
            del self.monitoring_tasks[connection_id]

    async def _monitor_connection(self, connection: WebSocketConnection) -> Dict[str, Any]:
        """Monitor connection health with heartbeats"""
        missed_heartbeats = 0

        while connection.is_active:
            try:
                # Send heartbeat
                await self._send_heartbeat(connection)

                # Wait for heartbeat interval
                await asyncio.sleep(self.config.interval)

                # Check if heartbeat was received
                time_since_last = datetime.utcnow() - connection.last_heartbeat
                if time_since_last > timedelta(seconds=self.config.timeout):
                    missed_heartbeats += 1
                    logger.warning(
                        f"Missed heartbeat {missed_heartbeats} for connection {connection.id}"
                    )

                    if missed_heartbeats >= self.config.max_missed:
                        logger.error(f"Connection {connection.id} timed out")
                        connection.is_active = False
                        break
                else:
                    missed_heartbeats = 0

            except Exception as e:
                logger.error(
                    f"Error monitoring connection {connection.id}: {str(e)}")
                break

    async def _send_heartbeat(self, connection: WebSocketConnection) -> Dict[str, Any]:
        """Send heartbeat ping to connection"""
        try:
            heartbeat_message = {
                "type": "heartbeat",
                "timestamp": datetime.utcnow().isoformat(),
                "connection_id": connection.id,
            }
            await connection.websocket.send_json(heartbeat_message)
        except Exception as e:
            logger.error(
                f"Failed to send heartbeat to {connection.id}: {str(e)}")
            connection.is_active = False


class MessageQueue:
    """Message queue for WebSocket connections"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queues: Dict[str, List[Dict[str, Any]]] = {}

    async def enqueue_message(self, connection_id: str,
                              message: Dict[str, Any]):
         -> Dict[str, Any]:"""Enqueue message for a connection"""
        if connection_id not in self.queues:
            self.queues[connection_id] = []

        if len(self.queues[connection_id]) >= self.max_size:
            # Remove oldest message
            self.queues[connection_id].pop(0)

        self.queues[connection_id].append(message)

    async def dequeue_messages(
            self, connection_id: str) -> List[Dict[str, Any]]:
        """Dequeue all messages for a connection"""
        if connection_id not in self.queues:
            return []

        messages = self.queues[connection_id].copy()
        self.queues[connection_id].clear()
        return messages

    async def clear_queue(self, connection_id: str) -> Dict[str, Any]:
        """Clear message queue for a connection"""
        if connection_id in self.queues:
            self.queues[connection_id].clear()


class WebSocketManager:
    """Manage WebSocket connections and messaging"""

    def __init__(self, redis_client: redis.Redis = None):
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.connection_groups: Dict[str, Set[str]] = {}
        # user_id -> connection_ids
        self.user_connections: Dict[str, Set[str]] = {}
        self.heartbeat_manager = HeartbeatManager()
        self.message_queue = MessageQueue()
        self.redis_client = redis_client
        self.heartbeat_task: Optional[asyncio.Task] = None

    async def start_heartbeat_monitoring(self) -> Dict[str, Any]:
        """Start global heartbeat monitoring task"""
        self.heartbeat_task = asyncio.create_task(
            self._global_heartbeat_monitor())

    async def _global_heartbeat_monitor(self) -> Dict[str, Any]:
        """Global heartbeat monitoring for all connections"""
        while True:
            try:
    await asyncio.sleep(30)  # Check every 30 seconds

                # Clean up inactive connections
                inactive_connections = []
                for conn_id, connection in self.active_connections.items():
                    if not connection.is_active:
                        inactive_connections.append(conn_id)

                for conn_id in inactive_connections:
    await self.cleanup_connection(self.active_connections[conn_id])

            except Exception as e:
                logger.error(f"Error in global heartbeat monitor: {str(e)}")
                await asyncio.sleep(5)

    async def handle_websocket_connection(
            self, websocket: WebSocket, user_id: str):
         -> Dict[str, Any]:"""Handle new WebSocket connection"""
        await websocket.accept()

        connection = WebSocketConnection(
            id=str(uuid.uuid4()),
            websocket=websocket,
            user_id=user_id,
            connected_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
        )

        self.active_connections[connection.id] = connection

        # Add to user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection.id)

        try:
            # Send connection confirmation
            await self.send_message(
                connection,
                {
                    "type": "connection_established",
                    "connection_id": connection.id,
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            # Start heartbeat monitoring
            await self.heartbeat_manager.start_monitoring(connection)

            # Process queued messages
            queued_messages = await self.message_queue.dequeue_messages(connection.id)
            for message in queued_messages:
    await self.send_message(connection, message)

            # Process incoming messages
            async for message in self.receive_messages(connection):
    await self.handle_websocket_message(connection, message)

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection.id}")
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
        finally:
    await self.cleanup_connection(connection)

    async def receive_messages(self, connection: WebSocketConnection) -> Dict[str, Any]:
        """Receive messages from WebSocket connection"""
        while connection.is_active:
            try:
                message = await connection.websocket.receive_json()
                yield message
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(
                    f"Error receiving message from {connection.id}: {str(e)}")
                break

    async def handle_websocket_message(
        self, connection: WebSocketConnection, message: Dict[str, Any]
    ) -> None:
        """Handle incoming WebSocket message"""
        message_type = message.get("type")

        try:
            if message_type == "subscribe":
    await self.handle_subscription(connection, message)
            elif message_type == "unsubscribe":
    await self.handle_unsubscription(connection, message)
            elif message_type == "heartbeat":
    await self.handle_heartbeat(connection, message)
            elif message_type == "collaboration_event":
    await self.handle_collaboration_event(connection, message)
            elif message_type == "user_action":
    await self.handle_user_action(connection, message)
            elif message_type == "join_group":
    await self.handle_join_group(connection, message)
            elif message_type == "leave_group":
    await self.handle_leave_group(connection, message)
            else:
                logger.warning(f"Unknown message type: {message_type}")

        except Exception as e:
            logger.error(f"Error handling message {message_type}: {str(e)}")
            await self.send_error(connection, f"Error processing {message_type}", str(e))

    async def handle_subscription(
            self, connection: WebSocketConnection, message: Dict[str, Any]):
         -> Dict[str, Any]:"""Handle subscription request"""
        subscriptions = message.get("subscriptions", [])

        for subscription in subscriptions:
            connection.subscriptions.add(subscription)

            # Add to appropriate groups
            if subscription.startswith("group:"):
                group_id = subscription[6:]  # Remove 'group:' prefix
                await self.add_to_group(connection.id, group_id)

        await self.send_message(
            connection,
            {
                "type": "subscription_confirmed",
                "subscriptions": list(connection.subscriptions),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    async def handle_unsubscription(
            self, connection: WebSocketConnection, message: Dict[str, Any]):
         -> Dict[str, Any]:"""Handle unsubscription request"""
        subscriptions = message.get("subscriptions", [])

        for subscription in subscriptions:
            connection.subscriptions.discard(subscription)

            # Remove from groups
            if subscription.startswith("group:"):
                group_id = subscription[6:]
                await self.remove_from_group(connection.id, group_id)

        await self.send_message(
            connection,
            {
                "type": "unsubscription_confirmed",
                "subscriptions": list(connection.subscriptions),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    async def handle_heartbeat(
            self, connection: WebSocketConnection, message: Dict[str, Any]):
         -> Dict[str, Any]:"""Handle heartbeat response"""
        connection.last_heartbeat = datetime.utcnow()

        await self.send_message(
            connection, {"type": "heartbeat_ack", "timestamp": datetime.utcnow().isoformat()}
        )

    async def handle_collaboration_event(
        self, connection: WebSocketConnection, message: Dict[str, Any]
    ):
         -> Dict[str, Any]:"""Handle collaboration event"""
        # Forward to collaboration engine
        if hasattr(self, "collaboration_engine") and self.collaboration_engine:
    await self.collaboration_engine.handle_collaboration_event(
                message.get("session_id"), message.get("event")
            )

    async def handle_user_action(
            self, connection: WebSocketConnection, message: Dict[str, Any]):
         -> Dict[str, Any]:"""Handle user action"""
        action = message.get("action")
        data = message.get("data", {})

        # Process user action (e.g., save article, update preferences)
        logger.info(f"User action from {connection.user_id}: {action}")

        # Broadcast to other connections if needed
        if action in ["article_saved", "preferences_updated"]:
    await self.broadcast_to_user(
                connection.user_id,
                {
                    "type": "user_action_update",
                    "action": action,
                    "data": data,
                    "timestamp": datetime.utcnow().isoformat(),
                },
                exclude_connection=connection.id,
            )

    async def handle_join_group(
            self, connection: WebSocketConnection, message: Dict[str, Any]):
         -> Dict[str, Any]:"""Handle join group request"""
        group_id = message.get("group_id")
        if group_id:
    await self.add_to_group(connection.id, group_id)
            await self.send_message(
                connection,
                {
                    "type": "group_joined",
                    "group_id": group_id,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

    async def handle_leave_group(
            self, connection: WebSocketConnection, message: Dict[str, Any]):
         -> Dict[str, Any]:"""Handle leave group request"""
        group_id = message.get("group_id")
        if group_id:
    await self.remove_from_group(connection.id, group_id)
            await self.send_message(
                connection,
                {
                    "type": "group_left",
                    "group_id": group_id,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

    async def add_to_group(self, connection_id: str, group_id: str) -> Dict[str, Any]:
        """Add connection to group"""
        if group_id not in self.connection_groups:
            self.connection_groups[group_id] = set()

        self.connection_groups[group_id].add(connection_id)

        if connection_id in self.active_connections:
            self.active_connections[connection_id].groups.add(group_id)

    async def remove_from_group(self, connection_id: str, group_id: str) -> Dict[str, Any]:
        """Remove connection from group"""
        if group_id in self.connection_groups:
            self.connection_groups[group_id].discard(connection_id)

        if connection_id in self.active_connections:
            self.active_connections[connection_id].groups.discard(group_id)

    async def broadcast_to_group(
        self, group_id: str, message: Dict[str, Any], exclude_connection: str = None
    ) -> int:
        """Broadcast message to all connections in a group"""
        if group_id not in self.connection_groups:
            return 0

        connection_ids = self.connection_groups[group_id].copy()
        if exclude_connection:
            connection_ids.discard(exclude_connection)

        broadcast_tasks = []
        for connection_id in connection_ids:
            if connection_id in self.active_connections:
                connection = self.active_connections[connection_id]
                broadcast_tasks.append(self.send_message(connection, message))

        # Send to all connections concurrently
        if broadcast_tasks:
            results = await asyncio.gather(*broadcast_tasks, return_exceptions=True)
            return sum(
                1 for result in results if not isinstance(
                    result, Exception))

        return 0

    async def broadcast_to_user(
        self, user_id: str, message: Dict[str, Any], exclude_connection: str = None
    ) -> int:
        """Broadcast message to all connections of a user"""
        if user_id not in self.user_connections:
            return 0

        connection_ids = self.user_connections[user_id].copy()
        if exclude_connection:
            connection_ids.discard(exclude_connection)

        broadcast_tasks = []
        for connection_id in connection_ids:
            if connection_id in self.active_connections:
                connection = self.active_connections[connection_id]
                broadcast_tasks.append(self.send_message(connection, message))

        if broadcast_tasks:
            results = await asyncio.gather(*broadcast_tasks, return_exceptions=True)
            return sum(
                1 for result in results if not isinstance(
                    result, Exception))

        return 0

    async def send_message(self,
                           connection: WebSocketConnection,
                           message: Dict[str,
                                         Any]) -> bool:
        """Send message to specific connection with error handling"""
        try:
            if not connection.is_active:
                return False

            await connection.websocket.send_json(message)
            connection.last_message_sent = datetime.utcnow()
            return True
        except Exception as e:
            logger.error(
                f"Failed to send message to {connection.id}: {str(e)}")
            connection.is_active = False
            return False

    async def send_error(
            self,
            connection: WebSocketConnection,
            error_type: str,
            error_message: str):
         -> Dict[str, Any]:"""Send error message to connection"""
        await self.send_message(
            connection,
            {
                "type": "error",
                "error_type": error_type,
                "message": error_message,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    async def cleanup_connection(self, connection: WebSocketConnection) -> Dict[str, Any]:
        """Clean up connection resources"""
        try:
            # Stop heartbeat monitoring
            await self.heartbeat_manager.stop_monitoring(connection.id)

            # Remove from active connections
            if connection.id in self.active_connections:
                del self.active_connections[connection.id]

            # Remove from user connections
            if connection.user_id in self.user_connections:
                self.user_connections[connection.user_id].discard(
                    connection.id)
                if not self.user_connections[connection.user_id]:
                    del self.user_connections[connection.user_id]

            # Remove from all groups
            for group_id in connection.groups:
                if group_id in self.connection_groups:
                    self.connection_groups[group_id].discard(connection.id)

            # Close WebSocket if still open
            if connection.websocket.client_state.name == "CONNECTED":
    await connection.websocket.close()

            logger.info(f"Cleaned up connection {connection.id}")

        except Exception as e:
            logger.error(
                f"Error cleaning up connection {connection.id}: {str(e)}")

    async def get_connections_status(self) -> Dict[str, Any]:
        """Get status of active connections"""
        return {
            "total_connections": len(
                self.active_connections),
            "total_users": len(
                self.user_connections),
            "total_groups": len(
                self.connection_groups),
            "connections_by_user": {
                user_id: len(conn_ids) for user_id,
                conn_ids in self.user_connections.items()},
            "connections_by_group": {
                group_id: len(conn_ids) for group_id,
                conn_ids in self.connection_groups.items()},
        }

    async def create_connection(self, session) -> WebSocketConnection:
        """Create a new WebSocket connection (for API compatibility)"""
        # This would be used when creating connections via API
        # For now, return a placeholder
        return WebSocketConnection(
            id=str(uuid.uuid4()),
            websocket=None,  # Will be set when WebSocket is established
            user_id=session.user_id,
            connected_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
        )
