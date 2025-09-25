"""
Server-Sent Events Manager
Handles SSE connections for real-time updates
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

import redis.asyncio as redis
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


@dataclass
class SSEStream:
    """SSE stream representation"""

    id: str
    user_id: str
    created_at: datetime
    last_event_id: int = 0
    is_active: bool = True
    subscriptions: List[str] = None

    def __post_init__(self):
        if self.subscriptions is None:
            self.subscriptions = []


class EventFormatter:
    """Formats events for SSE transmission"""

    @staticmethod
    def format_sse_event(event_data: Dict[str, Any]) -> str:
        """Format event for SSE transmission"""
        event_lines = []

        # Add event ID
        if "id" in event_data:
            event_lines.append(f"id: {event_data['id']}")
        else:
            event_lines.append(
                f"id: {int(datetime.utcnow().timestamp() * 1000)}")

        # Add event type
        if "type" in event_data:
            event_lines.append(f"event: {event_data['type']}")

        # Add retry interval
        if "retry" in event_data:
            event_lines.append(f"retry: {event_data['retry']}")
        else:
            event_lines.append("retry: 30000")  # 30 seconds default

        # Add data
        data = event_data.get("data", event_data)
        if isinstance(data, dict):
            data = json.dumps(data)

        # Split data into lines and prefix each with "data: "
        for line in str(data).split("\n"):
            event_lines.append(f"data: {line}")

        return "\n".join(event_lines) + "\n\n"

    @staticmethod
    def format_heartbeat_event() -> str:
        """Format heartbeat event"""
        return EventFormatter.format_sse_event(
            {"type": "heartbeat", "data": {"timestamp": datetime.utcnow().isoformat()}}
        )


class SSEManager:
    """Server-Sent Events for real-time updates"""

    def __init__(self, redis_client: redis.Redis = None):
        self.active_streams: Dict[str, SSEStream] = {}
        self.event_formatter = EventFormatter()
        self.redis_client = redis_client
        self.heartbeat_interval = 30  # seconds
        self.heartbeat_task: Optional[asyncio.Task] = None

    async def start_heartbeat_monitoring(self) -> Dict[str, Any]:
    """Start heartbeat monitoring for all streams"""
        self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

    async def _heartbeat_monitor(self) -> Dict[str, Any]:
    """Monitor and send heartbeats to active streams"""
        while True:
            try:
    await asyncio.sleep(self.heartbeat_interval)

                # Send heartbeat to all active streams
                inactive_streams = []
                for stream_id, stream in self.active_streams.items():
                    if stream.is_active:
                        try:
                            # This would send heartbeat to the stream
                            # For now, just log
                            logger.debug(
                                f"Sending heartbeat to stream {stream_id}")
                        except Exception as e:
                            logger.error(
                                f"Error sending heartbeat to {stream_id}: {str(e)}")
                            stream.is_active = False
                            inactive_streams.append(stream_id)
                    else:
                        inactive_streams.append(stream_id)

                # Clean up inactive streams
                for stream_id in inactive_streams:
                    if stream_id in self.active_streams:
                        del self.active_streams[stream_id]

            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {str(e)}")
                await asyncio.sleep(5)

    async def create_sse_endpoint(
            self,
            request: Request,
            user_id: str) -> StreamingResponse:
        """Create SSE endpoint for real-time updates"""
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID required")

        stream_id = str(uuid.uuid4())

        async def event_generator() -> Dict[str, Any]:
    stream = SSEStream(
                id=stream_id,
                user_id=user_id,
                created_at=datetime.utcnow())

            self.active_streams[stream_id] = stream

            try:
                # Send connection established event
                yield self.event_formatter.format_sse_event(
                    {
                        "type": "connection_established",
                        "data": {
                            "stream_id": stream_id,
                            "user_id": user_id,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    }
                )

                # Send periodic updates
                async for event in self.get_user_event_stream(user_id, stream):
                    formatted_event = self.event_formatter.format_sse_event(
                        event)
                    yield formatted_event

            except GeneratorExit:
                logger.info(f"SSE stream closed: {stream_id}")
            except Exception as e:
                logger.error(f"Error in SSE stream {stream_id}: {str(e)}")
            finally:
                if stream_id in self.active_streams:
                    del self.active_streams[stream_id]

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    async def get_user_event_stream(
        self, user_id: str, stream: SSEStream
    ) -> AsyncIterator[Dict[str, Any]]:
        """Get event stream for user"""
        last_heartbeat = datetime.utcnow()

        while stream.is_active:
            try:
                current_time = datetime.utcnow()

                # Send heartbeat every 30 seconds
                if (current_time -
                        last_heartbeat).seconds >= self.heartbeat_interval:
                    yield {
                        "type": "heartbeat",
                        "data": {"timestamp": current_time.isoformat(), "stream_id": stream.id},
                    }
                    last_heartbeat = current_time

                # Check for user-specific events
                user_events = await self.get_user_events(user_id)
                for event in user_events:
                    yield event

                # Check for subscription events
                subscription_events = await self.get_subscription_events(stream.subscriptions)
                for event in subscription_events:
                    yield event

                # Wait before next check
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in user event stream: {str(e)}")
                await asyncio.sleep(5)

    async def get_user_events(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user-specific events"""
        events = []

        # Simulate user events (notifications, updates, etc.)
        if int(datetime.utcnow().timestamp()) % 10 == 0:  # Every 10 seconds
            events.append(
                {
                    "type": "user_notification",
                    "data": {
                        "id": str(
                            uuid.uuid4()),
                        "title": f"Notification for {user_id}",
                        "message": f'You have a new update at {datetime.utcnow().strftime("%H:%M:%S")}',
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                })

        return events

    async def get_subscription_events(
            self, subscriptions: List[str]) -> List[Dict[str, Any]]:
        """Get events for subscriptions"""
        events = []

        for subscription in subscriptions:
            if subscription == "articles":
                # Simulate article updates
                if int(datetime.utcnow().timestamp()
                       ) % 15 == 0:  # Every 15 seconds
                    events.append(
                        {
                            "type": "article_update",
                            "data": {
                                "id": str(
                                    uuid.uuid4()),
                                "title": f'Breaking News {datetime.utcnow().strftime("%H:%M:%S")}',
                                "summary": f"Latest breaking news update",
                                "timestamp": datetime.utcnow().isoformat(),
                            },
                        })

            elif subscription == "trending":
                # Simulate trending topics updates
                if int(datetime.utcnow().timestamp()
                       ) % 20 == 0:  # Every 20 seconds
                    events.append(
                        {
                            "type": "trending_update",
                            "data": {
                                "topics": [
                                    {"name": "AI Technology", "score": 0.95},
                                    {"name": "Climate Change", "score": 0.87},
                                    {"name": "Space Exploration", "score": 0.82},
                                ],
                                "timestamp": datetime.utcnow().isoformat(),
                            },
                        }
                    )

            elif subscription == "analytics":
                # Simulate analytics updates
                if int(datetime.utcnow().timestamp()
                       ) % 5 == 0:  # Every 5 seconds
                    events.append(
                        {
                            "type": "analytics_update",
                            "data": {
                                "metrics": {
                                    "active_users": 1250,
                                    "articles_processed": 5420,
                                    "response_time": 45.2,
                                },
                                "timestamp": datetime.utcnow().isoformat(),
                            },
                        }
                    )

        return events

    async def send_event_to_stream(
            self, stream_id: str, event: Dict[str, Any]) -> bool:
        """Send event to specific stream"""
        if stream_id not in self.active_streams:
            return False

        stream = self.active_streams[stream_id]
        if not stream.is_active:
            return False

        try:
            # This would send the event to the stream
            # For now, just log
            logger.info(
                f"Sending event to stream {stream_id}: {event['type']}")
            return True
        except Exception as e:
            logger.error(
                f"Error sending event to stream {stream_id}: {str(e)}")
            return False

    async def broadcast_event(
            self, event: Dict[str, Any], target_users: List[str] = None) -> int:
        """Broadcast event to multiple streams"""
        sent_count = 0

        for stream_id, stream in self.active_streams.items():
            if target_users and stream.user_id not in target_users:
                continue

            if await self.send_event_to_stream(stream_id, event):
                sent_count += 1

        return sent_count

    async def create_connection(self, session) -> Any:
        """Create SSE connection (for API compatibility)"""
        # This would create an SSE connection
        # For now, return a placeholder
        return type(
            "SSEConnection", (), {
                "id": str(
                    uuid.uuid4()), "send_initial_data": lambda data: None})()

    async def get_streams_status(self) -> Dict[str, Any]:
    """Get status of active streams"""
        return {
            "total_streams": len(self.active_streams),
            "active_streams": [
                {
                    "id": stream.id,
                    "user_id": stream.user_id,
                    "created_at": stream.created_at.isoformat(),
                    "subscriptions": stream.subscriptions,
                }
                for stream in self.active_streams.values()
                if stream.is_active
            ],
        }

    async def close_stream(self, stream_id: str) -> bool:
        """Close specific stream"""
        if stream_id in self.active_streams:
            stream = self.active_streams[stream_id]
            stream.is_active = False
            del self.active_streams[stream_id]
            return True
        return False

    async def close_user_streams(self, user_id: str) -> int:
        """Close all streams for a user"""
        closed_count = 0

        streams_to_close = [
            stream_id
            for stream_id, stream in self.active_streams.items()
            if stream.user_id == user_id
        ]

        for stream_id in streams_to_close:
            if await self.close_stream(stream_id):
                closed_count += 1

        return closed_count
