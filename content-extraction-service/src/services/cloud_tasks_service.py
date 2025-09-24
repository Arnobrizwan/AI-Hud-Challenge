"""
GCP Cloud Tasks service for asynchronous processing.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2
from loguru import logger

from ..exceptions import TaskProcessingError


class CloudTasksService:
    """GCP Cloud Tasks service for asynchronous content processing."""

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        queue_name: str = "content-extraction-queue"
    ):
        """Initialize Cloud Tasks service."""
        self.project_id = project_id
        self.location = location
        self.queue_name = queue_name
        
        # Initialize Cloud Tasks client
        self.client = tasks_v2.CloudTasksClient()
        
        # Queue path
        self.queue_path = self.client.queue_path(project_id, location, queue_name)

    async def create_extraction_task(
        self,
        url: str,
        task_data: Dict[str, Any],
        delay_seconds: int = 0,
        task_name: Optional[str] = None
    ) -> str:
        """
        Create a content extraction task.
        
        Args:
            url: URL to extract content from
            task_data: Additional task data
            delay_seconds: Delay before task execution
            task_name: Optional custom task name
            
        Returns:
            Task name/ID
        """
        try:
            logger.info(f"Creating extraction task for URL: {url}")
            
            # Prepare task payload
            payload = {
                'url': url,
                'timestamp': datetime.utcnow().isoformat(),
                **task_data
            }
            
            # Create task
            task = {
                'http_request': {
                    'http_method': tasks_v2.HttpMethod.POST,
                    'url': self._get_task_handler_url(),
                    'headers': {
                        'Content-Type': 'application/json'
                    },
                    'body': json.dumps(payload).encode()
                }
            }
            
            # Add delay if specified
            if delay_seconds > 0:
                schedule_time = datetime.utcnow() + timedelta(seconds=delay_seconds)
                timestamp = timestamp_pb2.Timestamp()
                timestamp.FromDatetime(schedule_time)
                task['schedule_time'] = timestamp
            
            # Create task name if not provided
            if not task_name:
                task_name = f"extract-{url.replace('://', '-').replace('/', '-')}-{int(datetime.utcnow().timestamp())}"
            
            # Create the task
            response = self.client.create_task(
                parent=self.queue_path,
                task=task,
                task_id=task_name
            )
            
            logger.info(f"Task created successfully: {response.name}")
            return response.name
            
        except Exception as e:
            logger.error(f"Task creation failed for {url}: {str(e)}")
            raise TaskProcessingError(f"Task creation failed: {str(e)}")

    async def create_batch_extraction_task(
        self,
        urls: List[str],
        task_data: Dict[str, Any],
        delay_seconds: int = 0,
        batch_size: int = 10
    ) -> List[str]:
        """
        Create batch content extraction tasks.
        
        Args:
            urls: List of URLs to extract content from
            task_data: Additional task data
            delay_seconds: Delay before task execution
            batch_size: Number of URLs per batch
            
        Returns:
            List of task names/IDs
        """
        try:
            logger.info(f"Creating batch extraction tasks for {len(urls)} URLs")
            
            task_names = []
            
            # Split URLs into batches
            for i in range(0, len(urls), batch_size):
                batch_urls = urls[i:i + batch_size]
                batch_task_data = {
                    **task_data,
                    'batch_urls': batch_urls,
                    'batch_index': i // batch_size,
                    'total_batches': (len(urls) + batch_size - 1) // batch_size
                }
                
                # Create batch task
                task_name = await self.create_extraction_task(
                    url=batch_urls[0],  # Use first URL as identifier
                    task_data=batch_task_data,
                    delay_seconds=delay_seconds,
                    task_name=f"batch-{i // batch_size}-{int(datetime.utcnow().timestamp())}"
                )
                
                task_names.append(task_name)
            
            logger.info(f"Created {len(task_names)} batch tasks")
            return task_names
            
        except Exception as e:
            logger.error(f"Batch task creation failed: {str(e)}")
            raise TaskProcessingError(f"Batch task creation failed: {str(e)}")

    async def get_task_status(self, task_name: str) -> Dict[str, Any]:
        """
        Get task status and information.
        
        Args:
            task_name: Task name/ID
            
        Returns:
            Task status information
        """
        try:
            # Get task details
            task = self.client.get_task(name=task_name)
            
            return {
                'name': task.name,
                'state': task.state.name,
                'create_time': task.create_time,
                'schedule_time': task.schedule_time,
                'dispatch_time': task.dispatch_time,
                'response_time': task.response_time,
                'attempt_count': task.dispatch_count,
                'max_attempts': task.max_attempts,
                'http_method': task.http_request.http_method.name,
                'url': task.http_request.url
            }
            
        except Exception as e:
            logger.error(f"Task status retrieval failed for {task_name}: {str(e)}")
            return {
                'name': task_name,
                'state': 'UNKNOWN',
                'error': str(e)
            }

    async def list_tasks(
        self,
        limit: int = 100,
        filter_expression: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List tasks in the queue.
        
        Args:
            limit: Maximum number of tasks to return
            filter_expression: Optional filter expression
            
        Returns:
            List of task information
        """
        try:
            tasks = []
            
            # List tasks
            request = {
                'parent': self.queue_path,
                'page_size': limit
            }
            
            if filter_expression:
                request['filter'] = filter_expression
            
            response = self.client.list_tasks(request=request)
            
            for task in response:
                task_info = {
                    'name': task.name,
                    'state': task.state.name,
                    'create_time': task.create_time,
                    'schedule_time': task.schedule_time,
                    'dispatch_time': task.dispatch_time,
                    'response_time': task.response_time,
                    'attempt_count': task.dispatch_count,
                    'max_attempts': task.max_attempts
                }
                tasks.append(task_info)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Task listing failed: {str(e)}")
            return []

    async def delete_task(self, task_name: str) -> bool:
        """
        Delete a task.
        
        Args:
            task_name: Task name/ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_task(name=task_name)
            logger.info(f"Task deleted successfully: {task_name}")
            return True
            
        except Exception as e:
            logger.error(f"Task deletion failed for {task_name}: {str(e)}")
            return False

    async def purge_queue(self) -> int:
        """
        Purge all tasks from the queue.
        
        Returns:
            Number of tasks purged
        """
        try:
            logger.info("Purging queue")
            
            # List all tasks
            tasks = await self.list_tasks(limit=1000)
            
            # Delete each task
            deleted_count = 0
            for task in tasks:
                if await self.delete_task(task['name']):
                    deleted_count += 1
            
            logger.info(f"Purged {deleted_count} tasks from queue")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Queue purging failed: {str(e)}")
            return 0

    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.
        
        Returns:
            Queue statistics
        """
        try:
            # Get queue information
            queue = self.client.get_queue(name=self.queue_path)
            
            # List tasks to get counts
            all_tasks = await self.list_tasks(limit=1000)
            
            # Count tasks by state
            state_counts = {}
            for task in all_tasks:
                state = task['state']
                state_counts[state] = state_counts.get(state, 0) + 1
            
            return {
                'queue_name': queue.name,
                'state': queue.state.name,
                'total_tasks': len(all_tasks),
                'state_counts': state_counts,
                'rate_limits': {
                    'max_dispatches_per_second': queue.rate_limits.max_dispatches_per_second,
                    'max_burst_size': queue.rate_limits.max_burst_size,
                    'max_concurrent_dispatches': queue.rate_limits.max_concurrent_dispatches
                },
                'retry_config': {
                    'max_attempts': queue.retry_config.max_attempts,
                    'max_retry_duration': queue.retry_config.max_retry_duration,
                    'max_backoff': queue.retry_config.max_backoff,
                    'min_backoff': queue.retry_config.min_backoff
                }
            }
            
        except Exception as e:
            logger.error(f"Queue stats retrieval failed: {str(e)}")
            return {
                'queue_name': self.queue_path,
                'state': 'UNKNOWN',
                'total_tasks': 0,
                'state_counts': {},
                'error': str(e)
            }

    def _get_task_handler_url(self) -> str:
        """Get the URL for the task handler endpoint."""
        # This would be the actual endpoint URL in production
        return f"https://{self.project_id}.run.app/api/v1/tasks/extract"

    async def create_retry_task(
        self,
        original_task_name: str,
        retry_data: Dict[str, Any],
        delay_seconds: int = 60
    ) -> str:
        """
        Create a retry task for a failed task.
        
        Args:
            original_task_name: Name of the original task
            retry_data: Additional retry data
            delay_seconds: Delay before retry
            
        Returns:
            Retry task name/ID
        """
        try:
            logger.info(f"Creating retry task for: {original_task_name}")
            
            retry_task_data = {
                **retry_data,
                'original_task': original_task_name,
                'retry_attempt': retry_data.get('retry_attempt', 0) + 1,
                'is_retry': True
            }
            
            retry_task_name = await self.create_extraction_task(
                url=retry_data.get('url', ''),
                task_data=retry_task_data,
                delay_seconds=delay_seconds,
                task_name=f"retry-{original_task_name}-{int(datetime.utcnow().timestamp())}"
            )
            
            return retry_task_name
            
        except Exception as e:
            logger.error(f"Retry task creation failed: {str(e)}")
            raise TaskProcessingError(f"Retry task creation failed: {str(e)}")

    async def schedule_periodic_task(
        self,
        task_data: Dict[str, Any],
        cron_expression: str,
        task_name: Optional[str] = None
    ) -> str:
        """
        Schedule a periodic task using cron expression.
        
        Args:
            task_data: Task data
            cron_expression: Cron expression for scheduling
            task_name: Optional custom task name
            
        Returns:
            Task name/ID
        """
        try:
            logger.info(f"Scheduling periodic task with cron: {cron_expression}")
            
            # For Cloud Tasks, we'll create a task that schedules itself
            # This is a simplified implementation
            periodic_task_data = {
                **task_data,
                'cron_expression': cron_expression,
                'is_periodic': True
            }
            
            task_name = await self.create_extraction_task(
                url=task_data.get('url', ''),
                task_data=periodic_task_data,
                delay_seconds=0,
                task_name=task_name or f"periodic-{int(datetime.utcnow().timestamp())}"
            )
            
            return task_name
            
        except Exception as e:
            logger.error(f"Periodic task scheduling failed: {str(e)}")
            raise TaskProcessingError(f"Periodic task scheduling failed: {str(e)}")
