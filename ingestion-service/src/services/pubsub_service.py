"""
Google Cloud Pub/Sub service for message publishing and subscription.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from google.api_core import retry
from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.subscriber.message import Message

from src.config.settings import settings
from src.models.content import NormalizedArticle, ProcessingBatch

logger = logging.getLogger(__name__)


class PubSubService:
    """Google Cloud Pub/Sub service for message handling."""

    def __init__(self):
        self.project_id = settings.GCP_PROJECT_ID
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()

        # Topic paths
        self.ingestion_topic_path = self.publisher.topic_path(self.project_id, settings.PUBSUB_TOPIC_INGESTION)
        self.normalization_topic_path = self.publisher.topic_path(self.project_id, settings.PUBSUB_TOPIC_NORMALIZATION)

        # Subscription path
        self.subscription_path = self.subscriber.subscription_path(
            self.project_id, settings.PUBSUB_SUBSCRIPTION_INGESTION
        )

    async def publish_article(self, article: NormalizedArticle, topic: str = None) -> str:
        """Publish a single article to Pub/Sub."""
        try:
            # Choose topic
            if topic == "normalization":
                topic_path = self.normalization_topic_path
            else:
                topic_path = self.ingestion_topic_path

            # Prepare message data
            message_data = {
                "article_id": article.id,
                "url": article.url,
                "title": article.title,
                "source": article.source,
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "language": article.language,
                "content_type": article.content_type.value,
                "word_count": article.word_count,
                "content_hash": article.content_hash,
                "processing_status": article.processing_status.value,
                "ingestion_metadata": article.ingestion_metadata,
                "published_timestamp": datetime.utcnow().isoformat(),
            }

            # Serialize message
            message_json = json.dumps(message_data, default=str)
            message_bytes = message_json.encode("utf-8")

            # Publish message
            future = self.publisher.publish(
                topic_path,
                message_bytes,
                article_id=article.id,
                source_id=article.ingestion_metadata.get("source_id", ""),
                content_type=article.content_type.value,
                language=article.language,
            )

            message_id = future.result()
            logger.debug(f"Published article {article.id} to {topic_path}: {message_id}")

            return message_id

        except Exception as e:
            logger.error(f"Error publishing article {article.id}: {e}")
            raise

    async def publish_articles(self, articles: List[NormalizedArticle], topic: str = None) -> List[str]:
        """Publish multiple articles to Pub/Sub."""
        message_ids = []

        for article in articles:
            try:
                message_id = await self.publish_article(article, topic)
                message_ids.append(message_id)
            except Exception as e:
                logger.error(f"Error publishing article {article.id}: {e}")
                continue

        return message_ids

    async def publish_batch(self, batch: ProcessingBatch, topic: str = None) -> str:
        """Publish a processing batch to Pub/Sub."""
        try:
            # Choose topic
            if topic == "normalization":
                topic_path = self.normalization_topic_path
            else:
                topic_path = self.ingestion_topic_path

            # Prepare batch data
            batch_data = {
                "batch_id": batch.batch_id,
                "source_id": batch.source_id,
                "total_count": batch.total_count,
                "processed_count": batch.processed_count,
                "failed_count": batch.failed_count,
                "duplicate_count": batch.duplicate_count,
                "status": batch.status.value,
                "created_at": batch.created_at.isoformat(),
                "started_at": batch.started_at.isoformat() if batch.started_at else None,
                "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
                "processing_time_seconds": batch.processing_time_seconds,
                "success_rate": batch.success_rate,
                "published_timestamp": datetime.utcnow().isoformat(),
            }

            # Serialize message
            message_json = json.dumps(batch_data, default=str)
            message_bytes = message_json.encode("utf-8")

            # Publish message
            future = self.publisher.publish(
                topic_path,
                message_bytes,
                batch_id=batch.batch_id,
                source_id=batch.source_id,
                status=batch.status.value,
            )

            message_id = future.result()
            logger.debug(f"Published batch {batch.batch_id} to {topic_path}: {message_id}")

            return message_id

        except Exception as e:
            logger.error(f"Error publishing batch {batch.batch_id}: {e}")
            raise

    async def subscribe_to_ingestion(self, callback, max_messages: int = 10) -> None:
        """Subscribe to ingestion messages."""
        try:

            def message_handler(message: Message):
                try:
                    # Decode message
                    message_data = json.loads(message.data.decode("utf-8"))

                    # Process message
                    asyncio.create_task(callback(message_data))

                    # Acknowledge message
                    message.ack()

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    message.nack()

            # Start subscription
            flow_control = pubsub_v1.types.FlowControl(max_messages=max_messages)

            streaming_pull_future = self.subscriber.pull(
                request={
                    "subscription": self.subscription_path,
                    "max_messages": max_messages,
                },
                callback=message_handler,
                flow_control=flow_control,
            )

            logger.info(f"Started subscription to {self.subscription_path}")

            # Keep the subscription running
            try:
                streaming_pull_future.result()
            except KeyboardInterrupt:
                streaming_pull_future.cancel()
                streaming_pull_future.result()

        except Exception as e:
            logger.error(f"Error setting up subscription: {e}")
            raise

    async def create_topic(self, topic_name: str) -> str:
        """Create a Pub/Sub topic."""
        try:
            topic_path = self.publisher.topic_path(self.project_id, topic_name)

            topic = self.publisher.create_topic(request={"name": topic_path})
            logger.info(f"Created topic: {topic.name}")

            return topic.name

        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Topic {topic_name} already exists")
                return self.publisher.topic_path(self.project_id, topic_name)
            else:
                logger.error(f"Error creating topic {topic_name}: {e}")
                raise

    async def create_subscription(self, topic_name: str, subscription_name: str) -> str:
        """Create a Pub/Sub subscription."""
        try:
            topic_path = self.publisher.topic_path(self.project_id, topic_name)
            subscription_path = self.subscriber.subscription_path(self.project_id, subscription_name)

            subscription = self.subscriber.create_subscription(
                request={
                    "name": subscription_path,
                    "topic": topic_path,
                }
            )

            logger.info(f"Created subscription: {subscription.name}")
            return subscription.name

        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Subscription {subscription_name} already exists")
                return self.subscriber.subscription_path(self.project_id, subscription_name)
            else:
                logger.error(f"Error creating subscription {subscription_name}: {e}")
                raise

    async def get_topic_info(self, topic_name: str) -> Dict[str, Any]:
        """Get information about a topic."""
        try:
            topic_path = self.publisher.topic_path(self.project_id, topic_name)
            topic = self.publisher.get_topic(request={"topic": topic_path})

            return {
                "name": topic.name,
                "labels": dict(topic.labels),
                "message_retention_duration": str(topic.message_retention_duration),
                "kms_key_name": topic.kms_key_name,
                "schema_settings": (
                    {
                        "schema": topic.schema_settings.schema,
                        "encoding": topic.schema_settings.encoding.name,
                    }
                    if topic.schema_settings
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"Error getting topic info for {topic_name}: {e}")
            return {}

    async def get_subscription_info(self, subscription_name: str) -> Dict[str, Any]:
        """Get information about a subscription."""
        try:
            subscription_path = self.subscriber.subscription_path(self.project_id, subscription_name)
            subscription = self.subscriber.get_subscription(request={"subscription": subscription_path})

            return {
                "name": subscription.name,
                "topic": subscription.topic,
                "push_config": (
                    {
                        "push_endpoint": subscription.push_config.push_endpoint,
                        "attributes": dict(subscription.push_config.attributes),
                    }
                    if subscription.push_config
                    else None
                ),
                "ack_deadline_seconds": subscription.ack_deadline_seconds,
                "retain_acked_messages": subscription.retain_acked_messages,
                "message_retention_duration": str(subscription.message_retention_duration),
                "labels": dict(subscription.labels),
            }

        except Exception as e:
            logger.error(f"Error getting subscription info for {subscription_name}: {e}")
            return {}

    async def list_topics(self) -> List[str]:
        """List all topics in the project."""
        try:
            project_path = self.publisher.project_path(self.project_id)
            topics = self.publisher.list_topics(request={"project": project_path})

            topic_names = []
            for topic in topics:
                topic_names.append(topic.name)

            return topic_names

        except Exception as e:
            logger.error(f"Error listing topics: {e}")
            return []

    async def list_subscriptions(self) -> List[str]:
        """List all subscriptions in the project."""
        try:
            project_path = self.subscriber.project_path(self.project_id)
            subscriptions = self.subscriber.list_subscriptions(request={"project": project_path})

            subscription_names = []
            for subscription in subscriptions:
                subscription_names.append(subscription.name)

            return subscription_names

        except Exception as e:
            logger.error(f"Error listing subscriptions: {e}")
            return []

    async def delete_topic(self, topic_name: str) -> bool:
        """Delete a Pub/Sub topic."""
        try:
            topic_path = self.publisher.topic_path(self.project_id, topic_name)
            self.publisher.delete_topic(request={"topic": topic_path})

            logger.info(f"Deleted topic: {topic_path}")
            return True

        except Exception as e:
            logger.error(f"Error deleting topic {topic_name}: {e}")
            return False

    async def delete_subscription(self, subscription_name: str) -> bool:
        """Delete a Pub/Sub subscription."""
        try:
            subscription_path = self.subscriber.subscription_path(self.project_id, subscription_name)
            self.subscriber.delete_subscription(request={"subscription": subscription_path})

            logger.info(f"Deleted subscription: {subscription_path}")
            return True

        except Exception as e:
            logger.error(f"Error deleting subscription {subscription_name}: {e}")
            return False
