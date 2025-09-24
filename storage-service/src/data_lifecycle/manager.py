"""
Data Lifecycle Manager - Comprehensive data lifecycle and compliance management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from models import (
    GDPRDeletionResult,
    GDPRRequest,
    GDPRResponse,
    RetentionPolicy,
    RetentionPolicyType,
    RetentionResult,
)

from .archival_service import ArchivalService
from .backup_manager import BackupManager
from .gdpr_processor import GDPRProcessor
from .retention_policies import RetentionPolicies

logger = logging.getLogger(__name__)


class DataLifecycleManager:
    """Manage data lifecycle, archival, and compliance"""

    def __init__(self):
        self.archival_service: Optional[ArchivalService] = None
        self.retention_policies: Optional[RetentionPolicies] = None
        self.gdpr_processor: Optional[GDPRProcessor] = None
        self.backup_manager: Optional[BackupManager] = None
        self._initialized = False
        self._retention_scheduler: Optional[asyncio.Task] = None

    async def initialize(self) -> Dict[str, Any]:
        """Initialize data lifecycle management components"""
        if self._initialized:
            return

        logger.info("Initializing Data Lifecycle Manager...")

        try:
            # Initialize archival service
            self.archival_service = ArchivalService()
            await self.archival_service.initialize()

            # Initialize retention policies
            self.retention_policies = RetentionPolicies()
            await self.retention_policies.initialize()

            # Initialize GDPR processor
            self.gdpr_processor = GDPRProcessor()
            await self.gdpr_processor.initialize()

            # Initialize backup manager
            self.backup_manager = BackupManager()
            await self.backup_manager.initialize()

            self._initialized = True
            logger.info("Data Lifecycle Manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Data Lifecycle Manager: {e}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup data lifecycle management components"""
        logger.info("Cleaning up Data Lifecycle Manager...")

        # Stop retention scheduler
        if self._retention_scheduler:
            self._retention_scheduler.cancel()
            try:
    await self._retention_scheduler
            except asyncio.CancelledError:
                pass

        cleanup_tasks = []

        if self.archival_service:
            cleanup_tasks.append(self.archival_service.cleanup())
        if self.retention_policies:
            cleanup_tasks.append(self.retention_policies.cleanup())
        if self.gdpr_processor:
            cleanup_tasks.append(self.gdpr_processor.cleanup())
        if self.backup_manager:
            cleanup_tasks.append(self.backup_manager.cleanup())

        if cleanup_tasks:
    await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        self._initialized = False
        logger.info("Data Lifecycle Manager cleanup complete")

    async def start_retention_scheduler(self) -> Dict[str, Any]:
        """Start background retention policy scheduler"""
        if not self._initialized:
            raise RuntimeError("Data Lifecycle Manager not initialized")

        if self._retention_scheduler and not self._retention_scheduler.done():
            return

        self._retention_scheduler = asyncio.create_task(
            self._retention_scheduler_loop())
        logger.info("Retention scheduler started")

    async def _retention_scheduler_loop(self) -> Dict[str, Any]:
        """Background loop for retention policy execution"""
        while True:
            try:
    await asyncio.sleep(3600)  # Run every hour

                if not self._initialized:
                    break

                # Apply retention policies
                await self.apply_retention_policies()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in retention scheduler: {e}")

    async def apply_retention_policies(self) -> RetentionResult:
        """Apply data retention policies"""
        if not self._initialized:
            raise RuntimeError("Data Lifecycle Manager not initialized")

        logger.info("Applying retention policies...")

        try:
            policies = await self.retention_policies.get_active_policies()
            retention_results = []

            for policy in policies:
                try:
                    if policy.policy_type == RetentionPolicyType.DELETE_OLD_DATA:
                        result = await self._delete_old_data(policy)
                    elif policy.policy_type == RetentionPolicyType.ARCHIVE_DATA:
                        result = await self._archive_old_data(policy)
                    elif policy.policy_type == RetentionPolicyType.ANONYMIZE_DATA:
                        result = await self._anonymize_old_data(policy)
                else:
                        logger.warning(
                            f"Unknown policy type: {policy.policy_type}")
                        continue

                    retention_results.append(result)

                except Exception as e:
                    logger.error(
                        f"Failed to apply policy {policy.policy_id}: {e}")
                    retention_results.append(
                        {"policy_id": policy.policy_id, "status": "failed", "error": str(e)}
                    )

            logger.info(f"Applied {len(policies)} retention policies")

            return RetentionResult(
                policies_applied=len(policies),
                retention_results=retention_results,
                execution_timestamp=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Failed to apply retention policies: {e}")
            raise

    async def _delete_old_data(
            self, policy: RetentionPolicy) -> Dict[str, Any]:
    """Delete old data based on policy"""
        try:
            logger.info(f"Deleting old data for policy {policy.policy_id}")

            # Calculate cutoff date
            cutoff_date = datetime.utcnow() - timedelta(days=policy.retention_period_days)

            # Delete from different data stores
            deletion_results = []

            # Delete from PostgreSQL
            if self.archival_service:
                pg_result = await self.archival_service.delete_old_articles(
                    policy.data_type, cutoff_date
                )
                deletion_results.append(
                    {"store": "postgresql", "deleted_count": pg_result.get("deleted_count", 0)}
                )

            # Delete from Elasticsearch
            if self.archival_service:
                es_result = await self.archival_service.delete_old_elasticsearch_data(
                    policy.data_type, cutoff_date
                )
                deletion_results.append(
                    {"store": "elasticsearch", "deleted_count": es_result.get("deleted_count", 0)}
                )

            # Delete from Redis
            if self.archival_service:
                redis_result = await self.archival_service.delete_old_redis_data(
                    policy.data_type, cutoff_date
                )
                deletion_results.append(
                    {"store": "redis", "deleted_count": redis_result.get("deleted_count", 0)}
                )

            total_deleted = sum(result["deleted_count"]
                                for result in deletion_results)

            return {
                "policy_id": policy.policy_id,
                "policy_type": policy.policy_type.value,
                "status": "completed",
                "cutoff_date": cutoff_date.isoformat(),
                "total_deleted": total_deleted,
                "deletion_results": deletion_results,
            }

        except Exception as e:
            logger.error(
                f"Failed to delete old data for policy {policy.policy_id}: {e}")
            return {
                "policy_id": policy.policy_id,
                "policy_type": policy.policy_type.value,
                "status": "failed",
                "error": str(e),
            }

    async def _archive_old_data(
            self, policy: RetentionPolicy) -> Dict[str, Any]:
    """Archive old data based on policy"""
        try:
            logger.info(f"Archiving old data for policy {policy.policy_id}")

            # Calculate cutoff date
            cutoff_date = datetime.utcnow() - timedelta(days=policy.retention_period_days)

            # Archive data
            if self.archival_service:
                archive_result = await self.archival_service.archive_old_data(
                    policy.data_type, cutoff_date
                )

                return {
                    "policy_id": policy.policy_id,
                    "policy_type": policy.policy_type.value,
                    "status": "completed",
                    "cutoff_date": cutoff_date.isoformat(),
                    "archived_count": archive_result.get(
                        "archived_count",
                        0),
                    "archive_location": archive_result.get(
                        "archive_location",
                        ""),
                }
            else:
                return {
                    "policy_id": policy.policy_id,
                    "policy_type": policy.policy_type.value,
                    "status": "skipped",
                    "reason": "Archival service not available",
                }

        except Exception as e:
            logger.error(
                f"Failed to archive old data for policy {policy.policy_id}: {e}")
            return {
                "policy_id": policy.policy_id,
                "policy_type": policy.policy_type.value,
                "status": "failed",
                "error": str(e),
            }

    async def _anonymize_old_data(
            self, policy: RetentionPolicy) -> Dict[str, Any]:
    """Anonymize old data based on policy"""
        try:
            logger.info(f"Anonymizing old data for policy {policy.policy_id}")

            # Calculate cutoff date
            cutoff_date = datetime.utcnow() - timedelta(days=policy.retention_period_days)

            # Anonymize data
            if self.archival_service:
                anonymize_result = await self.archival_service.anonymize_old_data(
                    policy.data_type, cutoff_date
                )

                return {
                    "policy_id": policy.policy_id,
                    "policy_type": policy.policy_type.value,
                    "status": "completed",
                    "cutoff_date": cutoff_date.isoformat(),
                    "anonymized_count": anonymize_result.get(
                        "anonymized_count",
                        0),
                }
            else:
                return {
                    "policy_id": policy.policy_id,
                    "policy_type": policy.policy_type.value,
                    "status": "skipped",
                    "reason": "Archival service not available",
                }

        except Exception as e:
            logger.error(
                f"Failed to anonymize old data for policy {policy.policy_id}: {e}")
            return {
                "policy_id": policy.policy_id,
                "policy_type": policy.policy_type.value,
                "status": "failed",
                "error": str(e),
            }

    async def handle_gdpr_request(
            self, gdpr_request: GDPRRequest) -> GDPRResponse:
        """Handle GDPR data requests"""
        if not self._initialized:
            raise RuntimeError("Data Lifecycle Manager not initialized")

        logger.info(
            f"Handling GDPR request {gdpr_request.request_id} for user {gdpr_request.user_id}"
        )

        try:
            if gdpr_request.request_type == "data_export":
                return await self._export_user_data(gdpr_request)
            elif gdpr_request.request_type == "data_deletion":
                return await self._delete_user_data(gdpr_request)
            elif gdpr_request.request_type == "data_rectification":
                return await self._rectify_user_data(gdpr_request)
        else:
                raise ValueError(
                    f"Unknown GDPR request type: {gdpr_request.request_type}")

        except Exception as e:
            logger.error(
                f"Failed to handle GDPR request {gdpr_request.request_id}: {e}")
            return GDPRResponse(
                request_id=gdpr_request.request_id,
                user_id=gdpr_request.user_id,
                request_type=gdpr_request.request_type,
                status="failed",
                data={"error": str(e)},
            )

    async def _export_user_data(
            self, gdpr_request: GDPRRequest) -> GDPRResponse:
        """Export user data for GDPR compliance"""
        try:
            if not self.gdpr_processor:
                raise RuntimeError("GDPR processor not available")

            user_data = await self.gdpr_processor.export_user_data(gdpr_request.user_id)

            return GDPRResponse(
                request_id=gdpr_request.request_id,
                user_id=gdpr_request.user_id,
                request_type=gdpr_request.request_type,
                status="completed",
                data=user_data,
            )

        except Exception as e:
            logger.error(
                f"Failed to export user data for {gdpr_request.user_id}: {e}")
            raise

    async def _delete_user_data(
            self, gdpr_request: GDPRRequest) -> GDPRResponse:
        """Delete user data for GDPR compliance"""
        try:
            if not self.gdpr_processor:
                raise RuntimeError("GDPR processor not available")

            deletion_result = await self.gdpr_processor.delete_user_data(gdpr_request.user_id)

            return GDPRResponse(
                request_id=gdpr_request.request_id,
                user_id=gdpr_request.user_id,
                request_type=gdpr_request.request_type,
                status="completed",
                data=deletion_result,
            )

        except Exception as e:
            logger.error(
                f"Failed to delete user data for {gdpr_request.user_id}: {e}")
            raise

    async def _rectify_user_data(
            self, gdpr_request: GDPRRequest) -> GDPRResponse:
        """Rectify user data for GDPR compliance"""
        try:
            if not self.gdpr_processor:
                raise RuntimeError("GDPR processor not available")

            rectification_result = await self.gdpr_processor.rectify_user_data(
                gdpr_request.user_id, gdpr_request.corrections or {}
            )

            return GDPRResponse(
                request_id=gdpr_request.request_id,
                user_id=gdpr_request.user_id,
                request_type=gdpr_request.request_type,
                status="completed",
                data=rectification_result,
            )

        except Exception as e:
            logger.error(
                f"Failed to rectify user data for {gdpr_request.user_id}: {e}")
            raise

    async def create_backup(self, backup_name: str,
                            data_types: List[str] = None) -> Dict[str, Any]:
    """Create comprehensive backup"""
        if not self._initialized:
            raise RuntimeError("Data Lifecycle Manager not initialized")

        try:
            if not self.backup_manager:
                raise RuntimeError("Backup manager not available")

            backup_result = await self.backup_manager.create_backup(backup_name, data_types)

            logger.info(f"Created backup: {backup_name}")
            return backup_result

        except Exception as e:
            logger.error(f"Failed to create backup {backup_name}: {e}")
            raise

    async def restore_backup(self, backup_name: str) -> Dict[str, Any]:
        """Restore from backup"""
        if not self._initialized:
            raise RuntimeError("Data Lifecycle Manager not initialized")

        try:
            if not self.backup_manager:
                raise RuntimeError("Backup manager not available")

            restore_result = await self.backup_manager.restore_backup(backup_name)

            logger.info(f"Restored from backup: {backup_name}")
            return restore_result

        except Exception as e:
            logger.error(f"Failed to restore backup {backup_name}: {e}")
            raise

    async def log_gdpr_request(self, gdpr_request: GDPRRequest) -> Dict[str, Any]:
        """Log GDPR request for audit purposes"""
        try:
            # This would typically log to an audit database
            logger.info(
                f"GDPR request logged: {gdpr_request.request_id} - {gdpr_request.request_type} - {gdpr_request.user_id}"
            )

            # In a real implementation, you would:
            # 1. Store in audit database
            # 2. Send to compliance monitoring system
            # 3. Generate audit reports

        except Exception as e:
            logger.error(f"Failed to log GDPR request: {e}")

    async def get_lifecycle_statistics(self) -> Dict[str, Any]:
        """Get data lifecycle statistics"""
        if not self._initialized:
            return {}

        try:
            stats = {
                "timestamp": datetime.utcnow().isoformat(),
                "retention_policies": {},
                "gdpr_requests": {},
                "backup_status": {},
                "archival_status": {},
            }

            # Get retention policy stats
            if self.retention_policies:
                stats["retention_policies"] = await self.retention_policies.get_statistics()

            # Get GDPR processor stats
            if self.gdpr_processor:
                stats["gdpr_requests"] = await self.gdpr_processor.get_statistics()

            # Get backup manager stats
            if self.backup_manager:
                stats["backup_status"] = await self.backup_manager.get_statistics()

            # Get archival service stats
            if self.archival_service:
                stats["archival_status"] = await self.archival_service.get_statistics()

            return stats

        except Exception as e:
            logger.error(f"Failed to get lifecycle statistics: {e}")
            return {}
