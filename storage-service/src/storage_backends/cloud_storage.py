"""
Cloud Storage Manager - Media files and backup storage
"""

import asyncio
import hashlib
import logging
import mimetypes
from datetime import datetime, timedelta
from typing import Any, BinaryIO, Dict, List, Optional
from urllib.parse import urlparse

import aiofiles
import aiohttp

from config import Settings

logger = logging.getLogger(__name__)


class MediaStorageManager:
    """Manage cloud storage for media files and backups"""

    def __init__(self):
        self.settings = Settings()
        self._initialized = False
        self._storage_client = None
        self._bucket_name = None

    async def initialize(self):
        """Initialize cloud storage client"""
        if self._initialized:
            return

        logger.info("Initializing Cloud Storage Manager...")

        try:
            cloud_config = self.settings.get_cloud_storage_config()
            self._bucket_name = cloud_config.bucket_name

            if cloud_config.provider == "aws":
                await self._initialize_aws_s3(cloud_config)
            elif cloud_config.provider == "gcp":
                await self._initialize_gcp_storage(cloud_config)
            elif cloud_config.provider == "azure":
                await self._initialize_azure_blob(cloud_config)
            else:
                raise ValueError(f"Unsupported cloud provider: {cloud_config.provider}")

            self._initialized = True
            logger.info("Cloud Storage Manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Cloud Storage Manager: {e}")
            raise

    async def cleanup(self):
        """Cleanup cloud storage client"""
        if self._storage_client:
            if hasattr(self._storage_client, "close"):
                await self._storage_client.close()
            self._storage_client = None

        self._initialized = False
        logger.info("Cloud Storage Manager cleanup complete")

    async def _initialize_aws_s3(self, config):
        """Initialize AWS S3 client"""
        try:
            import boto3
            from botocore.config import Config

            # Create S3 client
            self._storage_client = boto3.client(
                "s3",
                aws_access_key_id=config.access_key,
                aws_secret_access_key=config.secret_key,
                region_name=config.region,
                config=Config(max_pool_connections=50, retries={"max_attempts": 3}),
            )

            # Test connection
            self._storage_client.head_bucket(Bucket=self._bucket_name)

        except ImportError:
            raise ImportError("boto3 is required for AWS S3 support")
        except Exception as e:
            logger.error(f"Failed to initialize AWS S3: {e}")
            raise

    async def _initialize_gcp_storage(self, config):
        """Initialize Google Cloud Storage client"""
        try:
            from google.cloud import storage
            from google.oauth2 import service_account

            # Create credentials
            credentials = service_account.Credentials.from_service_account_info(
                {
                    "type": "service_account",
                    "project_id": config.project_id,
                    "private_key_id": config.private_key_id,
                    "private_key": config.private_key,
                    "client_email": config.client_email,
                    "client_id": config.client_id,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            )

            # Create storage client
            self._storage_client = storage.Client(
                credentials=credentials, project=config.project_id
            )

            # Test connection
            bucket = self._storage_client.bucket(self._bucket_name)
            bucket.exists()

        except ImportError:
            raise ImportError("google-cloud-storage is required for GCP support")
        except Exception as e:
            logger.error(f"Failed to initialize GCP Storage: {e}")
            raise

    async def _initialize_azure_blob(self, config):
        """Initialize Azure Blob Storage client"""
        try:
            from azure.storage.blob.aio import BlobServiceClient

            # Create blob service client
            connection_string = f"DefaultEndpointsProtocol=https;AccountName={config.account_name};AccountKey={config.account_key};EndpointSuffix=core.windows.net"

            self._storage_client = BlobServiceClient.from_connection_string(connection_string)

            # Test connection
            await self._storage_client.get_account_information()

        except ImportError:
            raise ImportError("azure-storage-blob is required for Azure support")
        except Exception as e:
            logger.error(f"Failed to initialize Azure Blob Storage: {e}")
            raise

    async def store_media_files(
        self, article_id: str, media_files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Store media files for an article"""
        if not self._initialized or not self._storage_client:
            raise RuntimeError("Cloud Storage Manager not initialized")

        logger.info(f"Storing {len(media_files)} media files for article {article_id}")

        try:
            stored_files = []

            for media_file in media_files:
                file_path = media_file.get("path")
                file_content = media_file.get("content")
                file_type = media_file.get("type", "unknown")

                if not file_path or not file_content:
                    continue

                # Generate unique file key
                file_key = self._generate_file_key(article_id, file_path)

                # Store file
                file_url = await self._store_file(file_key, file_content, file_type)

                stored_files.append(
                    {
                        "original_path": file_path,
                        "storage_key": file_key,
                        "url": file_url,
                        "type": file_type,
                        "size": len(file_content) if isinstance(file_content, bytes) else 0,
                    }
                )

            logger.info(f"Stored {len(stored_files)} media files for article {article_id}")

            return {
                "article_id": article_id,
                "stored_files": stored_files,
                "total_files": len(stored_files),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to store media files for article {article_id}: {e}")
            raise

    async def get_article_media(self, article_id: str) -> List[Dict[str, Any]]:
        """Get media files for an article"""
        if not self._initialized or not self._storage_client:
            raise RuntimeError("Cloud Storage Manager not initialized")

        try:
            # List files with article prefix
            prefix = f"articles/{article_id}/"
            files = await self._list_files(prefix)

            media_files = []
            for file_info in files:
                media_files.append(
                    {
                        "storage_key": file_info["key"],
                        "url": file_info["url"],
                        "type": file_info.get("type", "unknown"),
                        "size": file_info.get("size", 0),
                        "last_modified": file_info.get("last_modified"),
                    }
                )

            return media_files

        except Exception as e:
            logger.error(f"Failed to get media files for article {article_id}: {e}")
            raise

    async def delete_article_media(self, article_id: str) -> bool:
        """Delete all media files for an article"""
        if not self._initialized or not self._storage_client:
            raise RuntimeError("Cloud Storage Manager not initialized")

        try:
            # List files with article prefix
            prefix = f"articles/{article_id}/"
            files = await self._list_files(prefix)

            # Delete all files
            for file_info in files:
                await self._delete_file(file_info["key"])

            logger.info(f"Deleted {len(files)} media files for article {article_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete media files for article {article_id}: {e}")
            return False

    async def create_backup(self, data: Dict[str, Any], backup_name: str) -> str:
        """Create a backup of data"""
        if not self._initialized or not self._storage_client:
            raise RuntimeError("Cloud Storage Manager not initialized")

        try:
            import json

            # Generate backup key
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_key = f"backups/{backup_name}_{timestamp}.json"

            # Serialize data
            backup_data = json.dumps(data, default=str, indent=2)

            # Store backup
            await self._store_file(backup_key, backup_data.encode("utf-8"), "application/json")

            logger.info(f"Created backup: {backup_key}")
            return backup_key

        except Exception as e:
            logger.error(f"Failed to create backup {backup_name}: {e}")
            raise

    async def restore_backup(self, backup_key: str) -> Dict[str, Any]:
        """Restore data from backup"""
        if not self._initialized or not self._storage_client:
            raise RuntimeError("Cloud Storage Manager not initialized")

        try:
            import json

            # Get backup data
            backup_data = await self._get_file(backup_key)

            # Deserialize data
            data = json.loads(backup_data.decode("utf-8"))

            logger.info(f"Restored backup: {backup_key}")
            return data

        except Exception as e:
            logger.error(f"Failed to restore backup {backup_key}: {e}")
            raise

    def _generate_file_key(self, article_id: str, file_path: str) -> str:
        """Generate unique file key"""
        # Extract file extension
        file_ext = file_path.split(".")[-1] if "." in file_path else ""

        # Generate hash for uniqueness
        file_hash = hashlib.md5(f"{article_id}_{file_path}".encode()).hexdigest()[:8]

        return f"articles/{article_id}/{file_hash}.{file_ext}"

    async def _store_file(self, file_key: str, content: bytes, content_type: str) -> str:
        """Store file in cloud storage"""
        try:
            if self.settings.cloud_storage_provider == "aws":
                return await self._store_file_s3(file_key, content, content_type)
            elif self.settings.cloud_storage_provider == "gcp":
                return await self._store_file_gcp(file_key, content, content_type)
            elif self.settings.cloud_storage_provider == "azure":
                return await self._store_file_azure(file_key, content, content_type)
            else:
                raise ValueError(
                    f"Unsupported storage provider: {self.settings.cloud_storage_provider}"
                )

        except Exception as e:
            logger.error(f"Failed to store file {file_key}: {e}")
            raise

    async def _store_file_s3(self, file_key: str, content: bytes, content_type: str) -> str:
        """Store file in AWS S3"""
        self._storage_client.put_object(
            Bucket=self._bucket_name, Key=file_key, Body=content, ContentType=content_type
        )

        return f"https://{self._bucket_name}.s3.{self.settings.cloud_storage_region}.amazonaws.com/{file_key}"

    async def _store_file_gcp(self, file_key: str, content: bytes, content_type: str) -> str:
        """Store file in Google Cloud Storage"""
        bucket = self._storage_client.bucket(self._bucket_name)
        blob = bucket.blob(file_key)

        blob.upload_from_string(content, content_type=content_type)

        return f"https://storage.googleapis.com/{self._bucket_name}/{file_key}"

    async def _store_file_azure(self, file_key: str, content: bytes, content_type: str) -> str:
        """Store file in Azure Blob Storage"""
        container_client = self._storage_client.get_container_client("media")

        blob_client = container_client.get_blob_client(file_key)
        blob_client.upload_blob(content, content_type=content_type, overwrite=True)

        return blob_client.url

    async def _get_file(self, file_key: str) -> bytes:
        """Get file from cloud storage"""
        try:
            if self.settings.cloud_storage_provider == "aws":
                return await self._get_file_s3(file_key)
            elif self.settings.cloud_storage_provider == "gcp":
                return await self._get_file_gcp(file_key)
            elif self.settings.cloud_storage_provider == "azure":
                return await self._get_file_azure(file_key)
            else:
                raise ValueError(
                    f"Unsupported storage provider: {self.settings.cloud_storage_provider}"
                )

        except Exception as e:
            logger.error(f"Failed to get file {file_key}: {e}")
            raise

    async def _get_file_s3(self, file_key: str) -> bytes:
        """Get file from AWS S3"""
        response = self._storage_client.get_object(Bucket=self._bucket_name, Key=file_key)
        return response["Body"].read()

    async def _get_file_gcp(self, file_key: str) -> bytes:
        """Get file from Google Cloud Storage"""
        bucket = self._storage_client.bucket(self._bucket_name)
        blob = bucket.blob(file_key)
        return blob.download_as_bytes()

    async def _get_file_azure(self, file_key: str) -> bytes:
        """Get file from Azure Blob Storage"""
        container_client = self._storage_client.get_container_client("media")
        blob_client = container_client.get_blob_client(file_key)
        return blob_client.download_blob().readall()

    async def _list_files(self, prefix: str) -> List[Dict[str, Any]]:
        """List files with prefix"""
        try:
            if self.settings.cloud_storage_provider == "aws":
                return await self._list_files_s3(prefix)
            elif self.settings.cloud_storage_provider == "gcp":
                return await self._list_files_gcp(prefix)
            elif self.settings.cloud_storage_provider == "azure":
                return await self._list_files_azure(prefix)
            else:
                raise ValueError(
                    f"Unsupported storage provider: {self.settings.cloud_storage_provider}"
                )

        except Exception as e:
            logger.error(f"Failed to list files with prefix {prefix}: {e}")
            return []

    async def _list_files_s3(self, prefix: str) -> List[Dict[str, Any]]:
        """List files in AWS S3"""
        response = self._storage_client.list_objects_v2(Bucket=self._bucket_name, Prefix=prefix)

        files = []
        for obj in response.get("Contents", []):
            files.append(
                {
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"],
                    "url": f"https://{self._bucket_name}.s3.{self.settings.cloud_storage_region}.amazonaws.com/{obj['Key']}",
                }
            )

        return files

    async def _list_files_gcp(self, prefix: str) -> List[Dict[str, Any]]:
        """List files in Google Cloud Storage"""
        bucket = self._storage_client.bucket(self._bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        files = []
        for blob in blobs:
            files.append(
                {
                    "key": blob.name,
                    "size": blob.size,
                    "last_modified": blob.time_created,
                    "url": f"https://storage.googleapis.com/{self._bucket_name}/{blob.name}",
                }
            )

        return files

    async def _list_files_azure(self, prefix: str) -> List[Dict[str, Any]]:
        """List files in Azure Blob Storage"""
        container_client = self._storage_client.get_container_client("media")
        blobs = container_client.list_blobs(name_starts_with=prefix)

        files = []
        for blob in blobs:
            files.append(
                {
                    "key": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified,
                    "url": f"https://{self.settings.cloud_storage_bucket}.blob.core.windows.net/media/{blob.name}",
                }
            )

        return files

    async def _delete_file(self, file_key: str) -> bool:
        """Delete file from cloud storage"""
        try:
            if self.settings.cloud_storage_provider == "aws":
                await self._delete_file_s3(file_key)
            elif self.settings.cloud_storage_provider == "gcp":
                await self._delete_file_gcp(file_key)
            elif self.settings.cloud_storage_provider == "azure":
                await self._delete_file_azure(file_key)
            else:
                raise ValueError(
                    f"Unsupported storage provider: {self.settings.cloud_storage_provider}"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to delete file {file_key}: {e}")
            return False

    async def _delete_file_s3(self, file_key: str):
        """Delete file from AWS S3"""
        self._storage_client.delete_object(Bucket=self._bucket_name, Key=file_key)

    async def _delete_file_gcp(self, file_key: str):
        """Delete file from Google Cloud Storage"""
        bucket = self._storage_client.bucket(self._bucket_name)
        blob = bucket.blob(file_key)
        blob.delete()

    async def _delete_file_azure(self, file_key: str):
        """Delete file from Azure Blob Storage"""
        container_client = self._storage_client.get_container_client("media")
        blob_client = container_client.get_blob_client(file_key)
        blob_client.delete_blob()
