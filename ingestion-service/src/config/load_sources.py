"""
Source configuration loader and manager.
"""

import yaml
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from src.models.content import SourceConfig, SourceType
from src.config.settings import settings

logger = logging.getLogger(__name__)


class SourceConfigLoader:
    """Load and manage source configurations."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or settings.SOURCES_CONFIG_PATH
        self.sources: List[SourceConfig] = []
        self.global_config: Dict[str, Any] = {}
    
    def load_sources(self) -> List[SourceConfig]:
        """Load source configurations from YAML file."""
        try:
            # Check if config file exists
            if not os.path.exists(self.config_path):
                logger.warning(f"Source config file not found: {self.config_path}")
                return []
            
            # Load YAML file
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
            
            if not config_data or 'sources' not in config_data:
                logger.warning("No sources found in config file")
                return []
            
            # Load global config
            self.global_config = config_data.get('global_config', {})
            
            # Load sources
            sources_data = config_data['sources']
            self.sources = []
            
            for source_data in sources_data:
                try:
                    source_config = self._create_source_config(source_data)
                    if source_config:
                        self.sources.append(source_config)
                except Exception as e:
                    logger.error(f"Error loading source config: {e}")
                    continue
            
            logger.info(f"Loaded {len(self.sources)} source configurations")
            return self.sources
        
        except Exception as e:
            logger.error(f"Error loading source configurations: {e}")
            return []
    
    def _create_source_config(self, source_data: Dict[str, Any]) -> Optional[SourceConfig]:
        """Create SourceConfig from dictionary data."""
        try:
            # Get source type
            source_type_str = source_data.get('type', 'rss_feed')
            try:
                source_type = SourceType(source_type_str)
            except ValueError:
                logger.warning(f"Unknown source type: {source_type_str}")
                return None
            
            # Apply global defaults
            global_defaults = self.global_config.get('default_filters', {})
            global_processing = self.global_config.get('processing', {})
            
            # Merge filters with global defaults
            filters = source_data.get('filters', {})
            if 'filters' in global_defaults:
                filters = {**global_defaults['filters'], **filters}
            
            # Create source config
            source_config = SourceConfig(
                id=source_data.get('id', ''),
                name=source_data.get('name', ''),
                type=source_type,
                url=source_data.get('url', ''),
                enabled=source_data.get('enabled', True),
                priority=source_data.get('priority', 1),
                rate_limit=source_data.get('rate_limit', self.global_config.get('default_rate_limit', 60)),
                timeout=source_data.get('timeout', self.global_config.get('default_timeout', 30)),
                retry_attempts=source_data.get('retry_attempts', self.global_config.get('default_retry_attempts', 3)),
                backoff_factor=source_data.get('backoff_factor', self.global_config.get('default_backoff_factor', 2.0)),
                user_agent=source_data.get('user_agent', self.global_config.get('default_user_agent', settings.USER_AGENT)),
                headers=source_data.get('headers', {}),
                auth=source_data.get('auth'),
                filters=filters,
                last_checked=source_data.get('last_checked'),
                last_success=source_data.get('last_success'),
                error_count=source_data.get('error_count', 0),
                success_count=source_data.get('success_count', 0)
            )
            
            return source_config
        
        except Exception as e:
            logger.error(f"Error creating source config: {e}")
            return None
    
    def get_enabled_sources(self) -> List[SourceConfig]:
        """Get only enabled sources."""
        return [s for s in self.sources if s.enabled]
    
    def get_sources_by_type(self, source_type: SourceType) -> List[SourceConfig]:
        """Get sources by type."""
        return [s for s in self.sources if s.type == source_type]
    
    def get_source_by_id(self, source_id: str) -> Optional[SourceConfig]:
        """Get source by ID."""
        for source in self.sources:
            if source.id == source_id:
                return source
        return None
    
    def update_source_config(self, source_id: str, updates: Dict[str, Any]) -> bool:
        """Update source configuration."""
        source = self.get_source_by_id(source_id)
        if not source:
            return False
        
        try:
            # Update source attributes
            for key, value in updates.items():
                if hasattr(source, key):
                    setattr(source, key, value)
            
            # Save updated config to file
            self._save_sources_to_file()
            
            logger.info(f"Updated source config: {source_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error updating source config {source_id}: {e}")
            return False
    
    def add_source_config(self, source_config: SourceConfig) -> bool:
        """Add new source configuration."""
        try:
            # Check if source already exists
            if self.get_source_by_id(source_config.id):
                logger.warning(f"Source {source_config.id} already exists")
                return False
            
            # Add to sources list
            self.sources.append(source_config)
            
            # Save to file
            self._save_sources_to_file()
            
            logger.info(f"Added new source config: {source_config.id}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding source config: {e}")
            return False
    
    def remove_source_config(self, source_id: str) -> bool:
        """Remove source configuration."""
        try:
            # Find and remove source
            source = self.get_source_by_id(source_id)
            if not source:
                logger.warning(f"Source {source_id} not found")
                return False
            
            self.sources.remove(source)
            
            # Save to file
            self._save_sources_to_file()
            
            logger.info(f"Removed source config: {source_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error removing source config: {e}")
            return False
    
    def _save_sources_to_file(self):
        """Save sources to YAML file."""
        try:
            # Prepare data for YAML
            sources_data = []
            for source in self.sources:
                source_dict = {
                    'id': source.id,
                    'name': source.name,
                    'type': source.type.value,
                    'url': source.url,
                    'enabled': source.enabled,
                    'priority': source.priority,
                    'rate_limit': source.rate_limit,
                    'timeout': source.timeout,
                    'retry_attempts': source.retry_attempts,
                    'backoff_factor': source.backoff_factor,
                    'user_agent': source.user_agent,
                    'headers': source.headers,
                    'auth': source.auth,
                    'filters': source.filters,
                    'last_checked': source.last_checked,
                    'last_success': source.last_success,
                    'error_count': source.error_count,
                    'success_count': source.success_count
                }
                sources_data.append(source_dict)
            
            # Create complete config
            config_data = {
                'sources': sources_data,
                'global_config': self.global_config
            }
            
            # Write to file
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(config_data, file, default_flow_style=False, indent=2)
            
            logger.debug(f"Saved sources to {self.config_path}")
        
        except Exception as e:
            logger.error(f"Error saving sources to file: {e}")
    
    def get_global_config(self) -> Dict[str, Any]:
        """Get global configuration."""
        return self.global_config
    
    def update_global_config(self, updates: Dict[str, Any]) -> bool:
        """Update global configuration."""
        try:
            self.global_config.update(updates)
            self._save_sources_to_file()
            
            logger.info("Updated global configuration")
            return True
        
        except Exception as e:
            logger.error(f"Error updating global config: {e}")
            return False
    
    def validate_sources(self) -> List[Dict[str, Any]]:
        """Validate all source configurations."""
        validation_results = []
        
        for source in self.sources:
            result = {
                'source_id': source.id,
                'valid': True,
                'errors': [],
                'warnings': []
            }
            
            # Check required fields
            if not source.id:
                result['errors'].append("Source ID is required")
                result['valid'] = False
            
            if not source.name:
                result['errors'].append("Source name is required")
                result['valid'] = False
            
            if not source.url:
                result['errors'].append("Source URL is required")
                result['valid'] = False
            
            # Validate URL format
            if source.url and not source.url.startswith(('http://', 'https://')):
                result['errors'].append("Source URL must start with http:// or https://")
                result['valid'] = False
            
            # Check priority range
            if source.priority < 1 or source.priority > 10:
                result['warnings'].append("Priority should be between 1 and 10")
            
            # Check rate limit
            if source.rate_limit < 1:
                result['warnings'].append("Rate limit should be positive")
            
            # Check timeout
            if source.timeout < 1:
                result['warnings'].append("Timeout should be positive")
            
            validation_results.append(result)
        
        return validation_results
    
    def get_source_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded sources."""
        total_sources = len(self.sources)
        enabled_sources = len([s for s in self.sources if s.enabled])
        disabled_sources = total_sources - enabled_sources
        
        # Count by type
        type_counts = {}
        for source in self.sources:
            source_type = source.type.value
            type_counts[source_type] = type_counts.get(source_type, 0) + 1
        
        # Count by priority
        priority_counts = {}
        for source in self.sources:
            priority = source.priority
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        return {
            'total_sources': total_sources,
            'enabled_sources': enabled_sources,
            'disabled_sources': disabled_sources,
            'type_distribution': type_counts,
            'priority_distribution': priority_counts
        }


# Global source config loader instance
source_config_loader = SourceConfigLoader()
