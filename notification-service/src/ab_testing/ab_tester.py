"""
A/B testing framework for notification strategies.
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import structlog
from redis.asyncio import Redis

from ..models.schemas import ABTestVariant
from ..exceptions import ABTestingError

logger = structlog.get_logger()


class ABTestingFramework:
    """A/B testing framework for notification strategies."""
    
    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client
        self.experiments = {}
        self.variant_cache = {}  # Cache for user assignments
        self.cache_ttl = 3600  # 1 hour
    
    async def initialize(self) -> None:
        """Initialize A/B testing framework."""
        logger.info("Initializing A/B testing framework")
        
        # Load active experiments
        await self._load_active_experiments()
        
        logger.info("A/B testing framework initialized successfully")
    
    async def cleanup(self) -> None:
        """Cleanup A/B testing framework."""
        logger.info("Cleaning up A/B testing framework")
        self.experiments.clear()
        self.variant_cache.clear()
    
    async def get_variant(
        self, 
        user_id: str, 
        experiment: str
    ) -> str:
        """Get A/B test variant for user."""
        
        try:
            # Check cache first
            cache_key = f"ab_variant:{user_id}:{experiment}"
            if cache_key in self.variant_cache:
                cached_variant, timestamp = self.variant_cache[cache_key]
                if (datetime.utcnow() - timestamp).seconds < self.cache_ttl:
                    return cached_variant
            
            # Check if experiment exists
            if experiment not in self.experiments:
                logger.warning(f"Experiment {experiment} not found, using default variant")
                return "default"
            
            exp_config = self.experiments[experiment]
            
            # Check if experiment is active
            if not exp_config.get('active', False):
                return exp_config.get('default_variant', 'default')
            
            # Check if user is already assigned
            existing_variant = await self._get_existing_assignment(user_id, experiment)
            if existing_variant:
                # Cache the assignment
                self.variant_cache[cache_key] = (existing_variant, datetime.utcnow())
                return existing_variant
            
            # Assign new variant
            variant = await self._assign_variant(user_id, experiment, exp_config)
            
            # Cache the assignment
            self.variant_cache[cache_key] = (variant, datetime.utcnow())
            
            logger.debug(
                "Assigned A/B test variant",
                user_id=user_id,
                experiment=experiment,
                variant=variant
            )
            
            return variant
            
        except Exception as e:
            logger.error(
                "Error getting A/B test variant",
                user_id=user_id,
                experiment=experiment,
                error=str(e),
                exc_info=True
            )
            return "default"
    
    async def create_experiment(
        self, 
        experiment_name: str,
        variants: List[str],
        traffic_split: Optional[Dict[str, float]] = None,
        default_variant: str = "control"
    ) -> Dict[str, Any]:
        """Create new A/B test experiment."""
        
        try:
            # Validate experiment
            if not experiment_name or not variants:
                raise ABTestingError("Experiment name and variants are required")
            
            if len(variants) < 2:
                raise ABTestingError("At least 2 variants are required")
            
            # Set default traffic split if not provided
            if not traffic_split:
                traffic_split = {variant: 1.0 / len(variants) for variant in variants}
            
            # Validate traffic split
            total_split = sum(traffic_split.values())
            if abs(total_split - 1.0) > 0.01:
                raise ABTestingError("Traffic split must sum to 1.0")
            
            # Create experiment configuration
            experiment_config = {
                'name': experiment_name,
                'variants': variants,
                'traffic_split': traffic_split,
                'default_variant': default_variant,
                'active': True,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Store experiment
            await self.redis_client.hset(
                f"ab_experiment:{experiment_name}",
                mapping=experiment_config
            )
            
            # Update local cache
            self.experiments[experiment_name] = experiment_config
            
            logger.info(
                "Created A/B test experiment",
                experiment_name=experiment_name,
                variants=variants,
                traffic_split=traffic_split
            )
            
            return experiment_config
            
        except Exception as e:
            logger.error(
                "Error creating A/B test experiment",
                experiment_name=experiment_name,
                error=str(e),
                exc_info=True
            )
            raise ABTestingError(f"Failed to create experiment: {str(e)}")
    
    async def update_experiment(
        self, 
        experiment_name: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update A/B test experiment."""
        
        try:
            if experiment_name not in self.experiments:
                raise ABTestingError(f"Experiment {experiment_name} not found")
            
            # Update configuration
            experiment_config = self.experiments[experiment_name].copy()
            experiment_config.update(updates)
            experiment_config['updated_at'] = datetime.utcnow().isoformat()
            
            # Store updated experiment
            await self.redis_client.hset(
                f"ab_experiment:{experiment_name}",
                mapping=experiment_config
            )
            
            # Update local cache
            self.experiments[experiment_name] = experiment_config
            
            logger.info(
                "Updated A/B test experiment",
                experiment_name=experiment_name,
                updates=updates
            )
            
            return experiment_config
            
        except Exception as e:
            logger.error(
                "Error updating A/B test experiment",
                experiment_name=experiment_name,
                error=str(e),
                exc_info=True
            )
            raise ABTestingError(f"Failed to update experiment: {str(e)}")
    
    async def stop_experiment(self, experiment_name: str) -> None:
        """Stop A/B test experiment."""
        
        try:
            await self.update_experiment(experiment_name, {'active': False})
            
            logger.info(
                "Stopped A/B test experiment",
                experiment_name=experiment_name
            )
            
        except Exception as e:
            logger.error(
                "Error stopping A/B test experiment",
                experiment_name=experiment_name,
                error=str(e)
            )
            raise ABTestingError(f"Failed to stop experiment: {str(e)}")
    
    async def get_experiment_results(
        self, 
        experiment_name: str
    ) -> Dict[str, Any]:
        """Get A/B test experiment results."""
        
        try:
            if experiment_name not in self.experiments:
                raise ABTestingError(f"Experiment {experiment_name} not found")
            
            exp_config = self.experiments[experiment_name]
            
            # Get variant assignments
            assignments = await self._get_variant_assignments(experiment_name)
            
            # Calculate metrics for each variant
            results = {
                'experiment_name': experiment_name,
                'variants': {},
                'total_users': len(assignments),
                'active': exp_config.get('active', False),
                'created_at': exp_config.get('created_at'),
                'updated_at': exp_config.get('updated_at')
            }
            
            for variant in exp_config['variants']:
                variant_users = [user for user, v in assignments.items() if v == variant]
                results['variants'][variant] = {
                    'user_count': len(variant_users),
                    'percentage': len(variant_users) / len(assignments) * 100 if assignments else 0,
                    'users': variant_users[:10]  # Show first 10 users
                }
            
            return results
            
        except Exception as e:
            logger.error(
                "Error getting experiment results",
                experiment_name=experiment_name,
                error=str(e)
            )
            raise ABTestingError(f"Failed to get experiment results: {str(e)}")
    
    async def _load_active_experiments(self) -> None:
        """Load active experiments from Redis."""
        
        try:
            # Get all experiment keys
            experiment_keys = await self.redis_client.keys("ab_experiment:*")
            
            for key in experiment_keys:
                experiment_name = key.replace("ab_experiment:", "")
                experiment_data = await self.redis_client.hgetall(key)
                
                if experiment_data:
                    # Convert string values to appropriate types
                    config = {}
                    for k, v in experiment_data.items():
                        if k in ['active']:
                            config[k] = v.lower() == 'true'
                        elif k in ['traffic_split']:
                            # Parse traffic split JSON
                            import json
                            config[k] = json.loads(v) if v else {}
                        else:
                            config[k] = v
                    
                    self.experiments[experiment_name] = config
            
            logger.info(
                "Loaded active experiments",
                count=len(self.experiments),
                experiments=list(self.experiments.keys())
            )
            
        except Exception as e:
            logger.error(f"Error loading active experiments: {e}")
    
    async def _get_existing_assignment(
        self, 
        user_id: str, 
        experiment: str
    ) -> Optional[str]:
        """Get existing variant assignment for user."""
        
        try:
            assignment_key = f"ab_assignment:{user_id}:{experiment}"
            variant = await self.redis_client.get(assignment_key)
            return variant
            
        except Exception as e:
            logger.error(f"Error getting existing assignment: {e}")
            return None
    
    async def _assign_variant(
        self, 
        user_id: str, 
        experiment: str, 
        exp_config: Dict[str, Any]
    ) -> str:
        """Assign variant to user based on traffic split."""
        
        try:
            # Use consistent hashing for stable assignments
            hash_input = f"{user_id}:{experiment}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            hash_ratio = (hash_value % 10000) / 10000.0
            
            # Find variant based on traffic split
            cumulative_split = 0.0
            for variant, split in exp_config['traffic_split'].items():
                cumulative_split += split
                if hash_ratio <= cumulative_split:
                    # Store assignment
                    assignment_key = f"ab_assignment:{user_id}:{experiment}"
                    await self.redis_client.setex(assignment_key, 86400 * 30, variant)  # 30 days TTL
                    
                    return variant
            
            # Fallback to default variant
            return exp_config.get('default_variant', 'control')
            
        except Exception as e:
            logger.error(f"Error assigning variant: {e}")
            return exp_config.get('default_variant', 'control')
    
    async def _get_variant_assignments(self, experiment: str) -> Dict[str, str]:
        """Get all variant assignments for experiment."""
        
        try:
            # Get all assignment keys for experiment
            pattern = f"ab_assignment:*:{experiment}"
            assignment_keys = await self.redis_client.keys(pattern)
            
            assignments = {}
            for key in assignment_keys:
                # Extract user_id from key
                user_id = key.split(':')[1]
                variant = await self.redis_client.get(key)
                if variant:
                    assignments[user_id] = variant
            
            return assignments
            
        except Exception as e:
            logger.error(f"Error getting variant assignments: {e}")
            return {}
    
    async def get_user_experiments(self, user_id: str) -> Dict[str, str]:
        """Get all experiments and variants for user."""
        
        try:
            user_experiments = {}
            
            for experiment_name in self.experiments.keys():
                variant = await self.get_variant(user_id, experiment_name)
                user_experiments[experiment_name] = variant
            
            return user_experiments
            
        except Exception as e:
            logger.error(f"Error getting user experiments: {e}")
            return {}
    
    async def get_experiment_analytics(self, experiment_name: str) -> Dict[str, Any]:
        """Get detailed analytics for experiment."""
        
        try:
            if experiment_name not in self.experiments:
                raise ABTestingError(f"Experiment {experiment_name} not found")
            
            # Get basic results
            results = await self.get_experiment_results(experiment_name)
            
            # Add additional analytics
            analytics = {
                **results,
                'traffic_distribution': self.experiments[experiment_name].get('traffic_split', {}),
                'experiment_duration_days': self._calculate_experiment_duration(experiment_name),
                'user_retention': await self._calculate_user_retention(experiment_name),
                'conversion_rates': await self._calculate_conversion_rates(experiment_name)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting experiment analytics: {e}")
            return {}
    
    def _calculate_experiment_duration(self, experiment_name: str) -> int:
        """Calculate experiment duration in days."""
        
        try:
            exp_config = self.experiments[experiment_name]
            created_at = datetime.fromisoformat(exp_config.get('created_at', datetime.utcnow().isoformat()))
            duration = (datetime.utcnow() - created_at).days
            return max(0, duration)
            
        except Exception as e:
            logger.error(f"Error calculating experiment duration: {e}")
            return 0
    
    async def _calculate_user_retention(self, experiment_name: str) -> Dict[str, float]:
        """Calculate user retention by variant."""
        
        try:
            # Mock implementation - would calculate from actual user data
            retention = {}
            exp_config = self.experiments[experiment_name]
            
            for variant in exp_config['variants']:
                retention[variant] = 0.75  # Mock retention rate
            
            return retention
            
        except Exception as e:
            logger.error(f"Error calculating user retention: {e}")
            return {}
    
    async def _calculate_conversion_rates(self, experiment_name: str) -> Dict[str, float]:
        """Calculate conversion rates by variant."""
        
        try:
            # Mock implementation - would calculate from actual conversion data
            conversion_rates = {}
            exp_config = self.experiments[experiment_name]
            
            for variant in exp_config['variants']:
                conversion_rates[variant] = 0.15  # Mock conversion rate
            
            return conversion_rates
            
        except Exception as e:
            logger.error(f"Error calculating conversion rates: {e}")
            return {}
