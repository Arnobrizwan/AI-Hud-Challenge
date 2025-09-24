"""
Online Evaluator - A/B testing and online evaluation orchestration
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from .ab_testing import ABTestingFramework
from .bandit_testing import BanditTestingFramework
from .sequential_testing import SequentialTestingFramework
from .bayesian_testing import BayesianTestingFramework
from ..models import ExperimentStatus

logger = logging.getLogger(__name__)


class OnlineEvaluator:
    """Online evaluation orchestrator for A/B testing and online experiments"""
    
    def __init__(self):
        self.ab_tester = ABTestingFramework()
        self.bandit_tester = None  # Will be initialized
        self.sequential_tester = None  # Will be initialized
        self.bayesian_tester = None  # Will be initialized
        
        # Evaluation tracking
        self.active_evaluations: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize the online evaluator"""
        try:
            logger.info("Initializing online evaluator...")
            
            # Initialize A/B testing framework
            await self.ab_tester.initialize()
            
            # Initialize other testing frameworks
            self.bandit_tester = BanditTestingFramework()
            self.sequential_tester = SequentialTestingFramework()
            self.bayesian_tester = BayesianTestingFramework()
            
            await self.bandit_tester.initialize()
            await self.sequential_tester.initialize()
            await self.bayesian_tester.initialize()
            
            logger.info("Online evaluator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize online evaluator: {str(e)}")
            raise
    
    async def cleanup(self):
        """Cleanup online evaluator resources"""
        try:
            logger.info("Cleaning up online evaluator...")
            
            await self.ab_tester.cleanup()
            
            if self.bandit_tester:
                await self.bandit_tester.cleanup()
            if self.sequential_tester:
                await self.sequential_tester.cleanup()
            if self.bayesian_tester:
                await self.bayesian_tester.cleanup()
            
            logger.info("Online evaluator cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during online evaluator cleanup: {str(e)}")
    
    async def evaluate(self, 
                      experiments: List[Dict[str, Any]], 
                      evaluation_period: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive online evaluation of experiments"""
        
        logger.info(f"Starting online evaluation of {len(experiments)} experiments")
        
        evaluation_results = {
            'evaluation_id': f"online_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'started_at': datetime.utcnow(),
            'experiments': [],
            'evaluation_period': evaluation_period,
            'overall_summary': {}
        }
        
        try:
            # Process each experiment
            experiment_results = []
            for experiment_config in experiments:
                experiment_result = await self._evaluate_experiment(experiment_config, evaluation_period)
                experiment_results.append(experiment_result)
            
            evaluation_results['experiments'] = experiment_results
            evaluation_results['completed_at'] = datetime.utcnow()
            
            # Generate overall summary
            evaluation_results['overall_summary'] = await self._generate_overall_summary(experiment_results)
            
            logger.info("Online evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Error in online evaluation: {str(e)}")
            evaluation_results['error'] = str(e)
            evaluation_results['completed_at'] = datetime.utcnow()
            raise
        
        return evaluation_results
    
    async def _evaluate_experiment(self, 
                                 experiment_config: Dict[str, Any], 
                                 evaluation_period: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single experiment"""
        
        experiment_type = experiment_config.get('type', 'ab_test')
        experiment_id = experiment_config.get('id')
        
        logger.info(f"Evaluating {experiment_type} experiment: {experiment_id}")
        
        try:
            if experiment_type == 'ab_test':
                return await self._evaluate_ab_test(experiment_config, evaluation_period)
            elif experiment_type == 'bandit':
                return await self._evaluate_bandit_test(experiment_config, evaluation_period)
            elif experiment_type == 'sequential':
                return await self._evaluate_sequential_test(experiment_config, evaluation_period)
            elif experiment_type == 'bayesian':
                return await self._evaluate_bayesian_test(experiment_config, evaluation_period)
            else:
                raise ValueError(f"Unsupported experiment type: {experiment_type}")
                
        except Exception as e:
            logger.error(f"Error evaluating experiment {experiment_id}: {str(e)}")
            return {
                'experiment_id': experiment_id,
                'experiment_type': experiment_type,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _evaluate_ab_test(self, 
                              experiment_config: Dict[str, Any], 
                              evaluation_period: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate A/B test experiment"""
        
        experiment_id = experiment_config.get('id')
        
        # Get experiment from A/B tester
        experiment = await self.ab_tester.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"A/B test experiment {experiment_id} not found")
        
        # Analyze experiment
        analysis = await self.ab_tester.analyze_experiment(
            experiment_id, 
            analysis_type=experiment_config.get('analysis_type', 'frequentist')
        )
        
        # Determine experiment status
        status = 'completed' if experiment.status == ExperimentStatus.COMPLETED else 'running'
        
        # Extract key metrics
        primary_metric = experiment.primary_metric
        winner_variant = None
        winner_confidence = 0.0
        
        if isinstance(analysis.statistical_results, dict):
            # Find winning variant
            for variant, result in analysis.statistical_results.get('variant_results', {}).items():
                if result.get('primary_metric_result', {}).get('is_significant', False):
                    winner_variant = variant
                    winner_confidence = 1.0 - result.get('primary_metric_result', {}).get('p_value', 1.0)
                    break
        
        return {
            'experiment_id': experiment_id,
            'experiment_type': 'ab_test',
            'status': status,
            'primary_metric': primary_metric,
            'winner_variant': winner_variant,
            'winner_confidence': winner_confidence,
            'analysis': analysis.dict(),
            'recommendations': analysis.recommendations,
            'evaluation_timestamp': datetime.utcnow()
        }
    
    async def _evaluate_bandit_test(self, 
                                  experiment_config: Dict[str, Any], 
                                  evaluation_period: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate multi-armed bandit experiment"""
        
        experiment_id = experiment_config.get('id')
        
        # Analyze bandit experiment
        analysis = await self.bandit_tester.analyze_experiment(experiment_id)
        
        return {
            'experiment_id': experiment_id,
            'experiment_type': 'bandit',
            'status': 'running',  # Bandits typically run continuously
            'analysis': analysis,
            'evaluation_timestamp': datetime.utcnow()
        }
    
    async def _evaluate_sequential_test(self, 
                                     experiment_config: Dict[str, Any], 
                                     evaluation_period: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate sequential test experiment"""
        
        experiment_id = experiment_config.get('id')
        
        # Analyze sequential experiment
        analysis = await self.sequential_tester.analyze_experiment(experiment_id)
        
        return {
            'experiment_id': experiment_id,
            'experiment_type': 'sequential',
            'status': 'running' if not analysis.get('early_stopping', False) else 'completed',
            'analysis': analysis,
            'evaluation_timestamp': datetime.utcnow()
        }
    
    async def _evaluate_bayesian_test(self, 
                                    experiment_config: Dict[str, Any], 
                                    evaluation_period: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Bayesian test experiment"""
        
        experiment_id = experiment_config.get('id')
        
        # Analyze Bayesian experiment
        analysis = await self.bayesian_tester.analyze_experiment(experiment_id)
        
        return {
            'experiment_id': experiment_id,
            'experiment_type': 'bayesian',
            'status': 'running',
            'analysis': analysis,
            'evaluation_timestamp': datetime.utcnow()
        }
    
    async def _generate_overall_summary(self, experiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall summary of online evaluation results"""
        
        if not experiment_results:
            return {}
        
        # Count experiments by type and status
        experiment_types = {}
        experiment_statuses = {}
        
        for result in experiment_results:
            exp_type = result.get('experiment_type', 'unknown')
            exp_status = result.get('status', 'unknown')
            
            experiment_types[exp_type] = experiment_types.get(exp_type, 0) + 1
            experiment_statuses[exp_status] = experiment_statuses.get(exp_status, 0) + 1
        
        # Count experiments with winners
        experiments_with_winners = len([
            r for r in experiment_results 
            if r.get('winner_variant') is not None
        ])
        
        # Calculate average confidence for winning experiments
        winning_experiments = [
            r for r in experiment_results 
            if r.get('winner_variant') is not None
        ]
        
        avg_winner_confidence = 0.0
        if winning_experiments:
            avg_winner_confidence = sum(
                r.get('winner_confidence', 0) for r in winning_experiments
            ) / len(winning_experiments)
        
        return {
            'total_experiments': len(experiment_results),
            'experiment_types': experiment_types,
            'experiment_statuses': experiment_statuses,
            'experiments_with_winners': experiments_with_winners,
            'winning_rate': experiments_with_winners / len(experiment_results) if experiment_results else 0,
            'average_winner_confidence': avg_winner_confidence,
            'evaluation_timestamp': datetime.utcnow()
        }
    
    async def create_experiment(self, 
                              experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new online experiment"""
        
        experiment_type = experiment_config.get('type', 'ab_test')
        
        if experiment_type == 'ab_test':
            from ..models import ExperimentConfig
            config = ExperimentConfig(**experiment_config)
            experiment = await self.ab_tester.create_experiment(config)
            return {
                'experiment_id': experiment.id,
                'experiment_type': 'ab_test',
                'status': experiment.status.value,
                'created_at': experiment.created_at
            }
        else:
            raise ValueError(f"Unsupported experiment type for creation: {experiment_type}")
    
    async def start_experiment(self, experiment_id: str, experiment_type: str = 'ab_test') -> bool:
        """Start an online experiment"""
        
        if experiment_type == 'ab_test':
            return await self.ab_tester.start_experiment(experiment_id)
        else:
            raise ValueError(f"Unsupported experiment type for starting: {experiment_type}")
    
    async def stop_experiment(self, experiment_id: str, experiment_type: str = 'ab_test') -> bool:
        """Stop an online experiment"""
        
        if experiment_type == 'ab_test':
            return await self.ab_tester.stop_experiment(experiment_id)
        else:
            raise ValueError(f"Unsupported experiment type for stopping: {experiment_type}")
    
    async def record_event(self, 
                         experiment_id: str, 
                         user_id: str, 
                         variant: str, 
                         event_type: str, 
                         value: float = 1.0,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Record an event for an online experiment"""
        
        # Try A/B testing first (most common)
        success = await self.ab_tester.record_event(
            experiment_id, user_id, variant, event_type, value, metadata
        )
        
        if success:
            return True
        
        # Try other experiment types if A/B testing fails
        # This would be implemented based on specific requirements
        
        return False
