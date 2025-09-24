"""
Data aggregation and analytics for feedback insights
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from ..models.database import Feedback as FeedbackDB, PerformanceMetric

logger = structlog.get_logger(__name__)

class DataAggregator:
    """Aggregate and analyze feedback data"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.trend_analyzer = TrendAnalyzer()
        self.insight_generator = InsightGenerator()
        
    async def aggregate_feedback(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Aggregate feedback data for time period"""
        
        try:
            # Get feedback counts by type
            result = await self.db.execute(
                select(
                    FeedbackDB.feedback_type,
                    func.count(FeedbackDB.id).label('count'),
                    func.avg(FeedbackDB.rating).label('avg_rating')
                ).where(
                    and_(
                        FeedbackDB.created_at >= start_time,
                        FeedbackDB.created_at <= end_time
                    )
                ).group_by(FeedbackDB.feedback_type)
            )
            
            feedback_stats = result.fetchall()
            
            # Convert to dictionary
            summary = {
                "time_period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "total_feedback": sum(stat.count for stat in feedback_stats),
                "by_type": {
                    stat.feedback_type.value: {
                        "count": stat.count,
                        "avg_rating": float(stat.avg_rating) if stat.avg_rating else None
                    }
                    for stat in feedback_stats
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error("Error aggregating feedback", error=str(e))
            return {}
    
    async def generate_insights(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from feedback summary"""
        
        try:
            insights = []
            
            # Insight 1: Feedback volume trend
            total_feedback = summary.get("total_feedback", 0)
            if total_feedback > 100:
                insights.append({
                    "type": "volume",
                    "title": "High Feedback Volume",
                    "description": f"Received {total_feedback} feedback items in the time period",
                    "severity": "info"
                })
            elif total_feedback < 10:
                insights.append({
                    "type": "volume",
                    "title": "Low Feedback Volume",
                    "description": f"Only {total_feedback} feedback items received",
                    "severity": "warning"
                })
            
            # Insight 2: Rating distribution
            by_type = summary.get("by_type", {})
            for feedback_type, stats in by_type.items():
                avg_rating = stats.get("avg_rating")
                if avg_rating is not None:
                    if avg_rating < 2.0:
                        insights.append({
                            "type": "rating",
                            "title": f"Low {feedback_type} Ratings",
                            "description": f"Average rating is {avg_rating:.2f}",
                            "severity": "warning"
                        })
                    elif avg_rating > 4.0:
                        insights.append({
                            "type": "rating",
                            "title": f"High {feedback_type} Ratings",
                            "description": f"Average rating is {avg_rating:.2f}",
                            "severity": "info"
                        })
            
            return insights
            
        except Exception as e:
            logger.error("Error generating insights", error=str(e))
            return []
    
    async def get_metrics(self, metric_name: Optional[str] = None, 
                         model_name: Optional[str] = None,
                         time_window: timedelta = timedelta(hours=24)) -> List[Dict[str, Any]]:
        """Get performance metrics"""
        
        try:
            start_time = datetime.utcnow() - time_window
            
            query = select(PerformanceMetric).where(
                PerformanceMetric.timestamp >= start_time
            )
            
            if metric_name:
                query = query.where(PerformanceMetric.metric_name == metric_name)
            
            if model_name:
                query = query.where(PerformanceMetric.model_name == model_name)
            
            result = await self.db.execute(query)
            metrics = result.scalars().all()
            
            return [
                {
                    "metric_name": metric.metric_name,
                    "metric_value": float(metric.metric_value),
                    "model_name": metric.model_name,
                    "timestamp": metric.timestamp.isoformat(),
                    "metadata": metric.metadata
                }
                for metric in metrics
            ]
            
        except Exception as e:
            logger.error("Error getting metrics", error=str(e))
            return []

class TrendAnalyzer:
    """Analyze trends in feedback data"""
    
    def analyze_trends(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in time series data"""
        
        # This would implement trend analysis
        # For now, return empty trends
        return {}

class InsightGenerator:
    """Generate actionable insights from data"""
    
    def generate_insights(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from aggregated data"""
        
        # This would implement insight generation
        # For now, return empty insights
        return []
