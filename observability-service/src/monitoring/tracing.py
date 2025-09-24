"""
Distributed tracing with OpenTelemetry
"""

import asyncio
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, ContextManager
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.zipkin.json import ZipkinExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBasedSampler
from opentelemetry.trace import Status, StatusCode, Span
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

logger = logging.getLogger(__name__)


@dataclass
class TracingConfig:
    """Tracing configuration"""
    jaeger_enabled: bool = True
    jaeger_host: str = "localhost"
    jaeger_port: int = 14268
    zipkin_enabled: bool = False
    zipkin_endpoint: str = "http://localhost:9411/api/v2/spans"
    otlp_enabled: bool = False
    otlp_endpoint: str = "http://localhost:4317"
    sampling_rate: float = 0.1
    service_name: str = "observability-service"


@dataclass
class TraceAnalysis:
    """Trace performance analysis result"""
    trace_id: str
    total_duration: float
    span_count: int
    critical_path: List[str]
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[str]
    analysis_timestamp: datetime


class TraceManager:
    """Manage distributed tracing with OpenTelemetry"""
    
    def __init__(self):
        self.tracer_provider = None
        self.tracer = None
        self.span_processor = None
        self.sampler = None
        self.exporters = []
        self.is_initialized = False
    
    async def initialize(self, config: TracingConfig) -> None:
        """Initialize distributed tracing"""
        
        logger.info("Initializing distributed tracing...")
        
        # Set up sampling
        self.sampler = TraceIdRatioBasedSampler(rate=config.sampling_rate)
        
        # Create tracer provider
        self.tracer_provider = TracerProvider(sampler=self.sampler)
        
        # Set up span processor
        self.span_processor = BatchSpanProcessor()
        
        # Configure exporters
        await self._setup_exporters(config)
        
        # Add span processor to tracer provider
        self.tracer_provider.add_span_processor(self.span_processor)
        
        # Set as global tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        
        # Get tracer
        self.tracer = self.tracer_provider.get_tracer(config.service_name)
        
        # Instrument libraries
        await self._instrument_libraries()
        
        self.is_initialized = True
        logger.info("Distributed tracing initialized successfully")
    
    async def _setup_exporters(self, config: TracingConfig):
        """Set up trace exporters"""
        
        if config.jaeger_enabled:
            try:
                jaeger_exporter = JaegerExporter(
                    agent_host_name=config.jaeger_host,
                    agent_port=config.jaeger_port
                )
                self.span_processor.add_span_exporter(jaeger_exporter)
                self.exporters.append(jaeger_exporter)
                logger.info("Jaeger exporter configured")
            except Exception as e:
                logger.error(f"Failed to configure Jaeger exporter: {str(e)}")
        
        if config.zipkin_enabled:
            try:
                zipkin_exporter = ZipkinExporter(endpoint=config.zipkin_endpoint)
                self.span_processor.add_span_exporter(zipkin_exporter)
                self.exporters.append(zipkin_exporter)
                logger.info("Zipkin exporter configured")
            except Exception as e:
                logger.error(f"Failed to configure Zipkin exporter: {str(e)}")
        
        if config.otlp_enabled:
            try:
                otlp_exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
                self.span_processor.add_span_exporter(otlp_exporter)
                self.exporters.append(otlp_exporter)
                logger.info("OTLP exporter configured")
            except Exception as e:
                logger.error(f"Failed to configure OTLP exporter: {str(e)}")
    
    async def _instrument_libraries(self):
        """Instrument third-party libraries"""
        
        try:
            # FastAPI instrumentation
            FastAPIInstrumentor().instrument()
            
            # HTTP requests instrumentation
            RequestsInstrumentor().instrument()
            
            # Database instrumentation
            Psycopg2Instrumentor().instrument()
            
            # Redis instrumentation
            RedisInstrumentor().instrument()
            
            # HTTPX instrumentation
            HTTPXClientInstrumentor().instrument()
            
            logger.info("Library instrumentation completed")
            
        except Exception as e:
            logger.error(f"Failed to instrument libraries: {str(e)}")
    
    @contextmanager
    def trace_operation(self, operation_name: str, 
                       attributes: Dict[str, Any] = None) -> ContextManager[Span]:
        """Context manager for tracing operations"""
        
        if not self.is_initialized or not self.tracer:
            # Return a no-op context manager if tracing is not initialized
            yield None
            return
        
        with self.tracer.start_as_current_span(operation_name) as span:
            if span and attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            try:
                yield span
            except Exception as e:
                if span:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    async def trace_async_operation(self, operation_name: str, 
                                  attributes: Dict[str, Any] = None,
                                  operation_func=None):
        """Trace an async operation"""
        
        if not self.is_initialized or not self.tracer:
            return await operation_func() if operation_func else None
        
        with self.tracer.start_as_current_span(operation_name) as span:
            if span and attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            try:
                result = await operation_func() if operation_func else None
                if span:
                    span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                if span:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    async def analyze_trace_performance(self, trace_id: str) -> TraceAnalysis:
        """Analyze trace performance and identify bottlenecks"""
        
        try:
            # Get trace data (this would typically query a trace store)
            trace_data = await self._get_trace_data(trace_id)
            
            if not trace_data:
                raise ValueError(f"Trace {trace_id} not found")
            
            # Calculate span durations
            span_durations = self._calculate_span_durations(trace_data)
            
            # Identify critical path
            critical_path = self._identify_critical_path(trace_data)
            
            # Find bottlenecks
            bottlenecks = self._identify_bottlenecks(span_durations)
            
            # Generate recommendations
            recommendations = self._generate_performance_recommendations(bottlenecks)
            
            return TraceAnalysis(
                trace_id=trace_id,
                total_duration=trace_data.get('total_duration', 0),
                span_count=len(trace_data.get('spans', [])),
                critical_path=critical_path,
                bottlenecks=bottlenecks,
                recommendations=recommendations,
                analysis_timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze trace {trace_id}: {str(e)}")
            raise
    
    async def _get_trace_data(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get trace data by trace ID"""
        
        # This is a placeholder implementation
        # In practice, you would query your trace store (Jaeger, Zipkin, etc.)
        
        # For now, return mock data
        return {
            "trace_id": trace_id,
            "total_duration": 1.5,
            "spans": [
                {
                    "span_id": "span1",
                    "operation_name": "http_request",
                    "duration": 0.2,
                    "start_time": datetime.utcnow(),
                    "tags": {"http.method": "GET", "http.status_code": 200}
                },
                {
                    "span_id": "span2",
                    "operation_name": "database_query",
                    "duration": 0.8,
                    "start_time": datetime.utcnow(),
                    "tags": {"db.operation": "SELECT", "db.table": "articles"}
                },
                {
                    "span_id": "span3",
                    "operation_name": "ml_inference",
                    "duration": 0.5,
                    "start_time": datetime.utcnow(),
                    "tags": {"ml.model": "sentiment_analysis", "ml.version": "v1.0"}
                }
            ]
        }
    
    def _calculate_span_durations(self, trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate span durations and identify slow operations"""
        
        spans = trace_data.get('spans', [])
        span_durations = []
        
        for span in spans:
            duration = span.get('duration', 0)
            span_durations.append({
                'span_id': span.get('span_id'),
                'operation_name': span.get('operation_name'),
                'duration': duration,
                'is_slow': duration > 1.0  # Consider spans > 1s as slow
            })
        
        return sorted(span_durations, key=lambda x: x['duration'], reverse=True)
    
    def _identify_critical_path(self, trace_data: Dict[str, Any]) -> List[str]:
        """Identify the critical path through the trace"""
        
        spans = trace_data.get('spans', [])
        if not spans:
            return []
        
        # Sort spans by duration to find the critical path
        sorted_spans = sorted(spans, key=lambda x: x.get('duration', 0), reverse=True)
        
        critical_path = []
        for span in sorted_spans[:3]:  # Top 3 slowest operations
            critical_path.append(span.get('operation_name', 'unknown'))
        
        return critical_path
    
    def _identify_bottlenecks(self, span_durations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        
        bottlenecks = []
        
        for span in span_durations:
            if span['is_slow']:
                bottlenecks.append({
                    'operation': span['operation_name'],
                    'duration': span['duration'],
                    'severity': 'high' if span['duration'] > 2.0 else 'medium',
                    'recommendation': self._get_bottleneck_recommendation(span)
                })
        
        return bottlenecks
    
    def _get_bottleneck_recommendation(self, span: Dict[str, Any]) -> str:
        """Get recommendation for a bottleneck"""
        
        operation = span['operation_name']
        duration = span['duration']
        
        if 'database' in operation.lower():
            return "Consider database query optimization, indexing, or connection pooling"
        elif 'ml' in operation.lower() or 'inference' in operation.lower():
            return "Consider model optimization, caching, or async processing"
        elif 'http' in operation.lower():
            return "Consider request optimization, caching, or connection pooling"
        else:
            return f"Investigate {operation} performance - duration: {duration}s"
    
    def _generate_performance_recommendations(self, bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        
        if not bottlenecks:
            recommendations.append("No significant bottlenecks detected")
            return recommendations
        
        # Group bottlenecks by type
        db_bottlenecks = [b for b in bottlenecks if 'database' in b['operation'].lower()]
        ml_bottlenecks = [b for b in bottlenecks if 'ml' in b['operation'].lower() or 'inference' in b['operation'].lower()]
        http_bottlenecks = [b for b in bottlenecks if 'http' in b['operation'].lower()]
        
        if db_bottlenecks:
            recommendations.append("Database optimization needed - consider query optimization and indexing")
        
        if ml_bottlenecks:
            recommendations.append("ML model optimization needed - consider model caching and async processing")
        
        if http_bottlenecks:
            recommendations.append("HTTP request optimization needed - consider request batching and caching")
        
        # General recommendations
        if len(bottlenecks) > 3:
            recommendations.append("Multiple performance bottlenecks detected - consider system-wide optimization")
        
        return recommendations
    
    async def get_trace_statistics(self, time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Get trace statistics for a time window"""
        
        try:
            # This would typically query your trace store
            # For now, return mock statistics
            
            return {
                "total_traces": 1000,
                "average_duration": 0.5,
                "p95_duration": 1.2,
                "p99_duration": 2.1,
                "error_rate": 0.02,
                "top_operations": [
                    {"operation": "http_request", "count": 500, "avg_duration": 0.3},
                    {"operation": "database_query", "count": 300, "avg_duration": 0.8},
                    {"operation": "ml_inference", "count": 200, "avg_duration": 0.4}
                ],
                "time_window": str(time_window),
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get trace statistics: {str(e)}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup tracing resources"""
        
        if self.span_processor:
            self.span_processor.shutdown()
        
        if self.tracer_provider:
            self.tracer_provider.shutdown()
        
        logger.info("Trace manager cleaned up")
