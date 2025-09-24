"""
Circuit breaker implementation for fault tolerance and resilience.
"""

import asyncio
import time
from typing import Any, Callable, Dict, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
import functools

from src.config.settings import settings
from src.utils.logging import get_logger
from src.utils.metrics import metrics_collector

logger = get_logger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = 0      # Normal operation
    OPEN = 1        # Failing, requests rejected
    HALF_OPEN = 2   # Testing if service recovered


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    total_requests: int = 0


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = field(default_factory=lambda: settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD)
    recovery_timeout: float = field(default_factory=lambda: settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT)
    expected_exception: tuple = field(default_factory=lambda: settings.CIRCUIT_BREAKER_EXPECTED_EXCEPTION)
    success_threshold: int = 1  # Successes needed in half-open state
    timeout: float = 30.0  # Request timeout


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    def __init__(self, service_name: str, last_failure_time: float):
        self.service_name = service_name
        self.last_failure_time = last_failure_time
        super().__init__(f"Circuit breaker is open for {service_name}")


class CircuitBreakerTimeoutException(Exception):
    """Exception raised when request times out."""
    def __init__(self, service_name: str, timeout: float):
        self.service_name = service_name
        self.timeout = timeout
        super().__init__(f"Request to {service_name} timed out after {timeout}s")


class CircuitBreaker:
    """Circuit breaker implementation with async support."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        
        # Initialize metrics
        metrics_collector.set_circuit_breaker_state(name, self.state.value)
        
        logger.info(
            f"Circuit breaker initialized",
            service=name,
            config=self.config.__dict__
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._check_and_update_state()
        
        if self.state == CircuitBreakerState.OPEN:
            raise CircuitBreakerOpenException(self.name, self.stats.last_failure_time)
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type is None:
            await self._record_success()
        elif issubclass(exc_type, self.config.expected_exception):
            await self._record_failure()
        # Don't suppress exceptions
        return False
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self:
            try:
                # Apply timeout
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.timeout
                    )
                else:
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, functools.partial(func, *args, **kwargs)
                        ),
                        timeout=self.config.timeout
                    )
                return result
            
            except asyncio.TimeoutError:
                raise CircuitBreakerTimeoutException(self.name, self.config.timeout)
    
    def __call__(self, func: Callable):
        """Decorator for circuit breaker protection."""
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.call(func, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            async def sync_wrapper(*args, **kwargs):
                return await self.call(func, *args, **kwargs)
            return sync_wrapper
    
    async def _check_and_update_state(self):
        """Check and update circuit breaker state."""
        async with self._lock:
            current_time = time.time()
            
            if self.state == CircuitBreakerState.OPEN:
                # Check if recovery timeout has passed
                if (self.stats.last_failure_time and 
                    current_time - self.stats.last_failure_time >= self.config.recovery_timeout):
                    self._transition_to_half_open()
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # In half-open state, check if we should close or open
                if self.stats.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
    
    async def _record_success(self):
        """Record successful request."""
        async with self._lock:
            self.stats.success_count += 1
            self.stats.total_requests += 1
            self.stats.last_success_time = time.time()
            
            # Reset failure count on success
            if self.state == CircuitBreakerState.CLOSED:
                self.stats.failure_count = 0
            
            logger.debug(
                f"Circuit breaker success recorded",
                service=self.name,
                state=self.state.name,
                stats=self.stats.__dict__
            )
    
    async def _record_failure(self):
        """Record failed request."""
        async with self._lock:
            self.stats.failure_count += 1
            self.stats.total_requests += 1
            self.stats.last_failure_time = time.time()
            
            # Record failure metrics
            metrics_collector.record_circuit_breaker_failure(self.name)
            
            # Check if we should open the circuit
            if (self.state == CircuitBreakerState.CLOSED and 
                self.stats.failure_count >= self.config.failure_threshold):
                self._transition_to_open()
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open state opens the circuit
                self._transition_to_open()
            
            logger.warning(
                f"Circuit breaker failure recorded",
                service=self.name,
                state=self.state.name,
                stats=self.stats.__dict__
            )
    
    def _transition_to_open(self):
        """Transition to open state."""
        old_state = self.state
        self.state = CircuitBreakerState.OPEN
        self.stats.success_count = 0  # Reset success count
        
        # Update metrics
        metrics_collector.set_circuit_breaker_state(self.name, self.state.value)
        
        logger.warning(
            f"Circuit breaker opened",
            service=self.name,
            old_state=old_state.name,
            new_state=self.state.name,
            failure_count=self.stats.failure_count,
            threshold=self.config.failure_threshold
        )
    
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        old_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self.stats.success_count = 0  # Reset success count for testing
        
        # Update metrics
        metrics_collector.set_circuit_breaker_state(self.name, self.state.value)
        
        logger.info(
            f"Circuit breaker half-opened",
            service=self.name,
            old_state=old_state.name,
            new_state=self.state.name
        )
    
    def _transition_to_closed(self):
        """Transition to closed state."""
        old_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self.stats.failure_count = 0  # Reset failure count
        self.stats.success_count = 0  # Reset success count
        
        # Update metrics
        metrics_collector.set_circuit_breaker_state(self.name, self.state.value)
        
        logger.info(
            f"Circuit breaker closed",
            service=self.name,
            old_state=old_state.name,
            new_state=self.state.name
        )
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit breaker is closed."""
        return self.state == CircuitBreakerState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == CircuitBreakerState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open."""
        return self.state == CircuitBreakerState.HALF_OPEN
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            'name': self.name,
            'state': self.state.name,
            'failure_count': self.stats.failure_count,
            'success_count': self.stats.success_count,
            'total_requests': self.stats.total_requests,
            'last_failure_time': self.stats.last_failure_time,
            'last_success_time': self.stats.last_success_time,
            'config': self.config.__dict__
        }


class CircuitBreakerRegistry:
    """Registry for managing circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._configs: Dict[str, CircuitBreakerConfig] = {}
    
    def register(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Register a new circuit breaker."""
        if name in self._breakers:
            return self._breakers[name]
        
        breaker = CircuitBreaker(name, config)
        self._breakers[name] = breaker
        self._configs[name] = config or CircuitBreakerConfig()
        
        return breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)
    
    def get_or_create(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name not in self._breakers:
            return self.register(name, config)
        return self._breakers[name]
    
    def list_breakers(self) -> Dict[str, CircuitBreaker]:
        """List all registered circuit breakers."""
        return self._breakers.copy()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}
    
    def reset(self, name: str) -> bool:
        """Reset circuit breaker state."""
        if name in self._breakers:
            breaker = self._breakers[name]
            breaker.state = CircuitBreakerState.CLOSED
            breaker.stats = CircuitBreakerStats()
            metrics_collector.set_circuit_breaker_state(name, breaker.state.value)
            return True
        return False
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for name in self._breakers:
            self.reset(name)


# Global circuit breaker registry
circuit_breaker_registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker."""
    return circuit_breaker_registry.get_or_create(name, config)


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator for applying circuit breaker to functions."""
    breaker = get_circuit_breaker(name, config)
    return breaker
