"""
DataClient: High-performance async-aware data client wrapper with tracking.
"""

import logging
import time
import asyncio
from typing import Any, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MethodStatistics:
    """Aggregated statistics for a specific method."""
    call_count: int = 0
    first_called: Optional[float] = None
    last_called: Optional[float] = None
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    
    def record_call(self, timestamp: float, duration: float):
        """Record a method call."""
        self.call_count += 1
        
        if self.first_called is None:
            self.first_called = timestamp
        self.last_called = timestamp
        
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
    
    @property
    def avg_duration(self) -> float:
        """Average call duration."""
        return self.total_duration / self.call_count if self.call_count > 0 else 0.0


@dataclass
class DataFlowMetrics:
    """High-level metrics for understanding data flow patterns."""
    calls_per_second: float = 0.0
    read_write_ratio: float = 0.0
    total_operations: int = 0
    unique_methods_used: int = 0
    most_frequent_method: Optional[str] = None
    most_frequent_count: int = 0
    avg_call_duration: float = 0.0
    total_active_time: float = 0.0


class DataClient:
    """
    Transparent wrapper around data backend clients.
    Provides async-safe tracking for learner analysis.
    """
    
    __slots__ = (
        '_backend',
        '_enable_tracking',
        '_method_stats',
        '_stats_lock',
        '_session_start',
        '_write_methods',
        '_read_methods'
    )
    
    def __init__(
        self,
        backend_client: Any,
        enable_tracking: bool = True
    ):
        """
        Initialize DataClient with backend.
        
        Args:
            backend_client: Initialized backend (SmartRedis.Client, etc.)
            enable_tracking: If True, track method calls
        """
        object.__setattr__(self, '_backend', backend_client)
        object.__setattr__(self, '_enable_tracking', enable_tracking)
        object.__setattr__(self, '_method_stats', {} if enable_tracking else None)
        object.__setattr__(self, '_stats_lock', asyncio.Lock() if enable_tracking else None)
        object.__setattr__(self, '_session_start', time.time() if enable_tracking else None)
        
        # Method categorization keywords
        object.__setattr__(self, '_write_methods', frozenset({
            'put', 'set', 'store', 'save', 'upload', 'push', 'send', 'write', 'add', 'insert'
        }))
        object.__setattr__(self, '_read_methods', frozenset({
            'get', 'fetch', 'retrieve', 'load', 'pull', 'read', 'download', 'receive', 'unpack'
        }))
    
    def _categorize_method(self, method_name: str) -> str:
        """Categorize method as read, write, or other."""
        method_lower = method_name.lower()
        
        for keyword in self._write_methods:
            if keyword in method_lower:
                return 'write'
        
        for keyword in self._read_methods:
            if keyword in method_lower:
                return 'read'
        
        return 'other'
    
    def __getattr__(self, name: str) -> Any:
        """Delegate all attribute access to backend with tracking."""
        backend_attr = getattr(self._backend, name)
        
        if not callable(backend_attr):
            return backend_attr
        
        if not self._enable_tracking:
            return backend_attr
        
        def tracked_wrapper(*args, **kwargs):
            start_time = time.time()
            timestamp = start_time
            
            try:
                result = backend_attr(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Method {name} failed: {e}")
                raise
            finally:
                duration = time.time() - start_time
                self._record_call_sync(name, timestamp, duration)
        
        return tracked_wrapper
    
    def __setattr__(self, name: str, value: Any):
        """Allow setting attributes."""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._backend, name, value)
    
    def _record_call_sync(self, method_name: str, timestamp: float, duration: float):
        """Synchronous call recording for minimal overhead."""
        if method_name not in self._method_stats:
            self._method_stats[method_name] = MethodStatistics()
        
        stats = self._method_stats[method_name]
        stats.record_call(timestamp, duration)
    
    async def get_method_stats(self, method_name: str) -> Optional[MethodStatistics]:
        """Get statistics for specific method."""
        if not self._method_stats:
            return None
        
        async with self._stats_lock:
            return self._method_stats.get(method_name)
    
    async def get_all_stats(self) -> Dict[str, MethodStatistics]:
        """Get all method statistics."""
        if not self._method_stats:
            return {}
        
        async with self._stats_lock:
            return dict(self._method_stats)
    
    async def get_flow_metrics(self) -> DataFlowMetrics:
        """Compute high-level data flow metrics."""
        if not self._method_stats:
            return DataFlowMetrics()
        
        async with self._stats_lock:
            return self._compute_metrics()
    
    def _compute_metrics(self) -> DataFlowMetrics:
        """Internal metrics computation (assumes lock is held)."""
        metrics = DataFlowMetrics()
        
        read_count = 0
        write_count = 0
        total_calls = 0
        total_duration = 0.0
        
        most_frequent = None
        most_frequent_count = 0
        
        for method_name, stats in self._method_stats.items():
            total_calls += stats.call_count
            total_duration += stats.total_duration
            
            if stats.call_count > most_frequent_count:
                most_frequent = method_name
                most_frequent_count = stats.call_count
            
            category = self._categorize_method(method_name)
            if category == 'read':
                read_count += stats.call_count
            elif category == 'write':
                write_count += stats.call_count
        
        metrics.total_operations = total_calls
        metrics.unique_methods_used = len(self._method_stats)
        metrics.most_frequent_method = most_frequent
        metrics.most_frequent_count = most_frequent_count
        
        if write_count > 0:
            metrics.read_write_ratio = read_count / write_count
        
        session_duration = time.time() - self._session_start
        if session_duration > 0:
            metrics.calls_per_second = total_calls / session_duration
        
        metrics.avg_call_duration = total_duration / total_calls if total_calls > 0 else 0.0
        metrics.total_active_time = total_duration
        
        return metrics
    
    async def get_method_distribution(self) -> Dict[str, int]:
        """Get call count distribution across methods."""
        if not self._method_stats:
            return {}
        
        async with self._stats_lock:
            return {name: stats.call_count for name, stats in self._method_stats.items()}
    
    async def get_timing_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get detailed timing breakdown for each method."""
        if not self._method_stats:
            return {}
        
        async with self._stats_lock:
            return {
                name: {
                    'total_duration': stats.total_duration,
                    'avg_duration': stats.avg_duration,
                    'min_duration': stats.min_duration,
                    'max_duration': stats.max_duration
                }
                for name, stats in self._method_stats.items()
            }
    
    async def export_stats(self, filepath: str):
        """Export statistics to JSON file."""
        import json
        
        if not self._method_stats:
            logger.warning("No statistics to export")
            return
        
        async with self._stats_lock:
            metrics = self._compute_metrics()
            
            data = {
                'session_start': self._session_start,
                'session_end': time.time(),
                'session_duration': time.time() - self._session_start,
                'flow_metrics': {
                    'calls_per_second': metrics.calls_per_second,
                    'read_write_ratio': metrics.read_write_ratio,
                    'total_operations': metrics.total_operations,
                    'unique_methods_used': metrics.unique_methods_used,
                    'most_frequent_method': metrics.most_frequent_method,
                    'most_frequent_count': metrics.most_frequent_count,
                    'avg_call_duration': metrics.avg_call_duration,
                    'total_active_time': metrics.total_active_time
                },
                'method_stats': {
                    name: {
                        'call_count': stats.call_count,
                        'total_duration': stats.total_duration,
                        'avg_duration': stats.avg_duration,
                        'min_duration': stats.min_duration,
                        'max_duration': stats.max_duration,
                        'first_called': stats.first_called,
                        'last_called': stats.last_called
                    }
                    for name, stats in self._method_stats.items()
                }
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Statistics exported to {filepath}")
    
    async def log_summary(self):
        """Log comprehensive summary."""
        if not self._method_stats:
            logger.info("Tracking is disabled")
            return
        
        metrics = await self.get_flow_metrics()
        
        logger.info("DataClient tracking summary")
        logger.info(f"Total operations: {metrics.total_operations}")
        logger.info(f"Unique methods: {metrics.unique_methods_used}")
        logger.info(f"Operations per second: {metrics.calls_per_second:.2f}")
        logger.info(f"Average call duration: {metrics.avg_call_duration * 1000:.2f} ms")
        logger.info(f"Read/Write ratio: {metrics.read_write_ratio:.2f}")
        logger.info(f"Most frequent: {metrics.most_frequent_method} ({metrics.most_frequent_count} calls)")
    
    async def reset_stats(self):
        """Reset all statistics."""
        if self._method_stats:
            async with self._stats_lock:
                self._method_stats.clear()
                object.__setattr__(self, '_session_start', time.time())
            logger.info("Statistics reset")
    
    def get_backend(self) -> Any:
        """Get direct access to backend."""
        return self._backend
    
    def __repr__(self) -> str:
        if not self._method_stats:
            return "DataClient(tracking_disabled)"
        
        total = sum(s.call_count for s in self._method_stats.values())
        unique = len(self._method_stats)
        return f"DataClient(calls={total}, unique_methods={unique})"
