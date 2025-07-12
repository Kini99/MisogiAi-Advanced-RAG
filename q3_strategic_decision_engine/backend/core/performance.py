"""
Performance monitoring and optimization module for Strategic Decision Engine.
"""

import time
import psutil
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: datetime
    endpoint: str
    method: str
    duration_ms: float
    status_code: int
    memory_usage_mb: float
    cpu_percent: float
    request_size_bytes: int = 0
    response_size_bytes: int = 0
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class SystemMetrics:
    """Container for system-wide metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    active_connections: int
    total_requests: int
    avg_response_time_ms: float


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self, max_metrics: int = 10000):
        """Initialize the performance monitor."""
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.system_metrics: deque = deque(maxlen=1440)  # 24 hours of minute-by-minute data
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        self.active_requests = {}
        self.start_time = datetime.now()
        self.lock = threading.Lock()
        
        # Start system monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitoring_active = True
        self.monitoring_thread.start()
    
    def _monitor_system(self):
        """Background thread to monitor system metrics."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                
                system_metric = SystemMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / (1024 * 1024),
                    memory_available_mb=memory.available / (1024 * 1024),
                    disk_percent=disk.percent,
                    disk_used_gb=disk.used / (1024 * 1024 * 1024),
                    disk_free_gb=disk.free / (1024 * 1024 * 1024),
                    network_sent_mb=network.bytes_sent / (1024 * 1024),
                    network_recv_mb=network.bytes_recv / (1024 * 1024),
                    active_connections=len(self.active_requests),
                    total_requests=sum(self.request_counts.values()),
                    avg_response_time_ms=self._calculate_avg_response_time()
                )
                
                with self.lock:
                    self.system_metrics.append(system_metric)
                
                time.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(60)
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time across all endpoints."""
        if not self.response_times:
            return 0.0
        
        total_time = 0
        total_requests = 0
        
        for times in self.response_times.values():
            if times:
                total_time += sum(times)
                total_requests += len(times)
        
        return total_time / total_requests if total_requests > 0 else 0.0
    
    def record_request(self, metric: PerformanceMetrics):
        """Record a request performance metric."""
        with self.lock:
            self.metrics.append(metric)
            self.request_counts[metric.endpoint] += 1
            
            if metric.error_message:
                self.error_counts[metric.endpoint] += 1
            
            # Keep only recent response times for averaging
            if len(self.response_times[metric.endpoint]) > 100:
                self.response_times[metric.endpoint].pop(0)
            self.response_times[metric.endpoint].append(metric.duration_ms)
    
    def start_request(self, request_id: str, endpoint: str, method: str):
        """Start tracking a request."""
        with self.lock:
            self.active_requests[request_id] = {
                'start_time': time.time(),
                'endpoint': endpoint,
                'method': method
            }
    
    def end_request(self, request_id: str) -> Optional[float]:
        """End tracking a request and return duration."""
        with self.lock:
            if request_id in self.active_requests:
                start_time = self.active_requests[request_id]['start_time']
                duration = (time.time() - start_time) * 1000  # Convert to milliseconds
                del self.active_requests[request_id]
                return duration
            return None
    
    def get_metrics_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get performance metrics summary for the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
            recent_system = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"message": "No metrics available for the specified time period"}
        
        # Calculate summary statistics
        total_requests = len(recent_metrics)
        error_count = sum(1 for m in recent_metrics if m.error_message)
        avg_duration = sum(m.duration_ms for m in recent_metrics) / total_requests
        max_duration = max(m.duration_ms for m in recent_metrics)
        min_duration = min(m.duration_ms for m in recent_metrics)
        
        # Endpoint statistics
        endpoint_stats = defaultdict(lambda: {'count': 0, 'errors': 0, 'total_time': 0})
        for metric in recent_metrics:
            stats = endpoint_stats[metric.endpoint]
            stats['count'] += 1
            stats['total_time'] += metric.duration_ms
            if metric.error_message:
                stats['errors'] += 1
        
        # Convert to final format
        endpoint_summary = {}
        for endpoint, stats in endpoint_stats.items():
            endpoint_summary[endpoint] = {
                'requests': stats['count'],
                'errors': stats['errors'],
                'error_rate': stats['errors'] / stats['count'] * 100,
                'avg_response_time': stats['total_time'] / stats['count']
            }
        
        # System metrics summary
        system_summary = {}
        if recent_system:
            system_summary = {
                'avg_cpu_percent': sum(m.cpu_percent for m in recent_system) / len(recent_system),
                'avg_memory_percent': sum(m.memory_percent for m in recent_system) / len(recent_system),
                'avg_memory_used_mb': sum(m.memory_used_mb for m in recent_system) / len(recent_system),
                'max_active_connections': max(m.active_connections for m in recent_system),
                'avg_response_time_ms': sum(m.avg_response_time_ms for m in recent_system) / len(recent_system)
            }
        
        return {
            'time_period_minutes': minutes,
            'total_requests': total_requests,
            'error_count': error_count,
            'error_rate_percent': (error_count / total_requests) * 100,
            'avg_response_time_ms': avg_duration,
            'max_response_time_ms': max_duration,
            'min_response_time_ms': min_duration,
            'requests_per_minute': total_requests / minutes,
            'endpoint_summary': endpoint_summary,
            'system_summary': system_summary,
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }
    
    def get_slow_requests(self, threshold_ms: float = 1000, limit: int = 10) -> List[Dict]:
        """Get slowest requests above threshold."""
        with self.lock:
            slow_requests = [
                m for m in self.metrics 
                if m.duration_ms > threshold_ms
            ]
        
        # Sort by duration (slowest first)
        slow_requests.sort(key=lambda x: x.duration_ms, reverse=True)
        
        return [
            {
                'timestamp': m.timestamp.isoformat(),
                'endpoint': m.endpoint,
                'method': m.method,
                'duration_ms': m.duration_ms,
                'status_code': m.status_code,
                'memory_usage_mb': m.memory_usage_mb,
                'error_message': m.error_message
            }
            for m in slow_requests[:limit]
        ]
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Get detailed error analysis."""
        with self.lock:
            error_metrics = [m for m in self.metrics if m.error_message]
        
        if not error_metrics:
            return {"message": "No errors recorded"}
        
        # Group errors by endpoint and error message
        error_groups = defaultdict(lambda: defaultdict(int))
        for metric in error_metrics:
            error_groups[metric.endpoint][metric.error_message] += 1
        
        # Convert to structured format
        error_analysis = {}
        for endpoint, errors in error_groups.items():
            error_analysis[endpoint] = {
                'total_errors': sum(errors.values()),
                'error_types': dict(errors)
            }
        
        return {
            'total_errors': len(error_metrics),
            'error_rate_percent': len(error_metrics) / len(self.metrics) * 100 if self.metrics else 0,
            'by_endpoint': error_analysis
        }
    
    def cleanup_old_metrics(self, hours: int = 24):
        """Clean up metrics older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            # Remove old metrics
            self.metrics = deque(
                [m for m in self.metrics if m.timestamp >= cutoff_time],
                maxlen=self.max_metrics
            )
            
            # Remove old system metrics
            self.system_metrics = deque(
                [m for m in self.system_metrics if m.timestamp >= cutoff_time],
                maxlen=1440
            )
    
    def stop_monitoring(self):
        """Stop the background monitoring thread."""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def performance_track(endpoint_name: Optional[str] = None):
    """Decorator to track function performance."""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            error_message = None
            status_code = 200
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error_message = str(e)
                status_code = 500
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                duration_ms = (end_time - start_time) * 1000
                
                metric = PerformanceMetrics(
                    timestamp=datetime.now(),
                    endpoint=endpoint_name or func.__name__,
                    method='async_function',
                    duration_ms=duration_ms,
                    status_code=status_code,
                    memory_usage_mb=end_memory - start_memory,
                    cpu_percent=psutil.cpu_percent(),
                    error_message=error_message
                )
                
                performance_monitor.record_request(metric)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            error_message = None
            status_code = 200
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_message = str(e)
                status_code = 500
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                duration_ms = (end_time - start_time) * 1000
                
                metric = PerformanceMetrics(
                    timestamp=datetime.now(),
                    endpoint=endpoint_name or func.__name__,
                    method='function',
                    duration_ms=duration_ms,
                    status_code=status_code,
                    memory_usage_mb=end_memory - start_memory,
                    cpu_percent=psutil.cpu_percent(),
                    error_message=error_message
                )
                
                performance_monitor.record_request(metric)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


@asynccontextmanager
async def performance_context(operation_name: str):
    """Context manager for tracking performance of code blocks."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    error_message = None
    
    try:
        yield
    except Exception as e:
        error_message = str(e)
        raise
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        duration_ms = (end_time - start_time) * 1000
        
        metric = PerformanceMetrics(
            timestamp=datetime.now(),
            endpoint=operation_name,
            method='context',
            duration_ms=duration_ms,
            status_code=500 if error_message else 200,
            memory_usage_mb=end_memory - start_memory,
            cpu_percent=psutil.cpu_percent(),
            error_message=error_message
        )
        
        performance_monitor.record_request(metric)


class PerformanceOptimizer:
    """System optimization utilities."""
    
    @staticmethod
    def get_optimization_recommendations() -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Get current system metrics
        metrics_summary = performance_monitor.get_metrics_summary(minutes=60)
        
        # Check response time
        if metrics_summary.get('avg_response_time_ms', 0) > 1000:
            recommendations.append({
                'category': 'Response Time',
                'severity': 'high',
                'issue': 'Average response time is above 1 second',
                'recommendation': 'Consider implementing caching, database query optimization, or scaling resources',
                'current_value': f"{metrics_summary.get('avg_response_time_ms', 0):.2f}ms",
                'target_value': '<500ms'
            })
        
        # Check error rate
        error_rate = metrics_summary.get('error_rate_percent', 0)
        if error_rate > 5:
            recommendations.append({
                'category': 'Error Rate',
                'severity': 'high' if error_rate > 10 else 'medium',
                'issue': f'Error rate is {error_rate:.1f}%',
                'recommendation': 'Investigate error patterns and improve error handling',
                'current_value': f"{error_rate:.1f}%",
                'target_value': '<2%'
            })
        
        # Check system resources
        system_summary = metrics_summary.get('system_summary', {})
        
        if system_summary.get('avg_cpu_percent', 0) > 80:
            recommendations.append({
                'category': 'CPU Usage',
                'severity': 'high',
                'issue': 'High CPU utilization',
                'recommendation': 'Consider scaling horizontally or optimizing CPU-intensive operations',
                'current_value': f"{system_summary.get('avg_cpu_percent', 0):.1f}%",
                'target_value': '<70%'
            })
        
        if system_summary.get('avg_memory_percent', 0) > 85:
            recommendations.append({
                'category': 'Memory Usage',
                'severity': 'high',
                'issue': 'High memory utilization',
                'recommendation': 'Consider increasing memory or optimizing memory usage patterns',
                'current_value': f"{system_summary.get('avg_memory_percent', 0):.1f}%",
                'target_value': '<80%'
            })
        
        # Check slow requests
        slow_requests = performance_monitor.get_slow_requests(threshold_ms=2000, limit=5)
        if slow_requests:
            endpoints = set(req['endpoint'] for req in slow_requests)
            recommendations.append({
                'category': 'Slow Endpoints',
                'severity': 'medium',
                'issue': f'Found {len(slow_requests)} requests slower than 2 seconds',
                'recommendation': f'Optimize endpoints: {", ".join(endpoints)}',
                'current_value': f"{len(slow_requests)} slow requests",
                'target_value': '0 requests >2s'
            })
        
        return recommendations
    
    @staticmethod
    def auto_optimize():
        """Perform automatic optimization based on current metrics."""
        recommendations = PerformanceOptimizer.get_optimization_recommendations()
        optimizations_applied = []
        
        for rec in recommendations:
            if rec['category'] == 'Memory Usage' and rec['severity'] == 'high':
                # Clean up old metrics
                performance_monitor.cleanup_old_metrics(hours=12)
                optimizations_applied.append('Cleaned up old performance metrics')
            
            # Add more automatic optimizations as needed
        
        return optimizations_applied


# Export the performance monitoring utilities
__all__ = [
    'PerformanceMonitor',
    'PerformanceMetrics',
    'SystemMetrics',
    'PerformanceOptimizer',
    'performance_monitor',
    'performance_track',
    'performance_context'
] 