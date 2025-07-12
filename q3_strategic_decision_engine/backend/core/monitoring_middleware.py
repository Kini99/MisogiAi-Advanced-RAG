"""
Monitoring middleware for Strategic Decision Engine.
Integrates performance monitoring and enhanced logging.
"""

import time
import uuid
import asyncio
from typing import Callable, Dict, Any
from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
import psutil

from .performance import performance_monitor, PerformanceMetrics
from .enhanced_logging import enhanced_logger


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware to monitor and log request performance."""
    
    def __init__(self, app: FastAPI):
        """Initialize the monitoring middleware."""
        super().__init__(app)
        self.process = psutil.Process()
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request and monitor performance."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Extract request information
        method = request.method
        endpoint = str(request.url.path)
        user_id = getattr(request.state, 'user_id', None)
        session_id = request.headers.get('Session-ID')
        
        # Get request size
        request_size = 0
        if hasattr(request, 'body'):
            try:
                body = await request.body()
                request_size = len(body) if body else 0
            except Exception:
                request_size = 0
        
        # Start performance tracking
        start_time = time.time()
        start_memory = self.process.memory_info().rss / (1024 * 1024)
        cpu_percent = psutil.cpu_percent()
        
        # Start request tracking
        performance_monitor.start_request(request_id, endpoint, method)
        
        # Add request context to logs
        enhanced_logger.info(
            f"Request started: {method} {endpoint}",
            request_id=request_id,
            method=method,
            endpoint=endpoint,
            user_id=user_id,
            session_id=session_id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get('User-Agent', '')
        )
        
        # Process request
        error_message = None
        status_code = 200
        response_size = 0
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Get response size if possible
            if hasattr(response, 'body'):
                response_size = len(response.body) if response.body else 0
            
            return response
            
        except Exception as e:
            error_message = str(e)
            status_code = 500
            enhanced_logger.error(
                f"Request failed: {method} {endpoint}",
                request_id=request_id,
                error=error_message,
                exception=e
            )
            raise
            
        finally:
            # Calculate metrics
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            end_memory = self.process.memory_info().rss / (1024 * 1024)
            memory_usage = end_memory - start_memory
            
            # End request tracking
            performance_monitor.end_request(request_id)
            
            # Record performance metrics
            metric = PerformanceMetrics(
                timestamp=time.time(),
                endpoint=endpoint,
                method=method,
                duration_ms=duration_ms,
                status_code=status_code,
                memory_usage_mb=memory_usage,
                cpu_percent=cpu_percent,
                request_size_bytes=request_size,
                response_size_bytes=response_size,
                user_id=user_id,
                session_id=session_id,
                error_message=error_message
            )
            
            performance_monitor.record_request(metric)
            
            # Log performance metrics
            enhanced_logger.performance.log_request(
                endpoint=endpoint,
                method=method,
                duration_ms=duration_ms,
                status_code=status_code,
                user_id=user_id,
                session_id=session_id,
                request_size=request_size,
                response_size=response_size,
                error=error_message
            )
            
            # Log completion
            enhanced_logger.info(
                f"Request completed: {method} {endpoint}",
                request_id=request_id,
                duration_ms=duration_ms,
                status_code=status_code,
                memory_usage_mb=memory_usage,
                response_size_bytes=response_size
            )


class SecurityMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware to monitor security-related events."""
    
    def __init__(self, app: FastAPI):
        """Initialize the security monitoring middleware."""
        super().__init__(app)
        self.suspicious_patterns = [
            '/admin',
            '/.env',
            '/config',
            '/api/v1/debug',
            '/wp-admin',
            'DROP TABLE',
            'SELECT * FROM',
            '<script>',
            'javascript:',
            '../',
            '..\\',
        ]
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request and monitor for security issues."""
        # Extract security-relevant information
        ip_address = request.client.host if request.client else 'unknown'
        user_agent = request.headers.get('User-Agent', '')
        endpoint = str(request.url.path)
        method = request.method
        
        # Check for suspicious patterns
        suspicious_activity = self._check_suspicious_patterns(request)
        
        if suspicious_activity:
            enhanced_logger.security.log_authorization(
                user_id='anonymous',
                resource=endpoint,
                action=method,
                success=False,
                reason=f"Suspicious pattern detected: {suspicious_activity}"
            )
            
            enhanced_logger.warning(
                f"Suspicious request detected: {method} {endpoint}",
                ip_address=ip_address,
                user_agent=user_agent,
                suspicious_pattern=suspicious_activity
            )
        
        # Process request normally
        response = await call_next(request)
        
        # Log authentication events for auth endpoints
        if '/auth' in endpoint or '/login' in endpoint:
            success = response.status_code < 400
            enhanced_logger.security.log_authentication(
                user_id=getattr(request.state, 'user_id', None),
                success=success,
                ip_address=ip_address,
                user_agent=user_agent,
                reason=None if success else f"HTTP {response.status_code}"
            )
        
        return response
    
    def _check_suspicious_patterns(self, request: Request) -> str:
        """Check for suspicious patterns in the request."""
        # Check URL path
        path = str(request.url.path).lower()
        for pattern in self.suspicious_patterns:
            if pattern.lower() in path:
                return f"Suspicious path: {pattern}"
        
        # Check query parameters
        query_string = str(request.url.query).lower()
        for pattern in self.suspicious_patterns:
            if pattern.lower() in query_string:
                return f"Suspicious query: {pattern}"
        
        # Check headers for common attack patterns
        for header_name, header_value in request.headers.items():
            if any(pattern.lower() in header_value.lower() for pattern in self.suspicious_patterns):
                return f"Suspicious header: {header_name}"
        
        return None


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app: FastAPI, requests_per_minute: int = 60):
        """Initialize rate limiting middleware."""
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_history: Dict[str, list] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request with rate limiting."""
        # Get client identifier
        client_ip = request.client.host if request.client else 'unknown'
        
        # Clean up old entries periodically
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_requests()
            self.last_cleanup = current_time
        
        # Check rate limit
        if self._is_rate_limited(client_ip, current_time):
            enhanced_logger.warning(
                f"Rate limit exceeded for IP: {client_ip}",
                ip_address=client_ip,
                endpoint=str(request.url.path),
                requests_per_minute=self.requests_per_minute
            )
            
            from fastapi import HTTPException
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Record request
        if client_ip not in self.request_history:
            self.request_history[client_ip] = []
        self.request_history[client_ip].append(current_time)
        
        # Process request
        return await call_next(request)
    
    def _is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        """Check if client is rate limited."""
        if client_ip not in self.request_history:
            return False
        
        # Remove requests older than 1 minute
        minute_ago = current_time - 60
        self.request_history[client_ip] = [
            req_time for req_time in self.request_history[client_ip]
            if req_time > minute_ago
        ]
        
        # Check if limit exceeded
        return len(self.request_history[client_ip]) >= self.requests_per_minute
    
    def _cleanup_old_requests(self):
        """Clean up old request history."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        for client_ip in list(self.request_history.keys()):
            self.request_history[client_ip] = [
                req_time for req_time in self.request_history[client_ip]
                if req_time > minute_ago
            ]
            
            # Remove empty entries
            if not self.request_history[client_ip]:
                del self.request_history[client_ip]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to add request context for logging and monitoring."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Add request context."""
        # Generate request ID if not present
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        
        # Add to request state
        request.state.request_id = request_id
        request.state.start_time = time.time()
        
        # Extract user information from headers or auth
        user_id = request.headers.get('X-User-ID')
        if user_id:
            request.state.user_id = user_id
        
        # Process request
        response = await call_next(request)
        
        # Add response headers
        response.headers['X-Request-ID'] = request_id
        response.headers['X-Response-Time'] = str(
            int((time.time() - request.state.start_time) * 1000)
        )
        
        return response


def setup_monitoring_middleware(app: FastAPI, 
                               enable_performance: bool = True,
                               enable_security: bool = True,
                               enable_rate_limiting: bool = True,
                               rate_limit_per_minute: int = 60):
    """Setup all monitoring middleware."""
    
    # Add request context middleware first
    app.add_middleware(RequestContextMiddleware)
    
    # Add rate limiting if enabled
    if enable_rate_limiting:
        app.add_middleware(RateLimitingMiddleware, requests_per_minute=rate_limit_per_minute)
    
    # Add security monitoring if enabled
    if enable_security:
        app.add_middleware(SecurityMonitoringMiddleware)
    
    # Add performance monitoring if enabled (should be last)
    if enable_performance:
        app.add_middleware(PerformanceMonitoringMiddleware)
    
    enhanced_logger.info(
        "Monitoring middleware configured",
        performance_monitoring=enable_performance,
        security_monitoring=enable_security,
        rate_limiting=enable_rate_limiting,
        rate_limit_per_minute=rate_limit_per_minute if enable_rate_limiting else None
    )


# Health check endpoint for monitoring
async def health_check() -> Dict[str, Any]:
    """Health check endpoint with system status."""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get performance summary
        perf_summary = performance_monitor.get_metrics_summary(minutes=5)
        
        status = "healthy"
        issues = []
        
        # Check for issues
        if cpu_percent > 90:
            status = "degraded"
            issues.append("High CPU usage")
        
        if memory.percent > 90:
            status = "degraded"
            issues.append("High memory usage")
        
        if disk.percent > 90:
            status = "degraded"
            issues.append("High disk usage")
        
        if perf_summary.get('error_rate_percent', 0) > 10:
            status = "degraded"
            issues.append("High error rate")
        
        return {
            "status": status,
            "timestamp": time.time(),
            "issues": issues,
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "uptime_seconds": time.time() - performance_monitor.start_time.timestamp()
            },
            "performance": {
                "avg_response_time_ms": perf_summary.get('avg_response_time_ms', 0),
                "error_rate_percent": perf_summary.get('error_rate_percent', 0),
                "requests_per_minute": perf_summary.get('requests_per_minute', 0)
            }
        }
        
    except Exception as e:
        enhanced_logger.error("Health check failed", exception=e)
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }


# Metrics endpoint for monitoring systems
async def metrics_endpoint() -> Dict[str, Any]:
    """Metrics endpoint for external monitoring systems."""
    try:
        # Get comprehensive metrics
        perf_summary = performance_monitor.get_metrics_summary(minutes=60)
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
        }
        
        return {
            "timestamp": time.time(),
            "performance": perf_summary,
            "system": system_metrics,
            "version": "1.0.0"  # Could be loaded from config
        }
        
    except Exception as e:
        enhanced_logger.error("Metrics endpoint failed", exception=e)
        return {
            "timestamp": time.time(),
            "error": str(e)
        }


# Export middleware setup function and endpoint handlers
__all__ = [
    'setup_monitoring_middleware',
    'PerformanceMonitoringMiddleware',
    'SecurityMonitoringMiddleware',
    'RateLimitingMiddleware',
    'RequestContextMiddleware',
    'health_check',
    'metrics_endpoint'
] 