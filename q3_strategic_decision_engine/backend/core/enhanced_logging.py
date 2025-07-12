"""
Enhanced logging system for Strategic Decision Engine.
Provides structured logging, performance tracking, and advanced features.
"""

import json
import logging
import logging.handlers
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Union
from functools import wraps
from pathlib import Path
import asyncio
import structlog
from structlog.processors import JSONRenderer
import sys
import os


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process
        }
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from the record
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage'):
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class PerformanceLogger:
    """Logger specifically for performance metrics and timing."""
    
    def __init__(self, logger_name: str = 'performance'):
        """Initialize the performance logger."""
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Create performance log file handler
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            log_dir / 'performance.jsonl',
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5
        )
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
    
    def log_request(self, 
                   endpoint: str,
                   method: str,
                   duration_ms: float,
                   status_code: int,
                   user_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   request_size: int = 0,
                   response_size: int = 0,
                   error: Optional[str] = None):
        """Log request performance metrics."""
        self.logger.info(
            "Request processed",
            extra={
                'event_type': 'request',
                'endpoint': endpoint,
                'method': method,
                'duration_ms': duration_ms,
                'status_code': status_code,
                'user_id': user_id,
                'session_id': session_id,
                'request_size_bytes': request_size,
                'response_size_bytes': response_size,
                'error': error
            }
        )
    
    def log_operation(self,
                     operation: str,
                     duration_ms: float,
                     success: bool = True,
                     metadata: Optional[Dict[str, Any]] = None):
        """Log operation performance metrics."""
        log_data = {
            'event_type': 'operation',
            'operation': operation,
            'duration_ms': duration_ms,
            'success': success
        }
        
        if metadata:
            log_data.update(metadata)
        
        self.logger.info(
            f"Operation {operation} {'completed' if success else 'failed'}",
            extra=log_data
        )


class SecurityLogger:
    """Logger for security-related events."""
    
    def __init__(self, logger_name: str = 'security'):
        """Initialize the security logger."""
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Create security log file handler
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            log_dir / 'security.jsonl',
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10  # Keep more security logs
        )
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
    
    def log_authentication(self,
                          user_id: Optional[str],
                          success: bool,
                          ip_address: str,
                          user_agent: str,
                          reason: Optional[str] = None):
        """Log authentication attempts."""
        self.logger.info(
            f"Authentication {'successful' if success else 'failed'}",
            extra={
                'event_type': 'authentication',
                'user_id': user_id,
                'success': success,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'reason': reason
            }
        )
    
    def log_authorization(self,
                         user_id: str,
                         resource: str,
                         action: str,
                         success: bool,
                         reason: Optional[str] = None):
        """Log authorization attempts."""
        self.logger.info(
            f"Authorization {'granted' if success else 'denied'}",
            extra={
                'event_type': 'authorization',
                'user_id': user_id,
                'resource': resource,
                'action': action,
                'success': success,
                'reason': reason
            }
        )
    
    def log_data_access(self,
                       user_id: str,
                       resource_type: str,
                       resource_id: str,
                       action: str,
                       ip_address: str):
        """Log data access events."""
        self.logger.info(
            f"Data access: {action} on {resource_type}",
            extra={
                'event_type': 'data_access',
                'user_id': user_id,
                'resource_type': resource_type,
                'resource_id': resource_id,
                'action': action,
                'ip_address': ip_address
            }
        )


class BusinessLogger:
    """Logger for business logic and workflow events."""
    
    def __init__(self, logger_name: str = 'business'):
        """Initialize the business logger."""
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Create business log file handler
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            log_dir / 'business.jsonl',
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5
        )
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
    
    def log_document_processing(self,
                               document_id: str,
                               filename: str,
                               user_id: str,
                               processing_time_ms: float,
                               success: bool,
                               error: Optional[str] = None):
        """Log document processing events."""
        self.logger.info(
            f"Document processing {'completed' if success else 'failed'}",
            extra={
                'event_type': 'document_processing',
                'document_id': document_id,
                'filename': filename,
                'user_id': user_id,
                'processing_time_ms': processing_time_ms,
                'success': success,
                'error': error
            }
        )
    
    def log_analysis_generation(self,
                               analysis_id: str,
                               analysis_type: str,
                               user_id: str,
                               generation_time_ms: float,
                               model_used: str,
                               success: bool,
                               error: Optional[str] = None):
        """Log analysis generation events."""
        self.logger.info(
            f"Analysis generation {'completed' if success else 'failed'}",
            extra={
                'event_type': 'analysis_generation',
                'analysis_id': analysis_id,
                'analysis_type': analysis_type,
                'user_id': user_id,
                'generation_time_ms': generation_time_ms,
                'model_used': model_used,
                'success': success,
                'error': error
            }
        )
    
    def log_chat_interaction(self,
                            session_id: str,
                            user_id: str,
                            message_count: int,
                            response_time_ms: float,
                            model_used: str,
                            tokens_used: int):
        """Log chat interaction events."""
        self.logger.info(
            "Chat interaction completed",
            extra={
                'event_type': 'chat_interaction',
                'session_id': session_id,
                'user_id': user_id,
                'message_count': message_count,
                'response_time_ms': response_time_ms,
                'model_used': model_used,
                'tokens_used': tokens_used
            }
        )


class EnhancedLogger:
    """Main enhanced logging class that combines all loggers."""
    
    def __init__(self):
        """Initialize the enhanced logging system."""
        self.performance = PerformanceLogger()
        self.security = SecurityLogger()
        self.business = BusinessLogger()
        
        # Configure main application logger
        self.app_logger = self._setup_app_logger()
    
    def _setup_app_logger(self) -> logging.Logger:
        """Setup the main application logger."""
        logger = logging.getLogger('strategic_engine')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler for all logs
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'application.log',
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
        
        # Structured JSON handler
        json_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'application.jsonl',
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5
        )
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(StructuredFormatter())
        logger.addHandler(json_handler)
        
        # Error handler for critical issues
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'errors.log',
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        logger.addHandler(error_handler)
        
        return logger
    
    def info(self, message: str, **kwargs):
        """Log info message with extra data."""
        self.app_logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with extra data."""
        self.app_logger.warning(message, extra=kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with exception details."""
        if exception:
            self.app_logger.error(message, exc_info=exception, extra=kwargs)
        else:
            self.app_logger.error(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with extra data."""
        self.app_logger.debug(message, extra=kwargs)


def log_function_call(include_args: bool = False, include_result: bool = False):
    """Decorator to log function calls."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            log_data = {
                'function': func.__name__,
                'event_type': 'function_call'
            }
            
            if include_args:
                log_data['args'] = str(args)
                log_data['kwargs'] = str(kwargs)
            
            logger.info(f"Function {func.__name__} called", extra=log_data)
            
            try:
                result = await func(*args, **kwargs)
                
                if include_result:
                    log_data['result'] = str(result)[:500]  # Limit result length
                
                logger.info(f"Function {func.__name__} completed", extra=log_data)
                return result
                
            except Exception as e:
                log_data['error'] = str(e)
                logger.error(f"Function {func.__name__} failed", extra=log_data)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            log_data = {
                'function': func.__name__,
                'event_type': 'function_call'
            }
            
            if include_args:
                log_data['args'] = str(args)
                log_data['kwargs'] = str(kwargs)
            
            logger.info(f"Function {func.__name__} called", extra=log_data)
            
            try:
                result = func(*args, **kwargs)
                
                if include_result:
                    log_data['result'] = str(result)[:500]  # Limit result length
                
                logger.info(f"Function {func.__name__} completed", extra=log_data)
                return result
                
            except Exception as e:
                log_data['error'] = str(e)
                logger.error(f"Function {func.__name__} failed", extra=log_data)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class LogAnalyzer:
    """Analyze logs for patterns and issues."""
    
    @staticmethod
    def analyze_error_patterns(log_file: Path, hours: int = 24) -> Dict[str, Any]:
        """Analyze error patterns in logs."""
        error_patterns = {}
        error_count = 0
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        
                        # Check if this is an error entry
                        if log_entry.get('level') == 'ERROR':
                            error_count += 1
                            
                            # Extract error pattern
                            error_msg = log_entry.get('message', '')
                            function = log_entry.get('function', 'unknown')
                            module = log_entry.get('module', 'unknown')
                            
                            pattern_key = f"{module}.{function}"
                            
                            if pattern_key not in error_patterns:
                                error_patterns[pattern_key] = {
                                    'count': 0,
                                    'messages': [],
                                    'first_seen': log_entry.get('timestamp'),
                                    'last_seen': log_entry.get('timestamp')
                                }
                            
                            error_patterns[pattern_key]['count'] += 1
                            error_patterns[pattern_key]['last_seen'] = log_entry.get('timestamp')
                            
                            if len(error_patterns[pattern_key]['messages']) < 5:
                                error_patterns[pattern_key]['messages'].append(error_msg)
                    
                    except json.JSONDecodeError:
                        continue
        
        except FileNotFoundError:
            return {'error': f'Log file not found: {log_file}'}
        
        return {
            'total_errors': error_count,
            'error_patterns': error_patterns,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def get_performance_insights(log_file: Path, hours: int = 24) -> Dict[str, Any]:
        """Get performance insights from logs."""
        performance_data = []
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        
                        if log_entry.get('event_type') == 'request':
                            performance_data.append({
                                'endpoint': log_entry.get('endpoint'),
                                'duration_ms': log_entry.get('duration_ms'),
                                'status_code': log_entry.get('status_code'),
                                'timestamp': log_entry.get('timestamp')
                            })
                    
                    except json.JSONDecodeError:
                        continue
        
        except FileNotFoundError:
            return {'error': f'Log file not found: {log_file}'}
        
        if not performance_data:
            return {'message': 'No performance data found'}
        
        # Analyze performance
        total_requests = len(performance_data)
        avg_duration = sum(req['duration_ms'] for req in performance_data) / total_requests
        slow_requests = [req for req in performance_data if req['duration_ms'] > 1000]
        error_requests = [req for req in performance_data if req['status_code'] >= 400]
        
        return {
            'total_requests': total_requests,
            'avg_response_time_ms': avg_duration,
            'slow_requests_count': len(slow_requests),
            'error_requests_count': len(error_requests),
            'error_rate_percent': len(error_requests) / total_requests * 100,
            'analysis_timestamp': datetime.now().isoformat()
        }


# Global enhanced logger instance
enhanced_logger = EnhancedLogger()

# Export the logging utilities
__all__ = [
    'EnhancedLogger',
    'PerformanceLogger',
    'SecurityLogger',
    'BusinessLogger',
    'LogAnalyzer',
    'enhanced_logger',
    'log_function_call'
] 