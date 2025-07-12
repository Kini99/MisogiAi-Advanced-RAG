"""
Logging configuration for Strategic Decision Engine
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from datetime import datetime
import structlog
from typing import Any, Dict

from .config import settings


def setup_logging():
    """Setup application logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=settings.LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_dir / 'error.log'),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_file_handler)
    
    # Third-party loggers
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('uvicorn.access').setLevel(logging.INFO)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('anthropic').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    
    # Application logger
    app_logger = logging.getLogger('strategic_decision_engine')
    app_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    logging.info("Logging configuration initialized")


class AppLogger:
    """Application logger with structured logging"""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
    
    def info(self, msg: str, **kwargs):
        """Log info message"""
        self.logger.info(msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log error message"""
        self.logger.error(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message"""
        self.logger.warning(msg, **kwargs)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message"""
        self.logger.debug(msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        """Log critical message"""
        self.logger.critical(msg, **kwargs)


class RequestLogger:
    """Request logging middleware"""
    
    def __init__(self):
        self.logger = AppLogger('request')
    
    def log_request(self, method: str, url: str, status_code: int, 
                   response_time: float, user_id: str = None, **kwargs):
        """Log HTTP request"""
        self.logger.info(
            "HTTP Request",
            method=method,
            url=url,
            status_code=status_code,
            response_time=response_time,
            user_id=user_id,
            **kwargs
        )
    
    def log_error(self, method: str, url: str, error: str, **kwargs):
        """Log HTTP error"""
        self.logger.error(
            "HTTP Error",
            method=method,
            url=url,
            error=error,
            **kwargs
        )


class LLMLogger:
    """LLM operation logging"""
    
    def __init__(self):
        self.logger = AppLogger('llm')
    
    def log_completion(self, model: str, prompt_tokens: int, completion_tokens: int,
                      response_time: float, cost: float = None, **kwargs):
        """Log LLM completion"""
        self.logger.info(
            "LLM Completion",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            response_time=response_time,
            cost=cost,
            **kwargs
        )
    
    def log_error(self, model: str, error: str, **kwargs):
        """Log LLM error"""
        self.logger.error(
            "LLM Error",
            model=model,
            error=error,
            **kwargs
        )


class RAGLogger:
    """RAG operation logging"""
    
    def __init__(self):
        self.logger = AppLogger('rag')
    
    def log_retrieval(self, query: str, num_results: int, retrieval_time: float,
                     reranking_time: float = None, **kwargs):
        """Log retrieval operation"""
        self.logger.info(
            "RAG Retrieval",
            query=query,
            num_results=num_results,
            retrieval_time=retrieval_time,
            reranking_time=reranking_time,
            **kwargs
        )
    
    def log_generation(self, query: str, response_length: int, sources_count: int,
                      generation_time: float, **kwargs):
        """Log generation operation"""
        self.logger.info(
            "RAG Generation",
            query=query,
            response_length=response_length,
            sources_count=sources_count,
            generation_time=generation_time,
            **kwargs
        )


class EvaluationLogger:
    """Evaluation logging"""
    
    def __init__(self):
        self.logger = AppLogger('evaluation')
    
    def log_evaluation(self, query: str, metrics: Dict[str, float], 
                      evaluation_time: float, **kwargs):
        """Log evaluation results"""
        self.logger.info(
            "Evaluation Result",
            query=query,
            metrics=metrics,
            evaluation_time=evaluation_time,
            **kwargs
        )


# Global logger instances
request_logger = RequestLogger()
llm_logger = LLMLogger()
rag_logger = RAGLogger()
evaluation_logger = EvaluationLogger()


def get_logger(name: str) -> AppLogger:
    """Get application logger instance"""
    return AppLogger(name) 