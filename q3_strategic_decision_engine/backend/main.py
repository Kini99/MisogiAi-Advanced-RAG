"""
Strategic Decision Engine - Main FastAPI Application
A comprehensive AI-powered strategic planning platform for CEOs
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import os

# Local imports
from .core.config import settings
from .api.endpoints import documents, chat, analysis, evaluation
from .core.database import init_db, close_db
from .core.logging_config import setup_logging
from .core.monitoring_middleware import setup_monitoring_middleware, health_check as health_endpoint, metrics_endpoint
from .core.enhanced_logging import enhanced_logger
from .core.performance import performance_monitor
from .services.llm_service import LLMService
from .services.vector_store_service import VectorStoreService
from .services.cache_service import CacheService

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global services
llm_service = LLMService()
vector_store_service = VectorStoreService()
cache_service = CacheService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    enhanced_logger.info("Starting Strategic Decision Engine...")
    
    # Initialize database
    await init_db()
    
    # Initialize services
    await llm_service.initialize()
    await vector_store_service.initialize()
    await cache_service.initialize()
    
    enhanced_logger.info("Strategic Decision Engine started successfully",
                        services_initialized=True,
                        monitoring_enabled=True,
                        performance_tracking=True)
    
    yield
    
    # Cleanup
    enhanced_logger.info("Shutting down Strategic Decision Engine...")
    
    # Stop performance monitoring
    performance_monitor.stop_monitoring()
    
    # Close services
    await close_db()
    await cache_service.close()
    
    enhanced_logger.info("Strategic Decision Engine shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Strategic Decision Engine",
    description="AI-powered strategic planning platform for CEOs",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup monitoring middleware
setup_monitoring_middleware(
    app,
    enable_performance=True,
    enable_security=True,
    enable_rate_limiting=True,
    rate_limit_per_minute=100
)

# Include API routers
app.include_router(
    documents.router,
    prefix="/api/documents",
    tags=["documents"]
)

app.include_router(
    chat.router,
    prefix="/api/chat",
    tags=["chat"]
)

app.include_router(
    analysis.router,
    prefix="/api/analysis",
    tags=["analysis"]
)

app.include_router(
    evaluation.router,
    prefix="/api/evaluation",
    tags=["evaluation"]
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Strategic Decision Engine API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with system metrics"""
    return await health_endpoint()

@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring systems"""
    return await metrics_endpoint()

@app.get("/performance")
async def performance_summary():
    """Performance summary endpoint"""
    try:
        summary = performance_monitor.get_metrics_summary(minutes=60)
        return {
            "status": "success",
            "data": summary
        }
    except Exception as e:
        enhanced_logger.error("Performance summary failed", exception=e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e)
            }
        )

@app.get("/performance/slow-requests")
async def slow_requests():
    """Get slow requests for performance analysis"""
    try:
        slow_reqs = performance_monitor.get_slow_requests(threshold_ms=1000, limit=20)
        return {
            "status": "success",
            "data": slow_reqs
        }
    except Exception as e:
        enhanced_logger.error("Slow requests analysis failed", exception=e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "error": str(e)
            }
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    enhanced_logger.error(
        f"Unhandled exception: {str(exc)}",
        exception=exc,
        endpoint=str(request.url.path),
        method=request.method,
        client_ip=request.client.host if request.client else None
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An error occurred"
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    ) 