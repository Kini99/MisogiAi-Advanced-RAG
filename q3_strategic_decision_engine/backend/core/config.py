"""
Configuration settings for Strategic Decision Engine
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional, Dict, Any
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""
    
    # Application settings
    APP_NAME: str = "Strategic Decision Engine"
    DEBUG: bool = Field(default=False, env="DEBUG")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # Security settings
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS settings
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        env="CORS_ORIGINS"
    )
    
    # Database settings
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = Field(default=20, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=0, env="DATABASE_MAX_OVERFLOW")
    
    # Redis settings
    REDIS_URL: str = Field(..., env="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    
    # Vector Database settings
    CHROMA_DB_PATH: str = Field(default="./chroma_db", env="CHROMA_DB_PATH")
    CHROMA_COLLECTION_NAME: str = Field(default="strategic_documents", env="CHROMA_COLLECTION_NAME")
    
    # LLM API Keys
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = Field(..., env="ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: str = Field(..., env="GOOGLE_API_KEY")
    
    # LLM Model configurations
    OPENAI_MODEL: str = Field(default="gpt-4o", env="OPENAI_MODEL")
    ANTHROPIC_MODEL: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    GOOGLE_MODEL: str = Field(default="gemini-2.5-pro", env="GOOGLE_MODEL")
    
    # LLM Temperature and settings
    LLM_TEMPERATURE: float = Field(default=0.1, env="LLM_TEMPERATURE")
    LLM_MAX_TOKENS: int = Field(default=4000, env="LLM_MAX_TOKENS")
    LLM_TIMEOUT: int = Field(default=60, env="LLM_TIMEOUT")
    
    # Embedding model settings
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-large", env="EMBEDDING_MODEL")
    EMBEDDING_DIMENSION: int = Field(default=3072, env="EMBEDDING_DIMENSION")
    
    # RAG settings
    CHUNK_SIZE: int = Field(default=1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=200, env="CHUNK_OVERLAP")
    MAX_RETRIEVAL_RESULTS: int = Field(default=10, env="MAX_RETRIEVAL_RESULTS")
    SIMILARITY_THRESHOLD: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # Reranking settings
    RERANKER_MODEL: str = Field(default="ms-marco-MiniLM-L-12-v2", env="RERANKER_MODEL")
    RERANKER_TOP_K: int = Field(default=5, env="RERANKER_TOP_K")
    
    # File upload settings
    MAX_FILE_SIZE: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    ALLOWED_EXTENSIONS: List[str] = Field(
        default=["pdf", "docx", "pptx", "xlsx", "txt", "csv"],
        env="ALLOWED_EXTENSIONS"
    )
    UPLOAD_DIR: str = Field(default="./uploads", env="UPLOAD_DIR")
    
    # Cache settings
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    CACHE_PREFIX: str = Field(default="sde:", env="CACHE_PREFIX")
    
    # External API settings
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    FRED_API_KEY: Optional[str] = Field(default=None, env="FRED_API_KEY")
    QUANDL_API_KEY: Optional[str] = Field(default=None, env="QUANDL_API_KEY")
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="./logs/app.log", env="LOG_FILE")
    
    # RAGAS Evaluation settings
    RAGAS_METRICS: List[str] = Field(
        default=["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"],
        env="RAGAS_METRICS"
    )
    
    # Strategic Analysis settings
    SWOT_ANALYSIS_PROMPT: str = Field(
        default="Conduct a comprehensive SWOT analysis for the company based on the provided documents.",
        env="SWOT_ANALYSIS_PROMPT"
    )
    
    MARKET_EXPANSION_PROMPT: str = Field(
        default="Analyze market expansion opportunities based on company capabilities and market data.",
        env="MARKET_EXPANSION_PROMPT"
    )
    
    # Chart generation settings
    CHART_THEME: str = Field(default="plotly", env="CHART_THEME")
    CHART_HEIGHT: int = Field(default=500, env="CHART_HEIGHT")
    CHART_WIDTH: int = Field(default=800, env="CHART_WIDTH")
    
    # Monitoring settings
    METRICS_ENABLED: bool = Field(default=True, env="METRICS_ENABLED")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("ALLOWED_EXTENSIONS", pre=True)
    def parse_allowed_extensions(cls, v):
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v
    
    @validator("RAGAS_METRICS", pre=True)
    def parse_ragas_metrics(cls, v):
        if isinstance(v, str):
            return [metric.strip() for metric in v.split(",")]
        return v
    
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.UPLOAD_DIR,
            self.CHROMA_DB_PATH,
            Path(self.LOG_FILE).parent
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create global settings instance
settings = Settings()

# Create necessary directories
settings.create_directories() 