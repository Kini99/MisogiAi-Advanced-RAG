"""Configuration settings for the Advanced Assessment Generation System."""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # OpenAI Configuration
    openai_api_key: str = ""
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Vector Database Configuration
    chroma_persist_directory: str = "./chroma_db"
    
    # Embedding Models
    dense_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Chunking Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Assessment Configuration
    max_questions_per_assessment: int = 10
    difficulty_levels: list = ["easy", "medium", "hard"]
    question_types: list = ["multiple_choice", "true_false", "short_answer", "essay"]
    
    # Caching Configuration
    cache_ttl: int = 3600  # 1 hour
    assessment_cache_ttl: int = 7200  # 2 hours
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # File Upload Configuration
    upload_dir: str = "./uploads"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: list = [".pdf", ".docx", ".txt", ".md"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.chroma_persist_directory, exist_ok=True) 