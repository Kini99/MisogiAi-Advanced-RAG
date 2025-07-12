import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the Sports Analytics RAG system."""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    
    # Vector Database Configuration
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sports_analytics")
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # RAG Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "10"))
    TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))
    
    # Query Decomposition
    MAX_SUB_QUESTIONS = int(os.getenv("MAX_SUB_QUESTIONS", "5"))
    
    # Context Compression
    COMPRESSION_RATIO = float(os.getenv("COMPRESSION_RATIO", "0.7"))
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required") 