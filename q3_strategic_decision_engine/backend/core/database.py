"""
Database configuration and connection management
"""

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, DateTime, Boolean, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import asynccontextmanager
from typing import Generator, Optional
import logging
from datetime import datetime, timezone
import asyncio
import aioredis
from redis.asyncio import Redis

from .config import settings

logger = logging.getLogger(__name__)

# Database engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    echo=settings.DEBUG,
    pool_pre_ping=True,
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base
Base = declarative_base()

# Redis connection
redis_client: Optional[Redis] = None


class Document(Base):
    """Document model for storing uploaded documents"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    original_filename = Column(String)
    file_path = Column(String)
    file_size = Column(Integer)
    file_type = Column(String)
    mime_type = Column(String)
    upload_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    processed = Column(Boolean, default=False)
    processed_date = Column(DateTime, nullable=True)
    chunk_count = Column(Integer, default=0)
    metadata = Column(JSON, nullable=True)
    summary = Column(Text, nullable=True)
    key_topics = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)


class ChatSession(Base):
    """Chat session model for tracking user conversations"""
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    user_id = Column(String, index=True, nullable=True)
    title = Column(String, nullable=True)
    created_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_activity = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)
    metadata = Column(JSON, nullable=True)


class ChatMessage(Base):
    """Chat message model for storing conversation history"""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    message_type = Column(String)  # 'user' or 'assistant'
    content = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    tokens_used = Column(Integer, nullable=True)
    model_used = Column(String, nullable=True)
    response_time = Column(Float, nullable=True)
    metadata = Column(JSON, nullable=True)
    sources = Column(JSON, nullable=True)  # Document sources for citations


class AnalysisResult(Base):
    """Analysis result model for storing strategic analysis outputs"""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    analysis_type = Column(String)  # 'swot', 'market_expansion', etc.
    query = Column(Text)
    result = Column(Text)
    charts_data = Column(JSON, nullable=True)
    sources = Column(JSON, nullable=True)
    created_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    model_used = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)
    metadata = Column(JSON, nullable=True)


class EvaluationResult(Base):
    """Evaluation result model for storing RAGAS evaluation metrics"""
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    query = Column(Text)
    response = Column(Text)
    context = Column(JSON)  # Retrieved documents
    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    context_precision = Column(Float, nullable=True)
    context_recall = Column(Float, nullable=True)
    answer_correctness = Column(Float, nullable=True)
    overall_score = Column(Float, nullable=True)
    created_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    metadata = Column(JSON, nullable=True)


async def init_db():
    """Initialize database tables"""
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Initialize Redis connection
        global redis_client
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        
        # Test Redis connection
        await redis_client.ping()
        logger.info("Redis connection established successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise


async def close_db():
    """Close database connections"""
    try:
        global redis_client
        if redis_client:
            await redis_client.close()
            logger.info("Redis connection closed")
        
        # Close SQLAlchemy engine
        engine.dispose()
        logger.info("Database connections closed")
        
    except Exception as e:
        logger.error(f"Database cleanup failed: {str(e)}")


def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def get_async_db():
    """Get async database session context manager"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_redis() -> Redis:
    """Get Redis client"""
    global redis_client
    if redis_client is None:
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB,
            decode_responses=True,
        )
    return redis_client


class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self):
        self.db = SessionLocal()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
    
    def create_document(self, **kwargs) -> Document:
        """Create a new document record"""
        document = Document(**kwargs)
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document
    
    def get_document(self, document_id: int) -> Optional[Document]:
        """Get document by ID"""
        return self.db.query(Document).filter(Document.id == document_id).first()
    
    def get_documents(self, skip: int = 0, limit: int = 100) -> list[Document]:
        """Get list of documents"""
        return self.db.query(Document).filter(Document.is_active == True).offset(skip).limit(limit).all()
    
    def update_document_processed(self, document_id: int, chunk_count: int, summary: str = None, key_topics: list = None):
        """Update document processing status"""
        document = self.get_document(document_id)
        if document:
            document.processed = True
            document.processed_date = datetime.now(timezone.utc)
            document.chunk_count = chunk_count
            if summary:
                document.summary = summary
            if key_topics:
                document.key_topics = key_topics
            self.db.commit()
    
    def create_chat_session(self, session_id: str, user_id: str = None, title: str = None) -> ChatSession:
        """Create a new chat session"""
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            title=title
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session
    
    def add_chat_message(self, session_id: str, message_type: str, content: str, 
                        tokens_used: int = None, model_used: str = None, 
                        response_time: float = None, metadata: dict = None, 
                        sources: list = None) -> ChatMessage:
        """Add a message to chat session"""
        message = ChatMessage(
            session_id=session_id,
            message_type=message_type,
            content=content,
            tokens_used=tokens_used,
            model_used=model_used,
            response_time=response_time,
            metadata=metadata,
            sources=sources
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message
    
    def get_chat_history(self, session_id: str, limit: int = 50) -> list[ChatMessage]:
        """Get chat history for a session"""
        return (self.db.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.timestamp.desc())
                .limit(limit)
                .all())
    
    def save_analysis_result(self, session_id: str, analysis_type: str, query: str, 
                           result: str, charts_data: dict = None, sources: list = None,
                           model_used: str = None, confidence_score: float = None) -> AnalysisResult:
        """Save analysis result"""
        analysis = AnalysisResult(
            session_id=session_id,
            analysis_type=analysis_type,
            query=query,
            result=result,
            charts_data=charts_data,
            sources=sources,
            model_used=model_used,
            confidence_score=confidence_score
        )
        self.db.add(analysis)
        self.db.commit()
        self.db.refresh(analysis)
        return analysis
    
    def save_evaluation_result(self, session_id: str, query: str, response: str, 
                             context: list, metrics: dict) -> EvaluationResult:
        """Save RAGAS evaluation result"""
        evaluation = EvaluationResult(
            session_id=session_id,
            query=query,
            response=response,
            context=context,
            faithfulness=metrics.get('faithfulness'),
            answer_relevancy=metrics.get('answer_relevancy'),
            context_precision=metrics.get('context_precision'),
            context_recall=metrics.get('context_recall'),
            answer_correctness=metrics.get('answer_correctness'),
            overall_score=metrics.get('overall_score')
        )
        self.db.add(evaluation)
        self.db.commit()
        self.db.refresh(evaluation)
        return evaluation 