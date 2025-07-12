"""
Test configuration and fixtures for Strategic Decision Engine.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.main import app
from backend.core.config import Settings
from backend.core.database import Base, get_db
from backend.services.llm_service import LLMService
from backend.services.vector_store_service import VectorStoreService
from backend.services.cache_service import CacheService
from backend.services.document_service import DocumentService


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Create test settings."""
    return Settings(
        database_url="sqlite:///./test.db",
        redis_url="redis://localhost:6379/1",
        openai_api_key="test-key",
        anthropic_api_key="test-key",
        google_api_key="test-key",
        alpha_vantage_api_key="test-key",
        fred_api_key="test-key",
        quandl_api_key="test-key",
        environment="test"
    )


@pytest.fixture
def test_db_engine():
    """Create a test database engine."""
    engine = create_engine("sqlite:///./test.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_db_session(test_db_engine):
    """Create a test database session."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def test_client(test_db_session):
    """Create a test client."""
    def override_get_db():
        yield test_db_session
    
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    service = Mock(spec=LLMService)
    service.generate_response = AsyncMock()
    service.generate_stream = AsyncMock()
    service.get_available_models = Mock(return_value=["gpt-4o", "claude-3.5-sonnet", "gemini-2.5-pro"])
    service.select_best_model = Mock(return_value="gpt-4o")
    return service


@pytest.fixture
def mock_vector_store_service():
    """Create a mock vector store service."""
    service = Mock(spec=VectorStoreService)
    service.add_documents = AsyncMock()
    service.search_documents = AsyncMock()
    service.get_relevant_documents = AsyncMock()
    service.delete_document = AsyncMock()
    service.get_collection_stats = AsyncMock()
    return service


@pytest.fixture
def mock_cache_service():
    """Create a mock cache service."""
    service = Mock(spec=CacheService)
    service.get = AsyncMock()
    service.set = AsyncMock()
    service.delete = AsyncMock()
    service.clear = AsyncMock()
    service.get_stats = AsyncMock()
    return service


@pytest.fixture
def mock_document_service():
    """Create a mock document service."""
    service = Mock(spec=DocumentService)
    service.process_document = AsyncMock()
    service.extract_text = AsyncMock()
    service.summarize_document = AsyncMock()
    service.get_document_metadata = AsyncMock()
    return service


@pytest.fixture
def sample_document_content():
    """Sample document content for testing."""
    return """
    Strategic Planning Document
    
    Executive Summary:
    This document outlines the strategic direction for our technology company.
    
    Market Analysis:
    The technology sector shows strong growth potential with emerging AI trends.
    
    Financial Projections:
    Revenue is expected to grow by 25% year-over-year.
    
    SWOT Analysis:
    Strengths: Strong technical team, innovative products
    Weaknesses: Limited market presence
    Opportunities: AI market expansion, new partnerships
    Threats: Increased competition, economic uncertainty
    
    Strategic Recommendations:
    1. Invest in AI research and development
    2. Expand market reach through strategic partnerships
    3. Strengthen competitive positioning
    """


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "user", "content": "What are the key strategic opportunities for our company?"},
        {"role": "assistant", "content": "Based on the analysis, the key strategic opportunities include AI market expansion and new partnerships."},
        {"role": "user", "content": "Can you elaborate on the AI market expansion opportunity?"}
    ]


@pytest.fixture
def sample_analysis_request():
    """Sample analysis request for testing."""
    return {
        "content": "Analyze the competitive landscape for our technology company",
        "analysis_type": "market",
        "context": "We are a mid-size technology company focused on AI solutions",
        "requirements": ["competitive analysis", "market trends", "strategic recommendations"]
    }


@pytest.fixture
def sample_evaluation_data():
    """Sample evaluation data for testing."""
    return {
        "question": "What are the key strategic opportunities?",
        "answer": "The key strategic opportunities include AI market expansion and new partnerships.",
        "contexts": ["AI market shows strong growth potential", "Partnership opportunities are emerging"],
        "ground_truth": "AI market expansion and partnerships are key opportunities"
    } 