from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class SubQuestion(BaseModel):
    """Model for decomposed sub-questions."""
    question: str = Field(..., description="The sub-question text")
    reasoning: str = Field(..., description="Reasoning for this sub-question")
    priority: int = Field(..., description="Priority order for processing")

class QueryDecomposition(BaseModel):
    """Model for query decomposition results."""
    original_query: str = Field(..., description="Original complex query")
    sub_questions: List[SubQuestion] = Field(..., description="List of decomposed sub-questions")
    decomposition_strategy: str = Field(..., description="Strategy used for decomposition")

class DocumentChunk(BaseModel):
    """Model for document chunks with metadata."""
    content: str = Field(..., description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata for the chunk")
    source: str = Field(..., description="Source document identifier")
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    score: Optional[float] = Field(None, description="Similarity score")

class RetrievalResult(BaseModel):
    """Model for retrieval results."""
    query: str = Field(..., description="The query that was processed")
    documents: List[DocumentChunk] = Field(..., description="Retrieved document chunks")
    total_retrieved: int = Field(..., description="Total number of documents retrieved")
    retrieval_time: float = Field(..., description="Time taken for retrieval in seconds")

class CompressedContext(BaseModel):
    """Model for compressed context."""
    original_chunks: List[DocumentChunk] = Field(..., description="Original document chunks")
    compressed_content: str = Field(..., description="Compressed and relevant content")
    compression_ratio: float = Field(..., description="Ratio of compression achieved")
    relevance_score: float = Field(..., description="Overall relevance score")

class Citation(BaseModel):
    """Model for citations in responses."""
    claim: str = Field(..., description="The specific claim being cited")
    source: str = Field(..., description="Source document identifier")
    page_section: Optional[str] = Field(None, description="Page or section reference")
    confidence: float = Field(..., description="Confidence score for the citation")

class RAGResponse(BaseModel):
    """Model for RAG system responses."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(..., description="Citations supporting the answer")
    sub_questions: List[SubQuestion] = Field(..., description="Decomposed sub-questions")
    compressed_context: Optional[CompressedContext] = Field(None, description="Compressed context used")
    processing_time: float = Field(..., description="Total processing time in seconds")
    confidence_score: float = Field(..., description="Overall confidence in the response")

class QueryRequest(BaseModel):
    """Model for incoming query requests."""
    query: str = Field(..., description="The query to process", min_length=1)
    include_decomposition: bool = Field(True, description="Whether to include query decomposition")
    include_citations: bool = Field(True, description="Whether to include citations")
    max_results: int = Field(5, description="Maximum number of results to return", ge=1, le=20)

class DocumentUpload(BaseModel):
    """Model for document upload requests."""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    source: str = Field(..., description="Source identifier")

class SystemStatus(BaseModel):
    """Model for system status information."""
    status: str = Field(..., description="System status")
    total_documents: int = Field(..., description="Total documents in the database")
    last_updated: datetime = Field(..., description="Last update timestamp")
    version: str = Field(..., description="System version") 