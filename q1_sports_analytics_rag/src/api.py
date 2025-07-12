from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import uvicorn
from datetime import datetime

from .config import Config
from .models import (
    QueryRequest, RAGResponse, DocumentUpload, SystemStatus
)
from .rag_system import SportsAnalyticsRAG

# Initialize FastAPI app
app = FastAPI(
    title="Sports Analytics RAG System",
    description="Advanced RAG system for sports analytics with query decomposition, context compression, and citation extraction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
rag_system = None

def get_rag_system() -> SportsAnalyticsRAG:
    """Dependency to get RAG system instance."""
    global rag_system
    if rag_system is None:
        try:
            rag_system = SportsAnalyticsRAG()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize RAG system: {str(e)}"
            )
    return rag_system

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global rag_system
    try:
        rag_system = SportsAnalyticsRAG()
        print("Sports Analytics RAG System initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Sports Analytics RAG System API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint."""
    try:
        rag = get_rag_system()
        status = rag.get_system_status()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system_status": status
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.post("/query", response_model=RAGResponse)
async def process_query(
    request: QueryRequest,
    rag: SportsAnalyticsRAG = Depends(get_rag_system)
):
    """
    Process a sports analytics query through the RAG system.
    
    This endpoint handles:
    - Query decomposition for complex questions
    - Document retrieval and reranking
    - Context compression
    - Answer generation with citations
    """
    try:
        response = rag.process_query(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/documents", response_model=Dict[str, Any])
async def add_documents(
    documents: List[DocumentUpload],
    rag: SportsAnalyticsRAG = Depends(get_rag_system)
):
    """
    Add documents to the vector store.
    
    Documents will be chunked, embedded, and stored for retrieval.
    """
    try:
        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No documents provided"
            )
        
        result = rag.add_documents(documents)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error adding documents: {str(e)}"
        )

@app.get("/status", response_model=SystemStatus)
async def get_status(rag: SportsAnalyticsRAG = Depends(get_rag_system)):
    """
    Get system status and statistics.
    """
    try:
        status = rag.get_system_status()
        return SystemStatus(
            status=status["status"],
            total_documents=status["total_documents"],
            last_updated=datetime.now(),
            version=status["version"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting status: {str(e)}"
        )

@app.get("/documents/count", response_model=Dict[str, int])
async def get_document_count(rag: SportsAnalyticsRAG = Depends(get_rag_system)):
    """
    Get the total number of documents in the system.
    """
    try:
        count = rag.vector_store.get_document_count()
        return {"total_documents": count}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting document count: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

def run_server():
    """Run the FastAPI server."""
    config = Config()
    uvicorn.run(
        "src.api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )

if __name__ == "__main__":
    run_server() 