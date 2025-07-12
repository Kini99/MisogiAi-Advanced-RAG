"""FastAPI application for the Advanced Assessment Generation System."""

import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .models import (
    AssessmentRequest, AssessmentResponse, DocumentUpload, 
    UserPerformance, CacheStats, DifficultyLevel, QuestionType
)
from .config import settings
from .document_processor import document_processor
from .hybrid_rag import hybrid_rag
from .assessment_generator import assessment_generator
from .cache import assessment_cache


app = FastAPI(
    title="Advanced Assessment Generation System",
    description="An advanced system for generating personalized educational assessments using hybrid RAG",
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


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Advanced Assessment Generation System",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "redis_cache": "connected",
            "chroma_db": "connected",
            "hybrid_rag": "ready"
        }
    }


@app.post("/upload-document", response_model=DocumentUpload)
async def upload_document(
    file: UploadFile = File(...),
    topic: str = None,
    instructor_id: str = None
):
    """Upload educational document for processing."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Allowed: {settings.allowed_extensions}"
            )
        
        # Check file size
        file_size = 0
        file_path = os.path.join(settings.upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            file_size = os.path.getsize(file_path)
        
        if file_size > settings.max_file_size:
            os.remove(file_path)
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Max size: {settings.max_file_size} bytes"
            )
        
        # Validate file
        if not document_processor.validate_file(file_path):
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Invalid file")
        
        # Process document
        chunks = document_processor.process_document(file_path, topic, instructor_id)
        
        if not chunks:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="No content extracted from file")
        
        # Add to hybrid RAG system
        hybrid_rag.add_documents(chunks, topic)
        
        # Clear topic cache to ensure fresh content
        assessment_cache.clear_topic_cache(topic)
        
        return DocumentUpload(
            filename=file.filename,
            content_type=file.content_type,
            size=file_size,
            topic=topic,
            instructor_id=instructor_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/generate-assessment", response_model=AssessmentResponse)
async def generate_assessment(request: AssessmentRequest):
    """Generate assessment based on uploaded content."""
    try:
        # Validate request
        if not request.topic:
            raise HTTPException(status_code=400, detail="Topic is required")
        
        if request.num_questions <= 0 or request.num_questions > settings.max_questions_per_assessment:
            raise HTTPException(
                status_code=400, 
                detail=f"Number of questions must be between 1 and {settings.max_questions_per_assessment}"
            )
        
        # Generate assessment
        response = assessment_generator.generate_assessment(request)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating assessment: {str(e)}")


@app.post("/adjust-difficulty")
async def adjust_difficulty(performance: UserPerformance):
    """Adjust assessment difficulty based on user performance."""
    try:
        # Get the original assessment (in a real system, this would be from a database)
        # For now, we'll create a mock assessment
        from .models import Assessment, Question
        
        mock_assessment = Assessment(
            id=performance.assessment_id,
            title="Mock Assessment",
            description="Mock assessment for difficulty adjustment",
            topic="mock_topic",
            difficulty=DifficultyLevel.MEDIUM,
            questions=[],
            total_points=0,
            estimated_time=0,
            learning_objectives=[],
            created_at=performance.completed_at,
            instructor_id="mock_instructor"
        )
        
        # Adjust difficulty
        adjusted_assessment = assessment_generator.adjust_assessment_difficulty(
            performance, mock_assessment
        )
        
        return {
            "original_difficulty": mock_assessment.difficulty,
            "adjusted_difficulty": adjusted_assessment.difficulty,
            "performance_score": performance.score / performance.total_questions,
            "adjustment_reason": "Performance-based difficulty adjustment"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adjusting difficulty: {str(e)}")


@app.get("/cache-stats", response_model=CacheStats)
async def get_cache_stats():
    """Get cache statistics."""
    try:
        return assessment_cache.get_cache_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cache stats: {str(e)}")


@app.get("/collection-stats")
async def get_collection_stats():
    """Get document collection statistics."""
    try:
        return hybrid_rag.get_collection_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collection stats: {str(e)}")


@app.get("/difficulty-levels")
async def get_difficulty_levels():
    """Get available difficulty levels."""
    return {
        "difficulty_levels": [level.value for level in DifficultyLevel],
        "descriptions": {
            "easy": "Basic understanding and recall",
            "medium": "Application and analysis", 
            "hard": "Synthesis and evaluation"
        }
    }


@app.get("/question-types")
async def get_question_types():
    """Get available question types."""
    return {
        "question_types": [qt.value for qt in QuestionType],
        "descriptions": {
            "multiple_choice": "4 options, one correct answer",
            "true_false": "Simple true/false questions",
            "short_answer": "Brief written responses",
            "essay": "Longer, more detailed responses"
        }
    }


@app.delete("/clear-cache/{topic}")
async def clear_topic_cache(topic: str):
    """Clear cache for a specific topic."""
    try:
        cleared_count = assessment_cache.clear_topic_cache(topic)
        return {
            "message": f"Cleared {cleared_count} cache entries for topic: {topic}",
            "cleared_count": cleared_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


@app.get("/search-content")
async def search_content(query: str, topic: str, top_k: int = 5):
    """Search for relevant content using hybrid RAG."""
    try:
        results = hybrid_rag.retrieve(query, topic, top_k)
        
        return {
            "query": query,
            "topic": topic,
            "results": [
                {
                    "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    "score": score,
                    "metadata": chunk.metadata
                }
                for chunk, score in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching content: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    ) 