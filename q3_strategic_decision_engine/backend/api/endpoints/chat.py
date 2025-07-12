"""
Chat API endpoints for strategic conversations with AI assistant
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
import uuid
import json
from pydantic import BaseModel, Field

from ...services.llm_service import LLMService, LLMTask
from ...services.vector_store_service import VectorStoreService
from ...services.cache_service import CacheService
from ...core.database import DatabaseManager
from ...core.logging_config import get_logger

# Initialize router
router = APIRouter()
logger = get_logger('api.chat')

# Global services
llm_service = None
vector_store_service = None
cache_service = None

# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    context_documents: Optional[List[str]] = Field(None, description="Document IDs for context")
    analysis_type: Optional[str] = Field("general", description="Type of analysis: general, swot, market, financial")
    stream: bool = Field(False, description="Enable streaming response")

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[Dict[str, Any]]
    analysis_type: str
    token_count: int
    response_time: float

class ChatHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    created_at: str
    updated_at: str

class ChatSessionInfo(BaseModel):
    session_id: str
    title: str
    message_count: int
    created_at: str
    updated_at: str

# Dependency to get services
async def get_services():
    global llm_service, vector_store_service, cache_service
    if llm_service is None:
        llm_service = LLMService()
        await llm_service.initialize()
    
    if vector_store_service is None:
        vector_store_service = VectorStoreService()
        await vector_store_service.initialize()
    
    if cache_service is None:
        cache_service = CacheService()
        await cache_service.initialize()
    
    return llm_service, vector_store_service, cache_service

@router.post("/message", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    services = Depends(get_services)
):
    """Send a message to the AI assistant"""
    llm_service, vector_store_service, cache_service = services
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Check cache for similar queries
        query_hash = cache_service.generate_query_hash(request.message)
        cached_response = await cache_service.get_query_results(query_hash)
        
        if cached_response:
            logger.info(f"Returning cached response for session {session_id}")
            return ChatResponse(**cached_response)
        
        # Retrieve relevant documents
        relevant_docs = []
        sources = []
        
        if request.context_documents:
            # Use specific documents
            for doc_id in request.context_documents:
                doc_chunks = await vector_store_service.get_document_chunks(doc_id)
                relevant_docs.extend(doc_chunks)
        else:
            # Search for relevant documents
            search_results = await vector_store_service.hybrid_search(
                query=request.message,
                k=10,
                threshold=0.7
            )
            relevant_docs = search_results
        
        # Extract source information
        for doc in relevant_docs:
            source_info = {
                "document_id": doc.metadata.get("document_id", "unknown"),
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "similarity_score": doc.metadata.get("similarity_score", 0.0),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source_info)
        
        # Prepare context for LLM
        context = ""
        if relevant_docs:
            context = "\n\n".join([f"Document: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in relevant_docs[:5]])
        
        # Select appropriate LLM task based on analysis type
        task_mapping = {
            "general": LLMTask.GENERAL_CHAT,
            "swot": LLMTask.SWOT_ANALYSIS,
            "market": LLMTask.MARKET_ANALYSIS,
            "financial": LLMTask.FINANCIAL_ANALYSIS,
            "strategic": LLMTask.STRATEGIC_PLANNING,
            "document": LLMTask.DOCUMENT_ANALYSIS
        }
        
        task = task_mapping.get(request.analysis_type, LLMTask.GENERAL_CHAT)
        
        # Prepare system message
        system_message = f"""You are an expert strategic business consultant and AI assistant. 
        You help CEOs and business leaders make informed strategic decisions by analyzing their documents and providing actionable insights.
        
        When answering questions:
        1. Use the provided context documents to ground your responses
        2. Provide specific, actionable recommendations
        3. Reference the source documents when making claims
        4. Focus on strategic business value and ROI
        5. Structure your response clearly with headings and bullet points
        
        Context Documents:
        {context}
        
        Analysis Type: {request.analysis_type}
        """
        
        # Generate response
        start_time = time.time()
        response = await llm_service.generate_response(
            prompt=request.message,
            system_message=system_message,
            task=task,
            temperature=0.1,
            max_tokens=2000
        )
        
        response_time = time.time() - start_time
        
        # Save to database
        background_tasks.add_task(
            save_chat_message,
            session_id,
            request.message,
            response.content,
            sources,
            response.tokens_used,
            response.model,
            response_time
        )
        
        # Cache response
        cache_data = {
            "response": response.content,
            "session_id": session_id,
            "sources": sources,
            "analysis_type": request.analysis_type,
            "token_count": response.tokens_used,
            "response_time": response_time
        }
        
        await cache_service.set(f"query_results:{query_hash}", cache_data, ttl=3600)
        
        logger.info(f"Generated response for session {session_id} in {response_time:.2f}s")
        
        return ChatResponse(
            response=response.content,
            session_id=session_id,
            sources=sources,
            analysis_type=request.analysis_type,
            token_count=response.tokens_used,
            response_time=response_time
        )
        
    except Exception as e:
        logger.error(f"Chat message failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@router.post("/stream")
async def stream_message(
    request: ChatRequest,
    services = Depends(get_services)
):
    """Stream AI assistant response"""
    llm_service, vector_store_service, cache_service = services
    
    async def generate_stream():
        try:
            # Generate session ID if not provided
            session_id = request.session_id or str(uuid.uuid4())
            
            # Retrieve relevant documents (same as above)
            relevant_docs = []
            if request.context_documents:
                for doc_id in request.context_documents:
                    doc_chunks = await vector_store_service.get_document_chunks(doc_id)
                    relevant_docs.extend(doc_chunks)
            else:
                search_results = await vector_store_service.hybrid_search(
                    query=request.message,
                    k=10,
                    threshold=0.7
                )
                relevant_docs = search_results
            
            # Prepare context
            context = ""
            if relevant_docs:
                context = "\n\n".join([f"Document: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in relevant_docs[:5]])
            
            # Select task
            task_mapping = {
                "general": LLMTask.GENERAL_CHAT,
                "swot": LLMTask.SWOT_ANALYSIS,
                "market": LLMTask.MARKET_ANALYSIS,
                "financial": LLMTask.FINANCIAL_ANALYSIS,
                "strategic": LLMTask.STRATEGIC_PLANNING,
                "document": LLMTask.DOCUMENT_ANALYSIS
            }
            
            task = task_mapping.get(request.analysis_type, LLMTask.GENERAL_CHAT)
            
            # System message
            system_message = f"""You are an expert strategic business consultant and AI assistant. 
            Provide strategic business insights based on the provided context documents.
            
            Context Documents:
            {context}
            
            Analysis Type: {request.analysis_type}
            """
            
            # Generate streaming response
            async for chunk in llm_service.generate_streaming_response(
                prompt=request.message,
                system_message=system_message,
                task=task
            ):
                # Format as Server-Sent Event
                yield f"data: {json.dumps({'chunk': chunk, 'session_id': session_id})}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming failed: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/plain")

@router.get("/history/{session_id}", response_model=ChatHistory)
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    services = Depends(get_services)
):
    """Get chat history for a session"""
    try:
        with DatabaseManager() as db:
            messages = db.get_chat_history(session_id, limit)
            
            chat_messages = []
            for msg in messages:
                chat_messages.append(ChatMessage(
                    role=msg.message_type,
                    content=msg.content,
                    timestamp=msg.timestamp.isoformat(),
                    sources=msg.sources,
                    metadata=msg.metadata
                ))
            
            return ChatHistory(
                session_id=session_id,
                messages=chat_messages,
                created_at=messages[-1].timestamp.isoformat() if messages else "",
                updated_at=messages[0].timestamp.isoformat() if messages else ""
            )
            
    except Exception as e:
        logger.error(f"Failed to get chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@router.get("/sessions", response_model=List[ChatSessionInfo])
async def get_chat_sessions(
    limit: int = 20,
    services = Depends(get_services)
):
    """Get list of chat sessions"""
    try:
        with DatabaseManager() as db:
            sessions = db.get_chat_sessions(limit)
            
            session_info = []
            for session in sessions:
                session_info.append(ChatSessionInfo(
                    session_id=session.session_id,
                    title=session.title or "Chat Session",
                    message_count=session.message_count,
                    created_at=session.created_at.isoformat(),
                    updated_at=session.updated_at.isoformat()
                ))
            
            return session_info
            
    except Exception as e:
        logger.error(f"Failed to get chat sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")

@router.delete("/session/{session_id}")
async def delete_chat_session(
    session_id: str,
    services = Depends(get_services)
):
    """Delete a chat session"""
    try:
        with DatabaseManager() as db:
            # Delete all messages in session
            db.delete_chat_session(session_id)
        
        logger.info(f"Deleted chat session: {session_id}")
        
        return {"message": "Chat session deleted successfully", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"Failed to delete chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

@router.post("/session/{session_id}/title")
async def update_session_title(
    session_id: str,
    title: str,
    services = Depends(get_services)
):
    """Update chat session title"""
    try:
        with DatabaseManager() as db:
            db.update_session_title(session_id, title)
        
        return {"message": "Session title updated", "session_id": session_id, "title": title}
        
    except Exception as e:
        logger.error(f"Failed to update session title: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update title: {str(e)}")

# Background task helpers
async def save_chat_message(
    session_id: str,
    user_message: str,
    ai_response: str,
    sources: List[Dict[str, Any]],
    token_count: int,
    model: str,
    response_time: float
):
    """Save chat messages to database"""
    try:
        with DatabaseManager() as db:
            # Save user message
            db.add_chat_message(
                session_id=session_id,
                message_type="user",
                content=user_message,
                metadata={"timestamp": time.time()}
            )
            
            # Save AI response
            db.add_chat_message(
                session_id=session_id,
                message_type="assistant",
                content=ai_response,
                tokens_used=token_count,
                model_used=model,
                response_time=response_time,
                sources=sources,
                metadata={"timestamp": time.time()}
            )
            
        logger.info(f"Saved chat messages for session {session_id}")
        
    except Exception as e:
        logger.error(f"Failed to save chat messages: {str(e)}")

import time 