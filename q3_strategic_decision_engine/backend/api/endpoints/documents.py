"""
Document API endpoints for file upload, processing, and management
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
from pydantic import BaseModel

from ...services.document_service import DocumentService
from ...services.vector_store_service import VectorStoreService
from ...services.llm_service import LLMService
from ...core.database import get_db, DatabaseManager
from ...core.logging_config import get_logger

# Initialize router
router = APIRouter()
logger = get_logger('api.documents')

# Global services (these would be injected in a real application)
document_service = None
vector_store_service = None
llm_service = None


# Pydantic models
class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_size: int
    file_type: str
    upload_date: str
    processed: bool
    chunk_count: int
    summary: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    skip: int
    limit: int


class ProcessingStatus(BaseModel):
    document_id: int
    status: str
    progress: float
    message: str
    processing_time: Optional[float] = None


# Dependency to get services
async def get_document_service() -> DocumentService:
    global document_service, vector_store_service, llm_service
    if document_service is None:
        # Initialize services if not already done
        if vector_store_service is None:
            vector_store_service = VectorStoreService()
            await vector_store_service.initialize()
        
        if llm_service is None:
            llm_service = LLMService()
            await llm_service.initialize()
        
        document_service = DocumentService(
            vector_store_service=vector_store_service,
            llm_service=llm_service
        )
    
    return document_service


@router.post("/upload", response_model=Dict[str, Any])
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    doc_service: DocumentService = Depends(get_document_service)
):
    """Upload a document file"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Read file content
        file_content = await file.read()
        
        # Upload file
        upload_result = await doc_service.upload_file(
            file_content=file_content,
            filename=file.filename,
            content_type=file.content_type
        )
        
        if not upload_result['success']:
            raise HTTPException(status_code=400, detail=upload_result['error'])
        
        # Schedule background processing
        background_tasks.add_task(
            process_document_background,
            upload_result['document_id'],
            doc_service
        )
        
        logger.info(f"Document uploaded: {file.filename}")
        
        return {
            "message": "File uploaded successfully",
            "document_id": upload_result['document_id'],
            "filename": upload_result['filename'],
            "original_filename": upload_result['original_filename'],
            "file_size": upload_result['file_size'],
            "file_type": upload_result['file_type'],
            "processing_status": "queued"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def process_document_background(document_id: int, doc_service: DocumentService):
    """Background task for document processing"""
    try:
        logger.info(f"Starting background processing for document {document_id}")
        await doc_service.process_document(document_id, generate_summary=True)
        logger.info(f"Background processing completed for document {document_id}")
    except Exception as e:
        logger.error(f"Background processing failed for document {document_id}: {str(e)}")


@router.post("/process/{document_id}", response_model=Dict[str, Any])
async def process_document(
    document_id: int,
    generate_summary: bool = Query(default=True, description="Generate document summary"),
    doc_service: DocumentService = Depends(get_document_service)
):
    """Process a specific document"""
    try:
        # Check if document exists
        doc_info = await doc_service.get_document_info(document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if doc_info['processed']:
            return {
                "message": "Document already processed",
                "document_id": document_id,
                "status": "completed"
            }
        
        # Process document
        processed_doc = await doc_service.process_document(
            document_id=document_id,
            generate_summary=generate_summary
        )
        
        logger.info(f"Document processed: {document_id}")
        
        return {
            "message": "Document processed successfully",
            "document_id": document_id,
            "filename": processed_doc.filename,
            "content_length": len(processed_doc.content),
            "chunk_count": len(processed_doc.chunks),
            "processing_time": processed_doc.processing_time,
            "summary": processed_doc.summary,
            "key_topics": processed_doc.key_topics,
            "status": "completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(default=0, ge=0, description="Number of documents to skip"),
    limit: int = Query(default=50, ge=1, le=100, description="Number of documents to return"),
    doc_service: DocumentService = Depends(get_document_service)
):
    """List all uploaded documents"""
    try:
        documents = await doc_service.list_documents(skip=skip, limit=limit)
        
        document_responses = [
            DocumentResponse(
                id=doc['id'],
                filename=doc['filename'],
                file_size=doc['file_size'],
                file_type=doc['file_type'],
                upload_date=doc['upload_date'],
                processed=doc['processed'],
                chunk_count=doc['chunk_count'],
                summary=doc['summary']
            )
            for doc in documents
        ]
        
        return DocumentListResponse(
            documents=document_responses,
            total=len(document_responses),
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/{document_id}", response_model=Dict[str, Any])
async def get_document(
    document_id: int,
    doc_service: DocumentService = Depends(get_document_service)
):
    """Get detailed information about a specific document"""
    try:
        doc_info = await doc_service.get_document_info(document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return doc_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    doc_service: DocumentService = Depends(get_document_service)
):
    """Delete a document and its associated data"""
    try:
        # Check if document exists
        doc_info = await doc_service.get_document_info(document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete document
        success = await doc_service.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete document")
        
        logger.info(f"Document deleted: {document_id}")
        
        return {
            "message": "Document deleted successfully",
            "document_id": document_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.get("/{document_id}/content")
async def get_document_content(
    document_id: int,
    doc_service: DocumentService = Depends(get_document_service)
):
    """Get the extracted content of a document"""
    try:
        # Check if document exists and is processed
        doc_info = await doc_service.get_document_info(document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not doc_info['processed']:
            raise HTTPException(status_code=400, detail="Document not yet processed")
        
        # Get document chunks from vector store
        if doc_service.vector_store_service:
            chunks = await doc_service.vector_store_service.get_document_chunks(str(document_id))
            
            # Combine chunk content
            content = "\n\n".join([chunk.content for chunk in chunks])
            
            return {
                "document_id": document_id,
                "filename": doc_info['filename'],
                "content": content,
                "chunk_count": len(chunks),
                "total_length": len(content)
            }
        else:
            raise HTTPException(status_code=500, detail="Vector store not available")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get content: {str(e)}")


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: int,
    doc_service: DocumentService = Depends(get_document_service)
):
    """Get the chunks of a processed document"""
    try:
        # Check if document exists and is processed
        doc_info = await doc_service.get_document_info(document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not doc_info['processed']:
            raise HTTPException(status_code=400, detail="Document not yet processed")
        
        # Get document chunks from vector store
        if doc_service.vector_store_service:
            chunks = await doc_service.vector_store_service.get_document_chunks(str(document_id))
            
            chunk_data = []
            for chunk in chunks:
                chunk_data.append({
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "content_length": len(chunk.content),
                    "metadata": chunk.metadata
                })
            
            return {
                "document_id": document_id,
                "filename": doc_info['filename'],
                "chunks": chunk_data,
                "total_chunks": len(chunk_data)
            }
        else:
            raise HTTPException(status_code=500, detail="Vector store not available")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chunks: {str(e)}")


@router.get("/{document_id}/status", response_model=ProcessingStatus)
async def get_processing_status(
    document_id: int,
    doc_service: DocumentService = Depends(get_document_service)
):
    """Get the processing status of a document"""
    try:
        doc_info = await doc_service.get_document_info(document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if doc_info['processed']:
            status = "completed"
            progress = 100.0
            message = "Document processed successfully"
        else:
            status = "processing"
            progress = 50.0  # This would be more sophisticated in a real implementation
            message = "Document is being processed"
        
        return ProcessingStatus(
            document_id=document_id,
            status=status,
            progress=progress,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get processing status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/stats/overview")
async def get_document_stats(
    doc_service: DocumentService = Depends(get_document_service)
):
    """Get document processing statistics"""
    try:
        stats = await doc_service.get_stats()
        
        return {
            "total_documents": stats.total_documents,
            "total_chunks": stats.total_chunks,
            "total_size_bytes": stats.total_size,
            "total_size_mb": round(stats.total_size / (1024 * 1024), 2),
            "average_processing_time": stats.processing_time,
            "file_types": stats.file_types
        }
        
    except Exception as e:
        logger.error(f"Failed to get document stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/batch-upload")
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    doc_service: DocumentService = Depends(get_document_service)
):
    """Upload multiple documents at once"""
    try:
        if len(files) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
        
        upload_results = []
        
        for file in files:
            if not file.filename:
                continue
            
            try:
                file_content = await file.read()
                
                upload_result = await doc_service.upload_file(
                    file_content=file_content,
                    filename=file.filename,
                    content_type=file.content_type
                )
                
                if upload_result['success']:
                    # Schedule background processing
                    background_tasks.add_task(
                        process_document_background,
                        upload_result['document_id'],
                        doc_service
                    )
                
                upload_results.append({
                    "filename": file.filename,
                    "success": upload_result['success'],
                    "document_id": upload_result.get('document_id'),
                    "error": upload_result.get('error')
                })
                
            except Exception as e:
                upload_results.append({
                    "filename": file.filename,
                    "success": False,
                    "document_id": None,
                    "error": str(e)
                })
        
        successful_uploads = sum(1 for result in upload_results if result['success'])
        
        logger.info(f"Batch upload completed: {successful_uploads}/{len(files)} files successful")
        
        return {
            "message": f"Batch upload completed: {successful_uploads}/{len(files)} files successful",
            "results": upload_results,
            "total_files": len(files),
            "successful_uploads": successful_uploads
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}") 