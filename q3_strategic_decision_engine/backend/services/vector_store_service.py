"""
Vector Store Service for document storage and retrieval
Uses ChromaDB for persistent vector storage with OpenAI embeddings
"""

import asyncio
import time
from typing import List, Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib

# Vector store imports
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# BM25 for sparse retrieval
from rank_bm25 import BM25Okapi

from ..core.config import settings
from ..core.logging_config import rag_logger


@dataclass
class RetrievalResult:
    """Result from vector store retrieval"""
    documents: List[Document]
    scores: List[float]
    metadata: Dict[str, Any]
    retrieval_time: float
    reranking_time: Optional[float] = None


@dataclass
class DocumentChunk:
    """Document chunk with metadata"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    chunk_id: str = ""
    document_id: str = ""
    chunk_index: int = 0


class VectorStoreService:
    """Vector store service for document storage and retrieval"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embeddings = None
        self.text_splitter = None
        self.bm25 = None
        self.document_chunks: List[DocumentChunk] = []
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.sentence_transformer = None
    
    async def initialize(self):
        """Initialize vector store service"""
        if self.initialized:
            return
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_DB_PATH,
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                api_key=settings.OPENAI_API_KEY,
                dimensions=settings.EMBEDDING_DIMENSION
            )
            
            # Initialize sentence transformer for reranking
            self.sentence_transformer = SentenceTransformer(settings.RERANKER_MODEL)
            
            # Create or get collection
            try:
                self.collection = self.client.get_collection(
                    name=settings.CHROMA_COLLECTION_NAME
                )
                self.logger.info(f"Retrieved existing collection: {settings.CHROMA_COLLECTION_NAME}")
            except Exception:
                # Create new collection
                self.collection = self.client.create_collection(
                    name=settings.CHROMA_COLLECTION_NAME,
                    metadata={"description": "Strategic documents collection"}
                )
                self.logger.info(f"Created new collection: {settings.CHROMA_COLLECTION_NAME}")
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            # Load existing document chunks for BM25
            await self._load_existing_chunks()
            
            self.initialized = True
            self.logger.info("Vector store service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store service: {str(e)}")
            raise
    
    async def _load_existing_chunks(self):
        """Load existing document chunks from ChromaDB"""
        try:
            # Get all documents from collection
            results = self.collection.get(include=["documents", "metadatas"])
            
            if results['documents']:
                self.document_chunks = []
                for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                    chunk = DocumentChunk(
                        content=doc,
                        metadata=metadata,
                        chunk_id=metadata.get('chunk_id', f'chunk_{i}'),
                        document_id=metadata.get('document_id', 'unknown'),
                        chunk_index=metadata.get('chunk_index', i)
                    )
                    self.document_chunks.append(chunk)
                
                # Initialize BM25 with existing chunks
                corpus = [chunk.content for chunk in self.document_chunks]
                if corpus:
                    self.bm25 = BM25Okapi([doc.split() for doc in corpus])
                    self.logger.info(f"Loaded {len(self.document_chunks)} existing chunks")
            
        except Exception as e:
            self.logger.error(f"Failed to load existing chunks: {str(e)}")
    
    async def add_documents(self, documents: List[Document], document_id: str = None) -> int:
        """Add documents to vector store"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Split documents into chunks
            chunks = []
            for doc in documents:
                doc_chunks = self.text_splitter.split_documents([doc])
                chunks.extend(doc_chunks)
            
            if not chunks:
                return 0
            
            # Create document chunks with metadata
            document_chunks = []
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = self._generate_chunk_id(chunk.page_content, document_id or "unknown", i)
                
                # Enhanced metadata
                metadata = {
                    "chunk_id": chunk_id,
                    "document_id": document_id or "unknown",
                    "chunk_index": i,
                    "chunk_size": len(chunk.page_content),
                    "source": chunk.metadata.get("source", "unknown"),
                    "page": chunk.metadata.get("page", 0),
                    "total_chunks": len(chunks)
                }
                
                # Add original metadata
                metadata.update(chunk.metadata)
                
                document_chunk = DocumentChunk(
                    content=chunk.page_content,
                    metadata=metadata,
                    chunk_id=chunk_id,
                    document_id=document_id or "unknown",
                    chunk_index=i
                )
                
                document_chunks.append(document_chunk)
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk.page_content)
                chunk_metadatas.append(metadata)
            
            # Generate embeddings
            embeddings = await self.embeddings.aembed_documents(chunk_texts)
            
            # Add to ChromaDB
            self.collection.add(
                ids=chunk_ids,
                documents=chunk_texts,
                metadatas=chunk_metadatas,
                embeddings=embeddings
            )
            
            # Update local storage
            self.document_chunks.extend(document_chunks)
            
            # Update BM25 index
            corpus = [chunk.content for chunk in self.document_chunks]
            self.bm25 = BM25Okapi([doc.split() for doc in corpus])
            
            self.logger.info(f"Added {len(chunks)} chunks to vector store")
            return len(chunks)
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {str(e)}")
            raise
    
    async def similarity_search(
        self,
        query: str,
        k: int = None,
        threshold: float = None,
        filters: Dict[str, Any] = None
    ) -> List[Document]:
        """Perform similarity search"""
        if not self.initialized:
            await self.initialize()
        
        k = k or settings.MAX_RETRIEVAL_RESULTS
        threshold = threshold or settings.SIMILARITY_THRESHOLD
        
        try:
            # Generate query embedding
            query_embedding = await self.embeddings.aembed_query(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
                where=filters
            )
            
            # Convert to LangChain documents
            documents = []
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                # Convert distance to similarity score
                similarity_score = 1 - distance
                
                if similarity_score >= threshold:
                    documents.append(Document(
                        page_content=doc,
                        metadata={
                            **metadata,
                            'similarity_score': similarity_score
                        }
                    ))
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Similarity search failed: {str(e)}")
            raise
    
    async def hybrid_search(
        self,
        query: str,
        k: int = None,
        alpha: float = 0.5,
        threshold: float = None,
        filters: Dict[str, Any] = None
    ) -> List[Document]:
        """Perform hybrid search combining dense and sparse retrieval"""
        if not self.initialized:
            await self.initialize()
        
        k = k or settings.MAX_RETRIEVAL_RESULTS
        threshold = threshold or settings.SIMILARITY_THRESHOLD
        
        start_time = time.time()
        
        try:
            # Dense retrieval (semantic search)
            dense_results = await self.similarity_search(
                query=query,
                k=k * 2,  # Get more results for reranking
                threshold=0.0,  # No threshold filtering here
                filters=filters
            )
            
            # Sparse retrieval (BM25)
            sparse_results = []
            if self.bm25 and self.document_chunks:
                query_tokens = query.split()
                bm25_scores = self.bm25.get_scores(query_tokens)
                
                # Get top k results from BM25
                top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k * 2]
                
                for idx in top_indices:
                    if idx < len(self.document_chunks):
                        chunk = self.document_chunks[idx]
                        # Apply filters if provided
                        if filters and not self._match_filters(chunk.metadata, filters):
                            continue
                        
                        sparse_results.append(Document(
                            page_content=chunk.content,
                            metadata={
                                **chunk.metadata,
                                'bm25_score': bm25_scores[idx]
                            }
                        ))
            
            # Combine and rerank results
            combined_results = self._combine_results(dense_results, sparse_results, alpha)
            
            # Rerank using sentence transformer
            reranked_results = await self._rerank_results(query, combined_results[:k * 2])
            
            # Apply similarity threshold
            final_results = []
            for doc in reranked_results[:k]:
                similarity_score = doc.metadata.get('similarity_score', 0)
                if similarity_score >= threshold:
                    final_results.append(doc)
            
            retrieval_time = time.time() - start_time
            
            # Log retrieval
            rag_logger.log_retrieval(
                query=query,
                num_results=len(final_results),
                retrieval_time=retrieval_time,
                reranking_time=None
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {str(e)}")
            raise
    
    def _combine_results(
        self,
        dense_results: List[Document],
        sparse_results: List[Document],
        alpha: float = 0.5
    ) -> List[Document]:
        """Combine dense and sparse results with weighted scores"""
        
        # Create a dictionary to store combined results
        combined = {}
        
        # Add dense results
        for doc in dense_results:
            chunk_id = doc.metadata.get('chunk_id', '')
            similarity_score = doc.metadata.get('similarity_score', 0)
            
            combined[chunk_id] = {
                'document': doc,
                'dense_score': similarity_score,
                'sparse_score': 0.0
            }
        
        # Add sparse results
        for doc in sparse_results:
            chunk_id = doc.metadata.get('chunk_id', '')
            bm25_score = doc.metadata.get('bm25_score', 0)
            
            if chunk_id in combined:
                combined[chunk_id]['sparse_score'] = bm25_score
            else:
                combined[chunk_id] = {
                    'document': doc,
                    'dense_score': 0.0,
                    'sparse_score': bm25_score
                }
        
        # Calculate combined scores
        results = []
        for chunk_id, data in combined.items():
            # Normalize scores to [0, 1] range
            dense_score = data['dense_score']
            sparse_score = min(data['sparse_score'] / 10.0, 1.0)  # Normalize BM25 score
            
            # Calculate weighted combination
            combined_score = alpha * dense_score + (1 - alpha) * sparse_score
            
            doc = data['document']
            doc.metadata['combined_score'] = combined_score
            doc.metadata['dense_score'] = dense_score
            doc.metadata['sparse_score'] = sparse_score
            
            results.append(doc)
        
        # Sort by combined score
        results.sort(key=lambda x: x.metadata['combined_score'], reverse=True)
        
        return results
    
    async def _rerank_results(self, query: str, results: List[Document]) -> List[Document]:
        """Rerank results using sentence transformer"""
        if not results or not self.sentence_transformer:
            return results
        
        try:
            # Get similarity scores between query and documents
            documents_text = [doc.page_content for doc in results]
            query_embedding = self.sentence_transformer.encode([query])
            doc_embeddings = self.sentence_transformer.encode(documents_text)
            
            # Calculate similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # Add reranking scores to documents
            for i, doc in enumerate(results):
                doc.metadata['rerank_score'] = similarities[i]
            
            # Sort by reranking score
            results.sort(key=lambda x: x.metadata['rerank_score'], reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {str(e)}")
            return results
    
    def _match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters"""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def _generate_chunk_id(self, content: str, document_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{document_id}_{chunk_index}_{content_hash}"
    
    async def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a specific document"""
        return [
            chunk for chunk in self.document_chunks
            if chunk.document_id == document_id
        ]
    
    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a specific document"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Get chunk IDs for the document
            chunk_ids = [
                chunk.chunk_id for chunk in self.document_chunks
                if chunk.document_id == document_id
            ]
            
            if not chunk_ids:
                return 0
            
            # Delete from ChromaDB
            self.collection.delete(ids=chunk_ids)
            
            # Remove from local storage
            self.document_chunks = [
                chunk for chunk in self.document_chunks
                if chunk.document_id != document_id
            ]
            
            # Update BM25 index
            corpus = [chunk.content for chunk in self.document_chunks]
            if corpus:
                self.bm25 = BM25Okapi([doc.split() for doc in corpus])
            else:
                self.bm25 = None
            
            self.logger.info(f"Deleted {len(chunk_ids)} chunks for document {document_id}")
            return len(chunk_ids)
            
        except Exception as e:
            self.logger.error(f"Failed to delete document: {str(e)}")
            raise
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.initialized:
            await self.initialize()
        
        try:
            count = self.collection.count()
            
            # Get document count
            document_ids = set(chunk.document_id for chunk in self.document_chunks)
            
            return {
                'total_chunks': count,
                'total_documents': len(document_ids),
                'collection_name': settings.CHROMA_COLLECTION_NAME,
                'embedding_model': settings.EMBEDDING_MODEL,
                'chunk_size': settings.CHUNK_SIZE,
                'chunk_overlap': settings.CHUNK_OVERLAP
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {str(e)}")
            return {}
    
    async def search_with_metadata(
        self,
        query: str,
        k: int = None,
        metadata_filters: Dict[str, Any] = None,
        include_scores: bool = True
    ) -> RetrievalResult:
        """Search with detailed metadata and scoring"""
        start_time = time.time()
        
        documents = await self.hybrid_search(
            query=query,
            k=k,
            filters=metadata_filters
        )
        
        retrieval_time = time.time() - start_time
        
        # Extract scores
        scores = []
        if include_scores:
            for doc in documents:
                scores.append(doc.metadata.get('combined_score', 0.0))
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            metadata={
                'query': query,
                'num_results': len(documents),
                'filters': metadata_filters,
                'search_type': 'hybrid'
            },
            retrieval_time=retrieval_time
        ) 