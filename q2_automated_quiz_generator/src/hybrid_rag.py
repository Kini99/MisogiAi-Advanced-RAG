"""Hybrid RAG system combining dense and sparse retrieval methods."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings as ChromaSettings

from .models import DocumentChunk
from .config import settings
from .cache import assessment_cache


class HybridRAG:
    """Hybrid RAG system with dense and sparse retrieval."""
    
    def __init__(self):
        """Initialize hybrid RAG system."""
        # Initialize dense embeddings
        self.dense_encoder = SentenceTransformer(settings.dense_embedding_model)
        
        # Initialize cross-encoder for reranking
        self.cross_encoder = CrossEncoder(settings.cross_encoder_model)
        
        # Initialize ChromaDB for dense retrieval
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Initialize BM25 for sparse retrieval
        self.bm25_index = None
        self.bm25_documents = []
        
        # Collection names
        self.collection_name = "educational_content"
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Educational content for assessment generation"}
            )
    
    def _update_bm25_index(self, chunks: List[DocumentChunk]):
        """Update BM25 index with new chunks."""
        # Extract text content for BM25
        documents = [chunk.content for chunk in chunks]
        tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Store document references
        self.bm25_documents.extend(chunks)
        
        # Recreate BM25 index with all documents
        all_docs = [chunk.content for chunk in self.bm25_documents]
        all_tokenized_docs = [doc.lower().split() for doc in all_docs]
        self.bm25_index = BM25Okapi(all_tokenized_docs)
    
    def add_documents(self, chunks: List[DocumentChunk], topic: str):
        """Add documents to both dense and sparse indices."""
        if not chunks:
            return
        
        # Update BM25 index
        self._update_bm25_index(chunks)
        
        # Add to ChromaDB for dense retrieval
        documents = [chunk.content for chunk in chunks]
        ids = [chunk.id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.dense_encoder.encode(documents, show_progress_bar=False)
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=metadatas
        )
        
        # Cache chunks for topic
        assessment_cache.set_document_chunks(topic, chunks)
    
    def _dense_retrieval(self, query: str, top_k: int = 20) -> List[Tuple[DocumentChunk, float]]:
        """Perform dense retrieval using ChromaDB."""
        try:
            # Generate query embedding
            query_embedding = self.dense_encoder.encode([query])
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            # Process results
            dense_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    chunk = DocumentChunk(
                        id=doc_id,
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                        embedding=results['embeddings'][0][i] if results['embeddings'] else None
                    )
                    distance = results['distances'][0][i] if results['distances'] else 0.0
                    dense_results.append((chunk, 1.0 - distance))  # Convert distance to similarity
            
            return dense_results
        except Exception as e:
            print(f"Error in dense retrieval: {e}")
            return []
    
    def _sparse_retrieval(self, query: str, top_k: int = 20) -> List[Tuple[DocumentChunk, float]]:
        """Perform sparse retrieval using BM25."""
        if self.bm25_index is None:
            return []
        
        try:
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            sparse_results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include documents with positive scores
                    chunk = self.bm25_documents[idx]
                    # Normalize BM25 score to [0, 1] range
                    normalized_score = min(scores[idx] / 10.0, 1.0)
                    sparse_results.append((chunk, normalized_score))
            
            return sparse_results
        except Exception as e:
            print(f"Error in sparse retrieval: {e}")
            return []
    
    def _combine_results(self, dense_results: List[Tuple[DocumentChunk, float]], 
                        sparse_results: List[Tuple[DocumentChunk, float]], 
                        alpha: float = 0.7) -> List[Tuple[DocumentChunk, float]]:
        """Combine dense and sparse results using weighted fusion."""
        # Create document ID to score mapping
        combined_scores = {}
        
        # Add dense results
        for chunk, score in dense_results:
            combined_scores[chunk.id] = {
                'chunk': chunk,
                'dense_score': score,
                'sparse_score': 0.0
            }
        
        # Add sparse results
        for chunk, score in sparse_results:
            if chunk.id in combined_scores:
                combined_scores[chunk.id]['sparse_score'] = score
            else:
                combined_scores[chunk.id] = {
                    'chunk': chunk,
                    'dense_score': 0.0,
                    'sparse_score': score
                }
        
        # Calculate combined scores
        combined_results = []
        for doc_id, scores in combined_scores.items():
            combined_score = (alpha * scores['dense_score'] + 
                            (1 - alpha) * scores['sparse_score'])
            combined_results.append((scores['chunk'], combined_score))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results
    
    def _rerank_with_cross_encoder(self, query: str, 
                                  candidates: List[Tuple[DocumentChunk, float]], 
                                  top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Rerank candidates using cross-encoder."""
        if not candidates:
            return []
        
        try:
            # Prepare pairs for cross-encoder
            pairs = [(query, candidate[0].content) for candidate in candidates]
            
            # Get cross-encoder scores
            cross_scores = self.cross_encoder.predict(pairs)
            
            # Combine with original scores
            reranked_results = []
            for i, (chunk, original_score) in enumerate(candidates):
                cross_score = cross_scores[i]
                # Combine scores (you can adjust the weights)
                final_score = 0.3 * original_score + 0.7 * cross_score
                reranked_results.append((chunk, final_score))
            
            # Sort by final score
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            
            return reranked_results[:top_k]
        except Exception as e:
            print(f"Error in cross-encoder reranking: {e}")
            return candidates[:top_k]
    
    def retrieve(self, query: str, topic: str, top_k: int = 10, 
                use_cache: bool = True) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve relevant documents using hybrid approach."""
        # Check cache first
        if use_cache:
            cached_chunks = assessment_cache.get_document_chunks(topic)
            if cached_chunks:
                # Use cached chunks for retrieval
                self._update_bm25_index(cached_chunks)
        
        # Perform dense retrieval
        dense_results = self._dense_retrieval(query, top_k=top_k * 2)
        
        # Perform sparse retrieval
        sparse_results = self._sparse_retrieval(query, top_k=top_k * 2)
        
        # Combine results
        combined_results = self._combine_results(dense_results, sparse_results)
        
        # Rerank with cross-encoder
        final_results = self._rerank_with_cross_encoder(query, combined_results, top_k)
        
        return final_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "bm25_documents": len(self.bm25_documents),
                "collection_name": self.collection_name
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {"error": str(e)}


# Global hybrid RAG instance
hybrid_rag = HybridRAG() 