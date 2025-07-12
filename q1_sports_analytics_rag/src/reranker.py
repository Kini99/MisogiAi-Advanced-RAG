from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple
import time

from .config import Config
from .models import DocumentChunk

class Reranker:
    """Basic reranker using semantic similarity scores."""
    
    def __init__(self):
        """Initialize the reranker."""
        self.config = Config()
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[DocumentChunk],
        top_k: int = None
    ) -> List[DocumentChunk]:
        """
        Rerank documents based on semantic similarity to the query.
        
        Args:
            query: The search query
            documents: List of document chunks to rerank
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of document chunks
        """
        if not documents:
            return []
        
        if top_k is None:
            top_k = self.config.TOP_K_RERANK
        
        start_time = time.time()
        
        try:
            # Get embeddings for query and documents
            query_embedding = self.embedding_model.encode([query])
            doc_embeddings = self.embedding_model.encode([doc.content for doc in documents])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # Create tuples of (document, similarity_score)
            doc_similarity_pairs = list(zip(documents, similarities))
            
            # Sort by similarity score (descending)
            doc_similarity_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Update document scores and return top_k
            reranked_documents = []
            for doc, similarity_score in doc_similarity_pairs[:top_k]:
                doc.score = float(similarity_score)
                reranked_documents.append(doc)
            
            return reranked_documents
            
        except Exception as e:
            # Fallback: return original documents with default scores
            print(f"Reranking failed: {e}")
            for doc in documents:
                if doc.score is None:
                    doc.score = 0.5  # Default score
            
            # Sort by existing scores if available
            documents.sort(key=lambda x: x.score or 0.0, reverse=True)
            return documents[:top_k]
    
    def calculate_similarity_matrix(
        self, 
        documents: List[DocumentChunk]
    ) -> np.ndarray:
        """
        Calculate similarity matrix between all documents.
        
        Args:
            documents: List of document chunks
            
        Returns:
            Similarity matrix
        """
        if not documents:
            return np.array([])
        
        try:
            # Get embeddings for all documents
            doc_embeddings = self.embedding_model.encode([doc.content for doc in documents])
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(doc_embeddings)
            
            return similarity_matrix
            
        except Exception as e:
            print(f"Similarity matrix calculation failed: {e}")
            # Return identity matrix as fallback
            n_docs = len(documents)
            return np.eye(n_docs)
    
    def find_duplicate_documents(
        self, 
        documents: List[DocumentChunk],
        similarity_threshold: float = 0.9
    ) -> List[Tuple[int, int, float]]:
        """
        Find duplicate or highly similar documents.
        
        Args:
            documents: List of document chunks
            similarity_threshold: Threshold for considering documents similar
            
        Returns:
            List of (doc1_index, doc2_index, similarity) tuples
        """
        if len(documents) < 2:
            return []
        
        similarity_matrix = self.calculate_similarity_matrix(documents)
        duplicates = []
        
        # Find pairs above threshold (upper triangle only to avoid duplicates)
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarity_matrix[i, j]
                if similarity >= similarity_threshold:
                    duplicates.append((i, j, similarity))
        
        return duplicates
    
    def diversify_results(
        self, 
        query: str, 
        documents: List[DocumentChunk],
        top_k: int = None,
        diversity_threshold: float = 0.7
    ) -> List[DocumentChunk]:
        """
        Diversify results by removing highly similar documents.
        
        Args:
            query: The search query
            documents: List of document chunks
            top_k: Number of documents to return
            diversity_threshold: Threshold for considering documents too similar
            
        Returns:
            Diversified list of document chunks
        """
        if not documents:
            return []
        
        if top_k is None:
            top_k = self.config.TOP_K_RERANK
        
        # First rerank by similarity to query
        reranked_docs = self.rerank_documents(query, documents, len(documents))
        
        # Then diversify
        diversified_docs = []
        similarity_matrix = self.calculate_similarity_matrix(reranked_docs)
        
        for i, doc in enumerate(reranked_docs):
            # Check if this document is too similar to already selected documents
            is_diverse = True
            
            for selected_doc in diversified_docs:
                selected_idx = reranked_docs.index(selected_doc)
                similarity = similarity_matrix[i, selected_idx]
                
                if similarity > diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diversified_docs.append(doc)
                
                if len(diversified_docs) >= top_k:
                    break
        
        return diversified_docs 