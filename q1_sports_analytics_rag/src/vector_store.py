import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import uuid
import time

from .config import Config
from .models import DocumentChunk, RetrievalResult

class VectorStore:
    """Vector store implementation using ChromaDB."""
    
    def __init__(self):
        """Initialize the vector store."""
        self.config = Config()
        self.client = chromadb.PersistentClient(
            path=self.config.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            metadata={"description": "Sports analytics documents"}
        )
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents with 'content' and 'metadata' keys
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            texts.append(doc['content'])
            metadatas.append(doc['metadata'])
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        return len(documents)
    
    def search(
        self, 
        query: str, 
        top_k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            RetrievalResult with documents and metadata
        """
        start_time = time.time()
        
        if top_k is None:
            top_k = self.config.TOP_K_RETRIEVAL
        
        # Perform search
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_metadata
        )
        
        # Convert to DocumentChunk objects
        documents = []
        if results['documents'] and results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # Convert distance to similarity score (1 - distance)
                similarity_score = 1 - distance if distance is not None else None
                
                document_chunk = DocumentChunk(
                    content=doc,
                    metadata=metadata or {},
                    source=metadata.get('source', 'unknown') if metadata else 'unknown',
                    chunk_id=results['ids'][0][i],
                    score=similarity_score
                )
                documents.append(document_chunk)
        
        retrieval_time = time.time() - start_time
        
        return RetrievalResult(
            query=query,
            documents=documents,
            total_retrieved=len(documents),
            retrieval_time=retrieval_time
        )
    
    def get_document_count(self) -> int:
        """Get total number of documents in the collection."""
        return self.collection.count()
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(self.config.COLLECTION_NAME)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata
        } 