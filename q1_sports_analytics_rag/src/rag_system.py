from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import time

from .config import Config
from .models import (
    RAGResponse, QueryRequest, DocumentUpload, 
    QueryDecomposition, CompressedContext, Citation, DocumentChunk
)
from .vector_store import VectorStore
from .query_decomposition import QueryDecomposer
from .context_compression import ContextCompressor
from .reranker import Reranker
from .citation_extractor import CitationExtractor

class SportsAnalyticsRAG:
    """Main RAG system for sports analytics queries."""
    
    def __init__(self):
        """Initialize the RAG system."""
        self.config = Config()
        self.config.validate()
        
        # Initialize components
        self.vector_store = VectorStore()
        self.query_decomposer = QueryDecomposer()
        self.context_compressor = ContextCompressor()
        self.reranker = Reranker()
        self.citation_extractor = CitationExtractor()
        
        # Initialize LLM for answer generation
        self.llm = ChatOpenAI(
            model=self.config.OPENAI_MODEL,
            temperature=0.1
        )
        
        # Prompt template for answer generation
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert sports analytics assistant. Your task is to provide comprehensive, accurate answers to sports-related queries based on the provided context.

Guidelines:
1. Use only the information provided in the context
2. Be specific and include relevant statistics and facts
3. Structure your response logically and clearly
4. If information is not available in the context, acknowledge this
5. Focus on sports analytics, player performance, team statistics, and game insights
6. Be objective and factual in your analysis

Context: {context}

Query: {query}

Provide a comprehensive answer based on the context:""")
        ])
    
    def process_query(self, request: QueryRequest) -> RAGResponse:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            request: Query request with parameters
            
        Returns:
            RAGResponse with answer, citations, and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Query Decomposition
            decomposition = None
            if request.include_decomposition:
                decomposition = self.query_decomposer.decompose_query(request.query)
                sub_questions = decomposition.sub_questions
            else:
                # Create single sub-question for simple processing
                from .models import SubQuestion
                sub_questions = [SubQuestion(
                    question=request.query,
                    reasoning="Direct query processing",
                    priority=1
                )]
            
            # Step 2: Retrieve documents for each sub-question
            all_documents = []
            for sub_question in sub_questions:
                retrieval_result = self.vector_store.search(
                    sub_question.question,
                    top_k=request.max_results
                )
                all_documents.extend(retrieval_result.documents)
            
            # Remove duplicates based on chunk_id
            unique_documents = {}
            for doc in all_documents:
                if doc.chunk_id not in unique_documents:
                    unique_documents[doc.chunk_id] = doc
            
            documents = list(unique_documents.values())
            
            if not documents:
                return self._create_empty_response(request.query, start_time)
            
            # Step 3: Rerank documents
            reranked_documents = self.reranker.rerank_documents(
                request.query,
                documents,
                top_k=request.max_results
            )
            
            # Step 4: Context Compression
            compressed_context = self.context_compressor.compress_context(
                request.query,
                reranked_documents
            )
            
            # Step 5: Generate Answer
            answer = self._generate_answer(request.query, compressed_context.compressed_content)
            
            # Step 6: Extract Citations
            citations = []
            if request.include_citations:
                citations = self.citation_extractor.extract_citations(answer, reranked_documents)
                citations = self.citation_extractor.validate_citations(citations, reranked_documents)
            
            # Calculate processing time and confidence
            processing_time = time.time() - start_time
            confidence_score = self._calculate_confidence_score(
                compressed_context, citations, reranked_documents
            )
            
            return RAGResponse(
                query=request.query,
                answer=answer,
                citations=citations,
                sub_questions=sub_questions,
                compressed_context=compressed_context,
                processing_time=processing_time,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            # Handle errors gracefully
            processing_time = time.time() - start_time
            return RAGResponse(
                query=request.query,
                answer=f"I apologize, but I encountered an error while processing your query: {str(e)}. Please try again or rephrase your question.",
                citations=[],
                sub_questions=[],
                compressed_context=None,
                processing_time=processing_time,
                confidence_score=0.0
            )
    
    def add_documents(self, documents: List[DocumentUpload]) -> Dict[str, Any]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            Dictionary with status and count
        """
        try:
            # Convert to format expected by vector store
            docs_for_store = []
            for doc in documents:
                docs_for_store.append({
                    'content': doc.content,
                    'metadata': {
                        **doc.metadata,
                        'source': doc.source
                    }
                })
            
            count = self.vector_store.add_documents(docs_for_store)
            
            return {
                "status": "success",
                "documents_added": count,
                "total_documents": self.vector_store.get_document_count()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "documents_added": 0
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status information.
        
        Returns:
            Dictionary with system status
        """
        try:
            collection_info = self.vector_store.get_collection_info()
            
            return {
                "status": "operational",
                "total_documents": collection_info["count"],
                "collection_name": collection_info["name"],
                "version": "1.0.0"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "total_documents": 0,
                "version": "1.0.0"
            }
    
    def _generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer using LLM.
        
        Args:
            query: The original query
            context: Compressed context
            
        Returns:
            Generated answer
        """
        try:
            messages = self.answer_prompt.format_messages(
                query=query,
                context=context
            )
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            return f"I apologize, but I encountered an error while generating the answer: {str(e)}."
    
    def _create_empty_response(self, query: str, start_time: float) -> RAGResponse:
        """
        Create response when no documents are found.
        
        Args:
            query: The original query
            start_time: Start time for processing
            
        Returns:
            RAGResponse with empty result
        """
        processing_time = time.time() - start_time
        
        return RAGResponse(
            query=query,
            answer="I couldn't find any relevant information in my knowledge base to answer your question. Please try rephrasing your query or ask about a different topic.",
            citations=[],
            sub_questions=[],
            compressed_context=None,
            processing_time=processing_time,
            confidence_score=0.0
        )
    
    def _calculate_confidence_score(
        self, 
        compressed_context: CompressedContext,
        citations: List[Citation],
        documents: List[DocumentChunk]
    ) -> float:
        """
        Calculate overall confidence score for the response.
        
        Args:
            compressed_context: Compressed context
            citations: List of citations
            documents: Retrieved documents
            
        Returns:
            Confidence score between 0 and 1
        """
        score = 0.0
        
        # Base score from context relevance
        if compressed_context:
            score += compressed_context.relevance_score * 0.4
        
        # Score from citation quality
        if citations:
            avg_citation_confidence = sum(c.confidence for c in citations) / len(citations)
            score += avg_citation_confidence * 0.3
        
        # Score from document quality
        if documents:
            avg_doc_score = sum(d.score or 0.0 for d in documents) / len(documents)
            score += avg_doc_score * 0.3
        
        return min(score, 1.0) 