from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List
import time

from .config import Config
from .models import DocumentChunk, CompressedContext

class ContextCompressor:
    """Compresses and filters retrieved context for relevance."""
    
    def __init__(self):
        """Initialize the context compressor."""
        self.config = Config()
        self.llm = ChatOpenAI(
            model=self.config.OPENAI_MODEL,
            temperature=0.1
        )
        
        # Prompt template for context compression
        self.compression_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at compressing and filtering sports analytics information for relevance.

Your task is to compress a collection of document chunks into a concise, relevant summary that directly addresses the given query.

Guidelines:
1. Focus only on information that directly answers the query
2. Remove irrelevant details, redundant information, and noise
3. Preserve key statistics, facts, and insights
4. Maintain accuracy and avoid introducing new information
5. Structure the compressed content logically
6. Include source references where possible

The compressed content should be:
- Concise but comprehensive
- Well-structured and easy to read
- Focused on the specific query
- Factual and accurate

Return only the compressed content without any additional formatting or explanations."""),
            ("human", """Query: {query}

Document chunks to compress:
{chunks}

Compress these chunks into relevant, concise information that answers the query:""")
        ])
    
    def compress_context(
        self, 
        query: str, 
        documents: List[DocumentChunk]
    ) -> CompressedContext:
        """
        Compress retrieved documents for relevance to the query.
        
        Args:
            query: The original query
            documents: List of retrieved document chunks
            
        Returns:
            CompressedContext with filtered and compressed content
        """
        if not documents:
            return CompressedContext(
                original_chunks=[],
                compressed_content="No relevant documents found.",
                compression_ratio=1.0,
                relevance_score=0.0
            )
        
        start_time = time.time()
        
        # Prepare document chunks for compression
        chunks_text = self._prepare_chunks_for_compression(documents)
        
        # Calculate original content length
        original_length = sum(len(doc.content) for doc in documents)
        
        try:
            # Compress using LLM
            messages = self.compression_prompt.format_messages(
                query=query,
                chunks=chunks_text
            )
            response = self.llm.invoke(messages)
            compressed_content = response.content.strip()
            
            # Calculate compression metrics
            compressed_length = len(compressed_content)
            compression_ratio = compressed_length / original_length if original_length > 0 else 1.0
            
            # Calculate relevance score based on compression quality
            relevance_score = self._calculate_relevance_score(
                query, compressed_content, documents
            )
            
            return CompressedContext(
                original_chunks=documents,
                compressed_content=compressed_content,
                compression_ratio=compression_ratio,
                relevance_score=relevance_score
            )
            
        except Exception as e:
            # Fallback: simple concatenation with basic filtering
            fallback_content = self._fallback_compression(query, documents)
            
            return CompressedContext(
                original_chunks=documents,
                compressed_content=fallback_content,
                compression_ratio=0.8,  # Conservative estimate
                relevance_score=0.6    # Conservative estimate
            )
    
    def _prepare_chunks_for_compression(self, documents: List[DocumentChunk]) -> str:
        """
        Prepare document chunks for compression.
        
        Args:
            documents: List of document chunks
            
        Returns:
            Formatted string of chunks
        """
        chunks_text = []
        
        for i, doc in enumerate(documents, 1):
            chunk_text = f"Chunk {i} (Source: {doc.source}):\n{doc.content}\n"
            if doc.metadata:
                metadata_str = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
                chunk_text += f"Metadata: {metadata_str}\n"
            chunks_text.append(chunk_text)
        
        return "\n".join(chunks_text)
    
    def _calculate_relevance_score(
        self, 
        query: str, 
        compressed_content: str, 
        documents: List[DocumentChunk]
    ) -> float:
        """
        Calculate relevance score for compressed content.
        
        Args:
            query: Original query
            compressed_content: Compressed content
            documents: Original documents
            
        Returns:
            Relevance score between 0 and 1
        """
        # Simple heuristic-based scoring
        score = 0.0
        
        # Check if compressed content contains query keywords
        query_words = set(query.lower().split())
        content_words = set(compressed_content.lower().split())
        
        # Keyword overlap
        keyword_overlap = len(query_words.intersection(content_words))
        if len(query_words) > 0:
            score += min(keyword_overlap / len(query_words), 1.0) * 0.4
        
        # Check for sports-related terms in compressed content
        sports_terms = [
            'goal', 'assist', 'save', 'defense', 'attack', 'midfield',
            'clean sheet', 'possession', 'pass', 'shot', 'tackle',
            'player', 'team', 'league', 'season', 'match', 'game',
            'statistics', 'performance', 'rating', 'percentage'
        ]
        
        sports_term_count = sum(1 for term in sports_terms if term in compressed_content.lower())
        score += min(sports_term_count / 10, 1.0) * 0.3
        
        # Check compression quality (not too short, not too long)
        content_length = len(compressed_content)
        if 100 <= content_length <= 2000:
            score += 0.3
        elif 50 <= content_length <= 3000:
            score += 0.2
        else:
            score += 0.1
        
        return min(score, 1.0)
    
    def _fallback_compression(self, query: str, documents: List[DocumentChunk]) -> str:
        """
        Fallback compression method when LLM compression fails.
        
        Args:
            query: Original query
            documents: Document chunks
            
        Returns:
            Compressed content
        """
        # Simple keyword-based filtering
        query_words = set(query.lower().split())
        
        relevant_sentences = []
        
        for doc in documents:
            sentences = doc.content.split('.')
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                # Check if sentence contains query keywords
                if query_words.intersection(sentence_words):
                    relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return '. '.join(relevant_sentences[:10]) + '.'
        else:
            # If no keyword matches, return first few sentences from each document
            all_sentences = []
            for doc in documents:
                sentences = doc.content.split('.')
                all_sentences.extend(sentences[:3])  # First 3 sentences from each doc
            
            return '. '.join(all_sentences[:15]) + '.' 