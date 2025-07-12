from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import re
import json

from .config import Config
from .models import Citation, DocumentChunk

class CitationExtractor:
    """Extracts and formats citations from retrieved documents."""
    
    def __init__(self):
        """Initialize the citation extractor."""
        self.config = Config()
        self.llm = ChatOpenAI(
            model=self.config.OPENAI_MODEL,
            temperature=0.1
        )
        
        # Prompt template for citation extraction
        self.citation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting and formatting citations from sports analytics documents.

Your task is to identify specific claims in the answer and link them to supporting evidence from the provided documents.

Guidelines:
1. Identify factual claims, statistics, and insights in the answer
2. For each claim, find the most relevant supporting document
3. Extract specific information that supports the claim
4. Provide confidence scores based on the strength of evidence
5. Include page/section references if available in metadata
6. Focus on sports statistics, player performance, team data, and game insights

Return your response as a JSON object with this structure:
{
    "citations": [
        {
            "claim": "specific claim from the answer",
            "source": "document source identifier",
            "page_section": "page or section reference if available",
            "confidence": 0.95
        }
    ]
}

Example:
Answer: "Manchester City has the best defensive record with only 23 goals conceded this season."

Response:
{
    "citations": [
        {
            "claim": "Manchester City has the best defensive record with only 23 goals conceded this season",
            "source": "premier_league_stats_2024",
            "page_section": "Defensive Statistics",
            "confidence": 0.95
        }
    ]
}"""),
            ("human", """Answer: {answer}

Supporting documents:
{documents}

Extract citations for the claims in the answer:""")
        ])
    
    def extract_citations(
        self, 
        answer: str, 
        documents: List[DocumentChunk]
    ) -> List[Citation]:
        """
        Extract citations from the answer based on supporting documents.
        
        Args:
            answer: The generated answer
            documents: Supporting document chunks
            
        Returns:
            List of citations
        """
        if not documents or not answer.strip():
            return []
        
        try:
            # Prepare documents for citation extraction
            documents_text = self._prepare_documents_for_citation(documents)
            
            # Extract citations using LLM
            messages = self.citation_prompt.format_messages(
                answer=answer,
                documents=documents_text
            )
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            citation_data = json.loads(response.content)
            
            # Convert to Citation objects
            citations = []
            for cit_data in citation_data.get("citations", []):
                citation = Citation(
                    claim=cit_data["claim"],
                    source=cit_data["source"],
                    page_section=cit_data.get("page_section"),
                    confidence=cit_data["confidence"]
                )
                citations.append(citation)
            
            return citations
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            # Fallback: simple keyword-based citation extraction
            return self._fallback_citation_extraction(answer, documents)
    
    def _prepare_documents_for_citation(self, documents: List[DocumentChunk]) -> str:
        """
        Prepare documents for citation extraction.
        
        Args:
            documents: List of document chunks
            
        Returns:
            Formatted string of documents
        """
        docs_text = []
        
        for i, doc in enumerate(documents, 1):
            doc_text = f"Document {i} (Source: {doc.source}):\n{doc.content}\n"
            if doc.metadata:
                metadata_str = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
                doc_text += f"Metadata: {metadata_str}\n"
            docs_text.append(doc_text)
        
        return "\n".join(docs_text)
    
    def _fallback_citation_extraction(
        self, 
        answer: str, 
        documents: List[DocumentChunk]
    ) -> List[Citation]:
        """
        Fallback citation extraction using keyword matching.
        
        Args:
            answer: The generated answer
            documents: Supporting document chunks
            
        Returns:
            List of citations
        """
        citations = []
        
        # Extract sentences from answer
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Sports-related keywords that indicate factual claims
        claim_keywords = [
            'scored', 'assisted', 'saved', 'conceded', 'won', 'lost', 'drew',
            'percentage', 'average', 'total', 'record', 'best', 'worst',
            'goals', 'assists', 'saves', 'clean sheets', 'possession',
            'passes', 'tackles', 'shots', 'minutes', 'matches', 'games'
        ]
        
        for sentence in sentences:
            # Check if sentence contains claim keywords
            sentence_lower = sentence.lower()
            has_claim_keywords = any(keyword in sentence_lower for keyword in claim_keywords)
            
            if has_claim_keywords:
                # Find the most relevant document for this sentence
                best_doc = self._find_best_document_for_sentence(sentence, documents)
                
                if best_doc:
                    citation = Citation(
                        claim=sentence,
                        source=best_doc.source,
                        page_section=best_doc.metadata.get('page', best_doc.metadata.get('section')),
                        confidence=0.7  # Conservative confidence for fallback
                    )
                    citations.append(citation)
        
        return citations
    
    def _find_best_document_for_sentence(
        self, 
        sentence: str, 
        documents: List[DocumentChunk]
    ) -> DocumentChunk:
        """
        Find the best document for a given sentence using keyword overlap.
        
        Args:
            sentence: The sentence to find support for
            documents: List of document chunks
            
        Returns:
            Best matching document chunk
        """
        sentence_words = set(sentence.lower().split())
        best_doc = None
        best_overlap = 0
        
        for doc in documents:
            doc_words = set(doc.content.lower().split())
            overlap = len(sentence_words.intersection(doc_words))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_doc = doc
        
        return best_doc
    
    def format_citations_for_display(self, citations: List[Citation]) -> str:
        """
        Format citations for display in the response.
        
        Args:
            citations: List of citations
            
        Returns:
            Formatted citation string
        """
        if not citations:
            return ""
        
        citation_texts = []
        for i, citation in enumerate(citations, 1):
            citation_text = f"[{i}] {citation.claim}"
            if citation.page_section:
                citation_text += f" (Source: {citation.source}, {citation.page_section})"
            else:
                citation_text += f" (Source: {citation.source})"
            citation_texts.append(citation_text)
        
        return "\n\nCitations:\n" + "\n".join(citation_texts)
    
    def validate_citations(
        self, 
        citations: List[Citation], 
        documents: List[DocumentChunk]
    ) -> List[Citation]:
        """
        Validate citations against available documents.
        
        Args:
            citations: List of citations to validate
            documents: Available document chunks
            
        Returns:
            Validated list of citations
        """
        valid_citations = []
        available_sources = {doc.source for doc in documents}
        
        for citation in citations:
            # Check if source exists
            if citation.source in available_sources:
                # Validate confidence score
                if 0 <= citation.confidence <= 1:
                    valid_citations.append(citation)
                else:
                    # Fix confidence score
                    citation.confidence = max(0, min(1, citation.confidence))
                    valid_citations.append(citation)
            else:
                # Try to find alternative source
                alternative_source = self._find_alternative_source(citation, documents)
                if alternative_source:
                    citation.source = alternative_source
                    citation.confidence *= 0.8  # Reduce confidence for alternative source
                    valid_citations.append(citation)
        
        return valid_citations
    
    def _find_alternative_source(
        self, 
        citation: Citation, 
        documents: List[DocumentChunk]
    ) -> str:
        """
        Find alternative source for a citation.
        
        Args:
            citation: The citation to find alternative for
            documents: Available document chunks
            
        Returns:
            Alternative source identifier or None
        """
        # Simple heuristic: find document with most keyword overlap
        claim_words = set(citation.claim.lower().split())
        best_source = None
        best_overlap = 0
        
        for doc in documents:
            doc_words = set(doc.content.lower().split())
            overlap = len(claim_words.intersection(doc_words))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_source = doc.source
        
        return best_source if best_overlap > 0 else None 