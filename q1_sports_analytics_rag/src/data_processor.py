from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import re

from .config import Config
from .models import DocumentUpload

class DataProcessor:
    """Processes and chunks documents for the vector store."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.config = Config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def process_documents(self, documents: List[DocumentUpload]) -> List[Dict[str, Any]]:
        """
        Process documents by chunking and adding metadata.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of processed document chunks
        """
        processed_chunks = []
        
        for doc in documents:
            # Clean and preprocess content
            cleaned_content = self._clean_content(doc.content)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(cleaned_content)
            
            # Create processed chunks with metadata
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Skip empty chunks
                    processed_chunk = {
                        'content': chunk.strip(),
                        'metadata': {
                            **doc.metadata,
                            'source': doc.source,
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'chunk_size': len(chunk)
                        }
                    }
                    processed_chunks.append(processed_chunk)
        
        return processed_chunks
    
    def _clean_content(self, content: str) -> str:
        """
        Clean and preprocess document content.
        
        Args:
            content: Raw document content
            
        Returns:
            Cleaned content
        """
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters that might cause issues
        content = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', content)
        
        # Normalize line breaks
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove multiple consecutive line breaks
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
    
    def extract_sports_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extract sports-related metadata from content.
        
        Args:
            content: Document content
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}
        
        # Extract team names
        team_patterns = [
            r'\b(Manchester United|Manchester City|Liverpool|Chelsea|Arsenal|Tottenham|Barcelona|Real Madrid|Bayern Munich|PSG)\b',
            r'\b(Team|Club)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        ]
        
        teams = []
        for pattern in team_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            teams.extend(matches)
        
        if teams:
            metadata['teams'] = list(set(teams))
        
        # Extract player names
        player_patterns = [
            r'\b(Messi|Ronaldo|Neymar|Mbappé|Haaland|De Bruyne|Salah|Kane|Benzema|Lewandowski)\b',
            r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'  # Simple name pattern
        ]
        
        players = []
        for pattern in player_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            players.extend(matches)
        
        if players:
            metadata['players'] = list(set(players))
        
        # Extract statistics
        stat_patterns = {
            'goals': r'\b(\d+)\s+goals?\b',
            'assists': r'\b(\d+)\s+assists?\b',
            'saves': r'\b(\d+)\s+saves?\b',
            'clean_sheets': r'\b(\d+)\s+clean\s+sheets?\b',
            'percentage': r'\b(\d+(?:\.\d+)?)\s*%\b',
            'matches': r'\b(\d+)\s+matches?\b',
            'minutes': r'\b(\d+)\s+minutes?\b'
        }
        
        for stat_name, pattern in stat_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                metadata[stat_name] = [float(match) for match in matches]
        
        # Extract seasons/years
        year_pattern = r'\b(20\d{2}|19\d{2})\b'
        years = re.findall(year_pattern, content)
        if years:
            metadata['years'] = list(set(years))
        
        # Extract leagues/competitions
        league_patterns = [
            r'\b(Premier League|La Liga|Bundesliga|Serie A|Ligue 1|Champions League|Europa League|World Cup|European Championship)\b'
        ]
        
        leagues = []
        for pattern in league_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            leagues.extend(matches)
        
        if leagues:
            metadata['leagues'] = list(set(leagues))
        
        return metadata
    
    def validate_document(self, document: DocumentUpload) -> bool:
        """
        Validate document before processing.
        
        Args:
            document: Document to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check if content is not empty
        if not document.content or not document.content.strip():
            return False
        
        # Check content length
        if len(document.content) < 10:
            return False
        
        # Check if source is provided
        if not document.source:
            return False
        
        return True
    
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about processed chunks.
        
        Args:
            chunks: List of processed chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
        
        chunk_sizes = [len(chunk['content']) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes)
        } 