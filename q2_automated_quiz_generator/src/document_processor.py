"""Document processing with dynamic chunk sizing and context compression."""

import os
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import pypdf
from docx import Document
import markdown
from bs4 import BeautifulSoup
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument

from .models import DocumentChunk
from .config import settings


class DocumentProcessor:
    """Document processing with dynamic chunk sizing."""
    
    def __init__(self):
        """Initialize document processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _calculate_optimal_chunk_size(self, content: str, topic: str) -> int:
        """Calculate optimal chunk size based on content characteristics."""
        # Base chunk size
        base_size = settings.chunk_size
        
        # Adjust based on content length
        content_length = len(content)
        if content_length < 5000:
            return min(base_size, 500)
        elif content_length > 50000:
            return min(base_size * 2, 2000)
        
        # Adjust based on topic complexity
        complexity_keywords = {
            "mathematics": 0.8,
            "physics": 0.8,
            "chemistry": 0.9,
            "biology": 0.9,
            "history": 1.1,
            "literature": 1.2,
            "philosophy": 1.3
        }
        
        multiplier = 1.0
        for keyword, factor in complexity_keywords.items():
            if keyword.lower() in topic.lower():
                multiplier = factor
                break
        
        return int(base_size * multiplier)
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from DOCX {file_path}: {e}")
            return ""
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error extracting text from TXT {file_path}: {e}")
            return ""
    
    def _extract_text_from_md(self, file_path: str) -> str:
        """Extract text from Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                # Convert markdown to HTML, then extract text
                html = markdown.markdown(md_content)
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text()
        except Exception as e:
            print(f"Error extracting text from MD {file_path}: {e}")
            return ""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from various file formats."""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self._extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return self._extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            return self._extract_text_from_txt(file_path)
        elif file_extension == '.md':
            return self._extract_text_from_md(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]]', '', text)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
    
    def _extract_metadata(self, file_path: str, topic: str, instructor_id: str) -> Dict[str, Any]:
        """Extract metadata from document."""
        file_info = Path(file_path)
        
        return {
            "filename": file_info.name,
            "file_path": str(file_path),
            "file_size": file_info.stat().st_size,
            "file_extension": file_info.suffix.lower(),
            "topic": topic,
            "instructor_id": instructor_id,
            "upload_timestamp": str(Path(file_path).stat().st_mtime),
            "content_hash": self._generate_content_hash(file_path)
        }
    
    def _generate_content_hash(self, file_path: str) -> str:
        """Generate hash for file content."""
        try:
            with open(file_path, 'rb') as file:
                content = file.read()
                return hashlib.md5(content).hexdigest()
        except Exception:
            return ""
    
    def _compress_context(self, chunks: List[str], max_length: int = 4000) -> List[str]:
        """Compress context by removing redundant information."""
        compressed_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            # If adding this chunk would exceed max length, start a new chunk
            if len(current_chunk) + len(chunk) > max_length:
                if current_chunk:
                    compressed_chunks.append(current_chunk.strip())
                current_chunk = chunk
            else:
                current_chunk += " " + chunk
        
        # Add the last chunk
        if current_chunk:
            compressed_chunks.append(current_chunk.strip())
        
        return compressed_chunks
    
    def process_document(self, file_path: str, topic: str, instructor_id: str) -> List[DocumentChunk]:
        """Process document and return chunks."""
        # Extract text
        raw_text = self.extract_text(file_path)
        if not raw_text:
            return []
        
        # Clean text
        cleaned_text = self._clean_text(raw_text)
        
        # Calculate optimal chunk size
        optimal_chunk_size = self._calculate_optimal_chunk_size(cleaned_text, topic)
        
        # Create temporary text splitter with optimal size
        temp_splitter = RecursiveCharacterTextSplitter(
            chunk_size=optimal_chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split text into chunks
        langchain_docs = temp_splitter.split_text(cleaned_text)
        
        # Compress context
        compressed_chunks = self._compress_context(langchain_docs)
        
        # Extract metadata
        metadata = self._extract_metadata(file_path, topic, instructor_id)
        
        # Create DocumentChunk objects
        document_chunks = []
        for i, chunk_content in enumerate(compressed_chunks):
            chunk_id = f"{metadata['content_hash']}_{i}"
            
            chunk = DocumentChunk(
                id=chunk_id,
                content=chunk_content,
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(compressed_chunks),
                    "chunk_size": len(chunk_content)
                }
            )
            document_chunks.append(chunk)
        
        return document_chunks
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if file can be processed."""
        if not os.path.exists(file_path):
            return False
        
        file_extension = Path(file_path).suffix.lower()
        if file_extension not in settings.allowed_extensions:
            return False
        
        file_size = Path(file_path).stat().st_size
        if file_size > settings.max_file_size:
            return False
        
        return True


# Global document processor instance
document_processor = DocumentProcessor() 