"""Basic tests for the Advanced Assessment Generation System."""

import pytest
import os
from pathlib import Path

from src.config import settings
from src.models import DifficultyLevel, QuestionType
from src.document_processor import document_processor


def test_config_loading():
    """Test that configuration loads correctly."""
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200


def test_models():
    """Test that models work correctly."""
    assert DifficultyLevel.EASY.value == "easy"
    assert DifficultyLevel.MEDIUM.value == "medium"
    assert DifficultyLevel.HARD.value == "hard"
    
    assert QuestionType.MULTIPLE_CHOICE.value == "multiple_choice"
    assert QuestionType.TRUE_FALSE.value == "true_false"
    assert QuestionType.SHORT_ANSWER.value == "short_answer"
    assert QuestionType.ESSAY.value == "essay"


def test_document_processor_validation():
    """Test document processor validation."""
    # Test with non-existent file
    assert not document_processor.validate_file("non_existent_file.txt")
    
    # Test with valid file
    test_file = Path("test_sample.txt")
    test_content = "This is a test document for validation."
    
    try:
        with open(test_file, "w") as f:
            f.write(test_content)
        
        assert document_processor.validate_file(str(test_file))
        
        # Test with oversized file (create a large file)
        large_file = Path("test_large.txt")
        large_content = "x" * (settings.max_file_size + 1000)
        
        with open(large_file, "w") as f:
            f.write(large_content)
        
        assert not document_processor.validate_file(str(large_file))
        
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()
        if large_file.exists():
            large_file.unlink()


def test_document_processing():
    """Test document processing functionality."""
    test_file = Path("test_processing.txt")
    test_content = """
    Introduction to Python Programming
    
    Python is a high-level, interpreted programming language known for its simplicity and readability.
    It was created by Guido van Rossum and first released in 1991.
    
    Key Features:
    - Easy to learn and use
    - Extensive standard library
    - Cross-platform compatibility
    - Strong community support
    
    Python is widely used in:
    - Web development
    - Data science and machine learning
    - Scientific computing
    - Automation and scripting
    """
    
    try:
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)
        
        # Process document
        chunks = document_processor.process_document(
            str(test_file), 
            topic="python_programming", 
            instructor_id="test_instructor"
        )
        
        assert len(chunks) > 0
        assert all(chunk.content for chunk in chunks)
        assert all(chunk.metadata for chunk in chunks)
        
        # Check metadata
        first_chunk = chunks[0]
        assert first_chunk.metadata["topic"] == "python_programming"
        assert first_chunk.metadata["instructor_id"] == "test_instructor"
        assert first_chunk.metadata["filename"] == "test_processing.txt"
        
    finally:
        if test_file.exists():
            test_file.unlink()


def test_directory_creation():
    """Test that required directories are created."""
    assert os.path.exists(settings.upload_dir)
    assert os.path.exists(settings.chroma_persist_directory)


if __name__ == "__main__":
    # Run basic tests
    print("Running basic tests...")
    
    test_config_loading()
    print("✅ Configuration loading test passed")
    
    test_models()
    print("✅ Models test passed")
    
    test_document_processor_validation()
    print("✅ Document processor validation test passed")
    
    test_document_processing()
    print("✅ Document processing test passed")
    
    test_directory_creation()
    print("✅ Directory creation test passed")
    
    print("\n🎉 All basic tests passed!") 