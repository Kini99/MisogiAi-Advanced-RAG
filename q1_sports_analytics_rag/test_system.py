#!/usr/bin/env python3
"""
Test script for Sports Analytics RAG System

This script tests individual components and the complete system.
"""

import os
import sys
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from models import DocumentUpload, QueryRequest
from sample_data import SampleDataGenerator

def test_configuration():
    """Test configuration loading."""
    print("🔧 Testing configuration...")
    try:
        config = Config()
        print(f"✅ Configuration loaded successfully")
        print(f"   - OpenAI Model: {config.OPENAI_MODEL}")
        print(f"   - Embedding Model: {config.EMBEDDING_MODEL}")
        print(f"   - Chunk Size: {config.CHUNK_SIZE}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_sample_data():
    """Test sample data generation."""
    print("\n📊 Testing sample data...")
    try:
        documents = SampleDataGenerator.get_sample_documents()
        print(f"✅ Generated {len(documents)} sample documents")
        
        # Check document structure
        for i, doc in enumerate(documents, 1):
            if not doc.content or not doc.source:
                print(f"❌ Document {i} has missing content or source")
                return False
        
        print("✅ All documents have valid structure")
        return True
    except Exception as e:
        print(f"❌ Sample data error: {e}")
        return False

def test_models():
    """Test Pydantic models."""
    print("\n📋 Testing data models...")
    try:
        # Test DocumentUpload
        doc = DocumentUpload(
            content="Test content",
            metadata={"type": "test"},
            source="test_source"
        )
        print("✅ DocumentUpload model works")
        
        # Test QueryRequest
        query = QueryRequest(
            query="Test query",
            include_decomposition=True,
            include_citations=True,
            max_results=5
        )
        print("✅ QueryRequest model works")
        
        return True
    except Exception as e:
        print(f"❌ Model error: {e}")
        return False

def test_rag_system():
    """Test the complete RAG system."""
    print("\n🏈 Testing RAG system...")
    try:
        from rag_system import SportsAnalyticsRAG
        
        # Initialize system
        rag = SportsAnalyticsRAG()
        print("✅ RAG system initialized")
        
        # Add sample documents
        documents = SampleDataGenerator.get_sample_documents()
        result = rag.add_documents(documents)
        
        if result["status"] == "success":
            print(f"✅ Added {result['documents_added']} documents")
        else:
            print(f"❌ Failed to add documents: {result.get('error')}")
            return False
        
        # Test simple query
        request = QueryRequest(
            query="What are the top defensive teams?",
            include_decomposition=False,
            include_citations=False,
            max_results=3
        )
        
        response = rag.process_query(request)
        print(f"✅ Query processed successfully")
        print(f"   - Answer length: {len(response.answer)} characters")
        print(f"   - Processing time: {response.processing_time:.2f} seconds")
        
        return True
    except Exception as e:
        print(f"❌ RAG system error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Sports Analytics RAG System - Component Tests")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Sample Data", test_sample_data),
        ("Data Models", test_models),
        ("RAG System", test_rag_system)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 