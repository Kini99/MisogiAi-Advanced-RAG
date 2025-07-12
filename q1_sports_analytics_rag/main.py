#!/usr/bin/env python3
"""
Sports Analytics RAG System - Main Application

This script demonstrates the complete RAG system with sample data and queries.
"""

import asyncio
import json
from typing import List
import uvicorn
from fastapi import FastAPI

from src.config import Config
from src.rag_system import SportsAnalyticsRAG
from src.sample_data import SampleDataGenerator
from src.models import QueryRequest, DocumentUpload

def setup_sample_data(rag_system: SportsAnalyticsRAG) -> None:
    """Set up sample data in the RAG system."""
    print("Setting up sample sports analytics data...")
    
    # Get sample documents
    sample_documents = SampleDataGenerator.get_sample_documents()
    
    # Add documents to the system
    result = rag_system.add_documents(sample_documents)
    
    if result["status"] == "success":
        print(f"✅ Successfully added {result['documents_added']} documents")
        print(f"📊 Total documents in system: {result['total_documents']}")
    else:
        print(f"❌ Error adding documents: {result.get('error', 'Unknown error')}")

def run_sample_queries(rag_system: SportsAnalyticsRAG) -> None:
    """Run sample queries to demonstrate the RAG system."""
    print("\n" + "="*60)
    print("RUNNING SAMPLE QUERIES")
    print("="*60)
    
    # Sample queries from requirements
    sample_queries = [
        "What are the top 3 teams in defense and their key defensive statistics?",
        "Compare Messi's goal-scoring rate in the last season vs previous seasons",
        "Which goalkeeper has the best save percentage in high-pressure situations?",
        "Which team has the best defense and how does their goalkeeper compare to the league average?",
        "What are Haaland's goal-scoring statistics compared to Mbappé?",
        "How did Manchester City perform in the Champions League knockout stages?"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 50)
        
        try:
            # Create query request
            request = QueryRequest(
                query=query,
                include_decomposition=True,
                include_citations=True,
                max_results=5
            )
            
            # Process query
            response = rag_system.process_query(request)
            
            # Display results
            print(f"📝 Answer: {response.answer}")
            print(f"⏱️  Processing time: {response.processing_time:.2f} seconds")
            print(f"🎯 Confidence score: {response.confidence_score:.2f}")
            
            # Display sub-questions if decomposition was used
            if response.sub_questions and len(response.sub_questions) > 1:
                print(f"\n🔧 Query Decomposition ({len(response.sub_questions)} sub-questions):")
                for j, sub_q in enumerate(response.sub_questions, 1):
                    print(f"  {j}. {sub_q.question}")
                    print(f"     Reasoning: {sub_q.reasoning}")
            
            # Display citations
            if response.citations:
                print(f"\n📚 Citations ({len(response.citations)}):")
                for j, citation in enumerate(response.citations, 1):
                    print(f"  [{j}] {citation.claim}")
                    print(f"      Source: {citation.source}")
                    if citation.page_section:
                        print(f"      Section: {citation.page_section}")
                    print(f"      Confidence: {citation.confidence:.2f}")
            
            # Display context compression info
            if response.compressed_context:
                print(f"\n📦 Context Compression:")
                print(f"   Compression ratio: {response.compressed_context.compression_ratio:.2f}")
                print(f"   Relevance score: {response.compressed_context.relevance_score:.2f}")
            
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
        
        print("\n" + "="*60)

def run_api_server() -> None:
    """Run the FastAPI server."""
    print("🚀 Starting Sports Analytics RAG API Server...")
    
    config = Config()
    uvicorn.run(
        "src.api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        log_level="info"
    )

def main():
    """Main function to run the Sports Analytics RAG system."""
    print("🏈 Sports Analytics RAG System")
    print("=" * 50)
    
    try:
        # Initialize configuration
        config = Config()
        config.validate()
        print("✅ Configuration validated")
        
        # Initialize RAG system
        print("🔧 Initializing RAG system...")
        rag_system = SportsAnalyticsRAG()
        print("✅ RAG system initialized")
        
        # Set up sample data
        setup_sample_data(rag_system)
        
        # Get system status
        status = rag_system.get_system_status()
        print(f"\n📊 System Status: {status['status']}")
        print(f"📚 Total Documents: {status['total_documents']}")
        print(f"🔧 Version: {status['version']}")
        
        # Run sample queries
        run_sample_queries(rag_system)
        
        print("\n🎉 Demo completed successfully!")
        print("\nTo start the API server, run: python main.py --api")
        print("To run the demo again, run: python main.py")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nMake sure you have:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Installed all dependencies: pip install -r requirements.txt")
        print("3. Python 3.11 or higher")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        run_api_server()
    else:
        main() 