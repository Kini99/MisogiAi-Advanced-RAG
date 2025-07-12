"""
Integration tests for Strategic Decision Engine.
These tests verify the complete workflow from document upload to analysis.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os
from fastapi.testclient import TestClient
from io import BytesIO


class TestDocumentToAnalysisWorkflow:
    """Test the complete workflow from document upload to analysis."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, test_client):
        """Test complete workflow: upload document -> process -> analyze -> evaluate."""
        
        # Step 1: Upload document
        test_document = b"Strategic Planning Document\n\nOur company needs to analyze market opportunities..."
        
        with patch('backend.services.document_service.DocumentService.process_document') as mock_process:
            mock_process.return_value = {
                'document_id': 'doc-123',
                'content': 'Strategic Planning Document\n\nOur company needs to analyze market opportunities...',
                'metadata': {'title': 'Strategic Plan', 'pages': 1}
            }
            
            response = test_client.post(
                "/api/documents/upload",
                files={"file": ("strategy.txt", BytesIO(test_document), "text/plain")}
            )
        
        assert response.status_code == 200
        upload_data = response.json()
        document_id = upload_data['document_id']
        
        # Step 2: Process document for vector storage
        with patch('backend.services.vector_store_service.VectorStoreService.add_documents') as mock_add:
            mock_add.return_value = True
            
            response = test_client.post(f"/api/documents/{document_id}/process")
        
        assert response.status_code == 200
        
        # Step 3: Generate strategic analysis
        with patch('backend.services.llm_service.LLMService.generate_response') as mock_llm:
            mock_llm.return_value = """
            Strategic Analysis:
            
            Market Opportunities:
            1. AI technology adoption is accelerating
            2. Digital transformation initiatives are increasing
            3. Strategic partnerships are available
            
            Recommendations:
            1. Invest in AI capabilities
            2. Develop strategic partnerships
            3. Focus on digital transformation solutions
            """
            
            response = test_client.post(
                "/api/analysis/generate",
                json={
                    "content": "Analyze strategic opportunities from the uploaded document",
                    "analysis_type": "strategic",
                    "document_ids": [document_id]
                }
            )
        
        assert response.status_code == 200
        analysis_data = response.json()
        assert "Strategic Analysis" in analysis_data['analysis']
        
        # Step 4: Evaluate the analysis quality
        with patch('backend.api.endpoints.evaluation.evaluate_faithfulness') as mock_eval:
            mock_eval.return_value = 0.85
            
            response = test_client.post(
                "/api/evaluation/evaluate",
                json={
                    "question": "What are the strategic opportunities?",
                    "answer": analysis_data['analysis'],
                    "contexts": ["Strategic Planning Document content"],
                    "ground_truth": "AI, partnerships, and digital transformation are key opportunities"
                }
            )
        
        assert response.status_code == 200
        evaluation_data = response.json()
        assert evaluation_data['evaluation_results']['faithfulness'] == 0.85

    @pytest.mark.asyncio
    async def test_multi_document_analysis(self, test_client):
        """Test analysis with multiple documents."""
        
        # Upload multiple documents
        documents = [
            (b"Market Research: AI market is growing at 25% annually", "market.txt"),
            (b"Financial Report: Revenue increased 30% last quarter", "financial.txt"),
            (b"Competitive Analysis: Main competitors are expanding", "competitive.txt")
        ]
        
        document_ids = []
        
        for doc_content, filename in documents:
            with patch('backend.services.document_service.DocumentService.process_document') as mock_process:
                mock_process.return_value = {
                    'document_id': f'doc-{len(document_ids) + 1}',
                    'content': doc_content.decode(),
                    'metadata': {'title': filename, 'pages': 1}
                }
                
                response = test_client.post(
                    "/api/documents/upload",
                    files={"file": (filename, BytesIO(doc_content), "text/plain")}
                )
            
            assert response.status_code == 200
            document_ids.append(response.json()['document_id'])
        
        # Generate comprehensive analysis
        with patch('backend.services.llm_service.LLMService.generate_response') as mock_llm:
            mock_llm.return_value = """
            Comprehensive Analysis:
            
            Market Analysis: Strong growth trajectory with 25% annual growth
            Financial Performance: Positive with 30% revenue increase
            Competitive Position: Need to respond to competitor expansion
            
            Strategic Recommendations:
            1. Capitalize on market growth
            2. Maintain financial momentum
            3. Strengthen competitive position
            """
            
            response = test_client.post(
                "/api/analysis/generate",
                json={
                    "content": "Provide comprehensive analysis based on all documents",
                    "analysis_type": "comprehensive",
                    "document_ids": document_ids
                }
            )
        
        assert response.status_code == 200
        analysis_data = response.json()
        assert "Comprehensive Analysis" in analysis_data['analysis']
        assert "Market Analysis" in analysis_data['analysis']
        assert "Financial Performance" in analysis_data['analysis']

    @pytest.mark.asyncio
    async def test_chat_with_rag_integration(self, test_client):
        """Test chat functionality with RAG integration."""
        
        # Setup: Upload and process a document
        with patch('backend.services.document_service.DocumentService.process_document') as mock_process:
            mock_process.return_value = {
                'document_id': 'doc-rag-test',
                'content': 'Company strategic plan focuses on AI development and market expansion',
                'metadata': {'title': 'Strategic Plan', 'pages': 1}
            }
            
            response = test_client.post(
                "/api/documents/upload",
                files={"file": ("strategy.txt", BytesIO(b"Strategic content"), "text/plain")}
            )
        
        document_id = response.json()['document_id']
        
        # Process for vector storage
        with patch('backend.services.vector_store_service.VectorStoreService.add_documents') as mock_add:
            mock_add.return_value = True
            test_client.post(f"/api/documents/{document_id}/process")
        
        # Start chat session
        session_response = test_client.post(
            "/api/chat/sessions",
            json={"title": "Strategic Discussion"}
        )
        session_id = session_response.json()['session_id']
        
        # Chat with RAG-enhanced responses
        with patch('backend.services.vector_store_service.VectorStoreService.search_documents') as mock_search:
            mock_search.return_value = [
                {
                    'content': 'Company strategic plan focuses on AI development and market expansion',
                    'metadata': {'source': 'Strategic Plan', 'page': 1}
                }
            ]
            
            with patch('backend.services.llm_service.LLMService.generate_response') as mock_llm:
                mock_llm.return_value = """
                Based on your strategic plan document, your company is focused on AI development and market expansion. 
                Here are the key strategic priorities:
                
                1. AI Development: Investing in artificial intelligence capabilities
                2. Market Expansion: Growing presence in new markets
                3. Strategic Positioning: Leveraging AI for competitive advantage
                
                Source: Strategic Plan, Page 1
                """
                
                response = test_client.post(
                    "/api/chat/message",
                    json={
                        "message": "What are our strategic priorities?",
                        "session_id": session_id,
                        "use_rag": True
                    }
                )
        
        assert response.status_code == 200
        chat_data = response.json()
        assert "AI development" in chat_data['response']
        assert "market expansion" in chat_data['response']
        assert "Source: Strategic Plan" in chat_data['response']

    @pytest.mark.asyncio
    async def test_analysis_export_workflow(self, test_client):
        """Test analysis generation and export workflow."""
        
        # Generate analysis
        with patch('backend.services.llm_service.LLMService.generate_response') as mock_llm:
            mock_llm.return_value = "Detailed strategic analysis with recommendations"
            
            response = test_client.post(
                "/api/analysis/generate",
                json={
                    "content": "Analyze our strategic position",
                    "analysis_type": "strategic"
                }
            )
        
        assert response.status_code == 200
        analysis_data = response.json()
        analysis_id = analysis_data['analysis_id']
        
        # Save analysis
        with patch('backend.api.endpoints.analysis.get_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            response = test_client.post(
                "/api/analysis/save",
                json={
                    "analysis_id": analysis_id,
                    "session_id": "test-session",
                    "title": "Strategic Analysis Report"
                }
            )
        
        assert response.status_code == 200
        
        # Export analysis as PDF
        with patch('backend.api.endpoints.analysis.get_db') as mock_db:
            mock_session = Mock()
            mock_analysis = Mock()
            mock_analysis.content = "Detailed strategic analysis with recommendations"
            mock_analysis.analysis_type = "strategic"
            mock_analysis.title = "Strategic Analysis Report"
            mock_session.query.return_value.filter.return_value.first.return_value = mock_analysis
            mock_db.return_value = mock_session
            
            response = test_client.get(f"/api/analysis/export/{analysis_id}?format=pdf")
        
        assert response.status_code == 200
        assert "application/pdf" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_evaluation_batch_workflow(self, test_client):
        """Test batch evaluation workflow."""
        
        # Generate multiple analyses
        analyses = []
        for i in range(3):
            with patch('backend.services.llm_service.LLMService.generate_response') as mock_llm:
                mock_llm.return_value = f"Analysis {i+1}: Strategic recommendations for growth"
                
                response = test_client.post(
                    "/api/analysis/generate",
                    json={
                        "content": f"Strategic analysis {i+1}",
                        "analysis_type": "strategic"
                    }
                )
            
            analyses.append(response.json())
        
        # Batch evaluate all analyses
        evaluation_requests = []
        for i, analysis in enumerate(analyses):
            evaluation_requests.append({
                "question": f"What are the strategic recommendations {i+1}?",
                "answer": analysis['analysis'],
                "contexts": [f"Strategic context {i+1}"],
                "ground_truth": f"Growth recommendations {i+1}"
            })
        
        with patch('backend.api.endpoints.evaluation.evaluate_faithfulness') as mock_eval:
            mock_eval.return_value = 0.85
            
            response = test_client.post(
                "/api/evaluation/batch",
                json={"evaluations": evaluation_requests}
            )
        
        assert response.status_code == 200
        batch_data = response.json()
        assert len(batch_data['batch_results']) == 3
        
        # Generate evaluation report
        with patch('backend.api.endpoints.evaluation.get_db') as mock_db:
            mock_session = Mock()
            mock_query = Mock()
            mock_results = [
                Mock(
                    evaluation_id=f"eval-{i}",
                    faithfulness=0.85,
                    answer_relevancy=0.88,
                    context_precision=0.92,
                    context_recall=0.87,
                    answer_correctness=0.91,
                    created_at="2024-01-01T00:00:00Z"
                ) for i in range(3)
            ]
            mock_query.filter.return_value.all.return_value = mock_results
            mock_session.query.return_value = mock_query
            mock_db.return_value = mock_session
            
            response = test_client.get("/api/evaluation/report/test-session")
        
        assert response.status_code == 200
        report_data = response.json()
        assert report_data['summary']['total_evaluations'] == 3
        assert 'average_scores' in report_data['summary']

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, test_client):
        """Test error handling in various workflow scenarios."""
        
        # Test document upload with unsupported format
        response = test_client.post(
            "/api/documents/upload",
            files={"file": ("test.exe", BytesIO(b"binary content"), "application/octet-stream")}
        )
        assert response.status_code == 400
        
        # Test analysis with missing document
        response = test_client.post(
            "/api/analysis/generate",
            json={
                "content": "Analyze non-existent document",
                "document_ids": ["non-existent-doc"]
            }
        )
        assert response.status_code == 404
        
        # Test chat with invalid session
        response = test_client.post(
            "/api/chat/message",
            json={
                "message": "Test message",
                "session_id": "invalid-session"
            }
        )
        assert response.status_code == 404
        
        # Test evaluation with invalid data
        response = test_client.post(
            "/api/evaluation/evaluate",
            json={
                "question": "",  # Empty question
                "answer": "Test answer"
            }
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_caching_integration(self, test_client):
        """Test caching integration across services."""
        
        # First request should generate and cache result
        with patch('backend.services.cache_service.CacheService.get') as mock_cache_get:
            with patch('backend.services.cache_service.CacheService.set') as mock_cache_set:
                with patch('backend.services.llm_service.LLMService.generate_response') as mock_llm:
                    mock_cache_get.return_value = None  # Cache miss
                    mock_llm.return_value = "Cached analysis result"
                    
                    response = test_client.post(
                        "/api/analysis/generate",
                        json={
                            "content": "Strategic analysis for caching test",
                            "analysis_type": "strategic"
                        }
                    )
                    
                    assert response.status_code == 200
                    mock_cache_set.assert_called_once()  # Result should be cached
        
        # Second identical request should use cache
        with patch('backend.services.cache_service.CacheService.get') as mock_cache_get:
            with patch('backend.services.llm_service.LLMService.generate_response') as mock_llm:
                mock_cache_get.return_value = "Cached analysis result"
                
                response = test_client.post(
                    "/api/analysis/generate",
                    json={
                        "content": "Strategic analysis for caching test",
                        "analysis_type": "strategic"
                    }
                )
                
                assert response.status_code == 200
                mock_llm.assert_not_called()  # Should not call LLM, use cache instead

    @pytest.mark.asyncio
    async def test_performance_workflow(self, test_client):
        """Test performance aspects of the workflow."""
        
        # Test concurrent document processing
        documents = [
            (b"Document 1 content", "doc1.txt"),
            (b"Document 2 content", "doc2.txt"),
            (b"Document 3 content", "doc3.txt")
        ]
        
        async def upload_document(doc_content, filename):
            with patch('backend.services.document_service.DocumentService.process_document') as mock_process:
                mock_process.return_value = {
                    'document_id': f'doc-{filename}',
                    'content': doc_content.decode(),
                    'metadata': {'title': filename, 'pages': 1}
                }
                
                response = test_client.post(
                    "/api/documents/upload",
                    files={"file": (filename, BytesIO(doc_content), "text/plain")}
                )
                return response
        
        # Upload documents concurrently
        tasks = [upload_document(content, name) for content, name in documents]
        responses = await asyncio.gather(*tasks)
        
        # All uploads should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Test concurrent analysis generation
        async def generate_analysis(content, analysis_type):
            with patch('backend.services.llm_service.LLMService.generate_response') as mock_llm:
                mock_llm.return_value = f"{analysis_type} analysis result"
                
                response = test_client.post(
                    "/api/analysis/generate",
                    json={
                        "content": content,
                        "analysis_type": analysis_type
                    }
                )
                return response
        
        analysis_tasks = [
            generate_analysis("Strategic content", "strategic"),
            generate_analysis("Financial content", "financial"),
            generate_analysis("Market content", "market")
        ]
        
        analysis_responses = await asyncio.gather(*analysis_tasks)
        
        # All analyses should succeed
        for response in analysis_responses:
            assert response.status_code == 200 