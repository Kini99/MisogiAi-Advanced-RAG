"""
API endpoint tests for Strategic Decision Engine.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import json
from fastapi.testclient import TestClient


class TestChatEndpoints:
    """Test cases for chat endpoints."""

    def test_send_message_success(self, test_client, mock_llm_service):
        """Test successful message sending."""
        # Mock the LLM service
        mock_llm_service.generate_response.return_value = "Test response"
        
        with patch('backend.api.endpoints.chat.llm_service', mock_llm_service):
            response = test_client.post(
                "/api/chat/message",
                json={
                    "message": "Test message",
                    "session_id": "test-session",
                    "context": []
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Test response"
        assert data["session_id"] == "test-session"

    def test_send_message_with_context(self, test_client, mock_llm_service):
        """Test message sending with context."""
        mock_llm_service.generate_response.return_value = "Response with context"
        
        with patch('backend.api.endpoints.chat.llm_service', mock_llm_service):
            response = test_client.post(
                "/api/chat/message",
                json={
                    "message": "Follow-up question",
                    "session_id": "test-session",
                    "context": [
                        {"role": "user", "content": "Previous question"},
                        {"role": "assistant", "content": "Previous answer"}
                    ]
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Response with context"

    def test_send_message_invalid_input(self, test_client):
        """Test message sending with invalid input."""
        response = test_client.post(
            "/api/chat/message",
            json={
                "message": "",  # Empty message
                "session_id": "test-session"
            }
        )
        
        assert response.status_code == 422

    def test_stream_message_success(self, test_client, mock_llm_service):
        """Test successful message streaming."""
        async def mock_stream():
            yield "Test "
            yield "stream "
            yield "response"
        
        mock_llm_service.generate_stream.return_value = mock_stream()
        
        with patch('backend.api.endpoints.chat.llm_service', mock_llm_service):
            response = test_client.post(
                "/api/chat/stream",
                json={
                    "message": "Test message",
                    "session_id": "test-session"
                }
            )
        
        assert response.status_code == 200
        # Stream response should be text/event-stream
        assert "text/event-stream" in response.headers.get("content-type", "")

    def test_get_chat_history_success(self, test_client):
        """Test successful chat history retrieval."""
        # Mock database session with chat history
        with patch('backend.api.endpoints.chat.get_db') as mock_db:
            mock_session = Mock()
            mock_query = Mock()
            mock_query.filter.return_value.all.return_value = [
                Mock(
                    message_id="msg-1",
                    user_message="Test question",
                    assistant_response="Test answer",
                    timestamp="2024-01-01T00:00:00Z"
                )
            ]
            mock_session.query.return_value = mock_query
            mock_db.return_value = mock_session
            
            response = test_client.get("/api/chat/history/test-session")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) == 1
        assert data["messages"][0]["user_message"] == "Test question"

    def test_get_chat_history_not_found(self, test_client):
        """Test chat history retrieval for non-existent session."""
        with patch('backend.api.endpoints.chat.get_db') as mock_db:
            mock_session = Mock()
            mock_query = Mock()
            mock_query.filter.return_value.all.return_value = []
            mock_session.query.return_value = mock_query
            mock_db.return_value = mock_session
            
            response = test_client.get("/api/chat/history/non-existent-session")
        
        assert response.status_code == 404

    def test_create_session_success(self, test_client):
        """Test successful session creation."""
        with patch('backend.api.endpoints.chat.get_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            response = test_client.post(
                "/api/chat/sessions",
                json={"title": "Test Session"}
            )
        
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Test Session"
        assert "session_id" in data

    def test_delete_session_success(self, test_client):
        """Test successful session deletion."""
        with patch('backend.api.endpoints.chat.get_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            response = test_client.delete("/api/chat/sessions/test-session")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Session deleted successfully"

    def test_clear_session_success(self, test_client):
        """Test successful session clearing."""
        with patch('backend.api.endpoints.chat.get_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            response = test_client.post("/api/chat/sessions/test-session/clear")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Session cleared successfully"


class TestAnalysisEndpoints:
    """Test cases for analysis endpoints."""

    def test_generate_analysis_success(self, test_client, mock_llm_service):
        """Test successful analysis generation."""
        mock_llm_service.generate_response.return_value = "Strategic analysis result"
        
        with patch('backend.api.endpoints.analysis.llm_service', mock_llm_service):
            response = test_client.post(
                "/api/analysis/generate",
                json={
                    "content": "Analyze our market position",
                    "analysis_type": "strategic",
                    "context": "Technology company",
                    "requirements": ["competitive analysis", "recommendations"]
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["analysis"] == "Strategic analysis result"
        assert data["analysis_type"] == "strategic"

    def test_generate_swot_analysis_success(self, test_client, mock_llm_service):
        """Test successful SWOT analysis generation."""
        mock_response = {
            "strengths": ["Strong technical team", "Innovative products"],
            "weaknesses": ["Limited market presence", "High costs"],
            "opportunities": ["AI market growth", "New partnerships"],
            "threats": ["Increased competition", "Economic uncertainty"]
        }
        mock_llm_service.generate_response.return_value = json.dumps(mock_response)
        
        with patch('backend.api.endpoints.analysis.llm_service', mock_llm_service):
            response = test_client.post(
                "/api/analysis/swot",
                json={
                    "content": "Analyze our company's SWOT",
                    "context": "Technology startup"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "swot_analysis" in data
        assert len(data["swot_analysis"]["strengths"]) == 2
        assert len(data["swot_analysis"]["weaknesses"]) == 2

    def test_generate_market_analysis_success(self, test_client, mock_llm_service):
        """Test successful market analysis generation."""
        mock_response = {
            "market_size": "$10B",
            "growth_rate": "15%",
            "key_trends": ["AI adoption", "Digital transformation"],
            "competitive_landscape": "Highly competitive"
        }
        mock_llm_service.generate_response.return_value = json.dumps(mock_response)
        
        with patch('backend.api.endpoints.analysis.llm_service', mock_llm_service):
            response = test_client.post(
                "/api/analysis/market",
                json={
                    "content": "Analyze the AI market",
                    "industry": "Technology",
                    "region": "North America"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "market_analysis" in data
        assert data["market_analysis"]["market_size"] == "$10B"

    def test_generate_financial_analysis_success(self, test_client, mock_llm_service):
        """Test successful financial analysis generation."""
        mock_response = {
            "revenue_forecast": [100000, 150000, 200000],
            "key_metrics": {
                "gross_margin": "70%",
                "operating_margin": "20%",
                "net_margin": "15%"
            },
            "recommendations": ["Increase marketing spend", "Optimize costs"]
        }
        mock_llm_service.generate_response.return_value = json.dumps(mock_response)
        
        with patch('backend.api.endpoints.analysis.llm_service', mock_llm_service):
            response = test_client.post(
                "/api/analysis/financial",
                json={
                    "content": "Analyze our financial performance",
                    "data": {"revenue": 100000, "expenses": 80000},
                    "period": "quarterly"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "financial_analysis" in data
        assert len(data["financial_analysis"]["revenue_forecast"]) == 3

    def test_get_analysis_results_success(self, test_client):
        """Test successful analysis results retrieval."""
        with patch('backend.api.endpoints.analysis.get_db') as mock_db:
            mock_session = Mock()
            mock_query = Mock()
            mock_query.filter.return_value.all.return_value = [
                Mock(
                    analysis_id="analysis-1",
                    analysis_type="strategic",
                    content="Test analysis",
                    created_at="2024-01-01T00:00:00Z"
                )
            ]
            mock_session.query.return_value = mock_query
            mock_db.return_value = mock_session
            
            response = test_client.get("/api/analysis/results/test-session")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["analysis_type"] == "strategic"

    def test_save_analysis_success(self, test_client):
        """Test successful analysis saving."""
        with patch('backend.api.endpoints.analysis.get_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session
            
            response = test_client.post(
                "/api/analysis/save",
                json={
                    "analysis_id": "analysis-1",
                    "session_id": "test-session",
                    "title": "Saved Analysis",
                    "tags": ["strategic", "important"]
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Analysis saved successfully"

    def test_export_analysis_success(self, test_client):
        """Test successful analysis export."""
        with patch('backend.api.endpoints.analysis.get_db') as mock_db:
            mock_session = Mock()
            mock_analysis = Mock()
            mock_analysis.content = "Test analysis content"
            mock_analysis.analysis_type = "strategic"
            mock_session.query.return_value.filter.return_value.first.return_value = mock_analysis
            mock_db.return_value = mock_session
            
            response = test_client.get("/api/analysis/export/analysis-1?format=pdf")
        
        assert response.status_code == 200
        assert "application/pdf" in response.headers.get("content-type", "")

    def test_analysis_invalid_input(self, test_client):
        """Test analysis with invalid input."""
        response = test_client.post(
            "/api/analysis/generate",
            json={
                "content": "",  # Empty content
                "analysis_type": "strategic"
            }
        )
        
        assert response.status_code == 422


class TestEvaluationEndpoints:
    """Test cases for evaluation endpoints."""

    def test_evaluate_response_success(self, test_client):
        """Test successful response evaluation."""
        with patch('backend.api.endpoints.evaluation.evaluate_faithfulness') as mock_eval:
            mock_eval.return_value = 0.85
            
            response = test_client.post(
                "/api/evaluation/evaluate",
                json={
                    "question": "What are the key opportunities?",
                    "answer": "AI market expansion and partnerships",
                    "contexts": ["AI market is growing", "Partnership opportunities exist"],
                    "ground_truth": "AI and partnerships are key opportunities"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "evaluation_results" in data
        assert data["evaluation_results"]["faithfulness"] == 0.85

    def test_evaluate_faithfulness_success(self, test_client):
        """Test successful faithfulness evaluation."""
        with patch('backend.api.endpoints.evaluation.evaluate_faithfulness') as mock_eval:
            mock_eval.return_value = 0.90
            
            response = test_client.post(
                "/api/evaluation/faithfulness",
                json={
                    "answer": "The company has strong AI capabilities",
                    "contexts": ["Company invests heavily in AI research", "AI team has 50+ engineers"]
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["faithfulness_score"] == 0.90

    def test_evaluate_answer_relevancy_success(self, test_client):
        """Test successful answer relevancy evaluation."""
        with patch('backend.api.endpoints.evaluation.evaluate_answer_relevancy') as mock_eval:
            mock_eval.return_value = 0.88
            
            response = test_client.post(
                "/api/evaluation/answer_relevancy",
                json={
                    "question": "What are our AI capabilities?",
                    "answer": "We have a strong AI team and significant investments in AI research"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer_relevancy_score"] == 0.88

    def test_evaluate_context_precision_success(self, test_client):
        """Test successful context precision evaluation."""
        with patch('backend.api.endpoints.evaluation.evaluate_context_precision') as mock_eval:
            mock_eval.return_value = 0.92
            
            response = test_client.post(
                "/api/evaluation/context_precision",
                json={
                    "question": "How is our AI team performing?",
                    "contexts": ["AI team delivered 5 projects this quarter", "Team has 95% satisfaction rate"],
                    "ground_truth": "AI team is performing well with high delivery and satisfaction"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["context_precision_score"] == 0.92

    def test_evaluate_context_recall_success(self, test_client):
        """Test successful context recall evaluation."""
        with patch('backend.api.endpoints.evaluation.evaluate_context_recall') as mock_eval:
            mock_eval.return_value = 0.87
            
            response = test_client.post(
                "/api/evaluation/context_recall",
                json={
                    "contexts": ["AI market is growing", "Company has AI expertise"],
                    "ground_truth": "AI market growth and company expertise are key factors"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["context_recall_score"] == 0.87

    def test_evaluate_answer_correctness_success(self, test_client):
        """Test successful answer correctness evaluation."""
        with patch('backend.api.endpoints.evaluation.evaluate_answer_correctness') as mock_eval:
            mock_eval.return_value = 0.91
            
            response = test_client.post(
                "/api/evaluation/answer_correctness",
                json={
                    "answer": "The company's revenue grew by 25% last year",
                    "ground_truth": "Company revenue increased 25% in the previous fiscal year"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer_correctness_score"] == 0.91

    def test_batch_evaluate_success(self, test_client):
        """Test successful batch evaluation."""
        with patch('backend.api.endpoints.evaluation.evaluate_faithfulness') as mock_eval:
            mock_eval.return_value = 0.85
            
            response = test_client.post(
                "/api/evaluation/batch",
                json={
                    "evaluations": [
                        {
                            "question": "Question 1",
                            "answer": "Answer 1",
                            "contexts": ["Context 1"],
                            "ground_truth": "Truth 1"
                        },
                        {
                            "question": "Question 2",
                            "answer": "Answer 2",
                            "contexts": ["Context 2"],
                            "ground_truth": "Truth 2"
                        }
                    ]
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["batch_results"]) == 2
        assert all("evaluation_id" in result for result in data["batch_results"])

    def test_get_evaluation_report_success(self, test_client):
        """Test successful evaluation report generation."""
        with patch('backend.api.endpoints.evaluation.get_db') as mock_db:
            mock_session = Mock()
            mock_query = Mock()
            mock_results = [
                Mock(
                    evaluation_id="eval-1",
                    faithfulness=0.85,
                    answer_relevancy=0.88,
                    context_precision=0.92,
                    context_recall=0.87,
                    answer_correctness=0.91,
                    created_at="2024-01-01T00:00:00Z"
                ),
                Mock(
                    evaluation_id="eval-2",
                    faithfulness=0.87,
                    answer_relevancy=0.85,
                    context_precision=0.89,
                    context_recall=0.84,
                    answer_correctness=0.88,
                    created_at="2024-01-01T01:00:00Z"
                )
            ]
            mock_query.filter.return_value.all.return_value = mock_results
            mock_session.query.return_value = mock_query
            mock_db.return_value = mock_session
            
            response = test_client.get("/api/evaluation/report/test-session")
        
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "detailed_results" in data
        assert len(data["detailed_results"]) == 2
        assert data["summary"]["total_evaluations"] == 2

    def test_evaluation_invalid_input(self, test_client):
        """Test evaluation with invalid input."""
        response = test_client.post(
            "/api/evaluation/evaluate",
            json={
                "question": "",  # Empty question
                "answer": "Test answer"
            }
        )
        
        assert response.status_code == 422

    def test_evaluation_missing_required_fields(self, test_client):
        """Test evaluation with missing required fields."""
        response = test_client.post(
            "/api/evaluation/faithfulness",
            json={
                "answer": "Test answer"
                # Missing contexts
            }
        )
        
        assert response.status_code == 422 