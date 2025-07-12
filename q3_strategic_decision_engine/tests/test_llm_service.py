"""
Unit tests for LLM Service.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import json

from backend.services.llm_service import LLMService
from backend.core.config import Settings


class TestLLMService:
    """Test cases for LLM Service."""

    @pytest.fixture
    def llm_service(self, test_settings):
        """Create LLM service instance."""
        return LLMService(test_settings)

    @pytest.mark.asyncio
    async def test_generate_response_openai(self, llm_service):
        """Test OpenAI response generation."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="Test response"))]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            response = await llm_service.generate_response(
                "Test prompt",
                model="gpt-4o",
                temperature=0.7,
                max_tokens=1000
            )
            
            assert response == "Test response"
            mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_anthropic(self, llm_service):
        """Test Anthropic response generation."""
        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.content = [Mock(text="Test response")]
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client
            
            response = await llm_service.generate_response(
                "Test prompt",
                model="claude-3.5-sonnet",
                temperature=0.7,
                max_tokens=1000
            )
            
            assert response == "Test response"
            mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_google(self, llm_service):
        """Test Google response generation."""
        with patch('google.generativeai.GenerativeModel') as mock_google:
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = "Test response"
            mock_model.generate_content_async.return_value = mock_response
            mock_google.return_value = mock_model
            
            response = await llm_service.generate_response(
                "Test prompt",
                model="gemini-2.5-pro",
                temperature=0.7,
                max_tokens=1000
            )
            
            assert response == "Test response"
            mock_model.generate_content_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_stream_openai(self, llm_service):
        """Test OpenAI streaming response."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = Mock()
            mock_stream = [
                Mock(choices=[Mock(delta=Mock(content="Test "))]),
                Mock(choices=[Mock(delta=Mock(content="stream "))]),
                Mock(choices=[Mock(delta=Mock(content="response"))])
            ]
            mock_client.chat.completions.create.return_value = mock_stream
            mock_openai.return_value = mock_client
            
            chunks = []
            async for chunk in llm_service.generate_stream("Test prompt", model="gpt-4o"):
                chunks.append(chunk)
            
            assert chunks == ["Test ", "stream ", "response"]

    def test_get_available_models(self, llm_service):
        """Test getting available models."""
        models = llm_service.get_available_models()
        expected_models = ["gpt-4o", "claude-3.5-sonnet", "gemini-2.5-pro"]
        assert all(model in models for model in expected_models)

    def test_select_best_model_strategic(self, llm_service):
        """Test model selection for strategic tasks."""
        model = llm_service.select_best_model("strategic")
        assert model == "claude-3.5-sonnet"

    def test_select_best_model_financial(self, llm_service):
        """Test model selection for financial tasks."""
        model = llm_service.select_best_model("financial")
        assert model == "gpt-4o"

    def test_select_best_model_market(self, llm_service):
        """Test model selection for market tasks."""
        model = llm_service.select_best_model("market")
        assert model == "gemini-2.5-pro"

    def test_select_best_model_default(self, llm_service):
        """Test default model selection."""
        model = llm_service.select_best_model("unknown")
        assert model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_generate_response_with_error_handling(self, llm_service):
        """Test error handling in response generation."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client
            
            with pytest.raises(Exception) as exc_info:
                await llm_service.generate_response("Test prompt", model="gpt-4o")
            
            assert "API Error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_response_with_system_message(self, llm_service):
        """Test response generation with system message."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="Test response"))]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            response = await llm_service.generate_response(
                "Test prompt",
                model="gpt-4o",
                system_message="You are a strategic advisor"
            )
            
            assert response == "Test response"
            # Verify system message was included in the call
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]['messages']
            assert messages[0]['role'] == 'system'
            assert messages[0]['content'] == "You are a strategic advisor"

    @pytest.mark.asyncio
    async def test_generate_response_with_context(self, llm_service):
        """Test response generation with context."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="Test response"))]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            context = [
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"}
            ]
            
            response = await llm_service.generate_response(
                "Test prompt",
                model="gpt-4o",
                context=context
            )
            
            assert response == "Test response"
            # Verify context was included in the call
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]['messages']
            assert len(messages) >= 3  # context + current message

    def test_model_config_validation(self, llm_service):
        """Test model configuration validation."""
        # Test valid model
        assert llm_service._validate_model("gpt-4o") is True
        
        # Test invalid model
        assert llm_service._validate_model("invalid-model") is False

    def test_token_limit_validation(self, llm_service):
        """Test token limit validation."""
        # Test valid token count
        assert llm_service._validate_token_limit("gpt-4o", 1000) is True
        
        # Test exceeding token limit
        assert llm_service._validate_token_limit("gpt-4o", 200000) is False

    @pytest.mark.asyncio
    async def test_rate_limiting(self, llm_service):
        """Test rate limiting functionality."""
        with patch('asyncio.sleep') as mock_sleep:
            # Simulate rate limiting
            llm_service._rate_limiter = {"last_request": 0, "requests_per_minute": 60}
            
            await llm_service._handle_rate_limiting()
            
            # Should not sleep if enough time has passed
            mock_sleep.assert_not_called()

    def test_cost_calculation(self, llm_service):
        """Test cost calculation for different models."""
        # Test OpenAI cost calculation
        cost = llm_service._calculate_cost("gpt-4o", 1000, 500)
        assert cost > 0
        
        # Test Anthropic cost calculation
        cost = llm_service._calculate_cost("claude-3.5-sonnet", 1000, 500)
        assert cost > 0
        
        # Test Google cost calculation
        cost = llm_service._calculate_cost("gemini-2.5-pro", 1000, 500)
        assert cost > 0

    @pytest.mark.asyncio
    async def test_response_quality_check(self, llm_service):
        """Test response quality checking."""
        # Test good response
        quality_score = await llm_service._check_response_quality("This is a detailed and helpful response about strategic planning.")
        assert quality_score > 0.5
        
        # Test poor response
        quality_score = await llm_service._check_response_quality("I don't know.")
        assert quality_score < 0.5

    def test_prompt_optimization(self, llm_service):
        """Test prompt optimization."""
        original_prompt = "Tell me about strategy"
        optimized_prompt = llm_service._optimize_prompt(original_prompt, "strategic")
        
        assert len(optimized_prompt) > len(original_prompt)
        assert "strategic" in optimized_prompt.lower()

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, llm_service):
        """Test retry mechanism for failed requests."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = Mock()
            # First call fails, second succeeds
            mock_client.chat.completions.create.side_effect = [
                Exception("Temporary error"),
                Mock(choices=[Mock(message=Mock(content="Success"))])
            ]
            mock_openai.return_value = mock_client
            
            response = await llm_service.generate_response(
                "Test prompt",
                model="gpt-4o",
                max_retries=2
            )
            
            assert response == "Success"
            assert mock_client.chat.completions.create.call_count == 2 