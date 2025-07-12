"""
LLM Service for managing multiple LLM providers
Supports OpenAI GPT-4o, Anthropic Claude 3.5 Sonnet, and Google Gemini 2.5 Pro
"""

import asyncio
import time
from typing import Dict, List, Optional, AsyncGenerator, Any, Tuple
from enum import Enum
import logging
from dataclasses import dataclass
from contextlib import asynccontextmanager

# LLM imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from ..core.config import settings
from ..core.logging_config import llm_logger


class LLMProvider(Enum):
    """LLM provider enumeration"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class LLMTask(Enum):
    """LLM task types for specialized model selection"""
    GENERAL_CHAT = "general_chat"
    DOCUMENT_ANALYSIS = "document_analysis"
    STRATEGIC_PLANNING = "strategic_planning"
    MARKET_ANALYSIS = "market_analysis"
    FINANCIAL_ANALYSIS = "financial_analysis"
    SWOT_ANALYSIS = "swot_analysis"
    QUERY_DECOMPOSITION = "query_decomposition"
    CONTEXT_COMPRESSION = "context_compression"
    CITATION_EXTRACTION = "citation_extraction"
    CHART_GENERATION = "chart_generation"


@dataclass
class LLMResponse:
    """LLM response data structure"""
    content: str
    model: str
    provider: LLMProvider
    tokens_used: int
    response_time: float
    cost: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMConfig:
    """LLM configuration"""
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    api_key: str
    provider: LLMProvider


class TokenUsageCallback(AsyncCallbackHandler):
    """Callback handler for tracking token usage"""
    
    def __init__(self):
        self.tokens_used = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0.0
    
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Handle LLM completion"""
        if response.llm_output and 'token_usage' in response.llm_output:
            usage = response.llm_output['token_usage']
            self.prompt_tokens = usage.get('prompt_tokens', 0)
            self.completion_tokens = usage.get('completion_tokens', 0)
            self.tokens_used = self.prompt_tokens + self.completion_tokens


class LLMService:
    """LLM service for managing multiple LLM providers"""
    
    def __init__(self):
        self.providers: Dict[LLMProvider, BaseLanguageModel] = {}
        self.configs: Dict[LLMProvider, LLMConfig] = {}
        self.task_routing: Dict[LLMTask, LLMProvider] = {}
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self):
        """Initialize LLM providers"""
        if self.initialized:
            return
        
        try:
            # Initialize OpenAI
            self.configs[LLMProvider.OPENAI] = LLMConfig(
                model=settings.OPENAI_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                timeout=settings.LLM_TIMEOUT,
                api_key=settings.OPENAI_API_KEY,
                provider=LLMProvider.OPENAI
            )
            
            self.providers[LLMProvider.OPENAI] = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                request_timeout=settings.LLM_TIMEOUT,
                api_key=settings.OPENAI_API_KEY,
                streaming=True,
                callbacks=[]
            )
            
            # Initialize Anthropic
            self.configs[LLMProvider.ANTHROPIC] = LLMConfig(
                model=settings.ANTHROPIC_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                timeout=settings.LLM_TIMEOUT,
                api_key=settings.ANTHROPIC_API_KEY,
                provider=LLMProvider.ANTHROPIC
            )
            
            self.providers[LLMProvider.ANTHROPIC] = ChatAnthropic(
                model=settings.ANTHROPIC_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                timeout=settings.LLM_TIMEOUT,
                api_key=settings.ANTHROPIC_API_KEY,
                streaming=True,
                callbacks=[]
            )
            
            # Initialize Google
            self.configs[LLMProvider.GOOGLE] = LLMConfig(
                model=settings.GOOGLE_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                timeout=settings.LLM_TIMEOUT,
                api_key=settings.GOOGLE_API_KEY,
                provider=LLMProvider.GOOGLE
            )
            
            self.providers[LLMProvider.GOOGLE] = ChatGoogleGenerativeAI(
                model=settings.GOOGLE_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_output_tokens=settings.LLM_MAX_TOKENS,
                timeout=settings.LLM_TIMEOUT,
                google_api_key=settings.GOOGLE_API_KEY,
                streaming=True,
                callbacks=[]
            )
            
            # Set up task routing - assign best model for each task
            self.task_routing = {
                LLMTask.GENERAL_CHAT: LLMProvider.OPENAI,
                LLMTask.DOCUMENT_ANALYSIS: LLMProvider.GOOGLE,  # Gemini 2.5 Pro excels at document analysis
                LLMTask.STRATEGIC_PLANNING: LLMProvider.ANTHROPIC,  # Claude excels at strategic thinking
                LLMTask.MARKET_ANALYSIS: LLMProvider.GOOGLE,  # Gemini 2.5 Pro for multimodal market data
                LLMTask.FINANCIAL_ANALYSIS: LLMProvider.OPENAI,  # GPT-4o for financial calculations
                LLMTask.SWOT_ANALYSIS: LLMProvider.ANTHROPIC,  # Claude for structured analysis
                LLMTask.QUERY_DECOMPOSITION: LLMProvider.OPENAI,  # GPT-4o for query understanding
                LLMTask.CONTEXT_COMPRESSION: LLMProvider.GOOGLE,  # Gemini 2.5 Pro for large context
                LLMTask.CITATION_EXTRACTION: LLMProvider.ANTHROPIC,  # Claude for precise citations
                LLMTask.CHART_GENERATION: LLMProvider.OPENAI,  # GPT-4o for code generation
            }
            
            self.initialized = True
            self.logger.info("LLM service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM service: {str(e)}")
            raise
    
    def get_model_for_task(self, task: LLMTask) -> Tuple[BaseLanguageModel, LLMProvider]:
        """Get the best model for a specific task"""
        provider = self.task_routing.get(task, LLMProvider.OPENAI)
        return self.providers[provider], provider
    
    def get_model_by_provider(self, provider: LLMProvider) -> BaseLanguageModel:
        """Get model by provider"""
        return self.providers.get(provider)
    
    async def generate_response(
        self,
        prompt: str,
        system_message: str = None,
        task: LLMTask = LLMTask.GENERAL_CHAT,
        provider: LLMProvider = None,
        temperature: float = None,
        max_tokens: int = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Generate response using appropriate LLM"""
        
        if not self.initialized:
            await self.initialize()
        
        # Select model
        if provider:
            model = self.get_model_by_provider(provider)
            selected_provider = provider
        else:
            model, selected_provider = self.get_model_for_task(task)
        
        # Override parameters if provided
        if temperature is not None:
            model.temperature = temperature
        if max_tokens is not None:
            if hasattr(model, 'max_tokens'):
                model.max_tokens = max_tokens
            elif hasattr(model, 'max_output_tokens'):
                model.max_output_tokens = max_tokens
        
        # Create messages
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))
        
        # Create callback for token tracking
        callback = TokenUsageCallback()
        
        start_time = time.time()
        
        try:
            if stream:
                # Streaming response
                response_chunks = []
                async for chunk in model.astream(messages, callbacks=[callback]):
                    response_chunks.append(chunk.content)
                    yield chunk.content
                
                content = ''.join(response_chunks)
            else:
                # Non-streaming response
                response = await model.ainvoke(messages, callbacks=[callback])
                content = response.content
            
            response_time = time.time() - start_time
            
            # Calculate cost (approximate)
            cost = self._calculate_cost(
                selected_provider,
                callback.prompt_tokens,
                callback.completion_tokens
            )
            
            # Log the completion
            llm_logger.log_completion(
                model=self.configs[selected_provider].model,
                prompt_tokens=callback.prompt_tokens,
                completion_tokens=callback.completion_tokens,
                response_time=response_time,
                cost=cost,
                task=task.value
            )
            
            if not stream:
                return LLMResponse(
                    content=content,
                    model=self.configs[selected_provider].model,
                    provider=selected_provider,
                    tokens_used=callback.tokens_used,
                    response_time=response_time,
                    cost=cost,
                    metadata={
                        'task': task.value,
                        'prompt_tokens': callback.prompt_tokens,
                        'completion_tokens': callback.completion_tokens
                    }
                )
                
        except Exception as e:
            llm_logger.log_error(
                model=self.configs[selected_provider].model,
                error=str(e),
                task=task.value
            )
            raise
    
    async def generate_streaming_response(
        self,
        prompt: str,
        system_message: str = None,
        task: LLMTask = LLMTask.GENERAL_CHAT,
        provider: LLMProvider = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        async for chunk in self.generate_response(
            prompt=prompt,
            system_message=system_message,
            task=task,
            provider=provider,
            stream=True,
            **kwargs
        ):
            yield chunk
    
    async def generate_multiple_responses(
        self,
        prompts: List[str],
        system_message: str = None,
        task: LLMTask = LLMTask.GENERAL_CHAT,
        provider: LLMProvider = None,
        **kwargs
    ) -> List[LLMResponse]:
        """Generate multiple responses in parallel"""
        
        tasks = [
            self.generate_response(
                prompt=prompt,
                system_message=system_message,
                task=task,
                provider=provider,
                **kwargs
            )
            for prompt in prompts
        ]
        
        responses = await asyncio.gather(*tasks)
        return responses
    
    async def compare_responses(
        self,
        prompt: str,
        system_message: str = None,
        providers: List[LLMProvider] = None,
        task: LLMTask = LLMTask.GENERAL_CHAT,
        **kwargs
    ) -> Dict[LLMProvider, LLMResponse]:
        """Compare responses from multiple providers"""
        
        if providers is None:
            providers = [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.GOOGLE]
        
        tasks = [
            self.generate_response(
                prompt=prompt,
                system_message=system_message,
                task=task,
                provider=provider,
                **kwargs
            )
            for provider in providers
        ]
        
        responses = await asyncio.gather(*tasks)
        
        return {
            provider: response
            for provider, response in zip(providers, responses)
        }
    
    def _calculate_cost(self, provider: LLMProvider, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate approximate cost based on token usage"""
        
        # Approximate pricing (as of 2024)
        pricing = {
            LLMProvider.OPENAI: {
                'prompt': 0.00001,  # $0.01 per 1K tokens
                'completion': 0.00003  # $0.03 per 1K tokens
            },
            LLMProvider.ANTHROPIC: {
                'prompt': 0.000015,  # $0.015 per 1K tokens
                'completion': 0.000075  # $0.075 per 1K tokens
            },
            LLMProvider.GOOGLE: {
                'prompt': 0.000125,  # $0.125 per 1K tokens
                'completion': 0.000375  # $0.375 per 1K tokens
            }
        }
        
        if provider in pricing:
            prompt_cost = (prompt_tokens / 1000) * pricing[provider]['prompt']
            completion_cost = (completion_tokens / 1000) * pricing[provider]['completion']
            return prompt_cost + completion_cost
        
        return 0.0
    
    async def health_check(self) -> Dict[LLMProvider, bool]:
        """Check health of all LLM providers"""
        health_status = {}
        
        for provider in self.providers:
            try:
                response = await self.generate_response(
                    prompt="Hello",
                    provider=provider,
                    max_tokens=10
                )
                health_status[provider] = True
            except Exception as e:
                self.logger.error(f"Health check failed for {provider}: {str(e)}")
                health_status[provider] = False
        
        return health_status
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available providers"""
        return list(self.providers.keys())
    
    def get_provider_info(self, provider: LLMProvider) -> Dict[str, Any]:
        """Get provider configuration info"""
        if provider not in self.configs:
            return {}
        
        config = self.configs[provider]
        return {
            'model': config.model,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'timeout': config.timeout,
            'provider': config.provider.value
        } 