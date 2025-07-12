from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from typing import List
import json
import time

from .config import Config
from .models import SubQuestion, QueryDecomposition

class QueryDecomposer:
    """Decomposes complex queries into simpler sub-questions."""
    
    def __init__(self):
        """Initialize the query decomposer."""
        self.config = Config()
        self.llm = ChatOpenAI(
            model=self.config.OPENAI_MODEL,
            temperature=0.1
        )
        
        # Prompt template for query decomposition
        self.decomposition_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at breaking down complex sports analytics queries into simpler, focused sub-questions.

Your task is to decompose a complex query into 2-5 sub-questions that can be answered independently and then combined to provide a comprehensive answer.

Guidelines:
1. Each sub-question should be specific and focused on one aspect
2. Sub-questions should be ordered by logical dependency
3. Include reasoning for why each sub-question is needed
4. Focus on sports analytics, player performance, team statistics, and game insights
5. Ensure sub-questions are answerable from sports documents and data

Return your response as a JSON object with this structure:
{
    "sub_questions": [
        {
            "question": "specific sub-question text",
            "reasoning": "why this sub-question is needed",
            "priority": 1
        }
    ],
    "strategy": "brief description of decomposition approach"
}

Example:
Query: "Which team has the best defense and how does their goalkeeper compare to the league average?"

Response:
{
    "sub_questions": [
        {
            "question": "What are the defensive statistics for all teams in the league?",
            "reasoning": "Need to identify teams with best defensive performance",
            "priority": 1
        },
        {
            "question": "Which team has the best defensive record based on goals conceded, clean sheets, and defensive efficiency?",
            "reasoning": "Determine the top defensive team using multiple metrics",
            "priority": 2
        },
        {
            "question": "What are the goalkeeper statistics for the team with the best defense?",
            "reasoning": "Focus on the specific goalkeeper's performance",
            "priority": 3
        },
        {
            "question": "What is the league average for goalkeeper save percentage and other key metrics?",
            "reasoning": "Establish baseline for comparison",
            "priority": 4
        },
        {
            "question": "How does the best defensive team's goalkeeper compare to the league average in save percentage, goals conceded, and other relevant metrics?",
            "reasoning": "Direct comparison to answer the original question",
            "priority": 5
        }
    ],
    "strategy": "Sequential decomposition focusing on team identification, goalkeeper analysis, and comparative metrics"
}"""),
            ("human", "Decompose this complex sports analytics query: {query}")
        ])
    
    def decompose_query(self, query: str) -> QueryDecomposition:
        """
        Decompose a complex query into sub-questions.
        
        Args:
            query: The complex query to decompose
            
        Returns:
            QueryDecomposition with sub-questions and strategy
        """
        start_time = time.time()
        
        # Check if query is simple enough (single question mark, no conjunctions)
        if self._is_simple_query(query):
            # For simple queries, create a single sub-question
            sub_question = SubQuestion(
                question=query,
                reasoning="Simple query that can be answered directly",
                priority=1
            )
            
            return QueryDecomposition(
                original_query=query,
                sub_questions=[sub_question],
                decomposition_strategy="Direct query - no decomposition needed"
            )
        
        # Decompose complex query
        try:
            # Generate decomposition using LLM
            messages = self.decomposition_prompt.format_messages(query=query)
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            decomposition_data = json.loads(response.content)
            
            # Convert to SubQuestion objects
            sub_questions = []
            for sq_data in decomposition_data.get("sub_questions", []):
                sub_question = SubQuestion(
                    question=sq_data["question"],
                    reasoning=sq_data["reasoning"],
                    priority=sq_data["priority"]
                )
                sub_questions.append(sub_question)
            
            # Sort by priority
            sub_questions.sort(key=lambda x: x.priority)
            
            # Limit to max sub-questions
            if len(sub_questions) > self.config.MAX_SUB_QUESTIONS:
                sub_questions = sub_questions[:self.config.MAX_SUB_QUESTIONS]
            
            return QueryDecomposition(
                original_query=query,
                sub_questions=sub_questions,
                decomposition_strategy=decomposition_data.get("strategy", "LLM-based decomposition")
            )
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            # Fallback: create a single sub-question
            sub_question = SubQuestion(
                question=query,
                reasoning=f"Fallback decomposition due to error: {str(e)}",
                priority=1
            )
            
            return QueryDecomposition(
                original_query=query,
                sub_questions=[sub_question],
                decomposition_strategy="Fallback decomposition"
            )
    
    def _is_simple_query(self, query: str) -> bool:
        """
        Check if a query is simple enough to not need decomposition.
        
        Args:
            query: The query to check
            
        Returns:
            True if query is simple, False otherwise
        """
        # Simple heuristics for query complexity
        query_lower = query.lower()
        
        # Check for multiple question marks
        if query.count('?') > 1:
            return False
        
        # Check for conjunction words that indicate complex queries
        conjunctions = ['and', 'or', 'but', 'however', 'while', 'whereas', 'compare', 'versus', 'vs']
        for conjunction in conjunctions:
            if conjunction in query_lower:
                return False
        
        # Check for multiple clauses (semicolons, multiple verbs)
        if ';' in query or query.count('?') > 1:
            return False
        
        # Check query length
        if len(query.split()) > 15:
            return False
        
        return True 