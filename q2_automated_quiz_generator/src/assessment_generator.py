"""Assessment generator with tool calling and dynamic difficulty adjustment."""

import json
import uuid
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import httpx
import asyncio

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate

from .models import (
    Assessment, Question, AssessmentRequest, AssessmentResponse,
    DifficultyLevel, QuestionType, UserPerformance
)
from .config import settings
from .cache import assessment_cache
from .hybrid_rag import hybrid_rag


class EducationalContentAPI:
    """Tool for fetching educational content from external APIs."""
    
    def __init__(self):
        """Initialize educational content API."""
        self.base_urls = {
            "wikipedia": "https://en.wikipedia.org/api/rest_v1/page/summary/",
            "khan_academy": "https://www.khanacademy.org/api/v1/",
            "openstax": "https://openstax.org/api/v2/"
        }
    
    async def search_wikipedia(self, topic: str) -> Dict[str, Any]:
        """Search Wikipedia for educational content."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_urls['wikipedia']}{topic.replace(' ', '_')}",
                    timeout=10.0
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "title": data.get("title", ""),
                        "extract": data.get("extract", ""),
                        "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                        "source": "wikipedia"
                    }
                return {"error": f"Wikipedia API error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Wikipedia search error: {str(e)}"}
    
    async def get_khan_academy_content(self, topic: str) -> Dict[str, Any]:
        """Get Khan Academy content for a topic."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_urls['khan_academy']}search",
                    params={"q": topic, "type": "video"},
                    timeout=10.0
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "videos": data.get("videos", [])[:3],
                        "source": "khan_academy"
                    }
                return {"error": f"Khan Academy API error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Khan Academy search error: {str(e)}"}
    
    def get_content_tool(self) -> Tool:
        """Get educational content tool."""
        return Tool(
            name="educational_content_search",
            description="Search for additional educational content from external sources like Wikipedia and Khan Academy",
            func=self._search_content_sync
        )
    
    def _search_content_sync(self, topic: str) -> str:
        """Synchronous wrapper for content search."""
        try:
            # Check if there's already an event loop running
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, we can't use run_until_complete
                # Return a simple response instead
                return json.dumps({
                    "wikipedia": {"title": f"Information about {topic}", "extract": f"Search results for {topic} would be available in async context.", "source": "wikipedia"},
                    "khan_academy": {"videos": [{"title": f"Video about {topic}", "description": f"Educational content about {topic}"}], "source": "khan_academy"}
                })
            except RuntimeError:
                # No event loop running, we can create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    wikipedia_result = loop.run_until_complete(self.search_wikipedia(topic))
                    khan_result = loop.run_until_complete(self.get_khan_academy_content(topic))
                    return json.dumps({
                        "wikipedia": wikipedia_result,
                        "khan_academy": khan_result
                    })
                finally:
                    loop.close()
        except Exception as e:
            return json.dumps({"error": str(e)})


class DifficultyAdjuster:
    """Dynamic difficulty adjustment based on user performance."""
    
    def __init__(self):
        """Initialize difficulty adjuster."""
        self.performance_thresholds = {
            "easy": {"min": 0.0, "max": 0.6},
            "medium": {"min": 0.4, "max": 0.8},
            "hard": {"min": 0.7, "max": 1.0}
        }
    
    def adjust_difficulty(self, user_performance: UserPerformance, 
                         current_difficulty: DifficultyLevel) -> DifficultyLevel:
        """Adjust difficulty based on user performance."""
        performance_ratio = user_performance.score / user_performance.total_questions
        
        # Get current difficulty range
        current_range = self.performance_thresholds[current_difficulty.value]
        
        # Determine if adjustment is needed
        if performance_ratio < current_range["min"]:
            # User is struggling, decrease difficulty
            if current_difficulty == DifficultyLevel.HARD:
                return DifficultyLevel.MEDIUM
            elif current_difficulty == DifficultyLevel.MEDIUM:
                return DifficultyLevel.EASY
        elif performance_ratio > current_range["max"]:
            # User is excelling, increase difficulty
            if current_difficulty == DifficultyLevel.EASY:
                return DifficultyLevel.MEDIUM
            elif current_difficulty == DifficultyLevel.MEDIUM:
                return DifficultyLevel.HARD
        
        return current_difficulty
    
    def get_question_distribution(self, difficulty: DifficultyLevel, 
                                 num_questions: int) -> Dict[str, int]:
        """Get question distribution based on difficulty."""
        if difficulty == DifficultyLevel.EASY:
            return {
                "easy": int(num_questions * 0.7),
                "medium": int(num_questions * 0.3),
                "hard": 0
            }
        elif difficulty == DifficultyLevel.MEDIUM:
            return {
                "easy": int(num_questions * 0.2),
                "medium": int(num_questions * 0.6),
                "hard": int(num_questions * 0.2)
            }
        else:  # HARD
            return {
                "easy": 0,
                "medium": int(num_questions * 0.3),
                "hard": int(num_questions * 0.7)
            }


class AssessmentGenerator:
    """Advanced assessment generator with tool calling."""
    
    def __init__(self):
        """Initialize assessment generator."""
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            openai_api_key=settings.openai_api_key
        )
        
        self.content_api = EducationalContentAPI()
        self.difficulty_adjuster = DifficultyAdjuster()
        
        # Create tools
        self.tools = [
            self.content_api.get_content_tool()
        ]
        
        # Create agent
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self._create_prompt()
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )
    
    def _create_prompt(self):
        """Create the system prompt for assessment generation."""
        system_template = """You are an expert educational assessment generator. Your task is to create high-quality, 
        personalized assessments based on educational content and learning objectives.

        Available tools:
        - educational_content_search: Search for additional educational content from external sources

        Guidelines for assessment generation:
        1. Create questions that test understanding, not just memorization
        2. Ensure questions are appropriate for the specified difficulty level
        3. Provide clear, detailed explanations for all answers
        4. Include a mix of question types as requested
        5. Make sure questions are relevant to the learning objectives
        6. Use the retrieved educational content as the primary source
        7. Supplement with external content when needed for better coverage

        Question types:
        - multiple_choice: 4 options, one correct answer
        - true_false: simple true/false questions
        - short_answer: brief written responses
        - essay: longer, more detailed responses

        Difficulty levels:
        - easy: Basic understanding and recall
        - medium: Application and analysis
        - hard: Synthesis and evaluation

        Always structure your response as a JSON object with the following format:
        {{
            "title": "Assessment title",
            "description": "Brief description",
            "questions": [
                {{
                    "question_text": "Question text",
                    "question_type": "multiple_choice|true_false|short_answer|essay",
                    "difficulty": "easy|medium|hard",
                    "learning_objective": "Specific learning objective",
                    "options": ["option1", "option2", "option3", "option4"],  // for multiple choice
                    "correct_answer": "correct option or answer",
                    "is_true": true/false,  // for true/false
                    "explanation": "Detailed explanation of the answer",
                    "points": 1,
                    "tags": ["tag1", "tag2"]
                }}
            ]
        }}
        """

        human_template = "{input}\n\n{agent_scratchpad}"

        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
    
    def _generate_request_hash(self, request: AssessmentRequest) -> str:
        """Generate hash for assessment request for caching."""
        import hashlib
        request_str = f"{request.topic}_{request.difficulty}_{request.num_questions}_{request.instructor_id}"
        return hashlib.md5(request_str.encode()).hexdigest()
    
    def _retrieve_relevant_content(self, request: AssessmentRequest) -> str:
        """Retrieve relevant content using hybrid RAG."""
        # Get relevant chunks
        chunks = hybrid_rag.retrieve(
            query=request.topic,
            topic=request.topic,
            top_k=10
        )
        
        # Combine content
        content = "\n\n".join([chunk[0].content for chunk in chunks])
        
        # Add learning objectives if provided
        if request.learning_objectives:
            content += f"\n\nLearning Objectives:\n" + "\n".join(request.learning_objectives)
        
        return content
    
    def _create_questions_from_content(self, content: str, request: AssessmentRequest) -> List[Question]:
        """Create questions from retrieved content."""
        # Prepare the prompt for question generation
        prompt = f"""
        Based on the following educational content, create {request.num_questions} questions.
        
        Topic: {request.topic}
        Difficulty: {request.difficulty}
        Question Types: {', '.join([qt.value for qt in request.question_types])}
        
        Educational Content:
        {content}
        
        Generate questions that:
        1. Test understanding of the key concepts
        2. Match the specified difficulty level
        3. Include the requested question types
        4. Have clear, detailed explanations
        5. Are relevant to the learning objectives
        """
        
        try:
            # Use the agent to generate questions
            response = self.agent_executor.invoke({"input": prompt})
            
            # Parse the response
            if "output" in response:
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', response["output"], re.DOTALL)
                if json_match:
                    question_data = json.loads(json_match.group())
                    
                    questions = []
                    for q_data in question_data.get("questions", []):
                        question = Question(
                            id=str(uuid.uuid4()),
                            question_text=q_data["question_text"],
                            question_type=QuestionType(q_data["question_type"]),
                            difficulty=DifficultyLevel(q_data["difficulty"]),
                            learning_objective=q_data["learning_objective"],
                            topic=request.topic,
                            options=q_data.get("options"),
                            correct_answer=q_data.get("correct_answer"),
                            is_true=q_data.get("is_true"),
                            explanation=q_data["explanation"],
                            points=q_data.get("points", 1),
                            tags=q_data.get("tags", [])
                        )
                        questions.append(question)
                    
                    return questions
            
            # Fallback: create simple questions
            return self._create_fallback_questions(content, request)
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            return self._create_fallback_questions(content, request)
    
    def _create_fallback_questions(self, content: str, request: AssessmentRequest) -> List[Question]:
        """Create fallback questions when LLM generation fails."""
        questions = []
        
        # Simple question generation logic
        lines = content.split('\n')
        relevant_lines = [line for line in lines if len(line.strip()) > 50][:request.num_questions]
        
        for i, line in enumerate(relevant_lines):
            question = Question(
                id=str(uuid.uuid4()),
                question_text=f"Question {i+1}: Explain the key concept mentioned in the content.",
                question_type=QuestionType.SHORT_ANSWER,
                difficulty=request.difficulty,
                learning_objective=f"Understand key concepts in {request.topic}",
                topic=request.topic,
                explanation="This question tests understanding of the educational content.",
                points=1,
                tags=[request.topic]
            )
            questions.append(question)
        
        return questions
    
    def generate_assessment(self, request: AssessmentRequest) -> AssessmentResponse:
        """Generate assessment based on request."""
        start_time = time.time()
        
        # Check cache first
        request_hash = self._generate_request_hash(request)
        cached_assessment = assessment_cache.get_assessment(request_hash)
        
        if cached_assessment:
            generation_time = time.time() - start_time
            return AssessmentResponse(
                assessment=cached_assessment,
                generation_time=generation_time,
                cache_hit=True
            )
        
        # Retrieve relevant content
        content = self._retrieve_relevant_content(request)
        
        # Generate questions
        questions = self._create_questions_from_content(content, request)
        
        # Create assessment
        assessment = Assessment(
            id=str(uuid.uuid4()),
            title=f"{request.topic} Assessment",
            description=f"Assessment covering {request.topic} at {request.difficulty} level",
            topic=request.topic,
            difficulty=request.difficulty,
            questions=questions,
            total_points=sum(q.points for q in questions),
            estimated_time=len(questions) * 2,  # 2 minutes per question
            learning_objectives=request.learning_objectives or [f"Understand {request.topic}"],
            created_at=datetime.now(),
            instructor_id=request.instructor_id
        )
        
        # Cache the assessment
        assessment_cache.set_assessment(request_hash, assessment)
        
        generation_time = time.time() - start_time
        
        return AssessmentResponse(
            assessment=assessment,
            generation_time=generation_time,
            cache_hit=False
        )
    
    def adjust_assessment_difficulty(self, user_performance: UserPerformance, 
                                   assessment: Assessment) -> Assessment:
        """Adjust assessment difficulty based on user performance."""
        new_difficulty = self.difficulty_adjuster.adjust_difficulty(
            user_performance, assessment.difficulty
        )
        
        if new_difficulty != assessment.difficulty:
            # Create new assessment with adjusted difficulty
            adjusted_request = AssessmentRequest(
                topic=assessment.topic,
                difficulty=new_difficulty,
                question_types=[q.question_type for q in assessment.questions],
                num_questions=len(assessment.questions),
                learning_objectives=assessment.learning_objectives,
                instructor_id=assessment.instructor_id
            )
            
            # Generate new assessment
            response = self.generate_assessment(adjusted_request)
            return response.assessment
        
        return assessment


# Global assessment generator instance
assessment_generator = AssessmentGenerator() 