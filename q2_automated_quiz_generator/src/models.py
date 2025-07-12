"""Data models for the Advanced Assessment Generation System."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class DifficultyLevel(str, Enum):
    """Difficulty levels for questions."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionType(str, Enum):
    """Types of questions."""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"


class LearningObjective(BaseModel):
    """Learning objective model."""
    id: str
    description: str
    topic: str
    difficulty: DifficultyLevel
    keywords: List[str] = []


class Question(BaseModel):
    """Question model."""
    id: str
    question_text: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    learning_objective: str
    topic: str
    
    # Multiple choice specific
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None
    
    # True/False specific
    is_true: Optional[bool] = None
    
    # All question types
    explanation: str
    points: int = 1
    tags: List[str] = []


class Assessment(BaseModel):
    """Assessment model."""
    id: str
    title: str
    description: str
    topic: str
    difficulty: DifficultyLevel
    questions: List[Question]
    total_points: int
    estimated_time: int  # in minutes
    learning_objectives: List[str]
    created_at: datetime
    instructor_id: str


class UserPerformance(BaseModel):
    """User performance tracking."""
    user_id: str
    assessment_id: str
    score: float
    total_questions: int
    correct_answers: int
    time_taken: int  # in minutes
    completed_at: datetime
    difficulty_adjustment: Optional[float] = None


class DocumentChunk(BaseModel):
    """Document chunk model for RAG."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    bm25_score: Optional[float] = None


class AssessmentRequest(BaseModel):
    """Request model for assessment generation."""
    topic: str
    difficulty: DifficultyLevel
    question_types: List[QuestionType]
    num_questions: int = Field(ge=1, le=20)
    learning_objectives: Optional[List[str]] = None
    instructor_id: str


class AssessmentResponse(BaseModel):
    """Response model for assessment generation."""
    assessment: Assessment
    generation_time: float
    cache_hit: bool = False


class DocumentUpload(BaseModel):
    """Document upload model."""
    filename: str
    content_type: str
    size: int
    topic: str
    instructor_id: str


class CacheStats(BaseModel):
    """Cache statistics model."""
    total_requests: int
    cache_hits: int
    cache_miss_rate: float
    average_response_time: float 