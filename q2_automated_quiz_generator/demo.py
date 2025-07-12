"""Demo script for the Advanced Assessment Generation System."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from src.models import (
    AssessmentRequest, DifficultyLevel, QuestionType, UserPerformance
)
from src.document_processor import document_processor
from src.hybrid_rag import hybrid_rag
from src.assessment_generator import assessment_generator
from src.cache import assessment_cache


def create_sample_document():
    """Create a sample educational document for demo."""
    sample_content = """
    Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.
    
    Types of Machine Learning:
    
    1. Supervised Learning: In supervised learning, the algorithm is trained on labeled data. The model learns to map input features to known output labels. Examples include classification and regression problems.
    
    2. Unsupervised Learning: Unsupervised learning works with unlabeled data. The algorithm tries to find hidden patterns or structures in the data. Clustering and dimensionality reduction are common unsupervised learning tasks.
    
    3. Reinforcement Learning: Reinforcement learning involves an agent learning to make decisions by taking actions in an environment to achieve maximum cumulative reward.
    
    Key Concepts:
    
    - Features: The input variables used to make predictions
    - Labels: The output variables we want to predict
    - Training Data: The dataset used to train the model
    - Testing Data: The dataset used to evaluate model performance
    - Overfitting: When a model performs well on training data but poorly on new data
    - Underfitting: When a model is too simple to capture the underlying patterns
    
    Common Algorithms:
    
    - Linear Regression: Used for predicting continuous values
    - Logistic Regression: Used for binary classification
    - Decision Trees: Tree-like model for classification and regression
    - Random Forest: Ensemble method using multiple decision trees
    - Support Vector Machines: Effective for classification tasks
    - Neural Networks: Deep learning models for complex patterns
    
    Model Evaluation:
    
    - Accuracy: Percentage of correct predictions
    - Precision: True positives divided by total predicted positives
    - Recall: True positives divided by total actual positives
    - F1-Score: Harmonic mean of precision and recall
    - Mean Squared Error: Average squared difference between predictions and actual values
    
    Applications:
    
    Machine learning is used in various fields including:
    - Healthcare: Disease diagnosis and drug discovery
    - Finance: Fraud detection and risk assessment
    - Marketing: Customer segmentation and recommendation systems
    - Transportation: Autonomous vehicles and route optimization
    - Entertainment: Content recommendation and gaming AI
    """
    
    # Create sample document file
    sample_file = Path("sample_machine_learning.txt")
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_content)
    
    return str(sample_file)


async def demo_document_processing():
    """Demo document processing functionality."""
    print("\n📄 Demo: Document Processing")
    print("=" * 50)
    
    # Create sample document
    sample_file = create_sample_document()
    print(f"✅ Created sample document: {sample_file}")
    
    # Process document
    topic = "machine_learning"
    instructor_id = "demo_instructor"
    
    chunks = document_processor.process_document(sample_file, topic, instructor_id)
    print(f"✅ Processed document into {len(chunks)} chunks")
    
    # Add to hybrid RAG
    hybrid_rag.add_documents(chunks, topic)
    print("✅ Added chunks to hybrid RAG system")
    
    return topic, instructor_id


async def demo_hybrid_rag():
    """Demo hybrid RAG functionality."""
    print("\n🔍 Demo: Hybrid RAG Retrieval")
    print("=" * 50)
    
    # Test retrieval
    query = "What are the types of machine learning?"
    topic = "machine_learning"
    
    results = hybrid_rag.retrieve(query, topic, top_k=3)
    print(f"✅ Retrieved {len(results)} relevant chunks for query: '{query}'")
    
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\nResult {i} (Score: {score:.3f}):")
        print(f"Content: {chunk.content[:150]}...")
        print(f"Metadata: {chunk.metadata.get('filename', 'N/A')}")


async def demo_assessment_generation(topic: str, instructor_id: str):
    """Demo assessment generation functionality."""
    print("\n📝 Demo: Assessment Generation")
    print("=" * 50)
    
    # Create assessment request
    request = AssessmentRequest(
        topic=topic,
        difficulty=DifficultyLevel.MEDIUM,
        question_types=[
            QuestionType.MULTIPLE_CHOICE,
            QuestionType.TRUE_FALSE,
            QuestionType.SHORT_ANSWER
        ],
        num_questions=5,
        learning_objectives=[
            "Understand the different types of machine learning",
            "Identify key concepts in machine learning",
            "Recognize common algorithms and their applications"
        ],
        instructor_id=instructor_id
    )
    
    print(f"✅ Created assessment request for topic: {topic}")
    print(f"   Difficulty: {request.difficulty}")
    print(f"   Question types: {[qt.value for qt in request.question_types]}")
    print(f"   Number of questions: {request.num_questions}")
    
    # Generate assessment
    response = assessment_generator.generate_assessment(request)
    
    print(f"\n✅ Generated assessment in {response.generation_time:.2f} seconds")
    print(f"   Cache hit: {response.cache_hit}")
    print(f"   Assessment ID: {response.assessment.id}")
    print(f"   Total points: {response.assessment.total_points}")
    print(f"   Estimated time: {response.assessment.estimated_time} minutes")
    
    # Display questions
    print(f"\n📋 Generated Questions:")
    for i, question in enumerate(response.assessment.questions, 1):
        print(f"\nQuestion {i}:")
        print(f"   Type: {question.question_type.value}")
        print(f"   Difficulty: {question.difficulty.value}")
        print(f"   Text: {question.question_text}")
        if question.options:
            print(f"   Options: {question.options}")
        print(f"   Points: {question.points}")
        print(f"   Explanation: {question.explanation[:100]}...")


async def demo_difficulty_adjustment():
    """Demo difficulty adjustment functionality."""
    print("\n🎯 Demo: Difficulty Adjustment")
    print("=" * 50)
    
    # Create mock user performance
    performance = UserPerformance(
        user_id="demo_user",
        assessment_id="demo_assessment",
        score=3.0,
        total_questions=5,
        correct_answers=3,
        time_taken=15,
        completed_at=datetime.now()
    )
    
    print(f"✅ Created user performance:")
    print(f"   Score: {performance.score}/{performance.total_questions}")
    print(f"   Performance ratio: {performance.score/performance.total_questions:.2f}")
    
    # Test difficulty adjustment
    from src.models import Assessment, DifficultyLevel
    
    mock_assessment = Assessment(
        id="demo_assessment",
        title="Demo Assessment",
        description="Demo assessment",
        topic="demo_topic",
        difficulty=DifficultyLevel.MEDIUM,
        questions=[],
        total_points=0,
        estimated_time=0,
        learning_objectives=[],
        created_at=datetime.now(),
        instructor_id="demo_instructor"
    )
    
    adjusted_assessment = assessment_generator.adjust_assessment_difficulty(
        performance, mock_assessment
    )
    
    print(f"\n✅ Difficulty adjustment:")
    print(f"   Original difficulty: {mock_assessment.difficulty.value}")
    print(f"   Adjusted difficulty: {adjusted_assessment.difficulty.value}")


async def demo_caching():
    """Demo caching functionality."""
    print("\n💾 Demo: Caching System")
    print("=" * 50)
    
    # Get cache stats
    stats = assessment_cache.get_cache_stats()
    print(f"✅ Cache Statistics:")
    print(f"   Total requests: {stats.total_requests}")
    print(f"   Cache hits: {stats.cache_hits}")
    print(f"   Cache miss rate: {stats.cache_miss_rate:.2f}")
    print(f"   Average response time: {stats.average_response_time:.3f} seconds")
    
    # Get collection stats
    collection_stats = hybrid_rag.get_collection_stats()
    print(f"\n✅ Collection Statistics:")
    print(f"   Total documents: {collection_stats.get('total_documents', 0)}")
    print(f"   BM25 documents: {collection_stats.get('bm25_documents', 0)}")


async def main():
    """Main demo function."""
    print("🎓 Advanced Assessment Generation System - Demo")
    print("=" * 60)
    
    try:
        # Demo document processing
        topic, instructor_id = await demo_document_processing()
        
        # Demo hybrid RAG
        await demo_hybrid_rag()
        
        # Demo assessment generation
        await demo_assessment_generation(topic, instructor_id)
        
        # Demo difficulty adjustment
        await demo_difficulty_adjustment()
        
        # Demo caching
        await demo_caching()
        
        print("\n🎉 Demo completed successfully!")
        print("\n📚 Next steps:")
        print("   1. Start the API server: python main.py")
        print("   2. Visit http://localhost:8000/docs for interactive API documentation")
        print("   3. Upload your own educational documents")
        print("   4. Generate personalized assessments")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Make sure all dependencies are installed and Redis is running.")


if __name__ == "__main__":
    asyncio.run(main()) 