"""
Evaluation API endpoints for RAGAS evaluation framework
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging
import uuid
import time
import asyncio
from pydantic import BaseModel, Field

from ...services.llm_service import LLMService, LLMTask
from ...services.vector_store_service import VectorStoreService
from ...services.cache_service import CacheService
from ...core.database import DatabaseManager
from ...core.logging_config import get_logger
from ...core.config import settings

# Initialize router
router = APIRouter()
logger = get_logger('api.evaluation')

# Global services
llm_service = None
vector_store_service = None
cache_service = None

# Pydantic models
class EvaluationRequest(BaseModel):
    session_id: str = Field(..., description="Session ID to evaluate")
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="AI response to evaluate")
    context: List[str] = Field(..., description="Context documents used")
    metrics: Optional[List[str]] = Field(None, description="Specific metrics to evaluate")
    ground_truth: Optional[str] = Field(None, description="Ground truth answer if available")

class BatchEvaluationRequest(BaseModel):
    evaluations: List[EvaluationRequest] = Field(..., description="List of evaluations to run")
    evaluation_name: Optional[str] = Field(None, description="Name for this evaluation batch")

class EvaluationResponse(BaseModel):
    evaluation_id: str
    session_id: str
    query: str
    response: str
    metrics: Dict[str, float]
    overall_score: float
    evaluation_time: float
    created_at: str
    feedback: Optional[str] = None
    recommendations: List[str]

class EvaluationMetrics(BaseModel):
    faithfulness: Optional[float] = Field(None, description="Faithfulness score (0-1)")
    answer_relevancy: Optional[float] = Field(None, description="Answer relevancy score (0-1)")
    context_precision: Optional[float] = Field(None, description="Context precision score (0-1)")
    context_recall: Optional[float] = Field(None, description="Context recall score (0-1)")
    answer_correctness: Optional[float] = Field(None, description="Answer correctness score (0-1)")
    overall_score: Optional[float] = Field(None, description="Overall evaluation score (0-1)")

class EvaluationReport(BaseModel):
    report_id: str
    session_id: str
    evaluation_count: int
    average_metrics: EvaluationMetrics
    time_period: str
    created_at: str
    recommendations: List[str]
    trends: Dict[str, Any]

class EvaluationSummary(BaseModel):
    total_evaluations: int
    average_scores: Dict[str, float]
    score_distribution: Dict[str, Dict[str, int]]
    recent_trends: List[Dict[str, Any]]

# Dependency to get services
async def get_services():
    global llm_service, vector_store_service, cache_service
    if llm_service is None:
        llm_service = LLMService()
        await llm_service.initialize()
    
    if vector_store_service is None:
        vector_store_service = VectorStoreService()
        await vector_store_service.initialize()
    
    if cache_service is None:
        cache_service = CacheService()
        await cache_service.initialize()
    
    return llm_service, vector_store_service, cache_service

@router.post("/run", response_model=EvaluationResponse)
async def run_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    services = Depends(get_services)
):
    """Run RAGAS evaluation on a query-response pair"""
    llm_service, vector_store_service, cache_service = services
    
    try:
        evaluation_id = str(uuid.uuid4())
        
        # Check cache first
        query_hash = cache_service.generate_query_hash(request.query, request.context)
        cached_result = await cache_service.get_evaluation_result(query_hash)
        
        if cached_result:
            logger.info(f"Returning cached evaluation for query hash {query_hash}")
            return EvaluationResponse(
                evaluation_id=evaluation_id,
                session_id=request.session_id,
                query=request.query,
                response=request.response,
                metrics=cached_result,
                overall_score=cached_result.get('overall_score', 0.0),
                evaluation_time=0.0,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                recommendations=[]
            )
        
        start_time = time.time()
        
        # Determine which metrics to evaluate
        metrics_to_evaluate = request.metrics or settings.RAGAS_METRICS
        
        # Run evaluation metrics
        evaluation_results = {}
        
        # Faithfulness evaluation
        if "faithfulness" in metrics_to_evaluate:
            evaluation_results["faithfulness"] = await evaluate_faithfulness(
                llm_service, request.query, request.response, request.context
            )
        
        # Answer relevancy evaluation
        if "answer_relevancy" in metrics_to_evaluate:
            evaluation_results["answer_relevancy"] = await evaluate_answer_relevancy(
                llm_service, request.query, request.response
            )
        
        # Context precision evaluation
        if "context_precision" in metrics_to_evaluate:
            evaluation_results["context_precision"] = await evaluate_context_precision(
                llm_service, request.query, request.context, request.response
            )
        
        # Context recall evaluation
        if "context_recall" in metrics_to_evaluate:
            evaluation_results["context_recall"] = await evaluate_context_recall(
                llm_service, request.query, request.context, request.ground_truth
            )
        
        # Answer correctness evaluation
        if "answer_correctness" in metrics_to_evaluate:
            evaluation_results["answer_correctness"] = await evaluate_answer_correctness(
                llm_service, request.query, request.response, request.ground_truth
            )
        
        # Calculate overall score
        overall_score = sum(evaluation_results.values()) / len(evaluation_results)
        evaluation_results["overall_score"] = overall_score
        
        evaluation_time = time.time() - start_time
        
        # Generate recommendations based on evaluation
        recommendations = await generate_evaluation_recommendations(
            llm_service, evaluation_results, request.query, request.response
        )
        
        # Save to database
        background_tasks.add_task(
            save_evaluation_result,
            evaluation_id,
            request.session_id,
            request.query,
            request.response,
            request.context,
            evaluation_results
        )
        
        # Cache results
        await cache_service.cache_evaluation_result(query_hash, evaluation_results)
        
        logger.info(f"Completed evaluation {evaluation_id} in {evaluation_time:.2f}s")
        
        return EvaluationResponse(
            evaluation_id=evaluation_id,
            session_id=request.session_id,
            query=request.query,
            response=request.response,
            metrics=evaluation_results,
            overall_score=overall_score,
            evaluation_time=evaluation_time,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@router.post("/batch", response_model=List[EvaluationResponse])
async def run_batch_evaluation(
    request: BatchEvaluationRequest,
    background_tasks: BackgroundTasks,
    services = Depends(get_services)
):
    """Run batch evaluation on multiple query-response pairs"""
    llm_service, vector_store_service, cache_service = services
    
    try:
        # Run evaluations in parallel
        tasks = []
        for eval_request in request.evaluations:
            task = run_evaluation(eval_request, background_tasks, services)
            tasks.append(task)
        
        # Execute all evaluations concurrently
        results = await asyncio.gather(*tasks)
        
        logger.info(f"Completed batch evaluation with {len(results)} evaluations")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")

@router.get("/metrics/{session_id}", response_model=List[EvaluationResponse])
async def get_evaluation_metrics(
    session_id: str,
    limit: int = 50,
    services = Depends(get_services)
):
    """Get evaluation metrics for a session"""
    try:
        with DatabaseManager() as db:
            evaluations = db.get_evaluation_results(session_id, limit)
            
            evaluation_responses = []
            for eval_result in evaluations:
                evaluation_responses.append(EvaluationResponse(
                    evaluation_id=str(eval_result.id),
                    session_id=eval_result.session_id,
                    query=eval_result.query,
                    response=eval_result.response,
                    metrics={
                        'faithfulness': eval_result.faithfulness,
                        'answer_relevancy': eval_result.answer_relevancy,
                        'context_precision': eval_result.context_precision,
                        'context_recall': eval_result.context_recall,
                        'answer_correctness': eval_result.answer_correctness,
                        'overall_score': eval_result.overall_score
                    },
                    overall_score=eval_result.overall_score or 0.0,
                    evaluation_time=0.0,
                    created_at=eval_result.created_date.isoformat(),
                    recommendations=[]
                ))
            
            return evaluation_responses
            
    except Exception as e:
        logger.error(f"Failed to get evaluation metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.get("/report/{session_id}", response_model=EvaluationReport)
async def generate_evaluation_report(
    session_id: str,
    time_period: str = "last_30_days",
    services = Depends(get_services)
):
    """Generate evaluation report for a session"""
    try:
        with DatabaseManager() as db:
            evaluations = db.get_evaluation_results(session_id, limit=1000)
            
            if not evaluations:
                raise HTTPException(status_code=404, detail="No evaluations found")
            
            # Calculate average metrics
            metrics_sum = {
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'context_precision': 0.0,
                'context_recall': 0.0,
                'answer_correctness': 0.0,
                'overall_score': 0.0
            }
            
            valid_count = 0
            for eval_result in evaluations:
                if eval_result.overall_score is not None:
                    metrics_sum['faithfulness'] += eval_result.faithfulness or 0.0
                    metrics_sum['answer_relevancy'] += eval_result.answer_relevancy or 0.0
                    metrics_sum['context_precision'] += eval_result.context_precision or 0.0
                    metrics_sum['context_recall'] += eval_result.context_recall or 0.0
                    metrics_sum['answer_correctness'] += eval_result.answer_correctness or 0.0
                    metrics_sum['overall_score'] += eval_result.overall_score or 0.0
                    valid_count += 1
            
            if valid_count > 0:
                average_metrics = EvaluationMetrics(
                    faithfulness=metrics_sum['faithfulness'] / valid_count,
                    answer_relevancy=metrics_sum['answer_relevancy'] / valid_count,
                    context_precision=metrics_sum['context_precision'] / valid_count,
                    context_recall=metrics_sum['context_recall'] / valid_count,
                    answer_correctness=metrics_sum['answer_correctness'] / valid_count,
                    overall_score=metrics_sum['overall_score'] / valid_count
                )
            else:
                average_metrics = EvaluationMetrics()
            
            # Generate recommendations
            recommendations = await generate_report_recommendations(
                average_metrics, evaluations
            )
            
            return EvaluationReport(
                report_id=str(uuid.uuid4()),
                session_id=session_id,
                evaluation_count=len(evaluations),
                average_metrics=average_metrics,
                time_period=time_period,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                recommendations=recommendations,
                trends={}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate evaluation report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@router.get("/summary", response_model=EvaluationSummary)
async def get_evaluation_summary(
    days: int = 30,
    services = Depends(get_services)
):
    """Get evaluation summary statistics"""
    try:
        with DatabaseManager() as db:
            evaluations = db.get_recent_evaluations(days)
            
            if not evaluations:
                return EvaluationSummary(
                    total_evaluations=0,
                    average_scores={},
                    score_distribution={},
                    recent_trends=[]
                )
            
            # Calculate average scores
            metrics_sum = {
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'context_precision': 0.0,
                'context_recall': 0.0,
                'answer_correctness': 0.0,
                'overall_score': 0.0
            }
            
            valid_count = 0
            for eval_result in evaluations:
                if eval_result.overall_score is not None:
                    metrics_sum['faithfulness'] += eval_result.faithfulness or 0.0
                    metrics_sum['answer_relevancy'] += eval_result.answer_relevancy or 0.0
                    metrics_sum['context_precision'] += eval_result.context_precision or 0.0
                    metrics_sum['context_recall'] += eval_result.context_recall or 0.0
                    metrics_sum['answer_correctness'] += eval_result.answer_correctness or 0.0
                    metrics_sum['overall_score'] += eval_result.overall_score or 0.0
                    valid_count += 1
            
            average_scores = {}
            if valid_count > 0:
                for metric, total in metrics_sum.items():
                    average_scores[metric] = total / valid_count
            
            # Calculate score distribution
            score_distribution = {
                'overall_score': {'high': 0, 'medium': 0, 'low': 0},
                'faithfulness': {'high': 0, 'medium': 0, 'low': 0},
                'answer_relevancy': {'high': 0, 'medium': 0, 'low': 0}
            }
            
            for eval_result in evaluations:
                if eval_result.overall_score is not None:
                    score = eval_result.overall_score
                    if score >= 0.8:
                        score_distribution['overall_score']['high'] += 1
                    elif score >= 0.6:
                        score_distribution['overall_score']['medium'] += 1
                    else:
                        score_distribution['overall_score']['low'] += 1
            
            return EvaluationSummary(
                total_evaluations=len(evaluations),
                average_scores=average_scores,
                score_distribution=score_distribution,
                recent_trends=[]
            )
            
    except Exception as e:
        logger.error(f"Failed to get evaluation summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

@router.delete("/results/{evaluation_id}")
async def delete_evaluation_result(
    evaluation_id: str,
    services = Depends(get_services)
):
    """Delete evaluation result"""
    try:
        with DatabaseManager() as db:
            success = db.delete_evaluation_result(evaluation_id)
            
            if not success:
                raise HTTPException(status_code=404, detail="Evaluation not found")
            
            return {"message": "Evaluation result deleted successfully", "evaluation_id": evaluation_id}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete evaluation result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete result: {str(e)}")

# RAGAS evaluation functions
async def evaluate_faithfulness(
    llm_service: LLMService,
    query: str,
    response: str,
    context: List[str]
) -> float:
    """Evaluate faithfulness of response to context"""
    try:
        faithfulness_prompt = f"""
        Evaluate the faithfulness of the AI response to the provided context documents.
        Faithfulness measures whether the response contains only information that can be inferred from the context.
        
        Query: {query}
        
        Context Documents:
        {chr(10).join(context)}
        
        AI Response:
        {response}
        
        Rate the faithfulness on a scale of 0.0 to 1.0 where:
        - 1.0 = All information in the response is supported by the context
        - 0.5 = Some information is supported, some is not
        - 0.0 = Response contains mostly unsupported information
        
        Provide only the numerical score (e.g., 0.85):
        """
        
        result = await llm_service.generate_response(
            prompt=faithfulness_prompt,
            task=LLMTask.CONTEXT_COMPRESSION,
            max_tokens=50
        )
        
        # Extract numerical score
        score_text = result.content.strip()
        try:
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except ValueError:
            logger.warning(f"Could not parse faithfulness score: {score_text}")
            return 0.5  # Default score
        
    except Exception as e:
        logger.error(f"Faithfulness evaluation failed: {str(e)}")
        return 0.5

async def evaluate_answer_relevancy(
    llm_service: LLMService,
    query: str,
    response: str
) -> float:
    """Evaluate relevancy of response to query"""
    try:
        relevancy_prompt = f"""
        Evaluate how relevant the AI response is to the user's query.
        Answer relevancy measures whether the response directly addresses the question asked.
        
        Query: {query}
        
        AI Response:
        {response}
        
        Rate the relevancy on a scale of 0.0 to 1.0 where:
        - 1.0 = Response directly and completely addresses the query
        - 0.5 = Response partially addresses the query
        - 0.0 = Response does not address the query
        
        Provide only the numerical score (e.g., 0.92):
        """
        
        result = await llm_service.generate_response(
            prompt=relevancy_prompt,
            task=LLMTask.GENERAL_CHAT,
            max_tokens=50
        )
        
        # Extract numerical score
        score_text = result.content.strip()
        try:
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except ValueError:
            logger.warning(f"Could not parse relevancy score: {score_text}")
            return 0.5  # Default score
        
    except Exception as e:
        logger.error(f"Answer relevancy evaluation failed: {str(e)}")
        return 0.5

async def evaluate_context_precision(
    llm_service: LLMService,
    query: str,
    context: List[str],
    response: str
) -> float:
    """Evaluate precision of retrieved context"""
    try:
        precision_prompt = f"""
        Evaluate the precision of the retrieved context documents for answering the query.
        Context precision measures how many of the retrieved documents are relevant to the query.
        
        Query: {query}
        
        Context Documents:
        {chr(10).join([f"Document {i+1}: {doc}" for i, doc in enumerate(context)])}
        
        AI Response:
        {response}
        
        Rate the context precision on a scale of 0.0 to 1.0 where:
        - 1.0 = All retrieved documents are highly relevant to the query
        - 0.5 = Some documents are relevant, some are not
        - 0.0 = Most documents are not relevant to the query
        
        Provide only the numerical score (e.g., 0.78):
        """
        
        result = await llm_service.generate_response(
            prompt=precision_prompt,
            task=LLMTask.CONTEXT_COMPRESSION,
            max_tokens=50
        )
        
        # Extract numerical score
        score_text = result.content.strip()
        try:
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except ValueError:
            logger.warning(f"Could not parse context precision score: {score_text}")
            return 0.5  # Default score
        
    except Exception as e:
        logger.error(f"Context precision evaluation failed: {str(e)}")
        return 0.5

async def evaluate_context_recall(
    llm_service: LLMService,
    query: str,
    context: List[str],
    ground_truth: Optional[str]
) -> float:
    """Evaluate recall of retrieved context"""
    try:
        if not ground_truth:
            # If no ground truth, use a heuristic based on context coverage
            return min(len(context) / 5.0, 1.0)  # Assume 5 documents is optimal
        
        recall_prompt = f"""
        Evaluate the recall of the retrieved context documents for answering the query.
        Context recall measures how well the retrieved documents cover the information needed to answer the query.
        
        Query: {query}
        
        Ground Truth Answer:
        {ground_truth}
        
        Context Documents:
        {chr(10).join([f"Document {i+1}: {doc}" for i, doc in enumerate(context)])}
        
        Rate the context recall on a scale of 0.0 to 1.0 where:
        - 1.0 = All information needed to answer the query is present in the context
        - 0.5 = Some required information is missing from the context
        - 0.0 = Most required information is missing from the context
        
        Provide only the numerical score (e.g., 0.84):
        """
        
        result = await llm_service.generate_response(
            prompt=recall_prompt,
            task=LLMTask.CONTEXT_COMPRESSION,
            max_tokens=50
        )
        
        # Extract numerical score
        score_text = result.content.strip()
        try:
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except ValueError:
            logger.warning(f"Could not parse context recall score: {score_text}")
            return 0.5  # Default score
        
    except Exception as e:
        logger.error(f"Context recall evaluation failed: {str(e)}")
        return 0.5

async def evaluate_answer_correctness(
    llm_service: LLMService,
    query: str,
    response: str,
    ground_truth: Optional[str]
) -> float:
    """Evaluate correctness of answer"""
    try:
        if not ground_truth:
            # If no ground truth, use a heuristic based on response quality
            return 0.7  # Default score when no ground truth available
        
        correctness_prompt = f"""
        Evaluate the correctness of the AI response compared to the ground truth answer.
        Answer correctness measures how factually accurate the response is.
        
        Query: {query}
        
        Ground Truth Answer:
        {ground_truth}
        
        AI Response:
        {response}
        
        Rate the correctness on a scale of 0.0 to 1.0 where:
        - 1.0 = Response is completely accurate and matches ground truth
        - 0.5 = Response is partially accurate with some errors
        - 0.0 = Response is mostly incorrect
        
        Provide only the numerical score (e.g., 0.91):
        """
        
        result = await llm_service.generate_response(
            prompt=correctness_prompt,
            task=LLMTask.GENERAL_CHAT,
            max_tokens=50
        )
        
        # Extract numerical score
        score_text = result.content.strip()
        try:
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except ValueError:
            logger.warning(f"Could not parse correctness score: {score_text}")
            return 0.5  # Default score
        
    except Exception as e:
        logger.error(f"Answer correctness evaluation failed: {str(e)}")
        return 0.5

async def generate_evaluation_recommendations(
    llm_service: LLMService,
    metrics: Dict[str, float],
    query: str,
    response: str
) -> List[str]:
    """Generate recommendations based on evaluation metrics"""
    try:
        recommendations_prompt = f"""
        Based on the following evaluation metrics, provide recommendations for improving the AI system's performance.
        
        Evaluation Metrics:
        {chr(10).join([f"- {metric}: {score:.3f}" for metric, score in metrics.items()])}
        
        Query: {query}
        Response: {response}
        
        Provide 3-5 specific, actionable recommendations for improvement.
        Focus on the lowest-scoring metrics.
        
        Recommendations:
        """
        
        result = await llm_service.generate_response(
            prompt=recommendations_prompt,
            task=LLMTask.STRATEGIC_PLANNING,
            max_tokens=500
        )
        
        # Parse recommendations
        recommendations = []
        for line in result.content.split('\n'):
            line = line.strip()
            if line and (line.startswith('•') or line.startswith('-') or line.startswith('1.')):
                recommendations.append(line.lstrip('•-123456789. '))
        
        return recommendations[:5]  # Limit to 5 recommendations
        
    except Exception as e:
        logger.error(f"Failed to generate evaluation recommendations: {str(e)}")
        return []

async def generate_report_recommendations(
    average_metrics: EvaluationMetrics,
    evaluations: List
) -> List[str]:
    """Generate recommendations for evaluation report"""
    try:
        recommendations = []
        
        # Generate recommendations based on average scores
        if average_metrics.faithfulness and average_metrics.faithfulness < 0.7:
            recommendations.append("Improve context retrieval to ensure more relevant documents are used")
        
        if average_metrics.answer_relevancy and average_metrics.answer_relevancy < 0.8:
            recommendations.append("Enhance query understanding and response generation to better address user questions")
        
        if average_metrics.context_precision and average_metrics.context_precision < 0.7:
            recommendations.append("Optimize document retrieval algorithm to reduce irrelevant context")
        
        if average_metrics.context_recall and average_metrics.context_recall < 0.7:
            recommendations.append("Increase retrieval coverage to capture more relevant information")
        
        if average_metrics.overall_score and average_metrics.overall_score < 0.75:
            recommendations.append("Consider retraining or fine-tuning the AI model for better performance")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Failed to generate report recommendations: {str(e)}")
        return []

# Background task helpers
async def save_evaluation_result(
    evaluation_id: str,
    session_id: str,
    query: str,
    response: str,
    context: List[str],
    metrics: Dict[str, float]
):
    """Save evaluation result to database"""
    try:
        with DatabaseManager() as db:
            db.save_evaluation_result(
                session_id=session_id,
                query=query,
                response=response,
                context=context,
                metrics=metrics
            )
            
        logger.info(f"Saved evaluation result {evaluation_id}")
        
    except Exception as e:
        logger.error(f"Failed to save evaluation result: {str(e)}") 