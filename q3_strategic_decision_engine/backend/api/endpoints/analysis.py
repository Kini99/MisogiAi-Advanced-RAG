"""
Analysis API endpoints for strategic business analysis
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging
import uuid
import time
import json
from pydantic import BaseModel, Field

from ...services.llm_service import LLMService, LLMTask
from ...services.vector_store_service import VectorStoreService
from ...services.cache_service import CacheService
from ...core.database import DatabaseManager
from ...core.logging_config import get_logger
from ...core.config import settings

# Initialize router
router = APIRouter()
logger = get_logger('api.analysis')

# Global services
llm_service = None
vector_store_service = None
cache_service = None

# Pydantic models
class AnalysisRequest(BaseModel):
    query: str = Field(..., description="Analysis query or question")
    analysis_type: str = Field(..., description="Type of analysis: swot, market, financial, strategic")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs to analyze")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional analysis parameters")
    include_charts: bool = Field(True, description="Include chart generation")

class SWOTAnalysisRequest(BaseModel):
    scope: str = Field("company", description="Analysis scope: company, product, market, project")
    session_id: Optional[str] = Field(None, description="Session ID")
    document_ids: Optional[List[str]] = Field(None, description="Document IDs to analyze")
    custom_prompt: Optional[str] = Field(None, description="Custom analysis prompt")

class MarketAnalysisRequest(BaseModel):
    market_type: str = Field("expansion", description="Market analysis type: expansion, competitive, trends")
    geographic_scope: Optional[str] = Field(None, description="Geographic scope for analysis")
    time_horizon: Optional[str] = Field("12 months", description="Time horizon for analysis")
    session_id: Optional[str] = Field(None, description="Session ID")
    document_ids: Optional[List[str]] = Field(None, description="Document IDs to analyze")

class FinancialAnalysisRequest(BaseModel):
    analysis_focus: str = Field("performance", description="Focus: performance, forecasting, risk, investment")
    time_period: Optional[str] = Field("quarterly", description="Time period for analysis")
    metrics: Optional[List[str]] = Field(None, description="Specific metrics to analyze")
    session_id: Optional[str] = Field(None, description="Session ID")
    document_ids: Optional[List[str]] = Field(None, description="Document IDs to analyze")

class AnalysisResponse(BaseModel):
    analysis_id: str
    session_id: str
    analysis_type: str
    result: str
    confidence_score: float
    sources: List[Dict[str, Any]]
    charts_data: Optional[Dict[str, Any]] = None
    recommendations: List[str]
    key_insights: List[str]
    processing_time: float
    created_at: str

class AnalysisStatus(BaseModel):
    analysis_id: str
    status: str
    progress: float
    message: str
    estimated_completion: Optional[str] = None

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

@router.post("/generate", response_model=AnalysisResponse)
async def generate_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    services = Depends(get_services)
):
    """Generate comprehensive strategic analysis"""
    llm_service, vector_store_service, cache_service = services
    
    try:
        # Generate analysis ID and session ID
        analysis_id = str(uuid.uuid4())
        session_id = request.session_id or str(uuid.uuid4())
        
        # Check cache
        cache_key = f"analysis:{request.analysis_type}:{hash(request.query)}"
        cached_result = await cache_service.get_analysis_result(session_id, request.analysis_type)
        
        if cached_result:
            logger.info(f"Returning cached analysis for {request.analysis_type}")
            return AnalysisResponse(**cached_result)
        
        start_time = time.time()
        
        # Retrieve relevant documents
        relevant_docs = []
        sources = []
        
        if request.document_ids:
            # Use specific documents
            for doc_id in request.document_ids:
                doc_chunks = await vector_store_service.get_document_chunks(doc_id)
                relevant_docs.extend(doc_chunks)
        else:
            # Search for relevant documents
            search_results = await vector_store_service.hybrid_search(
                query=request.query,
                k=15,
                threshold=0.6
            )
            relevant_docs = search_results
        
        # Extract source information
        for doc in relevant_docs:
            source_info = {
                "document_id": doc.metadata.get("document_id", "unknown"),
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "similarity_score": doc.metadata.get("similarity_score", 0.0),
                "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            }
            sources.append(source_info)
        
        # Prepare context for LLM
        context = ""
        if relevant_docs:
            context = "\n\n".join([f"Document: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in relevant_docs[:8]])
        
        # Select appropriate LLM task
        task_mapping = {
            "swot": LLMTask.SWOT_ANALYSIS,
            "market": LLMTask.MARKET_ANALYSIS,
            "financial": LLMTask.FINANCIAL_ANALYSIS,
            "strategic": LLMTask.STRATEGIC_PLANNING,
            "competitive": LLMTask.MARKET_ANALYSIS,
            "risk": LLMTask.STRATEGIC_PLANNING
        }
        
        task = task_mapping.get(request.analysis_type, LLMTask.STRATEGIC_PLANNING)
        
        # Generate system message based on analysis type
        system_message = generate_system_message(request.analysis_type, context)
        
        # Generate main analysis
        response = await llm_service.generate_response(
            prompt=request.query,
            system_message=system_message,
            task=task,
            temperature=0.1,
            max_tokens=3000
        )
        
        # Generate recommendations
        recommendations = await generate_recommendations(
            llm_service, response.content, request.analysis_type, context
        )
        
        # Extract key insights
        key_insights = await extract_key_insights(
            llm_service, response.content, request.analysis_type
        )
        
        # Generate charts data if requested
        charts_data = None
        if request.include_charts:
            charts_data = await generate_charts_data(
                llm_service, response.content, request.analysis_type
            )
        
        # Calculate confidence score
        confidence_score = calculate_confidence_score(relevant_docs, response.content)
        
        processing_time = time.time() - start_time
        
        # Save to database
        background_tasks.add_task(
            save_analysis_result,
            analysis_id,
            session_id,
            request.analysis_type,
            request.query,
            response.content,
            charts_data,
            sources,
            response.model,
            confidence_score,
            recommendations,
            key_insights
        )
        
        # Cache result
        cache_data = {
            "analysis_id": analysis_id,
            "session_id": session_id,
            "analysis_type": request.analysis_type,
            "result": response.content,
            "confidence_score": confidence_score,
            "sources": sources,
            "charts_data": charts_data,
            "recommendations": recommendations,
            "key_insights": key_insights,
            "processing_time": processing_time,
            "created_at": time.time()
        }
        
        await cache_service.cache_analysis_result(session_id, request.analysis_type, cache_data)
        
        logger.info(f"Generated {request.analysis_type} analysis in {processing_time:.2f}s")
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            session_id=session_id,
            analysis_type=request.analysis_type,
            result=response.content,
            confidence_score=confidence_score,
            sources=sources,
            charts_data=charts_data,
            recommendations=recommendations,
            key_insights=key_insights,
            processing_time=processing_time,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        logger.error(f"Analysis generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/swot", response_model=AnalysisResponse)
async def swot_analysis(
    request: SWOTAnalysisRequest,
    background_tasks: BackgroundTasks,
    services = Depends(get_services)
):
    """Generate SWOT analysis"""
    llm_service, vector_store_service, cache_service = services
    
    try:
        analysis_id = str(uuid.uuid4())
        session_id = request.session_id or str(uuid.uuid4())
        
        # Prepare SWOT-specific query
        swot_query = request.custom_prompt or f"""
        Conduct a comprehensive SWOT analysis for the {request.scope}. 
        Analyze the Strengths, Weaknesses, Opportunities, and Threats based on the provided documents.
        
        Structure your response as follows:
        ## SWOT Analysis Summary
        
        ### Strengths (Internal Positive Factors)
        - List specific strengths with supporting evidence
        
        ### Weaknesses (Internal Negative Factors)
        - List specific weaknesses with supporting evidence
        
        ### Opportunities (External Positive Factors)
        - List specific opportunities with supporting evidence
        
        ### Threats (External Negative Factors)
        - List specific threats with supporting evidence
        
        ## Strategic Recommendations
        - Provide actionable recommendations based on the SWOT analysis
        
        ## Risk Mitigation Strategies
        - Suggest strategies to address identified weaknesses and threats
        """
        
        # Create analysis request
        analysis_request = AnalysisRequest(
            query=swot_query,
            analysis_type="swot",
            session_id=session_id,
            document_ids=request.document_ids,
            include_charts=True
        )
        
        # Generate analysis
        result = await generate_analysis(analysis_request, background_tasks, services)
        
        return result
        
    except Exception as e:
        logger.error(f"SWOT analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SWOT analysis failed: {str(e)}")

@router.post("/market", response_model=AnalysisResponse)
async def market_analysis(
    request: MarketAnalysisRequest,
    background_tasks: BackgroundTasks,
    services = Depends(get_services)
):
    """Generate market analysis"""
    llm_service, vector_store_service, cache_service = services
    
    try:
        analysis_id = str(uuid.uuid4())
        session_id = request.session_id or str(uuid.uuid4())
        
        # Prepare market-specific query
        market_query = f"""
        Conduct a comprehensive market analysis focusing on {request.market_type} opportunities.
        Time horizon: {request.time_horizon}
        Geographic scope: {request.geographic_scope or 'Global'}
        
        Structure your analysis as follows:
        ## Market Analysis Summary
        
        ### Market Size and Growth
        - Current market size and growth trends
        - Key growth drivers and market dynamics
        
        ### Market Segmentation
        - Key market segments and their characteristics
        - Target customer profiles and needs
        
        ### Competitive Landscape
        - Major competitors and their market positions
        - Competitive advantages and disadvantages
        
        ### Market Opportunities
        - Emerging opportunities and market gaps
        - Potential for market expansion
        
        ### Market Threats and Challenges
        - Regulatory and economic challenges
        - Competitive threats and market risks
        
        ## Strategic Recommendations
        - Market entry or expansion strategies
        - Competitive positioning recommendations
        """
        
        # Create analysis request
        analysis_request = AnalysisRequest(
            query=market_query,
            analysis_type="market",
            session_id=session_id,
            document_ids=request.document_ids,
            include_charts=True
        )
        
        # Generate analysis
        result = await generate_analysis(analysis_request, background_tasks, services)
        
        return result
        
    except Exception as e:
        logger.error(f"Market analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Market analysis failed: {str(e)}")

@router.post("/financial", response_model=AnalysisResponse)
async def financial_analysis(
    request: FinancialAnalysisRequest,
    background_tasks: BackgroundTasks,
    services = Depends(get_services)
):
    """Generate financial analysis"""
    llm_service, vector_store_service, cache_service = services
    
    try:
        analysis_id = str(uuid.uuid4())
        session_id = request.session_id or str(uuid.uuid4())
        
        # Prepare financial-specific query
        metrics_text = ", ".join(request.metrics) if request.metrics else "all key financial metrics"
        
        financial_query = f"""
        Conduct a comprehensive financial analysis focusing on {request.analysis_focus}.
        Time period: {request.time_period}
        Key metrics to analyze: {metrics_text}
        
        Structure your analysis as follows:
        ## Financial Analysis Summary
        
        ### Financial Performance Overview
        - Key financial metrics and trends
        - Revenue and profitability analysis
        
        ### Financial Health Assessment
        - Liquidity and solvency analysis
        - Debt and capital structure evaluation
        
        ### Performance Benchmarking
        - Industry comparison and benchmarking
        - Historical performance trends
        
        ### Risk Assessment
        - Financial risks and vulnerabilities
        - Risk mitigation strategies
        
        ### Financial Forecasting
        - Future performance projections
        - Scenario analysis and sensitivity testing
        
        ## Strategic Financial Recommendations
        - Capital allocation strategies
        - Investment and financing recommendations
        """
        
        # Create analysis request
        analysis_request = AnalysisRequest(
            query=financial_query,
            analysis_type="financial",
            session_id=session_id,
            document_ids=request.document_ids,
            include_charts=True
        )
        
        # Generate analysis
        result = await generate_analysis(analysis_request, background_tasks, services)
        
        return result
        
    except Exception as e:
        logger.error(f"Financial analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Financial analysis failed: {str(e)}")

@router.get("/results/{session_id}")
async def get_analysis_results(
    session_id: str,
    analysis_type: Optional[str] = None,
    limit: int = 10,
    services = Depends(get_services)
):
    """Get analysis results for a session"""
    try:
        with DatabaseManager() as db:
            results = db.get_analysis_results(session_id, analysis_type, limit)
            
            return {
                "session_id": session_id,
                "results": results,
                "total": len(results)
            }
            
    except Exception as e:
        logger.error(f"Failed to get analysis results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get results: {str(e)}")

@router.get("/status/{analysis_id}", response_model=AnalysisStatus)
async def get_analysis_status(
    analysis_id: str,
    services = Depends(get_services)
):
    """Get analysis status"""
    try:
        with DatabaseManager() as db:
            analysis = db.get_analysis_by_id(analysis_id)
            
            if not analysis:
                raise HTTPException(status_code=404, detail="Analysis not found")
            
            return AnalysisStatus(
                analysis_id=analysis_id,
                status="completed",
                progress=100.0,
                message="Analysis completed successfully"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.delete("/results/{analysis_id}")
async def delete_analysis_result(
    analysis_id: str,
    services = Depends(get_services)
):
    """Delete analysis result"""
    try:
        with DatabaseManager() as db:
            success = db.delete_analysis_result(analysis_id)
            
            if not success:
                raise HTTPException(status_code=404, detail="Analysis not found")
            
            return {"message": "Analysis result deleted successfully", "analysis_id": analysis_id}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete analysis result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete result: {str(e)}")

# Helper functions
def generate_system_message(analysis_type: str, context: str) -> str:
    """Generate system message based on analysis type"""
    
    base_message = f"""You are an expert strategic business consultant with deep expertise in {analysis_type} analysis. 
    You provide actionable insights and recommendations based on document analysis.
    
    Context Documents:
    {context}
    
    Instructions:
    1. Analyze the provided documents thoroughly
    2. Provide specific, actionable recommendations
    3. Use data and evidence from the documents to support your analysis
    4. Structure your response clearly with headings and bullet points
    5. Focus on strategic business value and ROI
    6. Include risk assessment and mitigation strategies
    7. Provide quantitative insights where possible
    """
    
    type_specific_messages = {
        "swot": "Focus on internal strengths/weaknesses and external opportunities/threats. Provide strategic recommendations for leveraging strengths and addressing weaknesses.",
        "market": "Focus on market dynamics, competitive landscape, and growth opportunities. Provide market entry and expansion strategies.",
        "financial": "Focus on financial performance, risk assessment, and investment recommendations. Provide financial forecasting and strategic financial planning guidance.",
        "strategic": "Focus on overall strategic positioning and competitive advantage. Provide comprehensive strategic planning recommendations."
    }
    
    specific_message = type_specific_messages.get(analysis_type, "")
    
    return f"{base_message}\n\nSpecific Focus: {specific_message}"

async def generate_recommendations(
    llm_service: LLMService,
    analysis_content: str,
    analysis_type: str,
    context: str
) -> List[str]:
    """Generate strategic recommendations based on analysis"""
    
    try:
        recommendations_prompt = f"""
        Based on the following {analysis_type} analysis, generate 5-7 specific, actionable strategic recommendations.
        
        Analysis:
        {analysis_content}
        
        Context:
        {context}
        
        Format each recommendation as a clear, actionable statement. Focus on high-impact, implementable strategies.
        
        Recommendations:
        """
        
        response = await llm_service.generate_response(
            prompt=recommendations_prompt,
            task=LLMTask.STRATEGIC_PLANNING,
            max_tokens=800
        )
        
        # Parse recommendations from response
        recommendations = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line and (line.startswith('•') or line.startswith('-') or line.startswith('1.')):
                recommendations.append(line.lstrip('•-123456789. '))
        
        return recommendations[:7]  # Limit to 7 recommendations
        
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {str(e)}")
        return []

async def extract_key_insights(
    llm_service: LLMService,
    analysis_content: str,
    analysis_type: str
) -> List[str]:
    """Extract key insights from analysis"""
    
    try:
        insights_prompt = f"""
        Extract 5-6 key insights from the following {analysis_type} analysis.
        Focus on the most important findings and strategic implications.
        
        Analysis:
        {analysis_content}
        
        Format each insight as a concise, impactful statement.
        
        Key Insights:
        """
        
        response = await llm_service.generate_response(
            prompt=insights_prompt,
            task=LLMTask.STRATEGIC_PLANNING,
            max_tokens=600
        )
        
        # Parse insights from response
        insights = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line and (line.startswith('•') or line.startswith('-') or line.startswith('1.')):
                insights.append(line.lstrip('•-123456789. '))
        
        return insights[:6]  # Limit to 6 insights
        
    except Exception as e:
        logger.error(f"Failed to extract key insights: {str(e)}")
        return []

async def generate_charts_data(
    llm_service: LLMService,
    analysis_content: str,
    analysis_type: str
) -> Dict[str, Any]:
    """Generate charts data for visualization"""
    
    try:
        charts_prompt = f"""
        Based on the following {analysis_type} analysis, suggest appropriate charts and visualizations.
        Generate sample data and chart specifications in JSON format.
        
        Analysis:
        {analysis_content}
        
        Provide chart data in this format:
        {{
            "charts": [
                {{
                    "type": "bar|line|pie|scatter",
                    "title": "Chart Title",
                    "data": {{
                        "labels": ["Label1", "Label2", "Label3"],
                        "values": [10, 20, 30]
                    }},
                    "description": "Chart description"
                }}
            ]
        }}
        
        Generate 2-3 relevant charts:
        """
        
        response = await llm_service.generate_response(
            prompt=charts_prompt,
            task=LLMTask.CHART_GENERATION,
            max_tokens=1000
        )
        
        # Try to parse JSON response
        try:
            charts_data = json.loads(response.content)
            return charts_data
        except json.JSONDecodeError:
            # Return default chart structure if JSON parsing fails
            return {
                "charts": [
                    {
                        "type": "bar",
                        "title": f"{analysis_type.title()} Analysis Overview",
                        "data": {
                            "labels": ["Category 1", "Category 2", "Category 3"],
                            "values": [25, 35, 40]
                        },
                        "description": f"Overview of {analysis_type} analysis results"
                    }
                ]
            }
        
    except Exception as e:
        logger.error(f"Failed to generate charts data: {str(e)}")
        return None

def calculate_confidence_score(relevant_docs: List, analysis_content: str) -> float:
    """Calculate confidence score based on available data"""
    
    try:
        # Base confidence on number of relevant documents
        doc_score = min(len(relevant_docs) / 10.0, 1.0)  # Max score when 10+ docs
        
        # Base confidence on analysis length (more detailed = higher confidence)
        content_score = min(len(analysis_content) / 2000.0, 1.0)  # Max score at 2000+ chars
        
        # Average the scores
        confidence = (doc_score + content_score) / 2.0
        
        # Ensure minimum confidence of 0.5
        return max(confidence, 0.5)
        
    except Exception as e:
        logger.error(f"Failed to calculate confidence score: {str(e)}")
        return 0.7  # Default confidence

# Background task helpers
async def save_analysis_result(
    analysis_id: str,
    session_id: str,
    analysis_type: str,
    query: str,
    result: str,
    charts_data: Dict[str, Any],
    sources: List[Dict[str, Any]],
    model_used: str,
    confidence_score: float,
    recommendations: List[str],
    key_insights: List[str]
):
    """Save analysis result to database"""
    
    try:
        with DatabaseManager() as db:
            db.save_analysis_result(
                session_id=session_id,
                analysis_type=analysis_type,
                query=query,
                result=result,
                charts_data=charts_data,
                sources=sources,
                model_used=model_used,
                confidence_score=confidence_score
            )
            
        logger.info(f"Saved analysis result {analysis_id}")
        
    except Exception as e:
        logger.error(f"Failed to save analysis result: {str(e)}") 