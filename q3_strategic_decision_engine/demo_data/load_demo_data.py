#!/usr/bin/env python3
"""
Demo Data Loader for Strategic Decision Engine
This script loads sample strategic documents and creates demo scenarios.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict

# Add the parent directory to the path to import backend modules
sys.path.append(str(Path(__file__).parent.parent))

from backend.services.document_service import DocumentService
from backend.services.vector_store_service import VectorStoreService
from backend.services.cache_service import CacheService
from backend.core.config import Settings
from backend.core.database import SessionLocal
from backend.models.database_models import Document, ChatSession, AnalysisResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoDataLoader:
    """Loads demo data into the Strategic Decision Engine."""
    
    def __init__(self):
        """Initialize the demo data loader."""
        self.settings = Settings()
        self.demo_data_dir = Path(__file__).parent
        self.document_service = DocumentService(self.settings)
        self.vector_store_service = VectorStoreService(self.settings)
        self.cache_service = CacheService(self.settings)
        
        # Demo documents to load
        self.demo_documents = [
            {
                'filename': 'strategic_plan_2024.txt',
                'title': 'Strategic Planning Document 2024',
                'type': 'strategic',
                'description': 'Comprehensive strategic plan for TechForward Solutions Inc.'
            },
            {
                'filename': 'financial_analysis_2024.txt',
                'title': 'Financial Analysis & Forecast 2024',
                'type': 'financial',
                'description': 'Detailed financial analysis and projections for 2024'
            },
            {
                'filename': 'competitive_analysis_2024.txt',
                'title': 'Competitive Analysis 2024',
                'type': 'competitive',
                'description': 'Comprehensive competitive landscape analysis'
            }
        ]
    
    async def load_demo_data(self):
        """Load all demo data into the system."""
        logger.info("Starting demo data loading process...")
        
        try:
            # Clear existing demo data
            await self.clear_demo_data()
            
            # Load demo documents
            documents = await self.load_demo_documents()
            
            # Create demo chat sessions
            chat_sessions = await self.create_demo_chat_sessions()
            
            # Create demo analysis results
            analysis_results = await self.create_demo_analysis_results()
            
            # Create demo scenarios
            await self.create_demo_scenarios()
            
            logger.info("Demo data loading completed successfully!")
            
            return {
                'documents': len(documents),
                'chat_sessions': len(chat_sessions),
                'analysis_results': len(analysis_results),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error loading demo data: {str(e)}")
            raise
    
    async def clear_demo_data(self):
        """Clear existing demo data."""
        logger.info("Clearing existing demo data...")
        
        with SessionLocal() as db:
            # Delete demo documents
            demo_docs = db.query(Document).filter(
                Document.metadata.contains({'demo': True})
            ).all()
            
            for doc in demo_docs:
                # Delete from vector store
                await self.vector_store_service.delete_document(doc.id)
                # Delete from database
                db.delete(doc)
            
            # Delete demo chat sessions
            demo_sessions = db.query(ChatSession).filter(
                ChatSession.metadata.contains({'demo': True})
            ).all()
            
            for session in demo_sessions:
                db.delete(session)
            
            # Delete demo analysis results
            demo_analyses = db.query(AnalysisResult).filter(
                AnalysisResult.metadata.contains({'demo': True})
            ).all()
            
            for analysis in demo_analyses:
                db.delete(analysis)
            
            db.commit()
        
        # Clear cache
        await self.cache_service.clear()
        
        logger.info("Demo data cleared successfully")
    
    async def load_demo_documents(self) -> List[Dict]:
        """Load demo documents into the system."""
        logger.info("Loading demo documents...")
        
        loaded_documents = []
        
        for doc_info in self.demo_documents:
            file_path = self.demo_data_dir / doc_info['filename']
            
            if not file_path.exists():
                logger.warning(f"Demo file not found: {file_path}")
                continue
            
            try:
                # Read document content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Process document
                doc_result = await self.document_service.process_document(
                    content=content,
                    filename=doc_info['filename'],
                    metadata={
                        'title': doc_info['title'],
                        'type': doc_info['type'],
                        'description': doc_info['description'],
                        'demo': True,
                        'uploaded_at': '2024-01-01T00:00:00Z'
                    }
                )
                
                # Add to vector store
                vector_doc = {
                    'id': doc_result['document_id'],
                    'content': content,
                    'metadata': {
                        'title': doc_info['title'],
                        'type': doc_info['type'],
                        'source': doc_info['filename'],
                        'demo': True
                    }
                }
                
                await self.vector_store_service.add_documents([vector_doc])
                
                loaded_documents.append(doc_result)
                logger.info(f"Loaded demo document: {doc_info['title']}")
                
            except Exception as e:
                logger.error(f"Error loading document {doc_info['filename']}: {str(e)}")
                continue
        
        return loaded_documents
    
    async def create_demo_chat_sessions(self) -> List[Dict]:
        """Create demo chat sessions."""
        logger.info("Creating demo chat sessions...")
        
        demo_sessions = [
            {
                'title': 'Strategic Planning Discussion',
                'description': 'Discussion about 2024 strategic objectives',
                'messages': [
                    {
                        'role': 'user',
                        'content': 'What are our key strategic objectives for 2024?'
                    },
                    {
                        'role': 'assistant',
                        'content': 'Based on the strategic planning document, the key objectives for 2024 include: 1) Revenue Growth targeting $20M (60% growth), 2) Market Expansion into Europe and APAC, 3) Technology Leadership with 25% R&D investment, and 4) Operational Excellence scaling to 75 employees.'
                    },
                    {
                        'role': 'user',
                        'content': 'What are the main risks we should be concerned about?'
                    },
                    {
                        'role': 'assistant',
                        'content': 'The main risks identified include: Strategic risks like technology disruption and competitive pressure, Operational risks such as data security breaches and service outages, and Financial risks including cash flow management and currency fluctuations. Each has specific mitigation strategies outlined.'
                    }
                ]
            },
            {
                'title': 'Financial Performance Review',
                'description': 'Analysis of financial metrics and projections',
                'messages': [
                    {
                        'role': 'user',
                        'content': 'How is our financial performance looking for 2024?'
                    },
                    {
                        'role': 'assistant',
                        'content': 'Our 2024 financial outlook is strong with projected 60% revenue growth to $20M, improved gross margins to 75%, and EBITDA margin target of 25%. We have $8.2M in cash providing 18 months runway with positive cash flow generation.'
                    },
                    {
                        'role': 'user',
                        'content': 'What are our key customer metrics?'
                    },
                    {
                        'role': 'assistant',
                        'content': 'Key customer metrics show healthy performance: CAC of $18,500, CLV of $245,000 (13.2x ratio), 92% gross revenue retention, 118% net revenue retention, and 2.1% monthly churn rate. We acquired 52 new customers in 2023.'
                    }
                ]
            },
            {
                'title': 'Competitive Landscape Analysis',
                'description': 'Discussion about market competition and positioning',
                'messages': [
                    {
                        'role': 'user',
                        'content': 'Who are our main competitors and how do we compare?'
                    },
                    {
                        'role': 'assistant',
                        'content': 'Our main direct competitors include DataRobot (8.5% market share, high-end enterprise), H2O.ai (5.2% share, technical focus), and Palantir (12.3% share, government focus). We differentiate through industry-specific solutions, rapid implementation, and competitive pricing for mid-market.'
                    },
                    {
                        'role': 'user',
                        'content': 'What are our competitive advantages?'
                    },
                    {
                        'role': 'assistant',
                        'content': 'Our key competitive advantages include: 1) Industry specialization with vertical-specific AI models, 2) Rapid 90-day implementation, 3) Ethical AI framework and transparency, 4) Strong customer relationships with 85% retention, and 5) Competitive pricing for mid-market segment.'
                    }
                ]
            }
        ]
        
        created_sessions = []
        
        with SessionLocal() as db:
            for session_data in demo_sessions:
                session = ChatSession(
                    title=session_data['title'],
                    description=session_data['description'],
                    metadata={'demo': True, 'created_for': 'demonstration'}
                )
                
                db.add(session)
                db.flush()  # Get the session ID
                
                # Add messages to session
                for message in session_data['messages']:
                    # In a real implementation, you'd store messages in a separate table
                    # For demo purposes, we'll store them in session metadata
                    pass
                
                created_sessions.append({
                    'session_id': session.id,
                    'title': session.title,
                    'message_count': len(session_data['messages'])
                })
            
            db.commit()
        
        logger.info(f"Created {len(created_sessions)} demo chat sessions")
        return created_sessions
    
    async def create_demo_analysis_results(self) -> List[Dict]:
        """Create demo analysis results."""
        logger.info("Creating demo analysis results...")
        
        demo_analyses = [
            {
                'title': 'SWOT Analysis - TechForward Solutions',
                'analysis_type': 'swot',
                'content': {
                    'strengths': [
                        'Deep expertise in vertical AI applications',
                        'Strong customer relationships with 85% retention rate',
                        'Proprietary AI models optimized for specific industries',
                        'Agile development methodology enabling rapid deployment',
                        'Comprehensive AI ethics framework'
                    ],
                    'weaknesses': [
                        'Limited brand recognition compared to major competitors',
                        'Smaller marketing budget',
                        'Limited global presence',
                        'Fewer strategic partnerships',
                        'Smaller customer base'
                    ],
                    'opportunities': [
                        'AI market projected to reach $1.8 trillion by 2030',
                        'European and APAC market expansion',
                        'Healthcare and government sector opportunities',
                        'Growing demand for ethical AI solutions',
                        'Increasing need for industry-specific AI'
                    ],
                    'threats': [
                        'Intense competition from tech giants',
                        'Rapid technological changes',
                        'Economic downturn affecting IT spending',
                        'Talent shortage in AI field',
                        'Potential regulatory changes'
                    ]
                }
            },
            {
                'title': 'Market Analysis - AI Solutions Sector',
                'analysis_type': 'market',
                'content': {
                    'market_size': '$180B serviceable addressable market',
                    'growth_rate': '22.5% CAGR through 2030',
                    'key_trends': [
                        'Democratization of AI technology',
                        'Increased focus on ethical AI',
                        'Growth in edge AI applications',
                        'Regulatory framework development',
                        'Consolidation in mid-market segment'
                    ],
                    'opportunities': [
                        'AI-powered business intelligence ($4.2B market)',
                        'Automated document processing ($2.8B market)',
                        'Predictive maintenance ($6.3B market)',
                        'Conversational AI ($1.6B market)'
                    ]
                }
            },
            {
                'title': 'Financial Forecast Analysis',
                'analysis_type': 'financial',
                'content': {
                    'revenue_forecast': {
                        '2024_target': '$20M (60% growth)',
                        'q1_2024': '$4.2M',
                        'q2_2024': '$4.8M',
                        'q3_2024': '$5.4M',
                        'q4_2024': '$5.6M'
                    },
                    'margin_projections': {
                        'gross_margin': '75% (improvement from 72%)',
                        'ebitda_margin': '25% (improvement from 22%)',
                        'operating_margin': '20% (improvement from 17%)',
                        'net_margin': '16% (improvement from 14%)'
                    },
                    'key_metrics': {
                        'cash_position': '$8.2M (18 months runway)',
                        'customer_acquisition_cost': '$18,500',
                        'customer_lifetime_value': '$245,000',
                        'monthly_churn_rate': '2.1%'
                    }
                }
            }
        ]
        
        created_analyses = []
        
        with SessionLocal() as db:
            for analysis_data in demo_analyses:
                analysis = AnalysisResult(
                    title=analysis_data['title'],
                    analysis_type=analysis_data['analysis_type'],
                    content=str(analysis_data['content']),  # Convert to string for storage
                    metadata={'demo': True, 'created_for': 'demonstration'}
                )
                
                db.add(analysis)
                db.flush()
                
                created_analyses.append({
                    'analysis_id': analysis.id,
                    'title': analysis.title,
                    'type': analysis.analysis_type
                })
            
            db.commit()
        
        logger.info(f"Created {len(created_analyses)} demo analysis results")
        return created_analyses
    
    async def create_demo_scenarios(self):
        """Create demo scenarios for testing different features."""
        logger.info("Creating demo scenarios...")
        
        scenarios = [
            {
                'name': 'Strategic Planning Scenario',
                'description': 'CEO reviewing strategic plan and asking questions',
                'documents': ['strategic_plan_2024.txt'],
                'sample_queries': [
                    'What are our competitive advantages?',
                    'What risks should we be most concerned about?',
                    'How do we plan to achieve our revenue targets?',
                    'What are the key success factors for our strategy?'
                ]
            },
            {
                'name': 'Financial Review Scenario',
                'description': 'CFO analyzing financial performance and projections',
                'documents': ['financial_analysis_2024.txt'],
                'sample_queries': [
                    'What are our key financial metrics?',
                    'How does our performance compare to industry benchmarks?',
                    'What are the main assumptions in our 2024 forecast?',
                    'What funding requirements do we have?'
                ]
            },
            {
                'name': 'Competitive Analysis Scenario',
                'description': 'VP of Strategy reviewing competitive landscape',
                'documents': ['competitive_analysis_2024.txt'],
                'sample_queries': [
                    'Who are our biggest competitive threats?',
                    'How do we differentiate from competitors?',
                    'What market opportunities should we pursue?',
                    'How can we strengthen our competitive position?'
                ]
            },
            {
                'name': 'Comprehensive Business Review',
                'description': 'Board meeting preparation with all strategic documents',
                'documents': ['strategic_plan_2024.txt', 'financial_analysis_2024.txt', 'competitive_analysis_2024.txt'],
                'sample_queries': [
                    'Provide an executive summary of our business performance',
                    'What are the biggest opportunities and risks?',
                    'How well positioned are we for growth?',
                    'What should be our top priorities?'
                ]
            }
        ]
        
        # Store scenarios in cache for quick access
        for scenario in scenarios:
            cache_key = f"demo_scenario:{scenario['name'].lower().replace(' ', '_')}"
            await self.cache_service.set(cache_key, scenario, ttl=86400)  # 24 hours
        
        logger.info(f"Created {len(scenarios)} demo scenarios")
    
    def get_demo_statistics(self) -> Dict:
        """Get statistics about loaded demo data."""
        with SessionLocal() as db:
            demo_docs = db.query(Document).filter(
                Document.metadata.contains({'demo': True})
            ).count()
            
            demo_sessions = db.query(ChatSession).filter(
                ChatSession.metadata.contains({'demo': True})
            ).count()
            
            demo_analyses = db.query(AnalysisResult).filter(
                AnalysisResult.metadata.contains({'demo': True})
            ).count()
        
        return {
            'documents': demo_docs,
            'chat_sessions': demo_sessions,
            'analysis_results': demo_analyses
        }


async def main():
    """Main function to run the demo data loader."""
    loader = DemoDataLoader()
    
    try:
        result = await loader.load_demo_data()
        print("\nDemo Data Loading Results:")
        print("=" * 40)
        print(f"Documents loaded: {result['documents']}")
        print(f"Chat sessions created: {result['chat_sessions']}")
        print(f"Analysis results created: {result['analysis_results']}")
        print(f"Status: {result['status']}")
        
        # Show statistics
        stats = loader.get_demo_statistics()
        print("\nDemo Data Statistics:")
        print("=" * 40)
        print(f"Total demo documents: {stats['documents']}")
        print(f"Total demo chat sessions: {stats['chat_sessions']}")
        print(f"Total demo analysis results: {stats['analysis_results']}")
        
        print("\nDemo data loaded successfully! 🎉")
        print("\nYou can now:")
        print("- Use the Streamlit frontend to explore the documents")
        print("- Try the chat functionality with the loaded content")
        print("- Generate analyses using the demo documents")
        print("- Evaluate the system using RAGAS metrics")
        
    except Exception as e:
        print(f"Error loading demo data: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 