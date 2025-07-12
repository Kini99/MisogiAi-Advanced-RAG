# Strategic Decision Engine

A comprehensive AI-powered strategic planning platform designed specifically for CEOs and strategic decision-makers. This platform leverages cutting-edge AI technologies including Google Gemini 2.5 Pro, OpenAI GPT-4o, and Claude 3.5 Sonnet to provide intelligent insights for business strategy development.

## 🎯 Overview

The Strategic Decision Engine transforms how organizations approach strategic planning by combining:

- **Document Intelligence**: Upload and analyze company documents, reports, financial data, and market research
- **Multi-LLM Analysis**: Leverage specialized AI models for different strategic analysis tasks
- **Advanced RAG System**: Hybrid retrieval combining dense and sparse search with intelligent reranking
- **Strategic Frameworks**: SWOT analysis, market expansion analysis, financial forecasting
- **Real-time Evaluation**: RAGAS framework for measuring AI response quality
- **Interactive Dashboard**: Streamlit-based interface with data visualization and chat functionality

## ✨ Key Features

### 🔍 Core Capabilities
- **Document Upload & Processing**: Support for PDF, DOCX, PPTX, XLSX, TXT, CSV files
- **Query Decomposition**: Break complex strategic questions into analytical components
- **Contextual Compression**: Filter relevant business insights from large document sets
- **Hybrid RAG**: Dense (semantic) + sparse (BM25) retrieval for comprehensive results
- **Advanced Reranking**: Multi-model ensemble for business-relevant results
- **Citation-based Responses**: Source tracking for strategic recommendations

### 🧠 AI-Powered Analysis
- **SWOT Analysis**: Comprehensive strengths, weaknesses, opportunities, threats analysis
- **Market Analysis**: Market expansion opportunities and competitive landscape
- **Financial Analysis**: Financial performance insights and forecasting
- **Strategic Planning**: Data-driven strategic recommendations

### 📊 Visualization & Reporting
- **Interactive Charts**: Financial tables and strategic visualizations
- **Real-time Dashboard**: Key metrics and performance indicators
- **Export Capabilities**: Download analyses and reports

### 🔬 Quality Assurance
- **RAGAS Evaluation**: Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness
- **Multi-LLM Validation**: Cross-reference insights across different AI models
- **Source Verification**: Track and validate information sources

## 🏗️ Architecture

### Backend (FastAPI)
```
backend/
├── main.py                 # FastAPI application entry point
├── core/
│   ├── config.py          # Configuration management
│   ├── database.py        # Database models and connections
│   └── logging_config.py  # Structured logging setup
├── services/
│   ├── llm_service.py     # Multi-LLM management (OpenAI, Anthropic, Google)
│   ├── vector_store_service.py  # ChromaDB + hybrid search
│   ├── cache_service.py   # Redis caching layer
│   └── document_service.py # Document processing pipeline
└── api/endpoints/         # REST API endpoints
```

### Frontend (Streamlit)
```
frontend/
└── streamlit_app.py       # Interactive dashboard and chat interface
```

### Technology Stack

#### Core AI Technologies
- **LLM Providers**: OpenAI (GPT-4o), Anthropic (Claude 3.5 Sonnet), Google (Gemini 2.5 Pro)
- **Framework**: LangChain for LLM orchestration
- **Embeddings**: OpenAI text-embedding-3-large
- **Vector Store**: ChromaDB for document storage and retrieval
- **Search**: BM25 (sparse) + semantic search (dense) hybrid approach
- **Reranking**: Sentence-transformers for result optimization

#### Backend Infrastructure
- **API Framework**: FastAPI with async support
- **Database**: PostgreSQL for structured data, ChromaDB for vectors
- **Caching**: Redis for high-performance caching
- **Processing**: Advanced document parsing (PDF, DOCX, PPTX, XLSX)

#### Frontend & Visualization
- **Interface**: Streamlit for rapid dashboard development
- **Charts**: Plotly for interactive visualizations
- **Styling**: Custom CSS for professional appearance

#### Evaluation & Quality
- **RAGAS**: Comprehensive RAG evaluation framework
- **Metrics**: Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness
- **Monitoring**: Structured logging with performance tracking

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL database
- Redis server
- API keys for OpenAI, Anthropic, and Google AI

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd q3_strategic_decision_engine
   ```

2. **Set up virtual environment** (required - install only in project venv)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp env_template.txt .env
   # Edit .env with your actual configuration
   ```

### Environment Configuration

Create a `.env` file with the following required variables:

```env
# Security
SECRET_KEY=your-secret-key-here-make-it-long-and-random

# Database
DATABASE_URL=postgresql://username:password@localhost:5432/strategic_db
REDIS_URL=redis://localhost:6379

# AI API Keys
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key

# Optional: External APIs for market data
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key
FRED_API_KEY=your-fred-api-key
QUANDL_API_KEY=your-quandl-api-key
```

### Database Setup

1. **Create PostgreSQL database**
   ```sql
   CREATE DATABASE strategic_db;
   CREATE USER strategic_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE strategic_db TO strategic_user;
   ```

2. **Start Redis server**
   ```bash
   redis-server
   ```

### Running the Application

1. **Start the backend server**
   ```bash
   cd backend
   python main.py
   ```
   The API will be available at `http://localhost:8000`
   API documentation: `http://localhost:8000/api/docs`

2. **Start the frontend** (in a new terminal)
   ```bash
   cd frontend
   streamlit run streamlit_app.py
   ```
   The dashboard will be available at `http://localhost:8501`

## 📚 Usage Guide

### 1. Document Management
- **Upload Documents**: Use the Document Management page to upload company files
- **Supported Formats**: PDF, DOCX, PPTX, XLSX, TXT, CSV
- **Processing**: Documents are automatically processed and indexed for search
- **Management**: View, delete, and track processing status of uploaded documents

### 2. Strategic Chat
- **Interactive Queries**: Ask strategic questions in natural language
- **Context-Aware**: Responses based on your uploaded documents
- **Source Citations**: Track which documents informed each response
- **Multi-LLM**: Different AI models specialized for different analysis types

### 3. Analysis Modules

#### SWOT Analysis
- **Automated Generation**: AI-powered SWOT analysis based on your documents
- **Customizable Scope**: Company-wide, product line, market segment, or custom
- **Visualizations**: Charts and matrices for presentation
- **Actionable Insights**: Strategic recommendations based on SWOT findings

#### Market Analysis
- **Trend Analysis**: Market size, growth rate, and trend identification
- **Competitive Intelligence**: Competitive landscape analysis
- **Opportunity Assessment**: Market expansion opportunities
- **Risk Evaluation**: Market threats and mitigation strategies

#### Financial Analysis
- **Performance Metrics**: Revenue, profit, ROI, and cash flow analysis
- **Forecasting**: AI-powered financial projections
- **Trend Visualization**: Charts and graphs for financial data
- **Strategic Recommendations**: Data-driven financial insights

### 4. Quality Monitoring
- **RAGAS Dashboard**: Monitor AI response quality in real-time
- **Evaluation Metrics**: Track faithfulness, relevancy, precision, recall, and correctness
- **Performance Trends**: Historical quality metrics and improvements
- **Report Generation**: Detailed evaluation reports for stakeholders

## 🔧 Advanced Configuration

### LLM Task Routing
The system automatically routes different types of queries to specialized models:

- **General Chat**: OpenAI GPT-4o
- **Document Analysis**: Google Gemini 2.5 Pro (excels at long documents)
- **Strategic Planning**: Anthropic Claude (excels at structured thinking)
- **Market Analysis**: Google Gemini 2.5 Pro (multimodal capabilities)
- **Financial Analysis**: OpenAI GPT-4o (mathematical reasoning)
- **SWOT Analysis**: Anthropic Claude (systematic analysis)

### Vector Store Configuration
```python
# Customize chunk size and overlap
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Adjust retrieval parameters
MAX_RETRIEVAL_RESULTS=10
SIMILARITY_THRESHOLD=0.7

# Reranking settings
RERANKER_TOP_K=5
```

### Caching Strategy
- **Document Chunks**: 24-hour cache for processed documents
- **Query Results**: 1-hour cache for search results
- **Analysis Results**: 2-hour cache for generated analyses
- **Embeddings**: 48-hour cache for embedding vectors

## 📊 Sample Use Cases

### 1. SWOT Analysis Generation
```
User: "Create a comprehensive SWOT analysis for our company based on our latest quarterly reports and market research."

System: 
- Retrieves relevant documents
- Analyzes strengths, weaknesses, opportunities, threats
- Provides actionable strategic recommendations
- Cites specific sources for each insight
```

### 2. Market Expansion Analysis
```
User: "Analyze our potential for expanding into the Asian markets based on our current capabilities and market data."

System:
- Reviews company capability documents
- Analyzes market research and industry reports
- Identifies expansion opportunities and challenges
- Provides risk assessment and mitigation strategies
```

### 3. Financial Performance Review
```
User: "Generate insights on our financial performance trends and forecast next quarter's performance."

System:
- Analyzes financial statements and reports
- Identifies performance trends and patterns
- Generates forecasts based on historical data
- Provides strategic recommendations for improvement
```

## 🔍 API Documentation

### Document Endpoints
- `POST /api/documents/upload` - Upload documents
- `GET /api/documents/list` - List all documents
- `GET /api/documents/{id}` - Get document details
- `DELETE /api/documents/{id}` - Delete document
- `POST /api/documents/process/{id}` - Process document

### Chat Endpoints
- `POST /api/chat/message` - Send chat message
- `GET /api/chat/history/{session_id}` - Get chat history
- `DELETE /api/chat/session/{session_id}` - Clear chat session

### Analysis Endpoints
- `POST /api/analysis/generate` - Generate strategic analysis
- `GET /api/analysis/results/{session_id}` - Get analysis results
- `POST /api/analysis/swot` - SWOT analysis
- `POST /api/analysis/market` - Market analysis

### Evaluation Endpoints
- `POST /api/evaluation/run` - Run RAGAS evaluation
- `GET /api/evaluation/metrics` - Get evaluation metrics
- `GET /api/evaluation/reports` - Get evaluation reports

## 🧪 Testing

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/api/
```

### Test Coverage
- Unit tests for core services
- Integration tests for API endpoints
- End-to-end tests for complete workflows
- Performance tests for large document processing

## 🚀 Deployment

### Production Deployment
1. **Environment Setup**
   ```bash
   # Set production environment variables
   export DEBUG=false
   export LOG_LEVEL=INFO
   ```

2. **Database Migration**
   ```bash
   # Run database migrations
   python -c "from backend.core.database import init_db; init_db()"
   ```

3. **Service Deployment**
   ```bash
   # Use production WSGI server
   gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker
   
   # Or with Docker
   docker-compose up -d
   ```

### Docker Deployment
```dockerfile
# Example Dockerfile for backend
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ ./backend/
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Scaling Considerations
- **Load Balancing**: Use nginx or cloud load balancers
- **Database**: PostgreSQL with read replicas
- **Caching**: Redis cluster for high availability
- **Vector Store**: ChromaDB with persistent storage
- **Monitoring**: Prometheus and Grafana for metrics

## 🛡️ Security

### Data Protection
- **Encryption**: All data encrypted in transit and at rest
- **API Security**: JWT-based authentication and authorization
- **Input Validation**: Comprehensive input sanitization
- **Rate Limiting**: API rate limiting to prevent abuse

### Privacy Considerations
- **Data Anonymization**: Optional PII masking in documents
- **Audit Logging**: Comprehensive audit trails
- **Access Control**: Role-based access control (RBAC)
- **Compliance**: GDPR and SOC2 compliance features

## 📈 Performance Optimization

### Response Time Optimization
- **Caching Strategy**: Multi-layer caching for frequently accessed data
- **Connection Pooling**: Database and Redis connection pools
- **Async Processing**: Background document processing
- **Load Balancing**: Distribute requests across multiple instances

### Resource Management
- **Memory Optimization**: Efficient document chunking and embedding
- **Disk Usage**: Automatic cleanup of temporary files
- **API Rate Limits**: Prevent resource exhaustion
- **Monitoring**: Real-time performance metrics

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Code Standards
- **Python**: Follow PEP 8 style guidelines
- **Type Hints**: Use type hints for all functions
- **Documentation**: Comprehensive docstrings
- **Testing**: Minimum 80% test coverage

### Commit Convention
```
feat: add new SWOT analysis endpoint
fix: resolve document processing timeout
docs: update API documentation
test: add integration tests for chat system
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the excellent LLM orchestration framework
- **OpenAI**: For GPT-4o and embedding models
- **Anthropic**: For Claude 3.5 Sonnet
- **Google**: For Gemini 2.5 Pro
- **ChromaDB**: For vector storage and retrieval
- **Streamlit**: For rapid frontend development
- **RAGAS**: For RAG evaluation framework

## 📞 Support

For questions, issues, or contributions:

1. **Issues**: Create a GitHub issue for bugs or feature requests
2. **Discussions**: Use GitHub Discussions for questions and ideas
3. **Documentation**: Check the `/docs` folder for detailed guides
4. **API Reference**: Visit `/api/docs` when running the server

---

**Strategic Decision Engine** - Empowering strategic excellence through AI-driven insights.

Built with ❤️ for strategic decision-makers worldwide. 