# Sports Analytics RAG System

An advanced Retrieval-Augmented Generation (RAG) system specifically designed for sports analytics, capable of answering complex queries about player performance, team statistics, and game insights from a collection of sports documents.

## 🏈 Features

### Core RAG Capabilities
- **Query Decomposition**: Breaks complex multi-part questions into simpler sub-questions
- **Contextual Compression**: Reduces irrelevant information from retrieved documents
- **Basic Reranking**: Uses semantic similarity scores to improve document relevance
- **Citation-based Responses**: Shows which documents/sources support each claim
- **Vector Database**: ChromaDB for efficient document storage and retrieval

### Advanced Features
- **LLM-powered Query Decomposition**: Uses GPT-4 to intelligently break down complex queries
- **Semantic Reranking**: Cosine similarity-based document reranking
- **Context Compression**: AI-powered filtering of retrieved content for relevance
- **Citation Extraction**: Automatic identification and formatting of supporting evidence
- **FastAPI REST API**: Production-ready API with comprehensive endpoints

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- OpenAI API key
- 8GB+ RAM (for embedding models)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MisogiAi-Advanced-RAG
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

5. **Run the demo**
   ```bash
   python main.py
   ```

## 📖 Usage Examples

### Sample Queries

The system can handle complex sports analytics queries such as:

1. **"What are the top 3 teams in defense and their key defensive statistics?"**
2. **"Compare Messi's goal-scoring rate in the last season vs previous seasons"**
3. **"Which goalkeeper has the best save percentage in high-pressure situations?"**
4. **"Which team has the best defense and how does their goalkeeper compare to the league average?"**

### API Usage

#### Start the API Server
```bash
python main.py --api
```

#### Query the System
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What are the top 3 teams in defense and their key defensive statistics?",
       "include_decomposition": true,
       "include_citations": true,
       "max_results": 5
     }'
```

#### Add Documents
```bash
curl -X POST "http://localhost:8000/documents" \
     -H "Content-Type: application/json" \
     -d '[
       {
         "content": "Manchester City conceded only 23 goals in the 2023-24 season...",
         "metadata": {"type": "team_analysis", "league": "Premier League"},
         "source": "premier_league_stats_2024"
       }
     ]'
```

## 🏗️ System Architecture

### Components

1. **Vector Store** (`src/vector_store.py`)
   - ChromaDB integration for document storage
   - Sentence Transformers for embeddings
   - Efficient similarity search

2. **Query Decomposition** (`src/query_decomposition.py`)
   - LLM-powered query breakdown
   - Sub-question generation with reasoning
   - Priority-based processing

3. **Context Compression** (`src/context_compression.py`)
   - AI-powered content filtering
   - Relevance scoring
   - Fallback compression methods

4. **Reranker** (`src/reranker.py`)
   - Cosine similarity-based reranking
   - Document diversification
   - Duplicate detection

5. **Citation Extractor** (`src/citation_extractor.py`)
   - Automatic citation identification
   - Source validation
   - Confidence scoring

6. **RAG System** (`src/rag_system.py`)
   - Orchestrates all components
   - End-to-end query processing
   - Error handling and fallbacks

### Data Flow

```
Query Input → Query Decomposition → Document Retrieval → Reranking → Context Compression → Answer Generation → Citation Extraction → Response
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
OPENAI_MODEL=gpt-4-turbo-preview
CHROMA_PERSIST_DIRECTORY=./chroma_db
COLLECTION_NAME=sports_analytics
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=10
TOP_K_RERANK=5
MAX_SUB_QUESTIONS=5
COMPRESSION_RATIO=0.7
API_HOST=0.0.0.0
API_PORT=8000
```

### Configuration Options

- **CHUNK_SIZE**: Size of document chunks (default: 1000 characters)
- **CHUNK_OVERLAP**: Overlap between chunks (default: 200 characters)
- **TOP_K_RETRIEVAL**: Number of documents to retrieve (default: 10)
- **TOP_K_RERANK**: Number of documents after reranking (default: 5)
- **MAX_SUB_QUESTIONS**: Maximum sub-questions for decomposition (default: 5)

## 📊 API Endpoints

### Core Endpoints

- `GET /` - System information
- `GET /health` - Health check
- `POST /query` - Process sports analytics queries
- `POST /documents` - Add documents to the system
- `GET /status` - System status and statistics
- `GET /documents/count` - Total document count

### Query Request Format

```json
{
  "query": "Your sports analytics question",
  "include_decomposition": true,
  "include_citations": true,
  "max_results": 5
}
```

### Response Format

```json
{
  "query": "Original query",
  "answer": "Generated answer",
  "citations": [
    {
      "claim": "Specific claim",
      "source": "Document source",
      "page_section": "Page/section reference",
      "confidence": 0.95
    }
  ],
  "sub_questions": [
    {
      "question": "Sub-question",
      "reasoning": "Reasoning for sub-question",
      "priority": 1
    }
  ],
  "compressed_context": {
    "compressed_content": "Compressed context",
    "compression_ratio": 0.7,
    "relevance_score": 0.85
  },
  "processing_time": 2.34,
  "confidence_score": 0.88
}
```

## 🧪 Testing

### Run Demo
```bash
python main.py
```

### Run API Server
```bash
python main.py --api
```

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# System status
curl http://localhost:8000/status

# Document count
curl http://localhost:8000/documents/count
```

## 📁 Project Structure

```
MisogiAi-Advanced-RAG/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── models.py              # Pydantic data models
│   ├── vector_store.py        # ChromaDB vector store
│   ├── query_decomposition.py # Query decomposition logic
│   ├── context_compression.py # Context compression
│   ├── reranker.py           # Document reranking
│   ├── citation_extractor.py # Citation extraction
│   ├── rag_system.py         # Main RAG orchestrator
│   ├── api.py                # FastAPI application
│   ├── data_processor.py     # Document processing
│   └── sample_data.py        # Sample sports data
├── main.py                   # Main application entry point
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
└── README.md                # This file
```

## 🔍 Sample Data

The system includes comprehensive sample data covering:

- **Team Analysis**: Manchester City's defensive performance
- **Player Performance**: Messi's goal-scoring statistics
- **Goalkeeper Analysis**: Save percentages in high-pressure situations
- **Comparative Analysis**: Top defensive teams across leagues
- **Competition Analysis**: Champions League knockout stages
- **Player Comparisons**: Haaland vs Mbappé statistics

## 🛠️ Dependencies

### Core Dependencies
- **LangChain 0.1.0**: Latest version with LCEL syntax
- **ChromaDB 0.4.22**: Vector database for document storage
- **OpenAI 1.12.0**: Latest OpenAI API client
- **Sentence Transformers 2.2.2**: Embedding models
- **FastAPI 0.109.0**: Modern web framework
- **Pydantic 2.6.0**: Data validation

### Additional Dependencies
- **Uvicorn**: ASGI server
- **Python-dotenv**: Environment variable management
- **Scikit-learn**: Machine learning utilities
- **NumPy & Pandas**: Data processing
- **BeautifulSoup4**: Web scraping (if needed)

## 🚨 Error Handling

The system includes comprehensive error handling:

- **Configuration Validation**: Ensures required environment variables
- **API Error Handling**: Graceful error responses with status codes
- **Fallback Mechanisms**: Alternative processing when primary methods fail
- **Logging**: Detailed error logging for debugging