# Advanced Assessment Generation System

An advanced system for generating personalized educational assessments using hybrid RAG (Retrieval-Augmented Generation) with dense and sparse retrieval methods, advanced reranking, contextual compression, and Redis caching.

## 🚀 Features

### Core Functionality
- **Hybrid RAG System**: Combines dense embeddings (sentence-transformers) with sparse retrieval (BM25)
- **Advanced Reranking**: Uses cross-encoder models for sophisticated result reranking
- **Contextual Compression**: Dynamic chunk sizing based on content characteristics
- **Redis Caching**: Comprehensive caching layer for frequently accessed content
- **Tool Calling Integration**: Educational content APIs for enhanced question generation
- **Dynamic Difficulty Adjustment**: Personalized assessment difficulty based on user performance

### Assessment Generation
- **Multiple Question Types**: Multiple choice, true/false, short answer, and essay questions
- **Difficulty Levels**: Easy, medium, and hard with appropriate question distribution
- **Learning Objectives**: Aligned assessment generation with specific learning goals
- **Personalized Content**: Tailored assessments based on uploaded educational materials

### Document Processing
- **Multi-format Support**: PDF, DOCX, TXT, and Markdown files
- **Intelligent Chunking**: Dynamic chunk sizing based on topic complexity
- **Metadata Extraction**: Comprehensive document metadata tracking
- **Content Validation**: File validation and error handling

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  Document Proc  │    │  Hybrid RAG     │
│                 │    │                 │    │                 │
│  - Upload Docs  │───▶│  - Text Extract │───▶│  - Dense Search │
│  - Generate Qs  │    │  - Chunking     │    │  - Sparse Search│
│  - Cache Mgmt   │    │  - Compression  │    │  - Reranking    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Redis Cache   │    │  ChromaDB       │    │  BM25 Index     │
│                 │    │                 │    │                 │
│  - Assessments  │    │  - Embeddings   │    │  - Sparse Vecs  │
│  - Chunks       │    │  - Metadata     │    │  - Tokenization │
│  - User Prefs   │    │  - Collections  │    │  - Scoring      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Requirements

### System Requirements
- Python 3.11+
- Redis Server
- 8GB+ RAM (for model loading)
- 2GB+ free disk space

### Dependencies
All dependencies are specified in `requirements.txt` with compatible version ranges to avoid conflicts.

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd q2_automated_quiz_generator
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

4. **Set up Redis**
   ```bash
   # Install Redis (Ubuntu/Debian)
   sudo apt-get install redis-server
   
   # Or use Docker
   docker run -d -p 6379:6379 redis:latest
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and other settings
   ```

## ⚙️ Configuration

Create a `.env` file with the following settings:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# File Upload Configuration
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=10485760  # 10MB
```

## 🚀 Usage

### 1. Start the API Server
```bash
python main.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 2. Run the Demo
```bash
python demo.py
```

This will demonstrate all system features with sample data.

### 3. API Endpoints

#### Upload Educational Document
```bash
curl -X POST "http://localhost:8000/upload-document" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf" \
  -F "topic=machine_learning" \
  -F "instructor_id=instructor_123"
```

#### Generate Assessment
```bash
curl -X POST "http://localhost:8000/generate-assessment" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "machine_learning",
    "difficulty": "medium",
    "question_types": ["multiple_choice", "true_false", "short_answer"],
    "num_questions": 10,
    "learning_objectives": ["Understand ML types", "Identify algorithms"],
    "instructor_id": "instructor_123"
  }'
```

#### Get System Statistics
```bash
# Cache statistics
curl "http://localhost:8000/cache-stats"

# Collection statistics
curl "http://localhost:8000/collection-stats"

# Available difficulty levels
curl "http://localhost:8000/difficulty-levels"

# Available question types
curl "http://localhost:8000/question-types"
```

## 🔧 System Components

### 1. Document Processor (`src/document_processor.py`)
- Extracts text from multiple file formats
- Implements dynamic chunk sizing
- Performs contextual compression
- Validates and cleans content

### 2. Hybrid RAG (`src/hybrid_rag.py`)
- Combines dense (ChromaDB) and sparse (BM25) retrieval
- Uses cross-encoder models for reranking
- Implements weighted fusion of results
- Supports topic-based caching

### 3. Assessment Generator (`src/assessment_generator.py`)
- Generates questions using LLM with tool calling
- Implements dynamic difficulty adjustment
- Integrates with educational content APIs
- Provides fallback question generation

### 4. Caching Layer (`src/cache.py`)
- Redis-based caching for assessments and chunks
- User preferences and history caching
- Cache statistics and monitoring
- Topic-based cache invalidation

### 5. API Layer (`src/api.py`)
- FastAPI application with comprehensive endpoints
- File upload handling with validation
- Error handling and response formatting
- CORS support for frontend integration

## 📊 Performance Features

### Caching Strategy
- **Assessment Cache**: 2-hour TTL for generated assessments
- **Document Chunks**: 1-hour TTL for processed content
- **User Preferences**: 1-hour TTL for user settings
- **Cache Hit Rate**: Monitored and optimized

### Retrieval Optimization
- **Hybrid Fusion**: 70% dense + 30% sparse retrieval weight
- **Cross-encoder Reranking**: 70% reranking + 30% original score
- **Dynamic Chunking**: Topic-based optimal chunk sizes
- **Context Compression**: Intelligent content merging

## 🧪 Testing

### Run Tests
```bash
pytest tests/
```

### Test Coverage
```bash
pytest --cov=src tests/
```

## 📈 Monitoring

### Health Check
```bash
curl "http://localhost:8000/health"
```

### Cache Statistics
```bash
curl "http://localhost:8000/cache-stats"
```

### Collection Statistics
```bash
curl "http://localhost:8000/collection-stats"
```

## 🔒 Security

- File upload validation and sanitization
- API key management through environment variables
- Input validation and error handling
- CORS configuration for frontend security

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Production Considerations
- Use Redis cluster for high availability
- Implement database persistence for assessments
- Add authentication and authorization
- Set up monitoring and logging
- Configure load balancing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
1. Check the API documentation at `/docs`
2. Review the demo script for usage examples
3. Check the health endpoint for system status
4. Review logs for error details

## 🔄 Version History

- **v1.0.0**: Initial release with hybrid RAG, caching, and assessment generation
- Features: Document processing, hybrid retrieval, assessment generation, caching, API

## 📚 References

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Redis Documentation](https://redis.io/documentation) 