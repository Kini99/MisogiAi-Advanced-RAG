"""
Unit tests for Vector Store Service.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from backend.services.vector_store_service import VectorStoreService
from backend.core.config import Settings


class TestVectorStoreService:
    """Test cases for Vector Store Service."""

    @pytest.fixture
    def vector_store_service(self, test_settings):
        """Create vector store service instance."""
        return VectorStoreService(test_settings)

    @pytest.mark.asyncio
    async def test_add_documents_success(self, vector_store_service):
        """Test successful document addition."""
        documents = [
            {
                'id': 'doc1',
                'content': 'Strategic planning document about AI development',
                'metadata': {'title': 'Strategy Doc', 'type': 'strategic'}
            },
            {
                'id': 'doc2',
                'content': 'Financial analysis showing 25% growth',
                'metadata': {'title': 'Finance Doc', 'type': 'financial'}
            }
        ]
        
        with patch.object(vector_store_service.collection, 'add') as mock_add:
            result = await vector_store_service.add_documents(documents)
            
            assert result is True
            mock_add.assert_called_once()
            call_args = mock_add.call_args[1]
            assert len(call_args['documents']) == 2
            assert len(call_args['ids']) == 2
            assert len(call_args['metadatas']) == 2

    @pytest.mark.asyncio
    async def test_search_documents_success(self, vector_store_service):
        """Test successful document search."""
        with patch.object(vector_store_service.collection, 'query') as mock_query:
            mock_query.return_value = {
                'documents': [['Strategic planning document', 'AI development goals']],
                'metadatas': [[{'title': 'Strategy Doc'}, {'title': 'AI Doc'}]],
                'distances': [[0.1, 0.3]]
            }
            
            results = await vector_store_service.search_documents(
                query="strategic planning",
                n_results=2
            )
            
            assert len(results) == 2
            assert results[0]['content'] == 'Strategic planning document'
            assert results[0]['metadata']['title'] == 'Strategy Doc'
            assert results[0]['score'] > results[1]['score']  # Lower distance = higher score

    @pytest.mark.asyncio
    async def test_get_relevant_documents_success(self, vector_store_service):
        """Test getting relevant documents with filtering."""
        with patch.object(vector_store_service.collection, 'query') as mock_query:
            mock_query.return_value = {
                'documents': [['Relevant strategic content']],
                'metadatas': [[{'title': 'Strategy', 'type': 'strategic'}]],
                'distances': [[0.2]]
            }
            
            results = await vector_store_service.get_relevant_documents(
                query="strategic analysis",
                filter_metadata={"type": "strategic"},
                similarity_threshold=0.5
            )
            
            assert len(results) == 1
            assert results[0]['content'] == 'Relevant strategic content'
            mock_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_document_success(self, vector_store_service):
        """Test successful document deletion."""
        with patch.object(vector_store_service.collection, 'delete') as mock_delete:
            result = await vector_store_service.delete_document('doc1')
            
            assert result is True
            mock_delete.assert_called_once_with(ids=['doc1'])

    @pytest.mark.asyncio
    async def test_get_collection_stats_success(self, vector_store_service):
        """Test getting collection statistics."""
        with patch.object(vector_store_service.collection, 'count') as mock_count:
            mock_count.return_value = 150
            
            stats = await vector_store_service.get_collection_stats()
            
            assert stats['total_documents'] == 150
            assert 'collection_name' in stats
            assert 'embedding_model' in stats

    @pytest.mark.asyncio
    async def test_hybrid_search_success(self, vector_store_service):
        """Test hybrid search with both dense and sparse retrieval."""
        with patch.object(vector_store_service.collection, 'query') as mock_query:
            with patch.object(vector_store_service, '_sparse_search') as mock_sparse:
                # Mock dense search results
                mock_query.return_value = {
                    'documents': [['Dense result 1', 'Dense result 2']],
                    'metadatas': [[{'title': 'Dense 1'}, {'title': 'Dense 2'}]],
                    'distances': [[0.1, 0.2]]
                }
                
                # Mock sparse search results
                mock_sparse.return_value = [
                    {'content': 'Sparse result 1', 'metadata': {'title': 'Sparse 1'}, 'score': 0.8},
                    {'content': 'Sparse result 2', 'metadata': {'title': 'Sparse 2'}, 'score': 0.7}
                ]
                
                results = await vector_store_service.hybrid_search(
                    query="strategic planning",
                    n_results=4,
                    alpha=0.5  # Equal weighting
                )
                
                assert len(results) <= 4
                # Should contain results from both dense and sparse search
                assert any('Dense' in result['metadata']['title'] for result in results)

    @pytest.mark.asyncio
    async def test_rerank_documents_success(self, vector_store_service):
        """Test document reranking functionality."""
        documents = [
            {'content': 'Strategic planning document', 'metadata': {'title': 'Strategy'}},
            {'content': 'Financial analysis report', 'metadata': {'title': 'Finance'}},
            {'content': 'Market research findings', 'metadata': {'title': 'Market'}}
        ]
        
        with patch.object(vector_store_service.reranker, 'rank') as mock_rank:
            mock_rank.return_value = [
                {'corpus_id': 0, 'score': 0.9},
                {'corpus_id': 2, 'score': 0.7},
                {'corpus_id': 1, 'score': 0.5}
            ]
            
            reranked = await vector_store_service.rerank_documents(
                query="strategic planning",
                documents=documents
            )
            
            assert len(reranked) == 3
            assert reranked[0]['metadata']['title'] == 'Strategy'  # Highest score
            assert reranked[1]['metadata']['title'] == 'Market'    # Second highest
            assert reranked[2]['metadata']['title'] == 'Finance'   # Lowest score

    @pytest.mark.asyncio
    async def test_batch_add_documents_success(self, vector_store_service):
        """Test batch document addition with chunking."""
        # Create a large batch of documents
        documents = []
        for i in range(250):  # More than typical batch size
            documents.append({
                'id': f'doc{i}',
                'content': f'Document content {i}',
                'metadata': {'title': f'Doc {i}', 'batch': i // 100}
            })
        
        with patch.object(vector_store_service.collection, 'add') as mock_add:
            result = await vector_store_service.batch_add_documents(documents, batch_size=100)
            
            assert result is True
            assert mock_add.call_count == 3  # 3 batches for 250 documents

    @pytest.mark.asyncio
    async def test_update_document_success(self, vector_store_service):
        """Test document update functionality."""
        with patch.object(vector_store_service.collection, 'update') as mock_update:
            result = await vector_store_service.update_document(
                document_id='doc1',
                content='Updated strategic content',
                metadata={'title': 'Updated Strategy', 'version': 2}
            )
            
            assert result is True
            mock_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_similarity_search_with_threshold(self, vector_store_service):
        """Test similarity search with score threshold filtering."""
        with patch.object(vector_store_service.collection, 'query') as mock_query:
            mock_query.return_value = {
                'documents': [['High relevance doc', 'Low relevance doc']],
                'metadatas': [[{'title': 'High'}, {'title': 'Low'}]],
                'distances': [[0.1, 0.8]]  # 0.1 = high similarity, 0.8 = low similarity
            }
            
            results = await vector_store_service.similarity_search_with_threshold(
                query="strategic planning",
                threshold=0.5  # Only return documents with distance < 0.5
            )
            
            assert len(results) == 1
            assert results[0]['metadata']['title'] == 'High'

    @pytest.mark.asyncio
    async def test_get_documents_by_metadata(self, vector_store_service):
        """Test filtering documents by metadata."""
        with patch.object(vector_store_service.collection, 'get') as mock_get:
            mock_get.return_value = {
                'documents': ['Strategic doc 1', 'Strategic doc 2'],
                'metadatas': [{'title': 'Strategy 1'}, {'title': 'Strategy 2'}],
                'ids': ['doc1', 'doc2']
            }
            
            results = await vector_store_service.get_documents_by_metadata(
                filter_metadata={"type": "strategic"}
            )
            
            assert len(results) == 2
            assert all('Strategic' in doc['content'] for doc in results)

    @pytest.mark.asyncio
    async def test_error_handling(self, vector_store_service):
        """Test error handling in vector store operations."""
        with patch.object(vector_store_service.collection, 'add') as mock_add:
            mock_add.side_effect = Exception("ChromaDB error")
            
            with pytest.raises(Exception):
                await vector_store_service.add_documents([
                    {'id': 'doc1', 'content': 'test', 'metadata': {}}
                ])

    @pytest.mark.asyncio
    async def test_embedding_generation(self, vector_store_service):
        """Test embedding generation for documents."""
        with patch.object(vector_store_service.embedding_model, 'embed_documents') as mock_embed:
            mock_embed.return_value = [
                [0.1, 0.2, 0.3],  # Embedding for doc 1
                [0.4, 0.5, 0.6]   # Embedding for doc 2
            ]
            
            embeddings = await vector_store_service.generate_embeddings([
                'Strategic planning document',
                'Financial analysis report'
            ])
            
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 3
            assert embeddings[0] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_collection_reset(self, vector_store_service):
        """Test collection reset functionality."""
        with patch.object(vector_store_service.collection, 'delete') as mock_delete:
            with patch.object(vector_store_service.collection, 'count') as mock_count:
                mock_count.return_value = 0
                
                result = await vector_store_service.reset_collection()
                
                assert result is True
                mock_delete.assert_called_once()

    def test_chunk_text_success(self, vector_store_service):
        """Test text chunking functionality."""
        long_text = "This is a very long document. " * 100  # 500 words
        
        chunks = vector_store_service._chunk_text(long_text, max_chunk_size=100)
        
        assert len(chunks) > 1
        assert all(len(chunk.split()) <= 100 for chunk in chunks)
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_calculate_similarity_score(self, vector_store_service):
        """Test similarity score calculation."""
        # Test with different distance values
        assert vector_store_service._calculate_similarity_score(0.0) == 1.0  # Perfect match
        assert vector_store_service._calculate_similarity_score(1.0) == 0.0  # No similarity
        assert 0.0 < vector_store_service._calculate_similarity_score(0.5) < 1.0  # Partial match

    @pytest.mark.asyncio
    async def test_document_versioning(self, vector_store_service):
        """Test document versioning functionality."""
        with patch.object(vector_store_service.collection, 'add') as mock_add:
            with patch.object(vector_store_service.collection, 'get') as mock_get:
                mock_get.return_value = {
                    'documents': [],
                    'metadatas': [],
                    'ids': []
                }
                
                result = await vector_store_service.add_document_version(
                    document_id='doc1',
                    content='Version 2 content',
                    metadata={'version': 2, 'title': 'Updated Doc'}
                )
                
                assert result is True
                mock_add.assert_called_once()
                # Verify version is included in the document ID
                call_args = mock_add.call_args[1]
                assert 'doc1_v2' in call_args['ids'] 