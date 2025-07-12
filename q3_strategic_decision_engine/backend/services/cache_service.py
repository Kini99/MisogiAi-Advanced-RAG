"""
Cache Service for caching document chunks, analysis results, and user sessions
Uses Redis for high-performance caching with automatic expiration
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pickle
import hashlib

import aioredis
from redis.asyncio import Redis

from ..core.config import settings
from ..core.logging_config import get_logger


@dataclass
class CacheStats:
    """Cache statistics"""
    total_keys: int
    hits: int
    misses: int
    memory_usage: int
    hit_rate: float


class CacheService:
    """Cache service for managing Redis-based caching"""
    
    def __init__(self):
        self.client: Optional[Redis] = None
        self.logger = get_logger('cache')
        self.initialized = False
        self.hits = 0
        self.misses = 0
    
    async def initialize(self):
        """Initialize Redis cache service"""
        if self.initialized:
            return
        
        try:
            # Connect to Redis
            self.client = aioredis.from_url(
                settings.REDIS_URL,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.client.ping()
            
            self.initialized = True
            self.logger.info("Cache service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cache service: {str(e)}")
            raise
    
    async def close(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            self.logger.info("Cache service closed")
    
    def _get_key(self, key: str) -> str:
        """Get prefixed cache key"""
        return f"{settings.CACHE_PREFIX}{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for caching"""
        if isinstance(value, (str, int, float, bool)):
            return json.dumps(value).encode('utf-8')
        else:
            return pickle.dumps(value)
    
    def _deserialize_value(self, value: bytes) -> Any:
        """Deserialize cached value"""
        try:
            # Try JSON first
            return json.loads(value.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(value)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.initialized:
            await self.initialize()
        
        try:
            cache_key = self._get_key(key)
            value = await self.client.get(cache_key)
            
            if value is None:
                self.misses += 1
                return None
            
            self.hits += 1
            return self._deserialize_value(value)
            
        except Exception as e:
            self.logger.error(f"Cache get failed for key {key}: {str(e)}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """Set value in cache"""
        if not self.initialized:
            await self.initialize()
        
        try:
            cache_key = self._get_key(key)
            serialized_value = self._serialize_value(value)
            ttl = ttl or settings.CACHE_TTL
            
            result = await self.client.set(
                cache_key,
                serialized_value,
                ex=ttl,
                nx=nx,
                xx=xx
            )
            
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Cache set failed for key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.initialized:
            await self.initialize()
        
        try:
            cache_key = self._get_key(key)
            result = await self.client.delete(cache_key)
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Cache delete failed for key {key}: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.initialized:
            await self.initialize()
        
        try:
            cache_key = self._get_key(key)
            result = await self.client.exists(cache_key)
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Cache exists check failed for key {key}: {str(e)}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key"""
        if not self.initialized:
            await self.initialize()
        
        try:
            cache_key = self._get_key(key)
            result = await self.client.expire(cache_key, ttl)
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Cache expire failed for key {key}: {str(e)}")
            return False
    
    async def get_ttl(self, key: str) -> int:
        """Get time to live for key"""
        if not self.initialized:
            await self.initialize()
        
        try:
            cache_key = self._get_key(key)
            return await self.client.ttl(cache_key)
            
        except Exception as e:
            self.logger.error(f"Cache TTL check failed for key {key}: {str(e)}")
            return -1
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment numeric value"""
        if not self.initialized:
            await self.initialize()
        
        try:
            cache_key = self._get_key(key)
            result = await self.client.incrby(cache_key, amount)
            return result
            
        except Exception as e:
            self.logger.error(f"Cache increment failed for key {key}: {str(e)}")
            return None
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        if not self.initialized:
            await self.initialize()
        
        try:
            cache_keys = [self._get_key(key) for key in keys]
            values = await self.client.mget(cache_keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize_value(value)
                    self.hits += 1
                else:
                    self.misses += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Cache get_many failed: {str(e)}")
            return {}
    
    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Prepare pipeline
            async with self.client.pipeline() as pipe:
                ttl = ttl or settings.CACHE_TTL
                
                for key, value in mapping.items():
                    cache_key = self._get_key(key)
                    serialized_value = self._serialize_value(value)
                    pipe.set(cache_key, serialized_value, ex=ttl)
                
                await pipe.execute()
                return True
                
        except Exception as e:
            self.logger.error(f"Cache set_many failed: {str(e)}")
            return False
    
    async def delete_many(self, keys: List[str]) -> int:
        """Delete multiple keys from cache"""
        if not self.initialized:
            await self.initialize()
        
        try:
            cache_keys = [self._get_key(key) for key in keys]
            result = await self.client.delete(*cache_keys)
            return result
            
        except Exception as e:
            self.logger.error(f"Cache delete_many failed: {str(e)}")
            return 0
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        if not self.initialized:
            await self.initialize()
        
        try:
            cache_pattern = self._get_key(pattern)
            keys = await self.client.keys(cache_pattern)
            
            if keys:
                result = await self.client.delete(*keys)
                return result
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Cache clear_pattern failed: {str(e)}")
            return 0
    
    async def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Get Redis info
            info = await self.client.info()
            
            # Count keys with our prefix
            pattern = self._get_key("*")
            keys = await self.client.keys(pattern)
            total_keys = len(keys)
            
            # Calculate hit rate
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0
            
            return CacheStats(
                total_keys=total_keys,
                hits=self.hits,
                misses=self.misses,
                memory_usage=info.get('used_memory', 0),
                hit_rate=hit_rate
            )
            
        except Exception as e:
            self.logger.error(f"Cache stats failed: {str(e)}")
            return CacheStats(0, 0, 0, 0, 0.0)
    
    # Document-specific cache methods
    async def cache_document_chunks(self, document_id: str, chunks: List[Dict[str, Any]], ttl: int = None) -> bool:
        """Cache document chunks"""
        key = f"doc_chunks:{document_id}"
        return await self.set(key, chunks, ttl=ttl or settings.CACHE_TTL * 24)  # Cache for 24 hours
    
    async def get_document_chunks(self, document_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached document chunks"""
        key = f"doc_chunks:{document_id}"
        return await self.get(key)
    
    async def cache_query_results(self, query_hash: str, results: List[Dict[str, Any]], ttl: int = None) -> bool:
        """Cache query results"""
        key = f"query_results:{query_hash}"
        return await self.set(key, results, ttl=ttl or settings.CACHE_TTL)
    
    async def get_query_results(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached query results"""
        key = f"query_results:{query_hash}"
        return await self.get(key)
    
    async def cache_analysis_result(self, session_id: str, analysis_type: str, result: Dict[str, Any], ttl: int = None) -> bool:
        """Cache analysis result"""
        key = f"analysis:{session_id}:{analysis_type}"
        return await self.set(key, result, ttl=ttl or settings.CACHE_TTL * 2)  # Cache for 2 hours
    
    async def get_analysis_result(self, session_id: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result"""
        key = f"analysis:{session_id}:{analysis_type}"
        return await self.get(key)
    
    async def cache_chat_session(self, session_id: str, session_data: Dict[str, Any], ttl: int = None) -> bool:
        """Cache chat session data"""
        key = f"session:{session_id}"
        return await self.set(key, session_data, ttl=ttl or settings.CACHE_TTL * 12)  # Cache for 12 hours
    
    async def get_chat_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached chat session data"""
        key = f"session:{session_id}"
        return await self.get(key)
    
    async def cache_embeddings(self, text_hash: str, embeddings: List[float], ttl: int = None) -> bool:
        """Cache embeddings"""
        key = f"embeddings:{text_hash}"
        return await self.set(key, embeddings, ttl=ttl or settings.CACHE_TTL * 48)  # Cache for 48 hours
    
    async def get_embeddings(self, text_hash: str) -> Optional[List[float]]:
        """Get cached embeddings"""
        key = f"embeddings:{text_hash}"
        return await self.get(key)
    
    async def cache_evaluation_result(self, query_hash: str, metrics: Dict[str, float], ttl: int = None) -> bool:
        """Cache RAGAS evaluation result"""
        key = f"evaluation:{query_hash}"
        return await self.set(key, metrics, ttl=ttl or settings.CACHE_TTL * 6)  # Cache for 6 hours
    
    async def get_evaluation_result(self, query_hash: str) -> Optional[Dict[str, float]]:
        """Get cached evaluation result"""
        key = f"evaluation:{query_hash}"
        return await self.get(key)
    
    # Utility methods
    def generate_query_hash(self, query: str, context: List[str] = None) -> str:
        """Generate hash for query and context"""
        content = query
        if context:
            content += ''.join(context)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def generate_text_hash(self, text: str) -> str:
        """Generate hash for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    async def health_check(self) -> bool:
        """Check cache health"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Test basic operations
            test_key = "health_check"
            test_value = {"timestamp": time.time()}
            
            await self.set(test_key, test_value, ttl=10)
            retrieved_value = await self.get(test_key)
            await self.delete(test_key)
            
            return retrieved_value is not None
            
        except Exception as e:
            self.logger.error(f"Cache health check failed: {str(e)}")
            return False
    
    async def warm_up(self):
        """Warm up cache with frequently accessed data"""
        try:
            # This can be implemented to pre-load frequently accessed data
            # For example, popular queries, common document chunks, etc.
            self.logger.info("Cache warm-up completed")
            
        except Exception as e:
            self.logger.error(f"Cache warm-up failed: {str(e)}")
    
    async def cleanup_expired(self):
        """Clean up expired keys (Redis does this automatically, but we can add custom logic)"""
        try:
            # Get all keys with our prefix
            pattern = self._get_key("*")
            keys = await self.client.keys(pattern)
            
            expired_count = 0
            for key in keys:
                ttl = await self.client.ttl(key)
                if ttl == -2:  # Key has expired
                    expired_count += 1
            
            self.logger.info(f"Cleaned up {expired_count} expired keys")
            
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {str(e)}")
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information"""
        try:
            info = await self.client.info('memory')
            return {
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'used_memory_peak': info.get('used_memory_peak', 0),
                'used_memory_peak_human': info.get('used_memory_peak_human', '0B'),
                'total_system_memory': info.get('total_system_memory', 0),
                'total_system_memory_human': info.get('total_system_memory_human', '0B'),
                'memory_fragmentation_ratio': info.get('mem_fragmentation_ratio', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Cache memory usage check failed: {str(e)}")
            return {}
    
    async def reset_stats(self):
        """Reset hit/miss statistics"""
        self.hits = 0
        self.misses = 0
        self.logger.info("Cache statistics reset") 