"""Redis caching layer for the Advanced Assessment Generation System."""

import json
import pickle
from typing import Optional, Any, Dict, List
import redis
from datetime import datetime
import hashlib
import time

from .config import settings
from .models import Assessment, DocumentChunk, CacheStats


class RedisCache:
    """Redis caching implementation."""
    
    def __init__(self):
        """Initialize Redis connection."""
        self.redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            decode_responses=False  # Keep binary for pickle
        )
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "response_times": []
        }
    
    def _generate_key(self, prefix: str, identifier: str) -> str:
        """Generate a cache key."""
        return f"{prefix}:{identifier}"
    
    def _serialize(self, data: Any) -> bytes:
        """Serialize data for storage."""
        return pickle.dumps(data)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from storage."""
        return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        self.stats["total_requests"] += 1
        start_time = time.time()
        
        try:
            value = self.redis_client.get(key)
            if value:
                self.stats["cache_hits"] += 1
                response_time = time.time() - start_time
                self.stats["response_times"].append(response_time)
                return self._deserialize(value)
            return None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            serialized_value = self._serialize(value)
            if ttl:
                return self.redis_client.setex(key, ttl, serialized_value)
            else:
                return self.redis_client.set(key, serialized_value)
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            print(f"Cache exists error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            print(f"Cache clear pattern error: {e}")
            return 0


class AssessmentCache:
    """Assessment-specific caching operations."""
    
    def __init__(self):
        """Initialize assessment cache."""
        self.cache = RedisCache()
    
    def get_assessment(self, request_hash: str) -> Optional[Assessment]:
        """Get cached assessment."""
        key = self.cache._generate_key("assessment", request_hash)
        return self.cache.get(key)
    
    def set_assessment(self, request_hash: str, assessment: Assessment) -> bool:
        """Cache assessment."""
        key = self.cache._generate_key("assessment", request_hash)
        return self.cache.set(key, assessment, settings.assessment_cache_ttl)
    
    def get_document_chunks(self, topic: str) -> Optional[List[DocumentChunk]]:
        """Get cached document chunks for topic."""
        key = self.cache._generate_key("chunks", topic)
        return self.cache.get(key)
    
    def set_document_chunks(self, topic: str, chunks: List[DocumentChunk]) -> bool:
        """Cache document chunks for topic."""
        key = self.cache._generate_key("chunks", topic)
        return self.cache.set(key, chunks, settings.cache_ttl)
    
    def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user preferences."""
        key = self.cache._generate_key("preferences", user_id)
        return self.cache.get(key)
    
    def set_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Cache user preferences."""
        key = self.cache._generate_key("preferences", user_id)
        return self.cache.set(key, preferences, settings.cache_ttl)
    
    def get_user_history(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached user history."""
        key = self.cache._generate_key("history", user_id)
        return self.cache.get(key)
    
    def set_user_history(self, user_id: str, history: List[Dict[str, Any]]) -> bool:
        """Cache user history."""
        key = self.cache._generate_key("history", user_id)
        return self.cache.set(key, history, settings.cache_ttl)
    
    def clear_topic_cache(self, topic: str) -> int:
        """Clear all cache entries for a topic."""
        pattern = f"*:{topic}"
        return self.cache.clear_pattern(pattern)
    
    def get_cache_stats(self) -> CacheStats:
        """Get cache statistics."""
        stats = self.cache.stats
        total_requests = stats["total_requests"]
        cache_hits = stats["cache_hits"]
        
        cache_miss_rate = 0.0
        if total_requests > 0:
            cache_miss_rate = 1 - (cache_hits / total_requests)
        
        average_response_time = 0.0
        if stats["response_times"]:
            average_response_time = sum(stats["response_times"]) / len(stats["response_times"])
        
        return CacheStats(
            total_requests=total_requests,
            cache_hits=cache_hits,
            cache_miss_rate=cache_miss_rate,
            average_response_time=average_response_time
        )


# Global cache instance
assessment_cache = AssessmentCache() 