import redis, os
import hashlib
import json
import logging
import numpy as np
import faiss
from .embedding_service import get_embedding_model
import re

logger = logging.getLogger(__name__)

def _extract_keywords(query: str) -> set:
    """Extracts numerical and meaningful textual keywords from a query."""
    # Define a basic set of stop words to filter out noise
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'how', 'in', 
        'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was', 'what', 
        'when', 'where', 'who', 'will', 'with', 'show', 'me', 'tell', 'list', 'find'
    }
    
    # Clean the query by removing punctuation
    cleaned_query = re.sub(r'[^\w\s]', '', query)
    
    # Extract numbers (like years or other digits)
    numerical_keywords = re.findall(r'\b\d{4}\b|\b\d+\b', cleaned_query)
    
    # Extract textual words, lowercase them, and remove stop words
    words = re.findall(r'\b[a-z]+\b', cleaned_query.lower())
    textual_keywords = [word for word in words if word not in stop_words]
    
    # Combine and return a unique set of keywords
    return set(numerical_keywords + textual_keywords)

class SemanticCache:
    def __init__(self, embedding_dim=384, similarity_threshold=0.90):
        logger.info("Initializing Semantic Cache...")
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.entries = []
        self.similarity_threshold = similarity_threshold
        logger.info("Semantic Cache configured and ready.")

    def add(self, query: str, response: dict):
        """Adds a query and its response to the semantic cache."""
        model = get_embedding_model()
        embedding = model.encode([query])[0]
        keywords = _extract_keywords(query)
        
        self.index.add(np.array([embedding]))
        self.entries.append({"query": query, "response": response, "keywords": keywords, "embedding": embedding})
        logger.info(f"Added to semantic cache: '{query}' with keywords {keywords}")

    def search(self, query: str):
        """
        Searches for a semantically similar query in the cache using a hybrid approach.
        """
        if not self.entries:
            return None

        query_keywords = _extract_keywords(query)
        
        candidate_indices = [
            i for i, entry in enumerate(self.entries)
            if query_keywords.intersection(entry["keywords"])
        ]

        if not candidate_indices:
            logger.info(f"Hybrid cache miss (no keyword match) for query: {query}")
            return None

        logger.info(f"Found {len(candidate_indices)} candidates with keywords {query_keywords} for query: '{query}'")

        candidate_embeddings = np.array([self.entries[i]["embedding"] for i in candidate_indices])
        
        temp_index = faiss.IndexFlatL2(self.embedding_dim)
        temp_index.add(candidate_embeddings)

        model = get_embedding_model()
        query_embedding = model.encode([query])[0]
        distances, indices = temp_index.search(np.array([query_embedding]), 1)

        if len(indices) > 0:
            best_candidate_pos = indices[0][0]
            original_index = candidate_indices[best_candidate_pos]
            
            distance = distances[0][0]
            similarity = 1 / (1 + distance)

            if similarity > self.similarity_threshold:
                cached_entry = self.entries[original_index]
                logger.info(f"Hybrid semantic cache hit for query: '{query}' with similarity {similarity:.4f}. Found similar query: '{cached_entry['query']}'")
                return cached_entry['response']
            else:
                logger.info(f"Hybrid cache miss for query: '{query}'. Best match similarity {similarity:.4f} is below threshold {self.similarity_threshold}.")
        
        return None

class Cache:
    def __init__(self, redis_url=None, default_ttl=3600):
        self.default_ttl = default_ttl
        self.redis_client = None

        # Prioritize the provided redis_url, then environment variable, then default to None
        effective_redis_url = redis_url or os.getenv('REDIS_URL')

        if effective_redis_url:
            try:
                self.redis_client = redis.from_url(effective_redis_url)
                self.redis_client.ping()
                logger.info(f"Redis cache initialized at {effective_redis_url}.")
            except redis.exceptions.ConnectionError as e:
                logger.warning(f"Redis connection to {effective_redis_url} failed: {e}. Falling back to in-memory cache.")
                self.redis_client = None
        
        if not self.redis_client:
            self._in_memory_cache = {}
            logger.info("Using in-memory cache.")

    def _generate_key(self, components):
        key_string = "".join(str(c) for c in components)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(self, key):
        if self.redis_client:
            logger.info(f"REDIS GET: {key}")
            cached_value = self.redis_client.get(key)
            return json.loads(cached_value) if cached_value else None
        else:
            logger.info(f"IN-MEMORY GET: {key}")
            return self._in_memory_cache.get(key)

    def set(self, key, value, ttl=None):
        ttl = ttl or self.default_ttl
        if self.redis_client:
            logger.info(f"REDIS SET: {key}")
            self.redis_client.set(key, json.dumps(value), ex=ttl)
        else:
            logger.info(f"IN-MEMORY SET: {key}")
            self._in_memory_cache[key] = value

    def get_sql_key(self, dataset_hash, sql_query):
        return self._generate_key([dataset_hash, sql_query])

    def get_chart_key(self, dataset_hash, sql_query, chart_type):
        return self._generate_key([dataset_hash, sql_query, chart_type])

    def get_summary_key(self, dataset_hash, sql_query, summary_version):
        return self._generate_key([dataset_hash, sql_query, summary_version])

# Global cache instances
semantic_cache = SemanticCache()
cache = Cache()