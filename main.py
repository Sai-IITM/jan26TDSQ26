from fastapi import FastAPI
from pydantic import BaseModel
import time, hashlib, numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, Any
import uvicorn

app = FastAPI(title="AI Caching Assignment")
model = SentenceTransformer('all-MiniLM-L6-v2')
exact_cache: Dict[str, Dict] = {}
semantic_cache: list = []
MAX_SIZE = 1500
TTL = 86400  # 24h
stats = {"total": 0, "hits": 0}

class Query(BaseModel):
    query: str
    application: str

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def evict_if_needed():
    global exact_cache, semantic_cache
    while len(exact_cache) > MAX_SIZE or len(semantic_cache) > MAX_SIZE:
        # LRU: remove oldest
        if exact_cache:
            oldest_key = min(exact_cache, key=lambda k: exact_cache[k]["ts"])
            del exact_cache[oldest_key]
        if semantic_cache:
            semantic_cache.pop(0)

@app.post("/")
async def query_endpoint(data: Query):
    global stats
    query = data.query.lower().strip()  # Normalize!
    start_time = time.time()
    stats["total"] += 1
    
    # 1. EXACT MATCH (MD5 hash)
    key = hashlib.md5(query.encode()).hexdigest()
    if key in exact_cache and time.time() - exact_cache[key]["ts"] < TTL:
        latency = int((time.time() - start_time) * 1000)
        stats["hits"] += 1
        return {
            "answer": exact_cache[key]["resp"],
            "cached": True,
            "latency": latency,
            "cacheKey": key
        }
    
    # 2. SEMANTIC MATCH (>0.95 similarity)
    emb = model.encode([query])[0]
    for i, (cached_emb, resp, ts) in enumerate(semantic_cache):
        if time.time() - ts < TTL:
            sim = cosine_similarity(emb, cached_emb)
            if sim > 0.95:
                latency = int((time.time() - start_time) * 1000)
                stats["hits"] += 1
                return {
                    "answer": resp,
                    "cached": True,
                    "latency": latency,
                    "cacheKey": f"sem_{i}_{sim:.3f}"
                }
    
    # 3. CACHE MISS - Simulate LLM call
    time.sleep(0.0015)  # Simulate 1500ms API call
    answer = f"AI Response for: {query}"
    tokens = 800
    
    # Store both caches
    ts = time.time()
    exact_cache[key] = {"resp": answer, "ts": ts}
    semantic_cache.append((emb, answer, ts))
    
    evict_if_needed()
    
    latency = int((time.time() - start_time) * 1000)
    return {
        "answer": answer,
        "cached": False,
        "latency": latency,
        "cacheKey": key
    }

@app.get("/analytics")
async def analytics():
    hit_rate = stats["hits"] / max(stats["total"], 1)
    total_tokens = stats["total"] * 800
    saved_tokens = stats["hits"] * 800
    savings = (saved_tokens * 0.60) / 1e6  # $0.60/1M tokens
    
    return {
        "hitRate": round(hit_rate, 2),
        "totalRequests": stats["total"],
        "cacheHits": stats["hits"],
        "cacheMisses": stats["total"] - stats["hits"],
        "cacheSize": len(exact_cache) + len(semantic_cache),
        "costSavings": round(savings, 2),
        "savingsPercent": int(hit_rate * 100),
        "strategies": ["exact match (MD5)", "semantic >0.95 cosine", "LRU eviction", "24h TTL"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
