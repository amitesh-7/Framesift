"""
FrameSift Backend - Semantic Video Search Engine
Hybrid AI Architecture: Local Scout (CPU) + Cloud Intelligence (NVIDIA NIM)

Phase 2: Advanced Filtering
- Audio Trigger ("Clack" Detector)
- Physics Filter (Vertical Optical Flow)
- Robust Parallel Processing with Key Rotation
- Deep Scan Mode
"""

import os
import uuid
import json
import time
import base64
import tempfile
import itertools
from io import BytesIO
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

# Suppress transformers warnings
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings
import logging
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
# Suppress transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import cv2
import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from pymongo import MongoClient
import redis

# Phase 2 imports
from scout import AudioTrigger, PhysicsFilter, PriorityQueueManager, FramePriority, BrightnessSpikeDetector
from processor import ParallelFrameProcessor, DeepScanProcessor, create_parallel_processor

# Load environment variables from .env.local (development) or .env (production)
load_dotenv(".env.local")
if not os.getenv("MONGODB_URI"):  # Fallback to .env if .env.local doesn't exist
    load_dotenv()

# ============================================================================
# Configuration
# ============================================================================
# ============================================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "framesift")
NVIDIA_KEYS = json.loads(os.getenv("NVIDIA_KEYS", '[]'))
NVIDIA_VISION_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"
ADMIN_KEY = os.getenv("ADMIN_KEY", "admin-secret-key")

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "framesift")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "users")

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "")  # Cloud Redis URL (takes priority)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# Frame filtering thresholds
MOTION_THRESHOLD = 30.0  # Pixel change threshold for motion detection
SIMILARITY_THRESHOLD = 0.95  # CLIP similarity threshold for redundancy detection
FRAME_SKIP_INTERVAL = 5  # Process every Nth frame initially (default for long videos)

# Multi-Model Ensemble Configuration (P3)
# When enabled, uses multiple vision models to analyze frames for better accuracy
# This is slower but more accurate, especially for detecting fast events
ENSEMBLE_MODE = os.getenv("ENSEMBLE_MODE", "true").lower() == "true"
print(f"🔀 Multi-model ensemble: {'ENABLED' if ENSEMBLE_MODE else 'DISABLED'}")

# Enhanced Vision Prompt for better accuracy (Phase 2.1 - P0)
VISION_PROMPT = """Analyze this video frame comprehensively. Focus on ACTIONS and EVENTS:

**SCENE**: Environment, location, lighting, weather (indoor/outdoor/doorway/window)
**SUBJECTS**: People, animals, objects - appearance, clothing (color!), position, posture
**ACTIONS**: What are subjects DOING? Standing, sitting, walking, running, entering, exiting, falling?
**SPATIAL**: Where are subjects positioned? At door, by window, inside room, outside, in transition?
**EVENTS**: Any significant occurrences - flashes, movements, impacts, changes in state
**VISUAL PHENOMENA**: Lightning, fire, smoke, explosions, sparks, sudden brightness

CRITICAL TERMINOLOGY - USE THESE EXACT PHRASES:
- Person at/near doorway = "standing at doorway", "at the entrance", "about to exit"
- Person moving toward door = "heading to exit", "going outside", "leaving"
- Person in outdoor setting = "outside", "outdoors", "exterior"
- Bright flash in dark/cloudy sky = "lightning bolt" or "lightning strike"
- Fast downward movement = "falling", "dropping", "tumbling"
- Fire/flames = "fire", "burning", "flames"

DESCRIBE ACTIONS, NOT JUST SCENES. What is the subject DOING right now?"""

# ============================================================================
# Database Connections
# ============================================================================

# MongoDB Client
try:
    mongo_client = MongoClient(MONGODB_URI)
    mongo_db = mongo_client[MONGODB_DB_NAME]
    users_collection = mongo_db[MONGODB_COLLECTION_NAME]
    # Create index on user_id for faster lookups
    users_collection.create_index("id", unique=True)
    print(f"✅ MongoDB connected: {MONGODB_DB_NAME}.{MONGODB_COLLECTION_NAME}")
except Exception as e:
    print(f"⚠️ MongoDB connection failed: {e}")
    mongo_client = None
    users_collection = None

# Redis Client
try:
    if REDIS_URL:
        # Use Redis URL (for cloud Redis)
        redis_client = redis.Redis.from_url(
            REDIS_URL,
            db=REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=5
        )
    else:
        # Use host/port (for local Redis)
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD if REDIS_PASSWORD else None,
            decode_responses=True
        )
    redis_client.ping()
    connection_info = REDIS_URL[:40] + "..." if REDIS_URL else f"{REDIS_HOST}:{REDIS_PORT}"
    print(f"✅ Redis connected: {connection_info}")
except Exception as e:
    print(f"⚠️ Redis connection failed: {e}")
    redis_client = None

# ============================================================================
# Query Expansion & Keyword Boost (Phase 2.1 - Accuracy Improvements)
# ============================================================================

# Synonym expansions for common search terms (P1 - Query Expansion)
QUERY_EXPANSIONS = {
    # Weather/Natural phenomena
    "lightning": "lightning bolt electrical flash bright flash sky thunder storm electric discharge sudden brightness flash of light illumination bright sky",
    "thunder": "thunderstorm storm lightning rumble rain",
    "rain": "raining rainfall precipitation water drops storm weather wet",
    "snow": "snowing snowfall winter blizzard frost cold weather",
    
    # Fire/Danger
    "fire": "flames burning fire blaze inferno smoke hot",
    "explosion": "blast detonation explode fireball bright flash impact",
    "smoke": "smoking smoke fumes haze fire",
    
    # Movement actions
    "fall": "falling dropped tumbling descending crash dropping fell",
    "crash": "collision impact crash smash accident falling hit",
    "walk": "walking strolling moving person pedestrian stepping",
    "run": "running sprinting jogging fast movement rushing",
    "jump": "jumping leaping hopping bounce airborne",
    "dance": "dancing movement rhythm motion celebration moving",
    "fight": "fighting combat action confrontation violence hitting",
    
    # Location/Direction actions (CRITICAL for "goes outside" queries)
    "outside": "outside outdoors exterior exit leaving doorway door going out outdoor open air",
    "inside": "inside indoors interior entering indoor room",
    "leave": "leaving exiting departing going outside exit door doorway heading out venture outside",
    "enter": "entering coming in arriving indoor inside door",
    "goes": "going heading moving walking leaving exiting stepping venture traveling",
    "exit": "exiting leaving departure exit door doorway outside going out",
    "door": "doorway door entrance exit threshold entry gateway",
    
    # Light/Visual
    "bright": "bright flash light glow illumination lightning sudden brightness intense",
    "dark": "darkness shadow dim low light night evening",
    "flash": "flash bright light lightning sudden illumination brightness spike strobe",
    
    # People/Objects
    "person": "human people man woman individual figure someone",
    "girl": "girl woman female person young lady",
    "boy": "boy man male person young guy",
    "car": "vehicle automobile car truck driving",
}

def expand_search_query(query: str) -> str:
    """
    Expand user query with synonyms for better semantic matching (P1).
    
    Example: "when did the girl go outside" → 
             "when did the girl go outside girl woman female... outside outdoors exterior exit..."
    
    Now supports MULTIPLE term expansion for compound queries.
    """
    expanded = query
    query_lower = query.lower()
    added_expansions = []
    
    # Expand ALL matching terms (not just one)
    for term, expansion in QUERY_EXPANSIONS.items():
        if term in query_lower:
            added_expansions.append(expansion)
    
    # Combine all expansions
    if added_expansions:
        # Limit total expansion to avoid query bloating
        combined = " ".join(added_expansions)
        # Take only first 100 words of expansion
        expansion_words = combined.split()[:100]
        expanded = f"{query} {' '.join(expansion_words)}"
        print(f"  📝 Query expanded: '{query}' → +{len(expansion_words)} terms")
    
    return expanded

def apply_keyword_boost(original_query: str, results: List[Dict]) -> List[Dict]:
    """
    Boost scores for results that contain keywords from the original query.
    
    This helps when the embedding similarity is close but keyword presence
    indicates higher relevance.
    """
    if not results:
        return results
    
    # Extract meaningful keywords (skip common words)
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'did', 'do', 'does', 
                  'when', 'where', 'what', 'how', 'why', 'who', 'which', 'in', 'on', 
                  'at', 'to', 'for', 'of', 'with', 'by', 'from', 'it', 'this', 'that',
                  'occur', 'happen', 'show', 'see', 'find', 'time'}
    
    # Action word synonyms - if query contains action, boost related words in descriptions
    action_synonyms = {
        'goes': ['going', 'goes', 'went', 'leaving', 'exiting', 'stepping', 'walking out', 'heading', 'venture', 'preparing to'],
        'outside': ['outside', 'outdoor', 'outdoors', 'exterior', 'doorway', 'door', 'exit', 'leaving', 'out into'],
        'inside': ['inside', 'indoor', 'indoors', 'interior', 'entering', 'came in'],
        'falls': ['falling', 'falls', 'fell', 'dropped', 'dropping', 'tumbling'],
        'sits': ['sitting', 'sits', 'sat', 'seated'],
        'stands': ['standing', 'stands', 'stood'],
        'walks': ['walking', 'walks', 'walked', 'strolling'],
        'runs': ['running', 'runs', 'ran', 'sprinting'],
    }
    
    query_lower = original_query.lower()
    query_words = set(word.lower() for word in original_query.split() 
                      if word.lower() not in stop_words and len(word) > 2)
    
    # Expand query words with action synonyms
    expanded_keywords = set(query_words)
    for action, synonyms in action_synonyms.items():
        if action in query_lower or any(s in query_lower for s in synonyms[:2]):
            expanded_keywords.update(synonyms)
    
    if not expanded_keywords:
        return results
    
    for result in results:
        description_lower = result['description'].lower()
        
        # Count keyword matches (including synonyms)
        matches = sum(1 for word in expanded_keywords if word in description_lower)
        
        # Extra boost for action matches
        action_boost = 0
        for action, synonyms in action_synonyms.items():
            if action in query_lower:
                for syn in synonyms:
                    if syn in description_lower:
                        action_boost += 0.25  # Strong boost for action word matches
                        break
        
        if matches > 0 or action_boost > 0:
            # Boost score by 15% per matching keyword + action boost
            boost_factor = 1 + (0.15 * matches) + action_boost
            result['score'] *= boost_factor
            # print(f"  ⬆️ Boosted '{result['description'][:50]}...' by {boost_factor:.2f}x")
    
    # Re-sort by boosted scores
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results


def rerank_with_llm(query: str, results: List[Dict], api_key: str) -> List[Dict]:
    """
    Use LLM to re-rank search results based on semantic relevance to the query.
    
    This fixes the problem where embedding similarity doesn't capture intent well.
    For example: "when does the girl go outside" should rank "person in raincoat at doorway"
    higher than "girl sitting on bed", even if embedding similarity says otherwise.
    
    Args:
        query: User's original question
        results: List of search results with 'timestamp' and 'description'
        api_key: NVIDIA API key
        
    Returns:
        Re-ranked results with updated scores
    """
    if not results or not api_key or len(results) <= 1:
        return results
    
    # Build the re-ranking prompt
    descriptions_text = "\n".join([
        f"{i+1}. [{r['timestamp']:.1f}s] {r['description'][:300]}..."
        for i, r in enumerate(results[:10])  # Limit to top 10 for re-ranking
    ])
    
    prompt = f"""You are a video search re-ranking system. Given a user's query and a list of video frame descriptions, rank them by relevance.

USER QUERY: "{query}"

VIDEO FRAMES:
{descriptions_text}

TASK: Return ONLY the frame numbers in order of relevance to the query (most relevant first).
Format your response as just comma-separated numbers, nothing else.
Example response: 3,1,5,2,4

Consider:
- Which frame BEST answers or matches the user's query?
- Action queries (like "goes outside", "falls down") should match frames showing that ACTION
- For "goes outside": prioritize frames showing doorway, entrance, exiting, outdoor environment
- "Standing at doorway" or "leaving" is MORE relevant than "sitting by window"
- Time-based queries ("when does X happen") need frames showing the EVENT happening
- A frame showing preparation/result of an action is LESS relevant than the action itself
- IGNORE frames that just show similar setting but wrong action (e.g., sitting vs leaving)

Your ranking (numbers only):"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0.1  # Low temperature for consistent ranking
    }
    
    try:
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=20  # Increased timeout from 10 to 20 seconds
        )
        
        if response.status_code == 200:
            result = response.json()
            ranking_text = result['choices'][0]['message']['content'].strip()
            
            # Parse the ranking (e.g., "3,1,5,2,4")
            # Clean up any extra text
            ranking_text = ''.join(c for c in ranking_text if c.isdigit() or c == ',')
            ranking = [int(x.strip()) - 1 for x in ranking_text.split(',') if x.strip().isdigit()]
            
            print(f"  🔄 LLM Re-ranking: {[r+1 for r in ranking]}")
            
            # Re-order results based on LLM ranking with NEW position-based scores
            reranked = []
            seen = set()
            total_items = len(ranking) if ranking else len(results)
            
            for position, idx in enumerate(ranking):
                if 0 <= idx < len(results) and idx not in seen:
                    result_copy = results[idx].copy()
                    
                    # Assign NEW score based on ranking position (not multiplication)
                    # Position 0 (best) = 0.95, Position 1 = 0.85, etc.
                    # This ensures correct ordering regardless of original scores
                    new_score = max(0.95 - (position * 0.12), 0.20)
                    result_copy['score'] = new_score
                    
                    reranked.append(result_copy)
                    seen.add(idx)
                    print(f"    #{position+1}: [{result_copy['timestamp']:.1f}s] score: {new_score:.2f}")
            
            # Add any results that weren't ranked with low scores
            for i, r in enumerate(results):
                if i not in seen:
                    r_copy = r.copy()
                    r_copy['score'] = 0.15  # Low score for unranked
                    reranked.append(r_copy)
            
            return reranked
        else:
            print(f"  ⚠️ Re-ranking failed: {response.status_code}")
            return results
            
    except Exception as e:
        print(f"  ❌ Re-ranking error: {e}")
        return results


def generate_ai_answer(query: str, frame_descriptions: List[Dict], api_key: str) -> Optional[str]:
    """
    Generate a natural language answer to the user's question based on retrieved frame descriptions.
    
    This is the "Generation" part of RAG - using retrieved context to answer questions.
    
    Args:
        query: User's original question
        frame_descriptions: List of dicts with 'timestamp' and 'description'
        api_key: NVIDIA API key for LLM
        
    Returns:
        AI-generated answer string, or None if generation fails
    """
    if not frame_descriptions or not api_key:
        return None
    
    # Build context from frame descriptions
    context_parts = []
    for i, frame in enumerate(frame_descriptions[:5], 1):  # Limit to top 5
        timestamp = frame.get('timestamp', 0)
        description = frame.get('description', '')
        context_parts.append(f"[{timestamp:.1f}s] {description}")
    
    context = "\n".join(context_parts)
    
    # Create prompt for answer generation
    system_prompt = """You are a helpful video analysis assistant. Based on the video frame descriptions provided, answer the user's question naturally and conversationally.

Rules:
1. Answer based ONLY on the provided frame descriptions
2. Be concise but informative (2-4 sentences)
3. Mention specific timestamps when relevant (e.g., "At 5.2 seconds...")
4. If the frames don't contain enough information to answer, say so honestly
5. Don't make up information not present in the descriptions"""

    user_prompt = f"""Video Frame Descriptions:
{context}

User Question: {query}

Please provide a helpful answer based on what's shown in the video frames:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta/llama-3.1-70b-instruct",  # Using text model for answer generation
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            print(f"  🤖 AI Answer generated: {answer[:100]}...")
            return answer
        else:
            print(f"  ⚠️ AI Answer generation failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"  ❌ AI Answer error: {e}")
        return None


# ============================================================================
# Data Models
# ============================================================================

class SearchQuery(BaseModel):
    query: str
    video_id: Optional[str] = None
    user_id: Optional[str] = None  # Multi-user support
    top_k: int = 10

class SearchResult(BaseModel):
    timestamp: float
    score: float
    description: str
    frame_id: str

class SearchResponse(BaseModel):
    """Enhanced search response with AI-generated answer."""
    results: List[SearchResult]
    query: str
    ai_answer: Optional[str] = None  # AI-synthesized answer based on retrieved frames

class UploadResponse(BaseModel):
    job_id: str
    video_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    video_id: str
    status: str
    progress: float
    frames_processed: int
    frames_total: int
    message: Optional[str] = None
    error: Optional[str] = None
    user_id: Optional[str] = None  # Track which user owns this job

class UserLogin(BaseModel):
    id: str
    name: str
    email: str
    picture: str


class DeepScanRequest(BaseModel):
    """Request body for deep scan endpoint."""
    video_id: str
    start_time: float
    end_time: float
    fps: float = 1.0  # Frames per second to analyze

# ============================================================================
# In-Memory Job Storage (Use Redis in production)
# ============================================================================

jobs_store: Dict[str, JobStatus] = {}

# ============================================================================
# Database Helper Functions
# ============================================================================

def get_user_from_cache(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user from Redis cache."""
    if redis_client is None:
        return None
    try:
        user_data = redis_client.get(f"user:{user_id}")
        return json.loads(user_data) if user_data else None
    except Exception as e:
        print(f"Redis read error: {e}")
        return None

def cache_user(user_id: str, user_data: Dict[str, Any], ttl: int = 3600):
    """Cache user in Redis (1 hour TTL by default)."""
    if redis_client is None:
        return
    try:
        redis_client.setex(f"user:{user_id}", ttl, json.dumps(user_data))
    except Exception as e:
        print(f"Redis write error: {e}")

def save_user_to_db(user_data: Dict[str, Any]):
    """Save user to MongoDB."""
    if users_collection is None:
        return
    try:
        users_collection.update_one(
            {"id": user_data["id"]},
            {"$set": user_data},
            upsert=True
        )
    except Exception as e:
        print(f"MongoDB write error: {e}")

def get_all_users_from_db() -> List[Dict[str, Any]]:
    """Get all users from MongoDB."""
    if users_collection is None:
        return []
    try:
        return list(users_collection.find({}, {"_id": 0}).sort("lastLogin", -1))
    except Exception as e:
        print(f"MongoDB read error: {e}")
        return []

# ============================================================================
# SemanticScout - Local CPU-based Frame Filtering (Phase 2 Enhanced)
# ============================================================================

class SemanticScout:
    """
    Local "Scout" that filters video frames using:
    1. Audio Trigger - Detects sound spikes (CRITICAL priority)
    2. Brightness Spike Detector - Catches lightning/flashes (CRITICAL priority)
    3. Physics Filter - Vertical optical flow (falling vs walking)
    4. Optical Flow (Physics) - Detects motion between frames
    5. CLIP Embeddings (Meaning) - Removes semantically redundant frames
    
    Phase 2 Priority Queue:
    - CRITICAL (Audio Spikes OR Brightness Spikes) -> Bypass all filters
    - HIGH (Falling/Vertical) -> Send to NVIDIA
    - MEDIUM (General Motion) -> Send to NVIDIA
    - LOW/DISCARD (Static/Walking) -> Skip
    """
    
    def __init__(self):
        print("🔍 Loading CLIP model (clip-ViT-B-32)...")
        self.clip_model = SentenceTransformer('clip-ViT-B-32')
        print("✅ CLIP model loaded successfully!")
        self.last_embedding = None
        
        # Phase 2: Initialize advanced filters
        self.audio_trigger = AudioTrigger(
            rms_threshold=0.05,
            spike_multiplier=2.5,
            chunk_duration=1.0
        )
        self.physics_filter = PhysicsFilter(
            vertical_threshold=5.0,
            horizontal_threshold=3.0,
            cluster_min_pixels=500
        )
        # Phase 2.1: Brightness spike detector for lightning/flashes
        self.brightness_detector = BrightnessSpikeDetector(
            spike_threshold=25.0,       # Lower threshold to catch subtle flashes
            relative_threshold=0.25,    # 25% brightness increase = spike
            decay_frames=2,             # Shorter decay for animated content
            min_brightness=15.0         # Lower threshold for dark scenes
        )
        self.priority_manager = PriorityQueueManager()
    
    def _compute_motion_score(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute motion score between two frames using pixel difference."""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
        gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
        
        # Compute absolute difference
        diff = cv2.absdiff(gray1, gray2)
        motion_score = np.mean(diff)
        
        return motion_score
    
    def _get_clip_embedding(self, frame: np.ndarray) -> np.ndarray:
        """Generate CLIP embedding for a frame."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Generate embedding
        embedding = self.clip_model.encode(pil_image, convert_to_numpy=True)
        return embedding
    
    def _is_semantically_unique(self, embedding: np.ndarray) -> bool:
        """Check if frame is semantically unique compared to last processed frame."""
        if self.last_embedding is None:
            return True
        
        similarity = util.cos_sim(embedding, self.last_embedding).item()
        return similarity < SIMILARITY_THRESHOLD
    
    def process_video(self, video_path: str, job_id: str) -> List[Dict]:
        """
        Process video and extract semantically unique frames.
        
        Phase 2 Enhancement: Runs audio analysis first, then combines with
        physics and CLIP filters using priority queue.
        
        Returns:
            List of dicts with 'frame', 'timestamp', 'embedding' for surviving frames
        """
        print(f"🎬 Processing video: {video_path}")
        
        # =====================================================================
        # Phase 2: Audio Analysis (Run FIRST)
        # =====================================================================
        print("\n📢 Phase 2: Running audio spike detection...")
        critical_timestamps = self.audio_trigger.get_critical_timestamps(video_path)
        self.priority_manager.set_audio_critical_timestamps(critical_timestamps)
        
        if critical_timestamps:
            print(f"  🔊 Found {len(critical_timestamps)} audio-critical timestamps")
        else:
            print("  ℹ️ No audio spikes detected")
        
        # Reset brightness detector for new video
        self.brightness_detector.reset()
        print("  ⚡ Brightness spike detector ready (lightning/flash detection)")
        
        # =====================================================================
        # Video Processing
        # =====================================================================
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0
        
        # Dynamic frame skip interval based on video duration (P0 - High FPS for short videos)
        # Shorter videos get higher FPS to capture fast events like lightning, falls, etc.
        if video_duration <= 10:
            # Very short videos: Capture almost every second for maximum detail
            frame_skip = max(1, int(fps / 4))  # ~4 FPS for very short videos
            print(f"  🎯 Very short video ({video_duration:.1f}s) - using MAX FPS (~4 fps, skip every {frame_skip} frames)")
        elif video_duration <= 30:
            # Short videos: High detail capture
            frame_skip = max(1, int(fps / 3))  # ~3 FPS for short videos
            print(f"  ⚡ Short video ({video_duration:.1f}s) - using high FPS (~3 fps, skip every {frame_skip} frames)")
        elif video_duration <= 60:
            # Medium-short videos
            frame_skip = max(1, int(fps / 2.5))  # ~2.5 FPS
            print(f"  📹 Medium video ({video_duration:.1f}s) - using medium-high FPS (~2.5 fps, skip every {frame_skip} frames)")
        elif video_duration <= 120:
            # Medium videos
            frame_skip = max(1, int(fps / 2))  # ~2 FPS for medium videos
            print(f"  📹 Medium video ({video_duration:.1f}s) - using medium FPS (~2 fps, skip every {frame_skip} frames)")
        elif video_duration <= 300:
            # Long videos (up to 5 minutes)
            frame_skip = max(1, int(fps / 1.5))  # ~1.5 FPS
            print(f"  🎬 Long video ({video_duration:.1f}s) - using reduced FPS (~1.5 fps, skip every {frame_skip} frames)")
        else:
            # Very long videos (5+ minutes)
            frame_skip = max(1, int(fps / 1))  # ~1 FPS for very long videos
            print(f"  🎬 Very long video ({video_duration:.1f}s) - using standard FPS (~1 fps, skip every {frame_skip} frames)")
        
        surviving_frames = []
        prev_frame = None
        frame_count = 0
        processed_count = 0
        
        # Stats for Phase 2
        audio_bypassed = 0
        physics_passed = 0
        clip_passed = 0
        
        # Update job status
        jobs_store[job_id].frames_total = total_frames // frame_skip
        
        self.last_embedding = None  # Reset for new video
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for efficiency (using dynamic interval)
            if frame_count % frame_skip != 0:
                continue
            
            timestamp = frame_count / fps
            processed_count += 1
            
            # Update progress
            jobs_store[job_id].frames_processed = processed_count
            jobs_store[job_id].progress = processed_count / max(jobs_store[job_id].frames_total, 1)
            
            # =====================================================================
            # Phase 2: Check if timestamp is CRITICAL (audio spike)
            # If critical, BYPASS all other filters
            # =====================================================================
            is_audio_critical = self.priority_manager.is_audio_critical(timestamp, tolerance=0.5)
            
            if is_audio_critical:
                # BYPASS all filters - audio spike detected near this frame
                embedding = self._get_clip_embedding(frame)
                self.last_embedding = embedding
                
                surviving_frames.append({
                    'frame': frame,
                    'timestamp': timestamp,
                    'embedding': embedding,
                    'frame_id': f"{job_id}_frame_{len(surviving_frames)}",
                    'priority': 'CRITICAL_AUDIO'
                })
                
                audio_bypassed += 1
                prev_frame = frame
                print(f"  🔊 Frame at {timestamp:.2f}s BYPASSED (audio critical)")
                continue
            
            # =====================================================================
            # Phase 2.1: Check for BRIGHTNESS SPIKE (lightning/flash)
            # This catches sudden visual changes even without audio
            # =====================================================================
            brightness_result = self.brightness_detector.analyze_frame(frame, timestamp)
            
            if brightness_result.is_spike:
                # BYPASS all filters - brightness spike detected (LIGHTNING!)
                embedding = self._get_clip_embedding(frame)
                self.last_embedding = embedding
                
                surviving_frames.append({
                    'frame': frame,
                    'timestamp': timestamp,
                    'embedding': embedding,
                    'frame_id': f"{job_id}_frame_{len(surviving_frames)}",
                    'priority': 'CRITICAL_BRIGHTNESS'
                })
                
                prev_frame = frame
                # Don't increment audio_bypassed, this is a different type
                print(f"  ⚡ Frame at {timestamp:.2f}s BYPASSED (brightness spike - possible lightning!)")
                continue
            
            # =====================================================================
            # Logic Gate 1: Physics Filter (Phase 2)
            # Check for vertical motion (falling) vs horizontal (walking)
            # =====================================================================
            if prev_frame is not None:
                should_process, physics_result = self.physics_filter.should_process_frame(
                    prev_frame, frame
                )
                
                # Update priority manager
                self.priority_manager.update_priority(timestamp, physics_result.priority)
                
                # Skip walking/static motion
                if physics_result.priority == FramePriority.DISCARD:
                    prev_frame = frame
                    continue
                
                if physics_result.priority == FramePriority.LOW:
                    # Low priority but not discard - apply stricter threshold
                    motion_score = self._compute_motion_score(prev_frame, frame)
                    if motion_score < MOTION_THRESHOLD * 1.5:  # Stricter threshold
                        prev_frame = frame
                        continue
                
                # High priority (falling) passes immediately
                if physics_result.priority == FramePriority.HIGH:
                    physics_passed += 1
            
            # =====================================================================
            # Logic Gate 2: Classic Motion Detection (Fallback)
            # =====================================================================
            if prev_frame is not None:
                motion_score = self._compute_motion_score(prev_frame, frame)
                if motion_score < MOTION_THRESHOLD:
                    prev_frame = frame
                    continue
            
            # =====================================================================
            # Logic Gate 3: CLIP Similarity (Semantic Redundancy)
            # =====================================================================
            embedding = self._get_clip_embedding(frame)
            
            if not self._is_semantically_unique(embedding):
                prev_frame = frame
                continue
            
            # Frame survived all gates!
            self.last_embedding = embedding
            clip_passed += 1
            
            surviving_frames.append({
                'frame': frame,
                'timestamp': timestamp,
                'embedding': embedding,
                'frame_id': f"{job_id}_frame_{len(surviving_frames)}",
                'priority': 'STANDARD'
            })
            
            prev_frame = frame
            print(f"  ✓ Frame at {timestamp:.2f}s passed filters")
        
        cap.release()
        
        print(f"\n📊 Filtering complete: {len(surviving_frames)}/{total_frames} frames survived")
        print(f"   🔊 Audio bypassed: {audio_bypassed}")
        print(f"   📐 Physics (vertical): {physics_passed}")
        print(f"   🎯 CLIP passed: {clip_passed}")
        
        return surviving_frames

# ============================================================================
# NvidiaProcessor - Cloud Intelligence via NVIDIA NIM
# ============================================================================

class NvidiaProcessor:
    """
    Cloud "Intelligence" that processes surviving frames using NVIDIA NIM:
    1. VILA-40B for vision analysis (image -> text description)
    2. Mistral-7B for text embedding (description -> vector)
    """
    
    def __init__(self, api_keys: List[str]):
        if not api_keys:
            print("⚠️ No NVIDIA API keys provided. Cloud processing disabled.")
            self.key_cycle = None
        else:
            self.key_cycle = itertools.cycle(api_keys)
            print(f"🔑 Initialized with {len(api_keys)} NVIDIA API key(s)")
    
    def _get_next_key(self) -> str:
        """Get next API key from rotation pool."""
        if self.key_cycle is None:
            raise ValueError("No API keys available")
        return next(self.key_cycle)
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert OpenCV frame to base64 string."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _analyze_frame_with_model(self, frame: np.ndarray, model_name: str, prompt: str, max_retries: int = 3) -> str:
        """Send frame to a specific vision model for analysis."""
        base64_image = self._frame_to_base64(frame)
        
        for attempt in range(max_retries):
            api_key = self._get_next_key()
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.2
            }
            
            try:
                response = requests.post(NVIDIA_VISION_URL, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 429:
                    print(f"  ⚠️ Rate limit hit on key, rotating...")
                    time.sleep(1)
                    continue
                
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content']
                
            except requests.exceptions.RequestException as e:
                print(f"  ❌ {model_name} request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return ""
    
    def _analyze_frame_with_vila(self, frame: np.ndarray, max_retries: int = 3) -> str:
        """Send frame to primary vision model (Llama 3.2 90B Vision)."""
        return self._analyze_frame_with_model(
            frame, 
            "meta/llama-3.2-90b-vision-instruct", 
            VISION_PROMPT, 
            max_retries
        )
    
    def _analyze_frame_ensemble(self, frame: np.ndarray, use_ensemble: bool = True) -> str:
        """
        Multi-Model Ensemble (P3): Analyze frame using multiple vision models 
        and combine their outputs for better accuracy.
        
        Models used:
        1. Llama 3.2 90B Vision (Primary - detailed scene analysis)
        2. Llama 3.2 11B Vision (Secondary - fast action detection)
        
        The ensemble approach helps catch events that one model might miss,
        particularly fast-moving events like lightning, falls, or explosions.
        """
        if not use_ensemble:
            # Single model mode (faster)
            return self._analyze_frame_with_vila(frame)
        
        # Ensemble mode: Use multiple models in parallel
        primary_description = ""
        secondary_description = ""
        
        # Primary model: Detailed scene analysis
        primary_prompt = VISION_PROMPT
        
        # Secondary model: Focused on actions and events  
        action_prompt = """Describe ONLY the actions and events in this frame:
- What is happening RIGHT NOW?
- Any sudden movements, impacts, flashes, or changes?
- What are people/objects doing?
- Any visual phenomena: lightning, fire, explosions, falls?

Be concise but specific about ACTIONS and EVENTS."""
        
        # Run models in parallel for speed
        with ThreadPoolExecutor(max_workers=2) as executor:
            primary_future = executor.submit(
                self._analyze_frame_with_model,
                frame, "meta/llama-3.2-90b-vision-instruct", primary_prompt
            )
            secondary_future = executor.submit(
                self._analyze_frame_with_model,
                frame, "meta/llama-3.2-11b-vision-instruct", action_prompt
            )
            
            try:
                primary_description = primary_future.result(timeout=35)
            except Exception as e:
                print(f"  ⚠️ Primary model failed: {e}")
                primary_description = ""
            
            try:
                secondary_description = secondary_future.result(timeout=35)
            except Exception as e:
                print(f"  ⚠️ Secondary model failed: {e}")
                secondary_description = ""
        
        # Combine descriptions intelligently
        if primary_description and secondary_description:
            # Merge both descriptions, avoiding redundancy
            combined = f"{primary_description}\n\n**Action Focus:** {secondary_description}"
            return combined
        elif primary_description:
            return primary_description
        elif secondary_description:
            return secondary_description
        else:
            return "Frame analysis failed"
    
    def _embed_text_with_mistral(self, text: str, max_retries: int = 3) -> List[float]:
        """Generate embedding for text using Mistral-7B."""
        for attempt in range(max_retries):
            api_key = self._get_next_key()
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "nvidia/nv-embed-v1",
                "input": text,
                "input_type": "passage"
            }
            
            try:
                response = requests.post(NVIDIA_EMBED_URL, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 429:
                    print(f"  ⚠️ Rate limit hit on key, rotating...")
                    time.sleep(1)
                    continue
                
                response.raise_for_status()
                result = response.json()
                return result['data'][0]['embedding']
                
            except requests.exceptions.RequestException as e:
                print(f"  ❌ Mistral embed request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return []
    
    def _process_single_frame(self, frame_data: Dict, use_ensemble: bool = True) -> Dict:
        """Process a single frame through Vision + Embedding pipeline."""
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        frame_id = frame_data['frame_id']
        
        # Step 1: Visual analysis (ensemble or single model)
        if use_ensemble and ENSEMBLE_MODE:
            description = self._analyze_frame_ensemble(frame, use_ensemble=True)
        else:
            description = self._analyze_frame_with_vila(frame)
        
        # Step 2: Text embedding with NV-Embed
        embedding = self._embed_text_with_mistral(description)
        
        return {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'description': description,
            'embedding': embedding
        }
    
    def process_frames(self, frames: List[Dict], max_workers: int = 4) -> List[Dict]:
        """
        Process multiple frames in parallel using ThreadPoolExecutor.
        
        Args:
            frames: List of frame dicts from SemanticScout
            max_workers: Number of parallel threads
            
        Returns:
            List of processed frame dicts with descriptions and embeddings
        """
        if self.key_cycle is None:
            print("⚠️ Skipping NVIDIA processing - no API keys")
            # Return frames with CLIP embeddings as fallback
            return [{
                'frame_id': f['frame_id'],
                'timestamp': f['timestamp'],
                'description': 'Frame analysis unavailable (no API keys)',
                'embedding': f['embedding'].tolist()
            } for f in frames]
        
        print(f"🚀 Processing {len(frames)} frames with NVIDIA NIM...")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_frame = {
                executor.submit(self._process_single_frame, frame): frame 
                for frame in frames
            }
            
            for future in as_completed(future_to_frame):
                try:
                    result = future.result()
                    results.append(result)
                    print(f"  ✓ Processed frame at {result['timestamp']:.2f}s")
                except Exception as e:
                    frame = future_to_frame[future]
                    print(f"  ❌ Failed to process frame at {frame['timestamp']:.2f}s: {e}")
        
        # Sort by timestamp
        results.sort(key=lambda x: x['timestamp'])
        
        print(f"✅ NVIDIA processing complete: {len(results)} frames analyzed")
        return results

# ============================================================================
# Qdrant Vector Store
# ============================================================================

class VectorStore:
    """Qdrant Cloud vector database for storing and querying frame embeddings."""
    
    def __init__(self):
        if not PINECONE_API_KEY:
            print("⚠️ Pinecone API Key not set. Vector storage disabled.")
            self.pc = None
            self.index = None
            return
        
        try:
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Check if index exists, create if not
            existing_indexes = [i.name for i in self.pc.list_indexes()]
            
            if PINECONE_INDEX_NAME not in existing_indexes:
                print(f"📦 Creating Pinecone index: {PINECONE_INDEX_NAME}")
                self.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=4096, # NV-Embed dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            
            self.index = self.pc.Index(PINECONE_INDEX_NAME)
            print(f"✅ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
            
        except Exception as e:
            print(f"❌ Failed to connect to Pinecone: {e}")
            self.pc = None
            self.index = None
    
    def upsert_frames(self, video_id: str, frames: List[Dict], user_id: Optional[str] = None) -> int:
        """Store frame embeddings in Pinecone with optional user namespace for multi-tenancy."""
        if self.index is None:
            print("⚠️ Pinecone not configured, skipping upsert")
            return 0
        
        # Use user_id as namespace for multi-user isolation
        namespace = user_id if user_id else ""
        
        vectors = []
        for frame in frames:
            if not frame.get('embedding'):
                continue
            
            vectors.append({
                "id": frame['frame_id'],
                "values": frame['embedding'],
                "metadata": {
                    "video_id": video_id,
                    "user_id": user_id or "anonymous",
                    "timestamp": frame['timestamp'],
                    "description": frame['description'][:1000]
                }
            })
        
        if vectors:
            # Upsert in batches of 50
            batch_size = 50
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                print(f"  📤 Uploading batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}...")
                self.index.upsert(vectors=batch, namespace=namespace)
        
        print(f"📤 Upserted {len(vectors)} vectors to Pinecone (namespace: {namespace or 'default'})")
        return len(vectors)
    
    def clear_all(self, user_id: Optional[str] = None) -> bool:
        """Clear vectors from Pinecone index. If user_id provided, only clear that user's namespace."""
        if self.index is None:
            print("⚠️ Pinecone not configured, skipping clear")
            return False
        
        try:
            if user_id:
                # Clear only this user's namespace
                self.index.delete(delete_all=True, namespace=user_id)
                print(f"🗑️ Cleared vectors for user: {user_id}")
            else:
                # Clear all vectors (admin operation)
                self.index.delete(delete_all=True, namespace="")
                print("🗑️ Cleared all vectors from Pinecone")
            return True
        except Exception as e:
            # Handle 404 gracefully - namespace doesn't exist (already empty)
            if "404" in str(e) or "Not Found" in str(e) or "Namespace not found" in str(e):
                print(f"✅ Namespace already empty for user: {user_id if user_id else 'default'}")
                return True
            print(f"❌ Failed to clear Pinecone: {e}")
            return False
    
    def search(self, query_embedding: List[float], video_id: Optional[str] = None, user_id: Optional[str] = None, top_k: int = 10) -> List[Dict]:
        """Search for similar frames within a user's namespace."""
        if self.index is None:
            return []
        
        # Use user_id as namespace for multi-user isolation
        namespace = user_id if user_id else ""
        
        # Build filter if video_id is specified
        filter_dict = {}
        if video_id:
            filter_dict["video_id"] = video_id
        
        results = self.index.query(
            vector=query_embedding,
            filter=filter_dict if filter_dict else None,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        
        # Pinecone already returns results sorted by score (descending)
        return [{
            'frame_id': match['id'],
            'score': match['score'],
            'timestamp': match['metadata']['timestamp'],
            'description': match['metadata']['description']
        } for match in results['matches']]

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="FrameSift API",
    description="Semantic Video Search Engine with Hybrid AI Architecture",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add COOP headers for Google OAuth
@app.middleware("http")
async def add_coop_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin-allow-popups"
    response.headers["Cross-Origin-Embedder-Policy"] = "unsafe-none"
    return response

# Initialize components
scout = SemanticScout()
nvidia_processor = NvidiaProcessor(NVIDIA_KEYS)
vector_store = VectorStore()

# ============================================================================
# Background Processing
# ============================================================================

def process_video_task(video_path: str, job_id: str, user_id: str = "anonymous"):
    """Background task to process uploaded video with multi-user support."""
    try:
        jobs_store[job_id].status = "processing"
        jobs_store[job_id].message = "Extracting frames..."
        
        # Step 1: Local filtering with Scout
        print(f"\n{'='*60}")
        print(f"📹 Starting job: {job_id} (user: {user_id})")
        print(f"{'='*60}")
        
        surviving_frames = scout.process_video(video_path, job_id)
        
        if not surviving_frames:
            jobs_store[job_id].status = "completed"
            jobs_store[job_id].progress = 1.0
            jobs_store[job_id].message = "Processing complete"
            print("⚠️ No frames survived filtering")
            return
        
        # Step 2: Cloud analysis with NVIDIA
        jobs_store[job_id].message = "Analyzing frames with AI..."
        analyzed_frames = nvidia_processor.process_frames(surviving_frames)
        
        # Step 3: Store in Pinecone (in user's namespace for multi-tenancy)
        jobs_store[job_id].message = "Storing vectors..."
        vector_store.upsert_frames(job_id, analyzed_frames, user_id=user_id)
        
        # Don't delete video - keep it for playback
        # Videos are stored in ./videos/{user_id}/ directory
        
        jobs_store[job_id].status = "completed"
        jobs_store[job_id].progress = 1.0
        jobs_store[job_id].message = "Processing complete"
        
        print(f"\n✅ Job {job_id} completed successfully!")
        print(f"   User: {user_id}")
        print(f"   Frames analyzed: {len(analyzed_frames)}")
        
    except Exception as e:
        jobs_store[job_id].status = "failed"
        jobs_store[job_id].error = str(e)
        jobs_store[job_id].message = f"Processing failed: {str(e)}"
        print(f"❌ Job {job_id} failed: {e}")

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "FrameSift API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Render health check endpoint."""
    return {"status": "healthy"}

@app.post("/upload", response_model=UploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_user_id: Optional[str] = Header(None)  # Multi-user support
):
    """
    Upload a video for semantic processing.
    
    The video will be processed in the background:
    1. Local Scout filters frames using motion + CLIP similarity
    2. NVIDIA NIM analyzes surviving frames
    3. Embeddings stored in Pinecone for search (in user's namespace)
    
    Headers:
        x-user-id: Optional user ID for multi-tenancy
    """
    # Validate file type
    allowed_types = ["video/mp4", "video/avi", "video/mov", "video/webm", "video/quicktime"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    user_id = x_user_id or "anonymous"
    
    # Save file to user-specific directory for multi-user isolation
    videos_dir = os.path.join(os.getcwd(), "videos", user_id)
    os.makedirs(videos_dir, exist_ok=True)
    
    # Save with original extension
    file_ext = os.path.splitext(file.filename)[1]
    video_filename = f"{job_id}{file_ext}"
    video_path = os.path.join(videos_dir, video_filename)
    
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    print(f"📤 Video uploaded by user: {user_id} -> {video_filename}")
    
    # Initialize job status
    jobs_store[job_id] = JobStatus(
        job_id=job_id,
        video_id=job_id,
        status="queued",
        progress=0.0,
        frames_processed=0,
        frames_total=0,
        user_id=user_id
    )
    
    # Start background processing with user_id
    background_tasks.add_task(process_video_task, video_path, job_id, user_id)
    
    return UploadResponse(
        job_id=job_id,
        video_id=job_id,
        status="queued",
        message="Video upload successful. Processing started."
    )

@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a video processing job."""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs_store[job_id]

@app.post("/search", response_model=SearchResponse)
async def search_video(query: SearchQuery, x_user_id: Optional[str] = Header(None)):
    """
    Search for video moments matching a text query.
    
    Returns timestamps with matched descriptions, relevance scores,
    and an AI-generated answer based on the retrieved frames.
    
    Headers:
        x-user-id: Optional user ID for multi-tenancy (searches only user's videos)
    """
    if not query.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Use header user_id or query user_id
    user_id = x_user_id or query.user_id
    
    # =========================================================================
    # Query Expansion (Phase 2.1) - Improve accuracy by expanding with synonyms
    # =========================================================================
    expanded_query = expand_search_query(query.query)
    print(f"🔍 Original query: '{query.query}' (user: {user_id or 'anonymous'})")
    if expanded_query != query.query:
        print(f"   Expanded to: '{expanded_query}'")
    
    # Embed expanded query using Mistral
    if nvidia_processor.key_cycle:
        query_embedding = nvidia_processor._embed_text_with_mistral(expanded_query)
    else:
        # Fallback to CLIP for text embedding
        query_embedding = scout.clip_model.encode(expanded_query, convert_to_numpy=True).tolist()
    
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Failed to generate query embedding")
    
    # Search Pinecone with more results (within user's namespace for multi-tenancy)
    results = vector_store.search(
        query_embedding=query_embedding,
        video_id=query.video_id,
        user_id=user_id,  # Multi-user isolation
        top_k=query.top_k * 3  # Get more results for re-ranking
    )
    
    # Apply keyword boost to improve accuracy
    results = apply_keyword_boost(query.query, results)
    
    # =========================================================================
    # LLM Re-Ranking: Use AI to re-order results by semantic relevance
    # This fixes cases where embedding similarity doesn't match user intent
    # =========================================================================
    if results and nvidia_processor.key_cycle:
        api_key = next(nvidia_processor.key_cycle)
        results = rerank_with_llm(query.query, results, api_key)
    
    # Take top_k results after re-ranking
    top_results = results[:query.top_k]
    
    # =========================================================================
    # RAG: Generate AI Answer from retrieved frame descriptions
    # =========================================================================
    ai_answer = None
    if top_results and nvidia_processor.key_cycle:
        api_key = next(nvidia_processor.key_cycle)
        ai_answer = generate_ai_answer(
            query=query.query,
            frame_descriptions=top_results,
            api_key=api_key
        )
    
    # Return enhanced response with AI answer
    return SearchResponse(
        results=[
            SearchResult(
                timestamp=r['timestamp'],
                score=min(r['score'], 1.0),  # Cap at 1.0
                description=r['description'],
                frame_id=r['frame_id']
            )
            for r in top_results
        ],
        query=query.query,
        ai_answer=ai_answer
    )

@app.get("/jobs")
async def list_jobs():
    """List all processing jobs."""
    return list(jobs_store.values())

@app.get("/videos/{video_id}")
async def get_video(video_id: str):
    """Serve video file for playback (supports multi-user directories)."""
    videos_dir = os.path.join(os.getcwd(), "videos")
    
    # First, check user-specific directories for multi-user support
    if os.path.exists(videos_dir):
        # Search in all user directories
        for user_dir in os.listdir(videos_dir):
            user_video_dir = os.path.join(videos_dir, user_dir)
            if os.path.isdir(user_video_dir):
                for ext in [".mp4", ".avi", ".mov", ".webm", ".quicktime"]:
                    video_path = os.path.join(user_video_dir, f"{video_id}{ext}")
                    if os.path.exists(video_path):
                        from fastapi.responses import FileResponse
                        return FileResponse(video_path, media_type="video/mp4")
        
        # Fallback: check root videos directory (legacy support)
        for ext in [".mp4", ".avi", ".mov", ".webm", ".quicktime"]:
            video_path = os.path.join(videos_dir, f"{video_id}{ext}")
            if os.path.exists(video_path):
                from fastapi.responses import FileResponse
                return FileResponse(video_path, media_type="video/mp4")
    
    raise HTTPException(status_code=404, detail="Video not found")

# ============================================================================
# Admin Endpoints
# ============================================================================

def verify_admin_key(x_admin_key: Optional[str] = Header(None)):
    """Verify admin API key."""
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return True

@app.post("/admin/track-login")
async def track_user_login(user: UserLogin, x_admin_key: Optional[str] = Header(None)):
    """Track user login for admin portal."""
    verify_admin_key(x_admin_key)
    
    user_id = user.id
    
    # Check cache first
    cached_user = get_user_from_cache(user_id)
    
    if cached_user:
        # Update existing user
        user_data = {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "picture": user.picture,
            "lastLogin": datetime.now().isoformat(),
            "loginCount": cached_user.get("loginCount", 0) + 1,
        }
    else:
        # New user or cache miss - check MongoDB
        existing_user = None
        if users_collection is not None:
            existing_user = users_collection.find_one({"id": user_id})
        
        if existing_user:
            user_data = {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "picture": user.picture,
                "lastLogin": datetime.now().isoformat(),
                "loginCount": existing_user.get("loginCount", 0) + 1,
            }
        else:
            # Brand new user
            user_data = {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "picture": user.picture,
                "lastLogin": datetime.now().isoformat(),
                "loginCount": 1,
            }
    
    # Save to MongoDB (persistent)
    save_user_to_db(user_data)
    
    # Cache in Redis (fast access)
    cache_user(user_id, user_data)
    
    print(f"✅ User login tracked: {user.name} ({user.email}) - Login #{user_data['loginCount']}")
    return {"status": "success", "message": "Login tracked"}

@app.get("/admin/users")
async def get_all_users(x_admin_key: Optional[str] = Header(None)):
    """Get all logged-in users for admin portal."""
    verify_admin_key(x_admin_key)
    
    # Get from MongoDB (persistent storage)
    users_list = get_all_users_from_db()
    
    return {"users": users_list, "total": len(users_list)}

@app.post("/clear-database")
async def clear_database(x_user_id: Optional[str] = Header(None)):
    """
    Clear vectors and files for a specific user on logout.
    
    Multi-user aware:
    - If x-user-id is provided: Only clears that user's data
    - If no user_id: Clears all data (admin operation)
    
    Clears:
    - Pinecone vectors (in user's namespace)
    - User's video files
    - Temporary audio files
    """
    user_id = x_user_id
    
    try:
        # Clear Pinecone vectors (user-specific namespace if user_id provided)
        vector_success = vector_store.clear_all(user_id=user_id)
        
        # Delete video files
        videos_dir = os.path.join(os.getcwd(), "videos")
        videos_deleted = 0
        
        if os.path.exists(videos_dir):
            if user_id:
                # Delete only this user's video directory
                user_video_dir = os.path.join(videos_dir, user_id)
                if os.path.exists(user_video_dir):
                    import shutil
                    try:
                        shutil.rmtree(user_video_dir)
                        videos_deleted = len(os.listdir(user_video_dir)) if os.path.exists(user_video_dir) else 1
                        print(f"🗑️ Deleted user directory: {user_video_dir}")
                    except Exception as e:
                        print(f"⚠️ Failed to delete user directory: {e}")
            else:
                # Admin operation: delete all videos
                for item in os.listdir(videos_dir):
                    item_path = os.path.join(videos_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                            videos_deleted += 1
                        elif os.path.isdir(item_path):
                            import shutil
                            shutil.rmtree(item_path)
                            videos_deleted += 1
                    except Exception as e:
                        print(f"⚠️ Failed to delete {item}: {e}")
        
        # Delete temporary audio files
        temp_dir = tempfile.gettempdir()
        audio_deleted = 0
        
        try:
            for filename in os.listdir(temp_dir):
                if filename.endswith('.wav'):
                    file_path = os.path.join(temp_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            audio_deleted += 1
                    except Exception as e:
                        # Ignore files in use or permission errors
                        pass
        except Exception as e:
            print(f"⚠️ Failed to clean temp audio files: {e}")
        
        user_info = f" for user: {user_id}" if user_id else " (all users)"
        print(f"🗑️ Cleared {videos_deleted} video(s) and {audio_deleted} audio file(s) on logout{user_info}")
        
        if vector_success:
            return {
                "status": "success",
                "message": f"Cleared data{user_info}: {videos_deleted} video(s), {audio_deleted} audio file(s), vector database"
            }
        else:
            return {
                "status": "partial",
                "message": f"Cleared files{user_info}: {videos_deleted} video(s), {audio_deleted} audio file(s) (database not configured)"
            }
    except Exception as e:
        print(f"❌ Error clearing data: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

# ============================================================================
# Phase 2: Deep Scan Endpoint
# ============================================================================

@app.post("/deep-scan")
async def deep_scan_video(request: DeepScanRequest):
    """
    Phase 2: Deep Scan Mode
    
    Force-process a specific time range of a video, bypassing all filters.
    Useful when the AI missed important frames in a specific segment.
    
    Process:
    1. Extract every frame in the time range at specified FPS
    2. Analyze each frame with NVIDIA VILA (no filtering)
    3. Generate embeddings and upsert to Pinecone
    
    Returns:
        Success status and number of frames processed
    """
    video_id = request.video_id
    start_time = request.start_time
    end_time = request.end_time
    fps = request.fps
    
    print(f"\n🔬 Deep Scan requested for video: {video_id}")
    print(f"   Time range: {start_time:.2f}s - {end_time:.2f}s")
    print(f"   FPS: {fps}")
    
    # Find the video file
    videos_dir = os.path.join(os.getcwd(), "videos")
    video_path = None
    
    for filename in os.listdir(videos_dir):
        if video_id in filename:
            video_path = os.path.join(videos_dir, filename)
            break
    
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    
    try:
        # Initialize deep scan processor
        deep_processor = DeepScanProcessor(
            nvidia_keys=NVIDIA_KEYS,
            vila_url=NVIDIA_VLM_URL,
            embed_url=NVIDIA_EMBED_URL,
            max_workers=len(NVIDIA_KEYS)
        )
        
        # Run deep scan
        results = deep_processor.deep_scan(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            fps=fps
        )
        
        if not results:
            return {
                "status": "warning",
                "message": "No frames processed in the specified range",
                "frames_processed": 0
            }
        
        # Upsert results to Pinecone
        upsert_count = 0
        for result in results:
            if result.embedding:
                metadata = {
                    "video_id": video_id,
                    "timestamp": result.timestamp,
                    "description": result.description,
                    "source": "deep_scan"
                }
                
                # Generate unique frame ID for deep scan
                frame_id = f"{video_id}_deepscan_{result.timestamp:.2f}".replace(".", "_")
                
                success = vector_store.upsert_frame(
                    frame_id=frame_id,
                    embedding=result.embedding,
                    metadata=metadata
                )
                
                if success:
                    upsert_count += 1
        
        print(f"✅ Deep scan complete: {upsert_count}/{len(results)} frames indexed")
        
        return {
            "status": "success",
            "message": f"Deep scan processed {len(results)} frames, {upsert_count} indexed",
            "frames_processed": len(results),
            "frames_indexed": upsert_count
        }
        
    except Exception as e:
        print(f"❌ Deep scan failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deep scan failed: {str(e)}")

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    print("\n" + "="*60)
    print("🚀 FrameSift API Starting...")
    print("="*60)
    print(f"📦 Pinecone: {'Configured' if PINECONE_API_KEY else 'Not Configured'}")
    print(f"🔑 NVIDIA Keys: {len(NVIDIA_KEYS)} available")
    print("="*60 + "\n")
