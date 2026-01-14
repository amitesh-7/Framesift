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
from scout import AudioTrigger, PhysicsFilter, PriorityQueueManager, FramePriority
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
FRAME_SKIP_INTERVAL = 5  # Process every Nth frame initially

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
# Data Models
# ============================================================================

class SearchQuery(BaseModel):
    query: str
    video_id: Optional[str] = None
    top_k: int = 10

class SearchResult(BaseModel):
    timestamp: float
    score: float
    description: str
    frame_id: str

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
    2. Physics Filter - Vertical optical flow (falling vs walking)
    3. Optical Flow (Physics) - Detects motion between frames
    4. CLIP Embeddings (Meaning) - Removes semantically redundant frames
    
    Phase 2 Priority Queue:
    - CRITICAL (Audio Spikes) -> Bypass all filters, send immediately
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
        
        # =====================================================================
        # Video Processing
        # =====================================================================
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        surviving_frames = []
        prev_frame = None
        frame_count = 0
        processed_count = 0
        
        # Stats for Phase 2
        audio_bypassed = 0
        physics_passed = 0
        clip_passed = 0
        
        # Update job status
        jobs_store[job_id].frames_total = total_frames // FRAME_SKIP_INTERVAL
        
        self.last_embedding = None  # Reset for new video
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for efficiency
            if frame_count % FRAME_SKIP_INTERVAL != 0:
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
    
    def _analyze_frame_with_vila(self, frame: np.ndarray, max_retries: int = 3) -> str:
        """Send frame to VILA-40B for visual analysis."""
        base64_image = self._frame_to_base64(frame)
        
        for attempt in range(max_retries):
            api_key = self._get_next_key()
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta/llama-3.2-90b-vision-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this video frame in detail. Focus on: objects, actions, people, text, scene context. Be concise but comprehensive."
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
                print(f"  ❌ VILA request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
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
    
    def _process_single_frame(self, frame_data: Dict) -> Dict:
        """Process a single frame through VILA + Mistral pipeline."""
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        frame_id = frame_data['frame_id']
        
        # Step 1: Visual analysis with VILA
        description = self._analyze_frame_with_vila(frame)
        
        # Step 2: Text embedding with Mistral
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
    
    def upsert_frames(self, video_id: str, frames: List[Dict]) -> int:
        """Store frame embeddings in Pinecone."""
        if self.index is None:
            print("⚠️ Pinecone not configured, skipping upsert")
            return 0
        
        vectors = []
        for frame in frames:
            if not frame.get('embedding'):
                continue
            
            vectors.append({
                "id": frame['frame_id'],
                "values": frame['embedding'],
                "metadata": {
                    "video_id": video_id,
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
                self.index.upsert(vectors=batch)
        
        print(f"📤 Upserted {len(vectors)} vectors to Pinecone")
        return len(vectors)
    
    def clear_all(self) -> bool:
        """Clear all vectors from Pinecone index."""
        if self.index is None:
            print("⚠️ Pinecone not configured, skipping clear")
            return False
        
        try:
            # Delete all vectors by deleting entire namespace (or all if no namespace)
            self.index.delete(delete_all=True)
            print("🗑️ Cleared all vectors from Pinecone")
            return True
        except Exception as e:
            print(f"❌ Failed to clear Pinecone: {e}")
            return False
    
    def search(self, query_embedding: List[float], video_id: Optional[str] = None, top_k: int = 10) -> List[Dict]:
        """Search for similar frames."""
        if self.index is None:
            return []
        
        # Build filter if video_id is specified
        filter_dict = {}
        if video_id:
            filter_dict["video_id"] = video_id
        
        results = self.index.query(
            vector=query_embedding,
            filter=filter_dict if filter_dict else None,
            top_k=top_k,
            include_metadata=True
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

def process_video_task(video_path: str, job_id: str):
    """Background task to process uploaded video."""
    try:
        jobs_store[job_id].status = "processing"
        jobs_store[job_id].message = "Extracting frames..."
        
        # Step 1: Local filtering with Scout
        print(f"\n{'='*60}")
        print(f"📹 Starting job: {job_id}")
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
        
        # Step 3: Store in Pinecone
        jobs_store[job_id].message = "Storing vectors..."
        vector_store.upsert_frames(job_id, analyzed_frames)
        
        # Don't delete video - keep it for playback
        # Videos are stored in ./videos/ directory
        
        jobs_store[job_id].status = "completed"
        jobs_store[job_id].progress = 1.0
        jobs_store[job_id].message = "Processing complete"
        
        print(f"\n✅ Job {job_id} completed successfully!")
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

@app.post("/upload", response_model=UploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a video for semantic processing.
    
    The video will be processed in the background:
    1. Local Scout filters frames using motion + CLIP similarity
    2. NVIDIA NIM analyzes surviving frames
    3. Embeddings stored in Pinecone for search
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
    
    # Save file to persistent storage
    videos_dir = os.path.join(os.getcwd(), "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # Save with original extension
    file_ext = os.path.splitext(file.filename)[1]
    video_filename = f"{job_id}{file_ext}"
    video_path = os.path.join(videos_dir, video_filename)
    
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Initialize job status
    jobs_store[job_id] = JobStatus(
        job_id=job_id,
        video_id=job_id,
        status="queued",
        progress=0.0,
        frames_processed=0,
        frames_total=0
    )
    
    # Start background processing
    background_tasks.add_task(process_video_task, video_path, job_id)
    
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

@app.post("/search", response_model=List[SearchResult])
async def search_video(query: SearchQuery):
    """
    Search for video moments matching a text query.
    
    Returns timestamps with matched descriptions and relevance scores.
    """
    if not query.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Embed query using Mistral
    if nvidia_processor.key_cycle:
        query_embedding = nvidia_processor._embed_text_with_mistral(query.query)
    else:
        # Fallback to CLIP for text embedding
        query_embedding = scout.clip_model.encode(query.query, convert_to_numpy=True).tolist()
    
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Failed to generate query embedding")
    
    # Search Qdrant
    results = vector_store.search(
        query_embedding=query_embedding,
        video_id=query.video_id,
        top_k=query.top_k
    )
    
    return [
        SearchResult(
            timestamp=r['timestamp'],
            score=r['score'],
            description=r['description'],
            frame_id=r['frame_id']
        )
        for r in results
    ]

@app.get("/jobs")
async def list_jobs():
    """List all processing jobs."""
    return list(jobs_store.values())

@app.get("/videos/{video_id}")
async def get_video(video_id: str):
    """Serve video file for playback."""
    videos_dir = os.path.join(os.getcwd(), "videos")
    
    # Find the video file with this ID (any extension)
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
async def clear_database():
    """Clear all vectors from Pinecone database and delete all video/audio files on user logout."""
    try:
        # Clear Pinecone vectors
        vector_success = vector_store.clear_all()
        
        # Delete all video files
        videos_dir = os.path.join(os.getcwd(), "videos")
        videos_deleted = 0
        
        if os.path.exists(videos_dir):
            for filename in os.listdir(videos_dir):
                file_path = os.path.join(videos_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        videos_deleted += 1
                except Exception as e:
                    print(f"⚠️ Failed to delete {filename}: {e}")
        
        # Delete all temporary audio files
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
        
        print(f"🗑️ Cleared {videos_deleted} video file(s) and {audio_deleted} audio file(s) on logout")
        
        if vector_success:
            return {
                "status": "success",
                "message": f"Database, {videos_deleted} video(s), and {audio_deleted} audio file(s) cleared successfully"
            }
        else:
            return {
                "status": "partial",
                "message": f"{videos_deleted} video(s) and {audio_deleted} audio file(s) deleted, but database not configured"
            }
    except Exception as e:
        print(f"❌ Error clearing data: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
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
