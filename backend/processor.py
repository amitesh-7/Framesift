"""
FrameSift Phase 2 - Robust Parallel Processing
===============================================
This module implements advanced parallel frame processing with:
1. Round-Robin Key Rotation
2. True concurrent processing
3. Retry logic with exponential backoff
4. Rate limit handling
"""

import os
import time
import json
import itertools
from typing import List, Dict, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass
from threading import Lock
import base64
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image
from dotenv import load_dotenv

# Load environment
load_dotenv(".env.local")
if not os.getenv("NVIDIA_KEYS"):
    load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

NVIDIA_VISION_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"

# Load NVIDIA keys
NVIDIA_KEYS = json.loads(os.getenv("NVIDIA_KEYS", '[]'))

# Enhanced Vision Prompt for better accuracy (Phase 2.1)
VISION_PROMPT = """Analyze this video frame comprehensively:

**SCENE**: Describe the environment, location, lighting conditions, weather (if outdoor)
**SUBJECTS**: People, animals, objects - their appearance, position, actions
**EVENTS**: Any significant occurrences - flashes, movements, impacts, changes
**VISUAL PHENOMENA**: Lightning, fire, smoke, explosions, sparks, reflections, glows, brightness changes
**CONTEXT**: What's happening in this moment? What might have just occurred?

IMPORTANT: Use specific terminology:
- Bright flash in dark/cloudy sky = explicitly say "lightning bolt" or "lightning strike"
- Sudden bright light = "flash", "explosion", "camera flash", or "lightning"
- Fast downward movement = "falling", "dropping"
- Fire/flames = "fire", "burning", "flames"

Be detailed and precise in your description."""


# ============================================================================
# Thread-Safe Key Manager
# ============================================================================

class KeyManager:
    """
    Thread-safe round-robin API key manager with rate limit tracking.
    """
    
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.key_iterator = itertools.cycle(api_keys) if api_keys else None
        self.lock = Lock()
        self.rate_limited_keys: Dict[str, float] = {}  # key -> cooldown_until
        self.key_usage_count: Dict[str, int] = {k: 0 for k in api_keys}
    
    def get_next_key(self, max_retries: int = 10) -> Optional[str]:
        """
        Get next available API key using round-robin.
        Skips keys that are currently rate-limited.
        
        Args:
            max_retries: Maximum attempts to find available key
            
        Returns:
            API key string or None if all keys exhausted
        """
        if self.key_iterator is None:
            return None
        
        with self.lock:
            current_time = time.time()
            
            for _ in range(max_retries):
                key = next(self.key_iterator)
                
                # Check if key is rate-limited
                cooldown_until = self.rate_limited_keys.get(key, 0)
                if current_time >= cooldown_until:
                    self.key_usage_count[key] = self.key_usage_count.get(key, 0) + 1
                    return key
                
            # All keys rate-limited, return least-restricted key
            return min(self.rate_limited_keys, key=self.rate_limited_keys.get)
    
    def mark_rate_limited(self, key: str, cooldown_seconds: float = 2.0):
        """Mark a key as rate-limited for a period."""
        with self.lock:
            self.rate_limited_keys[key] = time.time() + cooldown_seconds
            print(f"  ⚠️ Key rate-limited for {cooldown_seconds}s")
    
    def clear_rate_limit(self, key: str):
        """Clear rate limit for a key."""
        with self.lock:
            self.rate_limited_keys.pop(key, None)
    
    def get_stats(self) -> Dict:
        """Get usage statistics."""
        with self.lock:
            return {
                "total_keys": len(self.api_keys),
                "usage_counts": dict(self.key_usage_count),
                "rate_limited": len(self.rate_limited_keys)
            }


# ============================================================================
# Processing Results
# ============================================================================

@dataclass
class ProcessingResult:
    """Result of processing a single frame."""
    success: bool
    frame_id: str
    timestamp: float
    description: str = ""
    embedding: List[float] = None
    error: Optional[str] = None
    retries: int = 0
    key_used: str = ""


# ============================================================================
# Frame Processor
# ============================================================================

class ParallelFrameProcessor:
    """
    Processes frames in parallel with robust error handling and key rotation.
    
    Features:
    - Round-robin key rotation
    - True concurrent processing (ThreadPoolExecutor)
    - Automatic retry on rate limits
    - Exponential backoff
    """
    
    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        max_retries: int = 3,
        base_timeout: int = 30,
        rate_limit_cooldown: float = 2.0
    ):
        self.keys = api_keys or NVIDIA_KEYS
        self.key_manager = KeyManager(self.keys)
        self.max_retries = max_retries
        self.base_timeout = base_timeout
        self.rate_limit_cooldown = rate_limit_cooldown
        
        print(f"🔑 ParallelFrameProcessor initialized with {len(self.keys)} key(s)")
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert OpenCV frame to base64 string."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _analyze_frame_vision(
        self, 
        base64_image: str, 
        api_key: str
    ) -> Tuple[str, bool, Optional[int]]:
        """
        Send frame to NVIDIA Vision API.
        
        Returns:
            Tuple of (description, success, http_status_code)
        """
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
                            "text": VISION_PROMPT
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
            response = requests.post(
                NVIDIA_VISION_URL,
                headers=headers,
                json=payload,
                timeout=self.base_timeout
            )
            
            if response.status_code == 429:
                return "", False, 429
            
            response.raise_for_status()
            result = response.json()
            description = result['choices'][0]['message']['content']
            return description, True, response.status_code
            
        except requests.exceptions.Timeout:
            return "Request timed out", False, None
        except requests.exceptions.RequestException as e:
            return str(e), False, getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
    
    def _embed_text(
        self, 
        text: str, 
        api_key: str
    ) -> Tuple[List[float], bool, Optional[int]]:
        """
        Generate embedding for text.
        
        Returns:
            Tuple of (embedding, success, http_status_code)
        """
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
            response = requests.post(
                NVIDIA_EMBED_URL,
                headers=headers,
                json=payload,
                timeout=self.base_timeout
            )
            
            if response.status_code == 429:
                return [], False, 429
            
            response.raise_for_status()
            result = response.json()
            embedding = result['data'][0]['embedding']
            return embedding, True, response.status_code
            
        except requests.exceptions.Timeout:
            return [], False, None
        except requests.exceptions.RequestException as e:
            return [], False, getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
    
    def process_single_frame(
        self, 
        frame_data: Dict,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """
        Process a single frame with retries and key rotation.
        
        Args:
            frame_data: Dict with 'frame', 'timestamp', 'frame_id'
            progress_callback: Optional callback(frame_id, status)
            
        Returns:
            ProcessingResult with description and embedding
        """
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        frame_id = frame_data['frame_id']
        
        base64_image = self._frame_to_base64(frame)
        
        description = ""
        embedding = []
        last_error = None
        retries = 0
        key_used = ""
        
        # Step 1: Get visual description with retries
        for attempt in range(self.max_retries):
            api_key = self.key_manager.get_next_key()
            if api_key is None:
                last_error = "No API keys available"
                break
            
            key_used = api_key[-8:]  # Last 8 chars for logging
            
            desc, success, status = self._analyze_frame_vision(base64_image, api_key)
            
            if success:
                description = desc
                break
            
            if status == 429:
                # Rate limited - mark key and retry with next key
                self.key_manager.mark_rate_limited(api_key, self.rate_limit_cooldown)
                retries += 1
                time.sleep(1)
                continue
            
            # Other error - exponential backoff
            retries += 1
            last_error = desc
            time.sleep(2 ** attempt)
        
        if not description:
            return ProcessingResult(
                success=False,
                frame_id=frame_id,
                timestamp=timestamp,
                error=last_error or "Vision analysis failed",
                retries=retries,
                key_used=key_used
            )
        
        # Step 2: Generate embedding with retries
        for attempt in range(self.max_retries):
            api_key = self.key_manager.get_next_key()
            if api_key is None:
                last_error = "No API keys available"
                break
            
            emb, success, status = self._embed_text(description, api_key)
            
            if success:
                embedding = emb
                break
            
            if status == 429:
                self.key_manager.mark_rate_limited(api_key, self.rate_limit_cooldown)
                retries += 1
                time.sleep(1)
                continue
            
            retries += 1
            last_error = f"Embedding failed: {status}"
            time.sleep(2 ** attempt)
        
        if not embedding:
            return ProcessingResult(
                success=False,
                frame_id=frame_id,
                timestamp=timestamp,
                description=description,
                error=last_error or "Embedding generation failed",
                retries=retries,
                key_used=key_used
            )
        
        return ProcessingResult(
            success=True,
            frame_id=frame_id,
            timestamp=timestamp,
            description=description,
            embedding=embedding,
            retries=retries,
            key_used=key_used
        )
    
    def process_frames_parallel(
        self,
        frames: List[Dict],
        progress_callback: Optional[Callable] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple frames in parallel using ThreadPoolExecutor.
        
        Uses Round-Robin key rotation with true concurrency.
        
        Args:
            frames: List of frame dicts with 'frame', 'timestamp', 'frame_id'
            progress_callback: Optional callback(completed, total, frame_id)
            
        Returns:
            List of ProcessingResult objects
        """
        if not self.keys:
            print("⚠️ No API keys available - skipping parallel processing")
            return [
                ProcessingResult(
                    success=False,
                    frame_id=f['frame_id'],
                    timestamp=f['timestamp'],
                    error="No API keys available"
                )
                for f in frames
            ]
        
        num_workers = len(self.keys)
        print(f"🚀 Processing {len(frames)} frames with {num_workers} worker(s)...")
        
        results: List[ProcessingResult] = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_frame: Dict[Future, Dict] = {
                executor.submit(self.process_single_frame, frame): frame
                for frame in frames
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_frame):
                frame = future_to_frame[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, len(frames), result.frame_id)
                        
                except Exception as e:
                    results.append(ProcessingResult(
                        success=False,
                        frame_id=frame['frame_id'],
                        timestamp=frame['timestamp'],
                        error=str(e)
                    ))
                    completed += 1
        
        # Sort by timestamp
        results.sort(key=lambda r: r.timestamp)
        
        # Print stats
        successful = sum(1 for r in results if r.success)
        print(f"✅ Parallel processing complete: {successful}/{len(results)} successful")
        
        return results
    
    def results_to_dicts(self, results: List[ProcessingResult]) -> List[Dict]:
        """Convert ProcessingResults to standard dict format for VectorStore."""
        return [
            {
                'frame_id': r.frame_id,
                'timestamp': r.timestamp,
                'description': r.description,
                'embedding': r.embedding
            }
            for r in results if r.success and r.embedding
        ]


# ============================================================================
# Deep Scan Processor
# ============================================================================

class DeepScanProcessor:
    """
    Processor for deep scan mode - processes every frame in a time range
    without applying any scout filters.
    """
    
    def __init__(self, parallel_processor: ParallelFrameProcessor):
        self.processor = parallel_processor
    
    def extract_frames_in_range(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        fps: float = 1.0,
        video_id: str = ""
    ) -> List[Dict]:
        """
        Extract all frames in a time range at specified FPS.
        
        Args:
            video_path: Path to video file
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            fps: Frames per second to extract (default 1.0)
            video_id: Video ID for frame naming
            
        Returns:
            List of frame dicts
        """
        print(f"📸 Deep scan: extracting frames from {start_time}s to {end_time}s...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)
        
        # Seek to start time
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        frame_count = start_frame
        extracted = 0
        
        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract at target FPS
            if (frame_count - start_frame) % frame_interval == 0:
                timestamp = frame_count / video_fps
                frames.append({
                    'frame': frame,
                    'timestamp': timestamp,
                    'frame_id': f"{video_id}_deep_{extracted}"
                })
                extracted += 1
            
            frame_count += 1
        
        cap.release()
        
        print(f"  📊 Extracted {len(frames)} frames for deep scan")
        return frames
    
    def deep_scan(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        video_id: str,
        fps: float = 1.0
    ) -> List[Dict]:
        """
        Perform deep scan on video segment.
        
        Extracts and processes EVERY frame in range without filters.
        
        Args:
            video_path: Path to video file
            start_time: Start timestamp
            end_time: End timestamp
            video_id: Video ID
            fps: Extraction FPS
            
        Returns:
            List of processed frame dicts
        """
        # Extract frames
        frames = self.extract_frames_in_range(
            video_path, start_time, end_time, fps, video_id
        )
        
        if not frames:
            return []
        
        # Process with parallel processor (no filters)
        results = self.processor.process_frames_parallel(frames)
        
        return self.processor.results_to_dicts(results)


# ============================================================================
# Factory Function
# ============================================================================

def create_parallel_processor(
    api_keys: Optional[List[str]] = None
) -> ParallelFrameProcessor:
    """Create a configured parallel processor."""
    keys = api_keys or NVIDIA_KEYS
    return ParallelFrameProcessor(api_keys=keys)
