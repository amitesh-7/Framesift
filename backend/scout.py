"""
FrameSift Phase 2 - Advanced Scout Filters
============================================
This module implements advanced frame filtering capabilities:
1. Audio Trigger (The "Clack" Detector) - Detects sudden sounds
2. Physics Filter (Vertical Optical Flow) - Distinguishes falling from walking
"""

import os
import tempfile
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

# Audio processing
try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("⚠️ moviepy not installed - audio trigger disabled")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ librosa not installed - using scipy fallback for audio")

from scipy.io import wavfile
from scipy import signal


# ============================================================================
# Priority Levels for Frame Processing
# ============================================================================

class FramePriority(Enum):
    """Priority levels for frame processing queue."""
    CRITICAL = 1    # Audio spike OR brightness spike - bypass all filters
    HIGH = 2        # Vertical motion (falling/dropping)
    MEDIUM = 3      # General motion (horizontal movement)
    LOW = 4         # Minimal motion
    DISCARD = 5     # Static/walking - skip


@dataclass
class AudioEvent:
    """Represents a detected audio event."""
    timestamp: float
    rms_value: float
    is_spike: bool
    priority: FramePriority


@dataclass
class PhysicsResult:
    """Result of physics-based motion analysis."""
    has_vertical_motion: bool
    vertical_magnitude: float
    horizontal_magnitude: float
    motion_type: str  # "falling", "walking", "static", "mixed"
    priority: FramePriority


# ============================================================================
# Feature A: Audio Trigger (The "Clack" Detector)
# ============================================================================

class AudioTrigger:
    """
    Detects sudden audio spikes (like chalk dropping, clapping, objects falling)
    that might indicate important moments even without visual motion.
    
    Logic:
    1. Extract audio from video using moviepy
    2. Calculate RMS amplitude for 1-second chunks
    3. If RMS > Threshold (sudden spike), mark as CRITICAL
    4. CRITICAL timestamps bypass all visual filters
    """
    
    def __init__(
        self,
        rms_threshold: float = 0.05,       # Base RMS threshold for spike detection
        spike_multiplier: float = 2.5,      # Spike = RMS > avg * multiplier
        chunk_duration: float = 1.0,        # Analyze 1-second chunks
        min_spike_gap: float = 0.5          # Minimum gap between spikes (seconds)
    ):
        self.rms_threshold = rms_threshold
        self.spike_multiplier = spike_multiplier
        self.chunk_duration = chunk_duration
        self.min_spike_gap = min_spike_gap
        
        if not MOVIEPY_AVAILABLE:
            print("⚠️ AudioTrigger: moviepy not available")
    
    def extract_audio(self, video_path: str) -> Tuple[Optional[np.ndarray], int]:
        """
        Extract audio track from video file.
        
        Returns:
            Tuple of (audio_samples, sample_rate) or (None, 0) if failed
        """
        if not MOVIEPY_AVAILABLE:
            return None, 0
        
        try:
            # Load video and extract audio
            video = VideoFileClip(video_path)
            
            if video.audio is None:
                print("  ℹ️ Video has no audio track")
                return None, 0
            
            # Create temp file for audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_audio_path = tmp.name
            
            # Extract audio to WAV
            video.audio.write_audiofile(
                temp_audio_path,
                fps=22050  # Standard audio sample rate
            )
            
            # Load the audio file
            if LIBROSA_AVAILABLE:
                audio_data, sample_rate = librosa.load(temp_audio_path, sr=22050, mono=True)
            else:
                sample_rate, audio_data = wavfile.read(temp_audio_path)
                # Normalize to float between -1 and 1
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                # Convert stereo to mono if needed
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
            
            # Cleanup
            video.close()
            os.unlink(temp_audio_path)
            
            return audio_data, sample_rate
            
        except Exception as e:
            print(f"  ❌ Audio extraction failed: {e}")
            return None, 0
    
    def calculate_rms(self, audio_chunk: np.ndarray) -> float:
        """Calculate Root Mean Square (RMS) of audio chunk."""
        return np.sqrt(np.mean(audio_chunk ** 2))
    
    def analyze_audio(self, video_path: str) -> List[AudioEvent]:
        """
        Analyze video audio and detect spike events.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of AudioEvent objects with timestamps and spike status
        """
        print("🎵 Analyzing audio for spike detection...")
        
        audio_data, sample_rate = self.extract_audio(video_path)
        
        if audio_data is None or sample_rate == 0:
            print("  ℹ️ Skipping audio analysis (no audio track)")
            return []
        
        # Calculate samples per chunk
        samples_per_chunk = int(sample_rate * self.chunk_duration)
        total_chunks = len(audio_data) // samples_per_chunk
        
        # Calculate RMS for each chunk
        rms_values = []
        for i in range(total_chunks):
            start = i * samples_per_chunk
            end = start + samples_per_chunk
            chunk = audio_data[start:end]
            rms = self.calculate_rms(chunk)
            rms_values.append(rms)
        
        if not rms_values:
            return []
        
        # Calculate statistics for adaptive thresholding
        avg_rms = np.mean(rms_values)
        std_rms = np.std(rms_values)
        
        # Dynamic spike threshold: base threshold OR avg + 2*std, whichever is higher
        spike_threshold = max(
            self.rms_threshold,
            avg_rms * self.spike_multiplier,
            avg_rms + 2 * std_rms
        )
        
        print(f"  📊 Audio stats: avg_rms={avg_rms:.4f}, spike_threshold={spike_threshold:.4f}")
        
        # Detect spikes
        events = []
        last_spike_time = -self.min_spike_gap
        spikes_detected = 0
        
        for i, rms in enumerate(rms_values):
            timestamp = i * self.chunk_duration
            
            is_spike = rms > spike_threshold
            
            # Enforce minimum gap between spikes
            if is_spike and (timestamp - last_spike_time) < self.min_spike_gap:
                is_spike = False
            
            if is_spike:
                last_spike_time = timestamp
                spikes_detected += 1
                priority = FramePriority.CRITICAL
            else:
                priority = FramePriority.LOW
            
            events.append(AudioEvent(
                timestamp=timestamp,
                rms_value=rms,
                is_spike=is_spike,
                priority=priority
            ))
        
        print(f"  🔊 Detected {spikes_detected} audio spikes")
        return events
    
    def get_critical_timestamps(self, video_path: str) -> List[float]:
        """
        Get list of timestamps that are CRITICAL due to audio spikes.
        These timestamps should bypass all visual filters.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of timestamps (in seconds) with audio spikes
        """
        events = self.analyze_audio(video_path)
        return [e.timestamp for e in events if e.is_spike]


# ============================================================================
# Feature B: Physics Filter (Vertical Optical Flow)
# ============================================================================

class PhysicsFilter:
    """
    Analyzes optical flow to distinguish motion types:
    - Walking (horizontal): Low priority, often filtered
    - Falling (vertical): High priority, indicates important events
    - Static: Discarded
    
    Logic:
    1. Convert frames to grayscale
    2. Use cv2.calcOpticalFlowFarneback for dense flow
    3. Calculate horizontal (fx) and vertical (fy) flow components
    4. Filter: Ignore if abs(fx) > abs(fy) (lateral motion)
    5. Trigger: Return True for high vertical (downward) acceleration
    """
    
    def __init__(
        self,
        vertical_threshold: float = 5.0,    # Minimum vertical flow for "falling"
        horizontal_threshold: float = 3.0,  # Horizontal flow threshold
        cluster_min_pixels: int = 500,      # Minimum pixels with vertical motion
        downward_bias: float = 1.5          # Weight for downward (+fy) vs upward
    ):
        self.vertical_threshold = vertical_threshold
        self.horizontal_threshold = horizontal_threshold
        self.cluster_min_pixels = cluster_min_pixels
        self.downward_bias = downward_bias
        
        # Farneback optical flow parameters
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    
    def compute_optical_flow(
        self, 
        prev_frame: np.ndarray, 
        curr_frame: np.ndarray
    ) -> np.ndarray:
        """
        Compute dense optical flow between two frames.
        
        Returns:
            Flow field with shape (H, W, 2) where [..., 0] is fx and [..., 1] is fy
        """
        # Convert to grayscale
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
            
        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame
        
        # Compute Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, **self.flow_params
        )
        
        return flow
    
    def analyze_physics(
        self, 
        prev_frame: np.ndarray, 
        curr_frame: np.ndarray
    ) -> PhysicsResult:
        """
        Analyze motion physics between two frames.
        
        Args:
            prev_frame: Previous video frame (BGR)
            curr_frame: Current video frame (BGR)
            
        Returns:
            PhysicsResult with motion analysis
        """
        flow = self.compute_optical_flow(prev_frame, curr_frame)
        
        # Separate horizontal (fx) and vertical (fy) components
        fx = flow[..., 0]  # Horizontal flow
        fy = flow[..., 1]  # Vertical flow (+y = downward)
        
        # Calculate magnitudes
        flow_magnitude = np.sqrt(fx**2 + fy**2)
        
        # Get regions with significant motion
        motion_mask = flow_magnitude > 1.0
        
        if not np.any(motion_mask):
            return PhysicsResult(
                has_vertical_motion=False,
                vertical_magnitude=0.0,
                horizontal_magnitude=0.0,
                motion_type="static",
                priority=FramePriority.DISCARD
            )
        
        # Calculate average motion components in moving regions
        avg_fx = np.mean(np.abs(fx[motion_mask]))
        avg_fy = np.mean(np.abs(fy[motion_mask]))
        
        # Detect DOWNWARD motion clusters (positive fy in OpenCV coordinates)
        # This catches falling objects, dropping items, etc.
        downward_mask = (fy > self.vertical_threshold) & motion_mask
        downward_pixels = np.sum(downward_mask)
        
        # Detect upward motion (jumping, throwing up)
        upward_mask = (fy < -self.vertical_threshold) & motion_mask
        upward_pixels = np.sum(upward_mask)
        
        # Calculate weighted vertical motion (bias toward downward)
        total_vertical_pixels = (
            downward_pixels * self.downward_bias + upward_pixels
        )
        
        # Detect primarily horizontal motion (walking, panning)
        horizontal_mask = (
            (np.abs(fx) > self.horizontal_threshold) & 
            (np.abs(fx) > np.abs(fy) * 1.5) &  # fx dominates fy
            motion_mask
        )
        horizontal_pixels = np.sum(horizontal_mask)
        
        # Determine motion type and priority
        has_vertical_motion = total_vertical_pixels >= self.cluster_min_pixels
        
        if has_vertical_motion and downward_pixels > horizontal_pixels:
            motion_type = "falling"
            priority = FramePriority.HIGH
        elif horizontal_pixels > total_vertical_pixels * 2:
            motion_type = "walking"
            priority = FramePriority.LOW
        elif has_vertical_motion:
            motion_type = "mixed"
            priority = FramePriority.MEDIUM
        elif np.sum(motion_mask) > self.cluster_min_pixels:
            motion_type = "general"
            priority = FramePriority.MEDIUM
        else:
            motion_type = "minimal"
            priority = FramePriority.LOW
        
        return PhysicsResult(
            has_vertical_motion=has_vertical_motion,
            vertical_magnitude=avg_fy,
            horizontal_magnitude=avg_fx,
            motion_type=motion_type,
            priority=priority
        )
    
    def should_process_frame(
        self, 
        prev_frame: np.ndarray, 
        curr_frame: np.ndarray
    ) -> Tuple[bool, PhysicsResult]:
        """
        Determine if a frame should be processed based on physics.
        
        Returns:
            Tuple of (should_process, physics_result)
        """
        result = self.analyze_physics(prev_frame, curr_frame)
        
        # Process if HIGH or MEDIUM priority
        should_process = result.priority in [
            FramePriority.CRITICAL,
            FramePriority.HIGH,
            FramePriority.MEDIUM
        ]
        
        return should_process, result


# ============================================================================
# Feature C: Brightness Spike Detector (Lightning/Flash Catcher)
# ============================================================================

@dataclass
class BrightnessResult:
    """Result of brightness spike analysis."""
    is_spike: bool
    brightness_delta: float
    current_brightness: float
    previous_brightness: float
    priority: FramePriority


class BrightnessSpikeDetector:
    """
    Detects sudden brightness changes between frames.
    This is specifically designed to catch:
    - Lightning flashes
    - Camera flashes
    - Explosions
    - Any sudden illumination changes
    
    Lightning typically causes a 50-200% brightness increase in a single frame.
    
    Logic:
    1. Convert frame to grayscale
    2. Calculate mean brightness (0-255)
    3. Compare to previous frame's brightness
    4. If delta > threshold, mark as CRITICAL (lightning detected!)
    """
    
    def __init__(
        self,
        spike_threshold: float = 30.0,      # Minimum brightness delta to detect
        relative_threshold: float = 0.3,    # 30% brightness increase = spike
        decay_frames: int = 3,              # Ignore next N frames after spike (flash afterglow)
        min_brightness: float = 20.0        # Ignore very dark frames (noise)
    ):
        self.spike_threshold = spike_threshold
        self.relative_threshold = relative_threshold
        self.decay_frames = decay_frames
        self.min_brightness = min_brightness
        
        # State tracking
        self.previous_brightness: Optional[float] = None
        self.frames_since_spike: int = 999  # Large number = ready to detect
        self.detected_spikes: List[float] = []  # Timestamps of detected spikes
    
    def reset(self):
        """Reset detector state for new video."""
        self.previous_brightness = None
        self.frames_since_spike = 999
        self.detected_spikes = []
    
    def calculate_brightness(self, frame: np.ndarray) -> float:
        """Calculate mean brightness of a frame."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return float(np.mean(gray))
    
    def analyze_frame(
        self, 
        frame: np.ndarray, 
        timestamp: float
    ) -> BrightnessResult:
        """
        Analyze frame for brightness spikes.
        
        Args:
            frame: Current video frame (BGR or grayscale)
            timestamp: Frame timestamp in seconds
            
        Returns:
            BrightnessResult with spike detection info
        """
        current_brightness = self.calculate_brightness(frame)
        
        # First frame - just store brightness
        if self.previous_brightness is None:
            self.previous_brightness = current_brightness
            return BrightnessResult(
                is_spike=False,
                brightness_delta=0.0,
                current_brightness=current_brightness,
                previous_brightness=current_brightness,
                priority=FramePriority.LOW
            )
        
        # Calculate brightness change
        brightness_delta = current_brightness - self.previous_brightness
        
        # Ignore if in decay period (after a spike, frames tend to be bright)
        self.frames_since_spike += 1
        if self.frames_since_spike <= self.decay_frames:
            self.previous_brightness = current_brightness
            return BrightnessResult(
                is_spike=False,
                brightness_delta=brightness_delta,
                current_brightness=current_brightness,
                previous_brightness=self.previous_brightness,
                priority=FramePriority.LOW
            )
        
        # Check for spike (both absolute and relative thresholds)
        is_spike = False
        priority = FramePriority.LOW
        
        # Absolute threshold check
        if brightness_delta >= self.spike_threshold:
            is_spike = True
        
        # Relative threshold check (handles varying base brightness)
        if (self.previous_brightness >= self.min_brightness and
            brightness_delta / max(self.previous_brightness, 1.0) >= self.relative_threshold):
            is_spike = True
        
        # Also detect sudden DECREASE (lightning aftermath - bright to dark)
        # This helps catch lightning even if we missed the peak
        if brightness_delta <= -self.spike_threshold * 1.5:
            is_spike = True
        
        if is_spike:
            priority = FramePriority.CRITICAL
            self.frames_since_spike = 0
            self.detected_spikes.append(timestamp)
            print(f"  ⚡ BRIGHTNESS SPIKE at {timestamp:.2f}s! "
                  f"Delta: {brightness_delta:.1f} "
                  f"({self.previous_brightness:.1f} → {current_brightness:.1f})")
        
        # Update state
        prev = self.previous_brightness
        self.previous_brightness = current_brightness
        
        return BrightnessResult(
            is_spike=is_spike,
            brightness_delta=brightness_delta,
            current_brightness=current_brightness,
            previous_brightness=prev,
            priority=priority
        )
    
    def get_spike_timestamps(self) -> List[float]:
        """Get list of all detected brightness spike timestamps."""
        return self.detected_spikes.copy()


# ============================================================================
# Priority Queue Manager
# ============================================================================

class PriorityQueueManager:
    """
    Manages the priority queue for frame processing.
    Combines audio and visual analysis to prioritize frames.
    """
    
    def __init__(self):
        self.audio_critical_timestamps: List[float] = []
        self.frame_priorities: Dict[float, FramePriority] = {}
    
    def set_audio_critical_timestamps(self, timestamps: List[float]):
        """Set timestamps marked as critical by audio analysis."""
        self.audio_critical_timestamps = timestamps
        for ts in timestamps:
            self.frame_priorities[ts] = FramePriority.CRITICAL
    
    def is_audio_critical(self, timestamp: float, tolerance: float = 0.5) -> bool:
        """
        Check if a timestamp is within tolerance of an audio-critical moment.
        
        Args:
            timestamp: Frame timestamp in seconds
            tolerance: Time window around critical timestamps
            
        Returns:
            True if timestamp should bypass filters
        """
        for critical_ts in self.audio_critical_timestamps:
            if abs(timestamp - critical_ts) <= tolerance:
                return True
        return False
    
    def update_priority(self, timestamp: float, priority: FramePriority):
        """Update priority for a timestamp (keeps highest priority)."""
        current = self.frame_priorities.get(timestamp, FramePriority.DISCARD)
        if priority.value < current.value:  # Lower value = higher priority
            self.frame_priorities[timestamp] = priority
    
    def get_priority(self, timestamp: float) -> FramePriority:
        """Get priority for a timestamp."""
        return self.frame_priorities.get(timestamp, FramePriority.DISCARD)
    
    def get_frames_to_process(self) -> List[float]:
        """Get list of timestamps to process, sorted by priority."""
        processable = [
            ts for ts, priority in self.frame_priorities.items()
            if priority.value <= FramePriority.MEDIUM.value
        ]
        # Sort by priority (CRITICAL first, then HIGH, then MEDIUM)
        return sorted(processable, key=lambda ts: self.frame_priorities[ts].value)


# ============================================================================
# Utility Functions
# ============================================================================

def visualize_optical_flow(
    frame: np.ndarray, 
    flow: np.ndarray, 
    step: int = 16
) -> np.ndarray:
    """
    Visualize optical flow as arrows on frame (for debugging).
    
    Args:
        frame: Original frame
        flow: Optical flow field
        step: Arrow spacing
        
    Returns:
        Frame with flow arrows drawn
    """
    h, w = flow.shape[:2]
    vis = frame.copy()
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            cv2.arrowedLine(
                vis,
                (x, y),
                (int(x + fx), int(y + fy)),
                (0, 255, 0),
                1,
                tipLength=0.3
            )
    
    return vis


def create_audio_waveform_plot(
    rms_values: List[float], 
    spike_threshold: float,
    chunk_duration: float
) -> np.ndarray:
    """
    Create a simple waveform visualization (for debugging).
    
    Returns:
        Image array with RMS plot
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    timestamps = [i * chunk_duration for i in range(len(rms_values))]
    
    ax.plot(timestamps, rms_values, 'b-', linewidth=0.5, label='RMS')
    ax.axhline(y=spike_threshold, color='r', linestyle='--', label='Spike Threshold')
    
    # Mark spikes
    spikes = [t for i, t in enumerate(timestamps) if rms_values[i] > spike_threshold]
    spike_vals = [rms_values[i] for i, t in enumerate(timestamps) if rms_values[i] > spike_threshold]
    ax.scatter(spikes, spike_vals, color='red', s=50, zorder=5, label='Spikes')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RMS Amplitude')
    ax.set_title('Audio Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Convert to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return img
