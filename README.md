# FrameSift 🔍

**FrameSift** is an AI-powered semantic video search engine that lets you find any moment in your videos using natural language queries.

> 🎯 **Example**: Search _"a person walking a dog in the park"_ and get exact timestamps where that scene appears.

## 📋 Table of Contents

- [Architecture Overview](#️-architecture-overview)
- [Key Features](#-key-features)
- [Phase 2: Advanced Filtering](#-phase-2-advanced-filtering)
- [Tech Stack](#️-tech-stack)
- [How It Works](#-how-it-works)
- [Getting Started](#-getting-started)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Configuration](#️-configuration)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🏗️ Architecture Overview

FrameSift uses a **Hybrid AI Architecture** that balances performance, cost, and accuracy:

1. **Local Edge Processing (CPU)**: Fast, cost-effective filtering
2. **Cloud AI (GPU)**: Powerful deep learning for semantic understanding

```
┌─────────────┐
│ Upload Video│
└──────┬──────┘
       ↓
┌──────────────────────────────────────────────┐
│   LOCAL PROCESSING (CPU) - Phase 2 Enhanced │
│  ┌──────────────────────────────────────┐    │
│  │ 1. Audio Spike Detection             │    │
│  │    - Extract audio with MoviePy      │    │
│  │    - RMS analysis (Librosa)          │    │
│  │    - Mark CRITICAL timestamps        │    │
│  └──────────────────────────────────────┘    │
│  ┌──────────────────────────────────────┐    │
│  │ 2. Physics Filter (Optical Flow)     │    │
│  │    - Farneback algorithm             │    │
│  │    - Vertical vs Horizontal motion   │    │
│  │    - HIGH: Falling | LOW: Walking    │    │
│  └──────────────────────────────────────┘    │
│  ┌──────────────────────────────────────┐    │
│  │ 3. CLIP Semantic Filter              │    │
│  │    - Eliminate duplicates            │    │
│  │    - Keep unique scenes              │    │
│  └──────────────────────────────────────┘    │
└──────────────┬───────────────────────────────┘
               ↓
┌──────────────────────────────────────────────┐
│   CLOUD PROCESSING (GPU) - Parallel         │
│  ┌──────────────────────────────────────┐    │
│  │ 4. NVIDIA NIM Analysis (Parallel)    │    │
│  │    - Llama Vision (90B params)       │    │
│  │    - ThreadPoolExecutor              │    │
│  │    - Round-Robin Key Rotation        │    │
│  │    - Auto retry on rate limits       │    │
│  └──────────────────────────────────────┘    │
│  ┌──────────────────────────────────────┐    │
│  │ 5. Vector Embeddings (Parallel)      │    │
│  │    - NV-Embed (4096-dim)             │    │
│  │    - Store in Pinecone               │    │
│  └──────────────────────────────────────┘    │
└──────────────┬───────────────────────────────┘
               ↓
       ┌───────────────┐
       │ Search Ready! │
       └───────────────┘
```

**Why Hybrid?**

- ✅ **70-80% cost reduction** - Process only unique frames
- ✅ **Faster processing** - Local filtering is instant
- ✅ **Better accuracy** - High-quality AI on important frames
- ✅ **Scalable** - Serverless infrastructure
- ✅ **Phase 2: Parallel processing** - True concurrency with key rotation

---

## ✨ Key Features

### 🎯 Semantic Search

- Natural language queries (e.g., "people dancing at a wedding")
- **Query display** with result count after each search
- Top 5 most relevant results with confidence scores
- Click to jump to exact timestamp in video
- Full description display for each result
- **Deep Scan UI** - Force-process specific time ranges when no results found

### 🎬 Video Management

- Drag & drop upload with live progress
- Real-time processing status
- Automatic video storage and streaming
- Supports MP4, MOV, AVI, WebM

### 🔐 Authentication & Security

- Google OAuth 2.0 integration
- Secure session management (Redis)
- Admin portal with user tracking
- **Auto cleanup** on logout (database + temp audio files)

### 👨‍💼 Admin Dashboard

- Track all user logins
- Monitor user activity
- Protected with admin key
- View user profiles

### 📊 Real-time Feedback

Processing stages:

1. 📤 Uploading video...
2. 🎞️ Extracting frames...
3. 🔊 Analyzing audio spikes...
4. 📐 Applying physics filters...
5. 🤖 Analyzing with AI (parallel)...
6. 💾 Storing vectors...
7. ✅ Complete!

### 🎨 Modern UI

- Responsive design
- Dark mode
- Glassmorphism effects
- Framer Motion animations

---

## 🚀 Phase 2: Advanced Filtering

### 🔊 Audio Trigger (The "Clack" Detector)

Detects sudden sounds (chalk dropping, clapping, objects falling) even without visual motion.

**How it works:**

1. Extract audio from video using **MoviePy**
2. Calculate RMS amplitude using **Librosa**
3. Detect spikes (RMS > threshold × spike_multiplier)
4. Mark timestamps as **CRITICAL** → bypass all visual filters

**Configuration:**

```python
AudioTrigger(
    rms_threshold=0.05,       # Base RMS threshold
    spike_multiplier=2.5,     # Spike = RMS > avg × 2.5
    chunk_duration=1.0,       # Analyze 1-second chunks
    min_spike_gap=0.5        # Minimum gap between spikes
)
```

### 📐 Physics Filter (Vertical Optical Flow)

Distinguishes **falling** (vertical motion) from **walking** (horizontal motion).

**How it works:**

1. Compute optical flow using **Farneback algorithm**
2. Analyze vertical (fy) vs horizontal (fx) flow
3. Vertical dominant → **HIGH** priority (falling/dropping)
4. Horizontal dominant → **LOW** priority (walking)
5. Static → **DISCARD**

**Priority Queue:**

```
CRITICAL (Audio Spikes)  → Bypass all filters
HIGH (Falling/Vertical)  → Send to NVIDIA
MEDIUM (General Motion)  → Send to NVIDIA
LOW (Walking)            → Stricter threshold
DISCARD (Static)         → Skip
```

### 🔄 Robust Parallel Processing

True concurrent processing with thread-safe round-robin key rotation.

**Features:**

- **ThreadPoolExecutor** with `max_workers = len(NVIDIA_KEYS)`
- **Round-robin key rotation** (thread-safe with Lock)
- **Automatic retry** on 429 rate limits
- **Exponential backoff** on other errors
- **Rate limit tracking** per key

**Example with 2 keys:**

```python
Frame 1 → Worker 1 → Key A → NVIDIA
Frame 2 → Worker 2 → Key B → NVIDIA (parallel!)
Frame 3 → Worker 1 → Key A → Rate limit → Key B
```

### 🔬 Deep Scan Mode

Force-process a specific time range without filters.

**Backend Endpoint:** `POST /deep-scan`

```json
{
  "video_id": "abc123",
  "start_time": 10.5,
  "end_time": 15.0,
  "fps": 1
}
```

**Frontend UI:**

- **Deep Scan Modal** - Appears when search returns no results
- Time range inputs (start/end seconds)
- Duration calculator with frame estimate
- Real-time loading state with spinner
- Success/error feedback
- Auto-refresh search results after completion

**Use case:** When AI missed important frames in a specific segment.

---

## 🛠️ Tech Stack

### Frontend

| Technology          | Version | Purpose          |
| :------------------ | :------ | :--------------- |
| React               | 18.3    | UI framework     |
| TypeScript          | 5.6     | Type safety      |
| Vite                | 7.3     | Build tool       |
| React Router        | 6.x     | Routing          |
| Tailwind CSS        | 3.4     | Styling          |
| Framer Motion       | 11.x    | Animations       |
| Zustand             | 5.x     | State management |
| @react-oauth/google | Latest  | OAuth            |
| Axios               | 1.x     | HTTP client      |

### Backend

| Technology   | Version   | Purpose                       |
| :----------- | :-------- | :---------------------------- |
| FastAPI      | 0.115+    | Web framework                 |
| Python       | 3.12      | Language                      |
| Uvicorn      | Latest    | ASGI server                   |
| PyMongo      | Latest    | MongoDB driver                |
| Redis        | Latest    | Cache client                  |
| OpenCV       | 4.x       | Video processing              |
| Transformers | Latest    | CLIP model                    |
| **MoviePy**  | **2.1+**  | **Phase 2: Audio extraction** |
| **Librosa**  | **0.10+** | **Phase 2: Audio analysis**   |
| **SciPy**    | **1.11+** | **Phase 2: Audio fallback**   |

### Databases

| Service       | Purpose                            |
| :------------ | :--------------------------------- |
| Pinecone      | Vector database (4096-dim, cosine) |
| MongoDB Atlas | User data & tracking               |
| Redis Cloud   | Session cache (1hr TTL)            |
| Local Storage | Video files                        |

### AI Models

| Model                     | Provider       | Purpose         | Size     |
| :------------------------ | :------------- | :-------------- | :------- |
| Llama 3.2 Vision Instruct | NVIDIA NIM     | Frame analysis  | 90B      |
| NV-Embed v1               | NVIDIA NIM     | Text embeddings | 4096-dim |
| CLIP ViT-B/32             | OpenAI (local) | Frame filtering | 151M     |

---

## 🎯 How It Works

### Video Processing Pipeline (Phase 2)

```
Upload → Save to backend/videos/ → Create background job
   ↓
📢 Audio Analysis (FIRST)
   ├─ Extract audio (MoviePy)
   ├─ RMS analysis (Librosa)
   └─ Mark CRITICAL timestamps
   ↓
Extract frames (OpenCV, 1 fps)
   ↓
📐 Physics Filter (per frame)
   ├─ Optical flow analysis
   ├─ Vertical → HIGH priority
   ├─ Horizontal → LOW priority
   └─ Static → DISCARD
   ↓
CLIP semantic filtering (remove duplicates)
   ↓
🚀 Parallel NVIDIA Processing
   ├─ ThreadPoolExecutor (max_workers = num_keys)
   ├─ Round-robin key rotation
   ├─ Vision analysis (Llama 3.2 90B)
   └─ Generate embeddings (NV-Embed)
   ↓
Store in Pinecone (with metadata + timestamps)
   ↓
Ready for search!
```

### Search Flow

```
Query: "people playing basketball"
   ↓
Convert to vector (NV-Embed)
   ↓
Pinecone similarity search (cosine)
   ↓
Get top 5 matches with scores
   ↓
Display with timestamps
   ↓
Click → Jump to moment
```

### Auto Cleanup on Logout

```
User logs out
   ↓
Clear Pinecone vectors
   ↓
Delete video files
   ↓
Delete temp audio files (.wav)
   ↓
Clear Redis session
   ↓
Done!
```

---

## ⚡ Getting Started

### Prerequisites

**Required Software:**

- Node.js v18+ ([Download](https://nodejs.org/))
- Python 3.12 ([Download](https://www.python.org/downloads/))
- Git ([Download](https://git-scm.com/))

**Required API Keys:**
| Service | Purpose | Get It |
|:--------|:--------|:-------|
| NVIDIA NIM | AI models (need 2 keys) | [Get Keys](https://build.nvidia.com/) |
| Pinecone | Vector database | [Sign Up](https://www.pinecone.io/) |
| Google Cloud | OAuth | [Console](https://console.cloud.google.com/) |
| MongoDB Atlas | User storage | [Free Cluster](https://www.mongodb.com/cloud/atlas) |
| Redis Cloud | Caching | [Try Free](https://redis.com/try-free/) |

---

### 🚀 Quick Start

#### 1. Clone Repository

```bash
git clone https://github.com/your-username/framesift.git
cd framesift
```

#### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create `backend/.env.local`:

```env
# NVIDIA NIM (get from https://build.nvidia.com/)
NVIDIA_KEYS=["nvapi-key1", "nvapi-key2"]

# Pinecone
PINECONE_API_KEY=pcsk_xxxxxx
PINECONE_INDEX_NAME=framesift

# MongoDB Atlas
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net
MONGODB_DB_NAME=framesift
MONGODB_COLLECTION_NAME=users

# Redis Cloud
REDIS_URL=redis://default:password@host:port

# Admin (choose your secret)
ADMIN_KEY=your-secret-key
```

Start backend:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### 3. Frontend Setup

```bash
# New terminal
cd frontend
npm install
```

Create `frontend/.env.local`:

```env
VITE_API_URL=http://localhost:8000
VITE_GOOGLE_CLIENT_ID=your-id.apps.googleusercontent.com
VITE_ADMIN_KEY=your-secret-key
VITE_ADMIN_EMAILS=admin@example.com
```

Start frontend:

```bash
npm run dev
```

#### 4. Setup Google OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create project
3. Enable **Google+ API**
4. Create **OAuth 2.0 Client ID**
5. Add origins: `http://localhost:5173`
6. Copy Client ID to `.env.local`

#### 5. Setup Pinecone

1. [Pinecone Console](https://app.pinecone.io/)
2. Create index:
   - Name: `framesift`
   - Dimensions: `4096`
   - Metric: `cosine`
   - Plan: Serverless
3. Copy API key

#### 6. Test It!

1. Visit http://localhost:5173
2. Sign in with Google
3. Upload a test video
4. Wait for processing
5. Search: "what's in the video?"
6. Click result to jump to timestamp

---

## 📡 API Documentation

### Video Processing

| Method | Endpoint       | Description                  |
| :----- | :------------- | :--------------------------- |
| GET    | `/`            | Health check                 |
| POST   | `/upload`      | Upload video for processing  |
| GET    | `/job/{id}`    | Get processing status        |
| POST   | `/search`      | Search query (returns top 5) |
| GET    | `/jobs`        | List all jobs                |
| GET    | `/videos/{id}` | Stream video file            |

### Admin (Protected)

| Method | Endpoint             | Description              |
| :----- | :------------------- | :----------------------- |
| POST   | `/admin/track-login` | Track user login         |
| GET    | `/admin/users`       | Get all users            |
| POST   | `/clear-database`    | Clear all data on logout |

### Example: Upload Video

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@video.mp4"
```

Response:

```json
{
  "job_id": "abc-123",
  "video_id": "abc-123",
  "status": "processing"
}
```

### Example: Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "people walking",
    "video_id": "abc-123",
    "top_k": 5
  }'
```

Response:

```json
{
  "results": [
    {
      "timestamp": 45.5,
      "description": "Two people walking in a park",
      "score": 0.92
    }
  ],
  "query": "people walking"
}
```

---

## 📂 Project Structure

```
framesift/
├── backend/
│   ├── main.py                 # FastAPI app with all endpoints
│   ├── scout.py                # 🆕 Phase 2: Audio & Physics filters
│   ├── processor.py            # 🆕 Phase 2: Parallel processing
│   ├── requirements.txt        # Python dependencies (Phase 2 updated)
│   ├── Dockerfile              # 🆕 Phase 2: Includes ffmpeg
│   ├── videos/                 # Uploaded videos (auto-created)
│   ├── .env.local              # Environment variables
│   └── venv/                   # Virtual environment
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── search/
│   │   │   │   ├── DeepScanModal.tsx       # 🆕 Deep Scan UI
│   │   │   │   └── SearchPanel.tsx         # 🆕 Query display + Deep Scan CTA
│   │   │   ├── LandingPage.tsx         # Landing page
│   │   │   ├── SearchComponent.tsx     # Search UI
│   │   │   ├── SearchPageWrapper.tsx   # Search page wrapper
│   │   │   ├── UploadModal.tsx         # Upload with progress
│   │   │   ├── VideoPlayer.tsx         # Video player
│   │   │   └── ui/                     # Reusable UI components
│   │   │       ├── background-paths.tsx
│   │   │       ├── button.tsx
│   │   │       ├── card.tsx
│   │   │       ├── expandable-tabs.tsx
│   │   │       ├── floating-navbar.tsx
│   │   │       ├── input.tsx
│   │   │       ├── modern-animated-hero-section.tsx
│   │   │       └── theme-toggle.tsx
│   │   ├── services/
│   │   │   └── videoService.ts         # 🆕 Added deepScan() function
│   │   ├── app/
│   │   │   ├── page.tsx                # Landing page
│   │   │   ├── layout.tsx              # Root layout
│   │   │   ├── globals.css             # Global styles
│   │   │   └── search/
│   │   │       └── page.tsx            # Search page
│   │   ├── lib/
│   │   │   └── utils.ts                # Utility functions
│   │   └── public/                     # Static assets
│   ├── .env.local              # Environment variables
│   ├── package.json
│   ├── next.config.ts          # Next.js config
│   ├── tsconfig.json           # TypeScript config
│   ├── eslint.config.mjs       # ESLint config
│   └── postcss.config.mjs      # PostCSS config
│
├── docker-compose.yml          # Docker orchestration
├── .gitignore
└── README.md                   # 🆕 Updated with Phase 2
```

### Backend File Details

#### `main.py` (1200+ lines)

- FastAPI application with all endpoints
- Warning suppression for transformers library at startup
- `SemanticScout` class - Phase 2 enhanced with audio/physics filters
- `NvidiaProcessor` class - Cloud AI processing
- `VectorStore` class - Pinecone integration
- Background video processing with priority queue
- Auto cleanup on logout (vectors, videos, temp audio files)
- Endpoints: `/upload`, `/search`, `/status`, `/video/{id}`, `/deep-scan` 🆕
- `VectorStore` class - Pinecone integration
- Background video processing with priority queue
- Endpoints: `/upload`, `/search`, `/status`, `/video/{id}`, `/deep-scan` 🆕

#### `scout.py` (570+ lines) - **Phase 2 NEW**

- `AudioTrigger` - Audio spike detection
  - Extract audio with MoviePy
  - RMS analysis with Librosa
  - Critical timestamp detection
- `PhysicsFilter` - Optical flow analysis
  - Farneback algorithm
  - Vertical vs horizontal motion classification
- `PriorityQueueManager` - Frame priority management
- `FramePriority` enum - CRITICAL/HIGH/MEDIUM/LOW/DISCARD

#### `processor.py` (590+ lines) - **Phase 2 NEW**

- `KeyManager` - Thread-safe round-robin key rotation
- `ParallelFrameProcessor` - Concurrent frame processing
  - ThreadPoolExecutor with max_workers = num_keys
  - Automatic retry on rate limits
  - Exponential backoff
- `DeepScanProcessor` - Force-process time ranges
- `ProcessingResult` - Structured results with retry tracking

---

## 📡 API Endpoints

### POST `/upload`

Upload a video for processing.

**Request:**

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@video.mp4"
```

**Response:**

```json
{
  "job_id": "abc-123-def-456",
  "video_id": "abc-123-def-456",
  "status": "processing"
}
```

### POST `/search`

Search for frames using natural language.

**Request:**

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "people walking a dog",
    "video_id": "abc-123-def-456",
    "top_k": 5
  }'
```

**Response:**

```json
{
  "results": [
    {
      "timestamp": 45.5,
      "description": "Two people walking a dog in a park",
      "score": 0.92
    }
  ],
  "query": "people walking a dog"
}
```

### GET `/status/{job_id}`

Check video processing status.

**Response:**

```json
{
  "status": "completed",
  "progress": 1.0,
  "frames_processed": 150,
  "frames_total": 150
}
```

### POST `/deep-scan` 🆕 **Phase 2**

Force-process a specific time range without filters.

**Request:**

```bash
curl -X POST http://localhost:8000/deep-scan \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "abc-123-def-456",
    "start_time": 10.5,
    "end_time": 15.0,
    "fps": 1
  }'
```

**Response:**

```json
{
  "status": "success",
  "message": "Deep scan processed 5 frames, 5 indexed",
  "frames_processed": 5,
  "frames_indexed": 5
}
```

**Use case:** When AI missed important frames in a specific segment, deep scan processes every frame in that range at the specified FPS (bypassing all filters).

### GET `/video/{video_id}`

Stream video file.

**Response:** Video file stream (MP4/MOV/AVI/WebM)

### POST `/logout-cleanup`

Clear all user data on logout.

**Response:**

```json
{
  "status": "success",
  "message": "Database and 3 video(s) cleared successfully"
}
```

---

## ⚙️ Configuration

### Backend Environment Variables

```env
# NVIDIA NIM (2 API keys required)
NVIDIA_KEYS=["key1", "key2"]

# Pinecone
PINECONE_API_KEY=pcsk_xxxxx
PINECONE_INDEX_NAME=framesift

# MongoDB
MONGODB_URI=mongodb+srv://...
MONGODB_DB_NAME=framesift
MONGODB_COLLECTION_NAME=users

# Redis
REDIS_URL=redis://...

# Admin
ADMIN_KEY=secret
```

### Frontend Environment Variables

```env
# Backend
VITE_API_URL=http://localhost:8000

# Google OAuth
VITE_GOOGLE_CLIENT_ID=xxxxx.apps.googleusercontent.com

# Admin
VITE_ADMIN_KEY=secret
VITE_ADMIN_EMAILS=admin@example.com
```

### Pinecone Index Configuration

- **Dimensions**: 4096 (NV-Embed v1)
- **Metric**: cosine
- **Cloud**: AWS (us-east-1 recommended)
- **Plan**: Serverless (free tier available)

### MongoDB Schema

```javascript
{
  email: String,
  name: String,
  picture: String,
  loginCount: Number,
  lastLogin: Date,
  firstLogin: Date
}
```

---

## 🔧 Troubleshooting

### Frontend won't start

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Backend errors

```bash
# Check Python version
python --version  # Should be 3.12+

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Upload fails

- Check video format (MP4, MOV, AVI, WebM)
- Check file size (recommended < 500MB)
- Check backend logs for errors
- Verify NVIDIA API keys are valid

### Search returns no results

- Check Pinecone index exists
- Verify index dimensions (4096)
- Check video was fully processed
- Look for errors in job status
- Use **Deep Scan UI** to force-process specific time ranges

### "Using a slow image processor" warning

- This is an informational warning from transformers library
- Does not affect performance or functionality
- Suppressed via environment variables in `main.py`:
  ```python
  os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
  logging.getLogger("transformers").setLevel(logging.ERROR)
  ```

### Google OAuth fails

- Verify Client ID in `.env.local`
- Check authorized origins in Google Console
- Clear browser cache and cookies
- Add test users in OAuth consent screen

### Pinecone connection issues

- Verify API key is correct
- Check index name matches `.env`
- Ensure index is "Active" status
- Try deleting and recreating index

---

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

Access:

- Frontend: http://localhost:3000
- Backend: http://localhost:8000

---

## 🚀 Production Deployment

### Vercel (Frontend)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd frontend
vercel --prod
```

### Railway/Render (Backend)

1. Push code to GitHub
2. Connect repository to Railway/Render
3. Add environment variables
4. Deploy

### Environment Variables

Make sure to set all `.env.local` variables in your deployment platform.

---

## 📝 License

MIT License - see [LICENSE](LICENSE)

Built with ❤️ by [Amitesh Vishwakarma](https://github.com/amitesh-7)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

---

## 📧 Support

- Issues: [GitHub Issues](https://github.com/your-username/framesift/issues)
- Email: amiteshvishwakarma2006@gmail.com
- Docs: [Full Documentation](https://framesift-docs.vercel.app)

---

## 🎯 Roadmap

- [ ] Multi-user support with isolated databases
- [ ] Video sharing and collaboration
- [ ] Real-time collaborative search
- [ ] Mobile app (React Native)
- [ ] Advanced filters (date, duration, quality)
- [ ] Batch video processing
- [ ] Custom AI model training
- [ ] WebSocket for live updates

---

**⭐ Star this repo if you find it useful!**
