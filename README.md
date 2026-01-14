# FrameSift 🔍

**FrameSift** is an AI-powered semantic video search engine that lets you find any moment in your videos using natural language queries.

> 🎯 **Example**: Search _"a person walking a dog in the park"_ and get exact timestamps where that scene appears.

## 📋 Table of Contents
- [Architecture Overview](#️-architecture-overview)
- [Key Features](#-key-features)
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
┌──────────────────────────────────────┐
│   LOCAL PROCESSING (CPU)             │
│  ┌────────────────────────────────┐  │
│  │ 1. OpenCV Frame Extraction     │  │
│  │    - Motion-based filtering    │  │
│  │    - Skip static frames        │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │ 2. CLIP Semantic Filter        │  │
│  │    - Eliminate duplicates      │  │
│  │    - Keep unique scenes        │  │
│  └────────────────────────────────┘  │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│   CLOUD PROCESSING (GPU)             │
│  ┌────────────────────────────────┐  │
│  │ 3. NVIDIA NIM Analysis         │  │
│  │    - Llama Vision (90B params) │  │
│  │    - Frame descriptions        │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │ 4. Vector Embeddings           │  │
│  │    - NV-Embed (4096-dim)       │  │
│  │    - Store in Pinecone         │  │
│  └────────────────────────────────┘  │
└──────────────┬───────────────────────┘
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

---

## ✨ Key Features

### 🎯 Semantic Search
- Natural language queries (e.g., "people dancing at a wedding")
- Top 5 most relevant results with confidence scores
- Click to jump to exact timestamp in video
- Full description display for each result

### 🎬 Video Management
- Drag & drop upload with live progress
- Real-time processing status
- Automatic video storage and streaming
- Supports MP4, MOV, AVI, WebM

### 🔐 Authentication & Security
- Google OAuth 2.0 integration
- Secure session management (Redis)
- Admin portal with user tracking
- **Auto cleanup** on logout

### 👨‍💼 Admin Dashboard
- Track all user logins
- Monitor user activity
- Protected with admin key
- View user profiles

### 📊 Real-time Feedback
Processing stages:
1. 📤 Uploading video...
2. 🎞️ Extracting frames...
3. 🤖 Analyzing with AI...
4. 💾 Storing vectors...
5. ✅ Complete!

### 🎨 Modern UI
- Responsive design
- Dark mode
- Glassmorphism effects
- Framer Motion animations

---

## 🛠️ Tech Stack

### Frontend
| Technology | Version | Purpose |
|:-----------|:--------|:--------|
| React | 18.3 | UI framework |
| TypeScript | 5.6 | Type safety |
| Vite | 7.3 | Build tool |
| React Router | 6.x | Routing |
| Tailwind CSS | 3.4 | Styling |
| Framer Motion | 11.x | Animations |
| Zustand | 5.x | State management |
| @react-oauth/google | Latest | OAuth |
| Axios | 1.x | HTTP client |

### Backend
| Technology | Version | Purpose |
|:-----------|:--------|:--------|
| FastAPI | 0.115+ | Web framework |
| Python | 3.12 | Language |
| Uvicorn | Latest | ASGI server |
| PyMongo | Latest | MongoDB driver |
| Redis | Latest | Cache client |
| OpenCV | 4.x | Video processing |
| Transformers | Latest | CLIP model |

### Databases
| Service | Purpose |
|:--------|:--------|
| Pinecone | Vector database (4096-dim, cosine) |
| MongoDB Atlas | User data & tracking |
| Redis Cloud | Session cache (1hr TTL) |
| Local Storage | Video files |

### AI Models
| Model | Provider | Purpose | Size |
|:------|:---------|:--------|:-----|
| Llama 3.2 Vision Instruct | NVIDIA NIM | Frame analysis | 90B |
| NV-Embed v1 | NVIDIA NIM | Text embeddings | 4096-dim |
| CLIP ViT-B/32 | OpenAI (local) | Frame filtering | 151M |

---

## 🎯 How It Works

### Video Processing Pipeline
```
Upload → Save to backend/videos/ → Create background job
   ↓
Extract frames (OpenCV, 1 fps)
   ↓
Filter by motion (skip static)
   ↓
CLIP semantic filtering (remove duplicates)
   ↓
NVIDIA Vision analysis (scene descriptions)
   ↓
Generate embeddings (NV-Embed, 4096-dim)
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
| Method | Endpoint | Description |
|:-------|:---------|:------------|
| GET | `/` | Health check |
| POST | `/upload` | Upload video for processing |
| GET | `/job/{id}` | Get processing status |
| POST | `/search` | Search query (returns top 5) |
| GET | `/jobs` | List all jobs |
| GET | `/videos/{id}` | Stream video file |

### Admin (Protected)
| Method | Endpoint | Description |
|:-------|:---------|:------------|
| POST | `/admin/track-login` | Track user login |
| GET | `/admin/users` | Get all users |
| POST | `/clear-database` | Clear all data on logout |

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
│   ├── requirements.txt        # Python dependencies
│   ├── videos/                 # Uploaded videos (auto-created)
│   ├── .env.local              # Environment variables
│   └── venv/                   # Virtual environment
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── search/
│   │   │   │   ├── SearchPanel.tsx     # Search UI
│   │   │   │   ├── VideoPlayer.tsx     # Video player
│   │   │   │   └── UploadModal.tsx     # Upload with progress
│   │   │   ├── ui/                     # Reusable components
│   │   │   ├── home/                   # Landing page
│   │   │   └── layout/                 # Layout components
│   │   ├── pages/
│   │   │   ├── Home.tsx                # Landing page
│   │   │   ├── Search.tsx              # Search page
│   │   │   ├── Admin.tsx               # Admin dashboard
│   │   │   ├── Features.tsx            # Features page
│   │   │   ├── HowItWorks.tsx          # How it works
│   │   │   └── Technology.tsx          # Tech stack
│   │   ├── services/
│   │   │   ├── api.ts                  # Axios instance
│   │   │   └── videoService.ts         # Video API
│   │   ├── store/
│   │   │   └── authStore.ts            # Zustand auth
│   │   └── App.tsx                     # Main app + routes
│   ├── .env.local              # Environment variables
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
│
├── docker-compose.yml          # Docker orchestration
├── .gitignore
└── README.md
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
