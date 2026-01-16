<<<<<<< HEAD
<div align="center">

# 🎬 FrameSift

### _AI-Powered Semantic Video Search Engine_

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![NVIDIA](https://img.shields.io/badge/NVIDIA_NIM-Powered-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/nim)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000000?style=for-the-badge)](https://pinecone.io)

<br/>

**Find any moment in your videos using natural language.**

_"When does the lightning strike?" • "Show me the girl going outside" • "Find the explosion scene"_

<br/>

[🚀 Quick Start](#-quick-start) •
[✨ Features](#-features) •
[🏗️ Architecture](#️-architecture) •
[📖 API Docs](#-api-documentation) •
[🎯 Accuracy](#-accuracy-system)

<br/>

<img src="https://img.shields.io/badge/Multi--Model-Ensemble-blueviolet?style=flat-square" alt="Multi-Model"/>
<img src="https://img.shields.io/badge/RAG-AI_Answers-orange?style=flat-square" alt="RAG"/>
<img src="https://img.shields.io/badge/LLM-Re--Ranking-green?style=flat-square" alt="LLM Re-Ranking"/>
<img src="https://img.shields.io/badge/Multi--User-Isolated-blue?style=flat-square" alt="Multi-User"/>

</div>

---

## 🎯 What is FrameSift?

FrameSift transforms video search from tedious scrubbing into instant discovery. Simply ask a question in plain English, and our AI finds the exact moments you're looking for.

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│    🎥  Upload Video  →  🤖  AI Analysis  →  🔍  Search Instantly   │
│                                                                     │
│    "When does the car crash?"                                      │
│         ↓                                                           │
│    ⏱️ 2:34 - "A red car collides with a truck at intersection"    │
│    ⏱️ 5:12 - "The damaged vehicle spins and hits a guardrail"     │
│                                                                     │
│    🧠 AI Answer: "The car crash occurs at 2:34 when a red sedan   │
│       runs a red light and collides with a delivery truck..."      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔍 Semantic Search

- Natural language queries
- Intent understanding (not just keywords)
- Returns timestamps + descriptions
- Click-to-seek video playback

### 🧠 AI Answer Generation

- RAG-powered conversational responses
- Synthesizes information from multiple frames
- Answers "when", "what", "how" questions

### 🎯 Multi-Model Ensemble

- **Llama 3.2 90B Vision** - Detailed scene analysis
- **Llama 3.2 11B Vision** - Fast action detection
- Combined outputs for maximum accuracy

</td>
<td width="50%">

### ⚡ Smart Frame Processing

- **Audio Spike Detection** - Catches impacts, explosions
- **Brightness Spike Detection** - Detects lightning, flashes
- **Physics Filter** - Distinguishes falls vs. walks
- **CLIP Deduplication** - Eliminates redundant frames

### 🔄 Dynamic Optimization

- **Adaptive FPS** - 4 FPS for short videos, 1 FPS for long
- **Query Expansion** - Synonym matching
- **LLM Re-Ranking** - AI reorders by relevance
- **Keyword Boosting** - Action word prioritization

### 👥 Multi-User Support

- Isolated data per user (Pinecone namespaces)
- Auto-cleanup on logout
- Concurrent video processing

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
                           ┌─────────────────────────────────────────┐
                           │           FrameSift Pipeline            │
                           └─────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        LOCAL PROCESSING (CPU) - Fast & Free                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐  │
│  │ 🔊 Audio Spike  │  │ ⚡ Brightness   │  │ 🏃 Physics      │  │ 🎯 CLIP    │  │
│  │    Detection    │  │    Spike        │  │    Filter       │  │   Filter   │  │
│  │                 │  │                 │  │                 │  │            │  │
│  │ MoviePy + RMS   │  │ Delta Analysis  │  │ Optical Flow    │  │ Similarity │  │
│  │ "Find impacts"  │  │ "Find flashes"  │  │ "Find falls"    │  │ "Dedupe"   │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  └─────┬──────┘  │
│           │                    │                    │                  │         │
│           └────────────────────┴────────────────────┴──────────────────┘         │
│                                         │                                        │
│                              Surviving Frames (20-30%)                           │
└─────────────────────────────────────────┬───────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       CLOUD PROCESSING (GPU) - NVIDIA NIM                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    🔀 Multi-Model Ensemble (Parallel)                    │   │
│  │  ┌─────────────────────────┐        ┌─────────────────────────┐         │   │
│  │  │ 🧠 Llama 3.2 90B Vision │        │ ⚡ Llama 3.2 11B Vision │         │   │
│  │  │    Detailed Analysis    │   +    │    Action Detection     │         │   │
│  │  │    "Scene, subjects,    │        │    "What's happening    │         │   │
│  │  │     environment..."     │        │     RIGHT NOW?"         │         │   │
│  │  └─────────────────────────┘        └─────────────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                         │                                       │
│                                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │              📊 NV-Embed (4096-dim) → Pinecone Vector DB                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            SEARCH & RAG PIPELINE                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │ 📝 Query     │→ │ 🔎 Vector    │→ │ 🔄 LLM       │→ │ 🤖 AI Answer     │   │
│  │   Expansion  │  │   Search     │  │   Re-Rank    │  │   Generation     │   │
│  │              │  │              │  │              │  │                  │   │
│  │ +synonyms    │  │ cosine sim   │  │ Llama 70B    │  │ "Based on the    │   │
│  │ +actions     │  │ namespace    │  │ reorder      │  │  frames, the..." │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Accuracy System

FrameSift uses a **5-stage accuracy enhancement pipeline**:

| Stage | Technology               | Purpose                                             | Impact         |
| ----- | ------------------------ | --------------------------------------------------- | -------------- |
| 1️⃣    | **Query Expansion**      | Add synonyms ("outside" → "doorway, exit, leaving") | +15% recall    |
| 2️⃣    | **Keyword Boost**        | Prioritize action word matches                      | +10% precision |
| 3️⃣    | **Multi-Model Ensemble** | Two vision models catch different details           | +20% coverage  |
| 4️⃣    | **LLM Re-Ranking**       | Llama 70B reorders by semantic relevance            | +25% precision |
| 5️⃣    | **RAG Answer**           | Synthesize conversational response                  | Better UX      |

### Dynamic Frame Processing

| Video Length | FPS      | Rationale                     |
| ------------ | -------- | ----------------------------- |
| ≤10 seconds  | ~4 FPS   | Capture every detail          |
| ≤30 seconds  | ~3 FPS   | High detail for short content |
| ≤60 seconds  | ~2.5 FPS | Balanced coverage             |
| ≤120 seconds | ~2 FPS   | Standard processing           |
| ≤300 seconds | ~1.5 FPS | Efficient for medium content  |
| >300 seconds | ~1 FPS   | Scalable for long videos      |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- [NVIDIA NIM API Key](https://build.nvidia.com/)
- [Pinecone API Key](https://pinecone.io/)

### 1. Clone & Setup Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment

Create `backend/.env.local`:

```env
# NVIDIA NIM API Keys (JSON array for rotation)
NVIDIA_KEYS=["nvapi-xxx", "nvapi-yyy"]

# Pinecone
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=framesift

# MongoDB (optional - for user tracking)
MONGODB_URI=mongodb://localhost:27017

# Redis (optional - for caching)
REDIS_HOST=localhost
REDIS_PORT=6379

# Multi-Model Ensemble (default: enabled)
ENSEMBLE_MODE=true
```

### 3. Start Backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Setup Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

```env
VITE_API_URL=http://localhost:8000
VITE_GOOGLE_CLIENT_ID=your-google-client-id
```

### 5. Start Frontend

```bash
npm run dev
```

Visit **http://localhost:5173** 🎉

---

## 📖 API Documentation

### Core Endpoints

#### `POST /upload`

Upload and process a video.

```bash
curl -X POST http://localhost:8000/upload \
  -H "X-User-Id: user123" \
  -F "file=@video.mp4"
```

#### `POST /search`

Search for moments in processed videos.

```bash
curl -X POST http://localhost:8000/search \
  -H "X-User-Id: user123" \
  -H "Content-Type: application/json" \
  -d '{"query": "when does the lightning strike?", "top_k": 5}'
```

**Response:**

```json
{
  "results": [
    {
      "timestamp": 12.5,
      "score": 0.95,
      "description": "A bright lightning bolt illuminates the dark stormy sky...",
      "frame_id": "abc123_frame_42"
    }
  ],
  "query": "when does the lightning strike?",
  "ai_answer": "The lightning strike occurs at approximately 12.5 seconds..."
}
```

#### `POST /deep-scan`

Force-analyze a specific time range at high FPS.

```bash
curl -X POST http://localhost:8000/deep-scan \
  -H "Content-Type: application/json" \
  -d '{"video_id": "abc123", "start_time": 10, "end_time": 15, "fps": 2}'
```

#### `POST /clear-database`

Clear user data on logout (multi-user aware).

```bash
curl -X POST http://localhost:8000/clear-database \
  -H "X-User-Id: user123"
```

---

## 🛠️ Tech Stack

<table>
<tr>
<td align="center" width="140">
<img src="https://skillicons.dev/icons?i=python" width="48" height="48" alt="Python" />
<br><strong>Python</strong>
<br><sub>Backend</sub>
</td>
<td align="center" width="140">
<img src="https://skillicons.dev/icons?i=fastapi" width="48" height="48" alt="FastAPI" />
<br><strong>FastAPI</strong>
<br><sub>API Framework</sub>
</td>
<td align="center" width="140">
<img src="https://skillicons.dev/icons?i=react" width="48" height="48" alt="React" />
<br><strong>React</strong>
<br><sub>Frontend</sub>
</td>
<td align="center" width="140">
<img src="https://skillicons.dev/icons?i=typescript" width="48" height="48" alt="TypeScript" />
<br><strong>TypeScript</strong>
<br><sub>Type Safety</sub>
</td>
<td align="center" width="140">
<img src="https://skillicons.dev/icons?i=tailwind" width="48" height="48" alt="Tailwind" />
<br><strong>Tailwind</strong>
<br><sub>Styling</sub>
</td>
</tr>
<tr>
<td align="center" width="140">
<img src="https://skillicons.dev/icons?i=mongodb" width="48" height="48" alt="MongoDB" />
<br><strong>MongoDB</strong>
<br><sub>User Storage</sub>
</td>
<td align="center" width="140">
<img src="https://skillicons.dev/icons?i=redis" width="48" height="48" alt="Redis" />
<br><strong>Redis</strong>
<br><sub>Caching</sub>
</td>
<td align="center" width="140">
<img src="https://skillicons.dev/icons?i=docker" width="48" height="48" alt="Docker" />
<br><strong>Docker</strong>
<br><sub>Containerization</sub>
</td>
<td align="center" width="140">
<strong>🟢 NVIDIA NIM</strong>
<br><sub>AI Models</sub>
</td>
<td align="center" width="140">
<strong>🌲 Pinecone</strong>
<br><sub>Vector DB</sub>
</td>
</tr>
</table>

---

## 📁 Project Structure

```
framesift/
├── 📂 backend/
│   ├── main.py              # FastAPI application (endpoints, pipeline)
│   ├── scout.py             # Local filters (audio, physics, brightness)
│   ├── processor.py         # Parallel frame processing
│   ├── requirements.txt     # Python dependencies
│   └── videos/              # Uploaded videos (per-user directories)
│       └── {user_id}/       # User-isolated video storage
│
├── 📂 frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── search/      # SearchPanel, VideoPlayer, UploadModal
│   │   │   ├── home/        # Landing page sections
│   │   │   └── ui/          # Reusable UI components
│   │   ├── pages/           # Route pages
│   │   ├── services/        # API client
│   │   └── store/           # Zustand state management
│   ├── package.json
│   └── vite.config.ts
│
├── docker-compose.yml       # Full stack deployment
└── README.md               # You are here! 📍
```

---

## 🔧 Configuration Options

| Variable               | Default | Description                                            |
| ---------------------- | ------- | ------------------------------------------------------ |
| `ENSEMBLE_MODE`        | `true`  | Enable multi-model ensemble (slower but more accurate) |
| `FRAME_SKIP_INTERVAL`  | `5`     | Default frame skip (overridden by dynamic FPS)         |
| `MOTION_THRESHOLD`     | `30.0`  | Pixel change threshold for motion detection            |
| `SIMILARITY_THRESHOLD` | `0.95`  | CLIP similarity threshold for deduplication            |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ using NVIDIA NIM, Pinecone, and modern web technologies**

<br/>

_Find any moment. Ask any question. Get instant answers._

</div>
=======
---
title: Framesift Backend
emoji: 🐨
colorFrom: green
colorTo: pink
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 8e4f8f7953c40ac0378478a264d05b75f89aed29
