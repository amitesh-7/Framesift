# FrameSift 🔍

**FrameSift** is a production-ready **Semantic Video Search Engine** that allows you to find any moment in your videos using natural language queries (e.g., _"a person walking a dog in the park"_).

It uses a **Hybrid AI Architecture** to balance performance and cost, combining local edge processing with powerful cloud intelligence.

![FrameSift Architecture](https://mermaid.ink/img/pako:eNp1UktqwzAQvYpRax9gC8E2gTQEQ8hC6aab4shTjC3FkoxcCznH6L16kxzHKd1kZjTvzZt5M4KVVgjWUOyrwN6JDWvhTzJjLC_JgT2yI7thL7CH8oW9wR_syP6wX9T5J5ZgT7TCOeOswB_sTj0eR5E9sQd2-v6d3e8jWGOl5KLAXn4wVpCV_xVjVWAJ5YmV2Nf6_g3W0oM1WKnwXjHqazB4NWhEClYJeijR4MUIg8ETeKk48eC0osSDF4oTD15rSjx4U3Hig7eKEw_eK0588EFx4sGHihMf_Kg48eCn4sSHf1Sc-PCPiit_4r81J16815x48UFz4sOHmhMf_Kg58eCn5sSHf9Sc-PCv5iqc-FdzE078q7kLJ_7V3IcT_2oeweBfzSMY_Kt5BIN_NY9g8K_mEQz-1TyCwb-aRzD41zJ4R6h0i2f-B4t_9Q8?type=png)

---

## 🚀 Features

- **Semantic Search**: Search video content by meaning, not just keywords.
- **Hybrid AI Pipeline**:
  - **Local Scout (CPU)**: Uses **OpenCV** (Motion Detection) and **CLIP** (Semantic Redundancy Check) to filter frames efficiently.
  - **Cloud Intelligence (GPU)**: Uses **NVIDIA NIM** (Llama 3.2 Vision) for detailed frame analysis and **NV-Embed** for high-dimensional text embeddings.
- **Vector Search**: Serverless **Pinecone** database (4096 dimensions) for lightning-fast retrieval.
- **Modern UI**: Built with **Next.js 16**, **Tailwind CSS v4**, and **Shadcn/UI**, featuring glassmorphism and smooth animations.

---

## 🛠️ Tech Stack

| Component    | Technology                                                              |
| :----------- | :---------------------------------------------------------------------- |
| **Frontend** | Next.js 16 (App Router), React 19, Tailwind v4, Framer Motion           |
| **Backend**  | FastAPI, Python 3.12, Uvicorn                                           |
| **Local AI** | CLIP (`clip-ViT-B-32`), OpenCV                                          |
| **Cloud AI** | NVIDIA NIM (`meta/llama-3.2-90b-vision-instruct`, `nvidia/nv-embed-v1`) |
| **Database** | Pinecone (Serverless Vector DB)                                         |
| **Infra**    | Docker, Docker Compose                                                  |

---

## ⚡ Getting Started

### Prerequisites

- **Node.js** v18+
- **Python** 3.10+
- **Docker** (Optional, for containerized run)
- **API Keys**:
  - [NVIDIA NIM API Key](https://build.nvidia.com/) (for Vision/Embeddings)
  - [Pinecone API Key](https://pinecone.io/) (Serverless)

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure Environment
cp .env.example .env
# Edit .env and add your API keys:
# PINECONE_API_KEY=...
# PINECONE_INDEX_NAME=framesift
# NVIDIA_KEYS=["nvapi-..."]

# Run Server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure Environment
cp .env.example .env.local
# (Optional) NEXT_PUBLIC_API_URL=http://localhost:8000

# Run Dev Server
npm run dev
```

Visit **http://localhost:3000** to start searching! 🎬

---

## 🐳 Docker Setup (Recommended)

Run the entire stack with a single command:

```bash
# 1. Set Environment Variables
# Create a .env file in the root directory (copy backend/.env.example)

# 2. Build and Run
docker-compose up --build -d
```

---

## 📡 API Endpoints

| Method | Endpoint    | Description                            |
| :----- | :---------- | :------------------------------------- |
| `GET`  | `/`         | Health check                           |
| `POST` | `/upload`   | Upload video for background processing |
| `GET`  | `/job/{id}` | Check processing status                |
| `POST` | `/search`   | Semantic search query                  |
| `GET`  | `/jobs`     | List recent jobs                       |

---

## 📝 License

MIT License. Built by [Amitesh Vishwakarma](https://github.com/amitesh-7).
