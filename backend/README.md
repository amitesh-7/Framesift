---
title: FrameSift Backend
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
---

# 🎬 FrameSift Backend API

AI-Powered Semantic Video Search Engine - Backend Service

## 🚀 Features

- **Semantic Video Search**: Natural language queries to find moments in videos
- **Multi-Model Ensemble**: Llama 3.2 90B Vision + 11B Vision for maximum accuracy
- **Smart Frame Processing**: Audio spike, brightness spike, physics filters
- **RAG-Powered Answers**: AI-generated conversational responses
- **Multi-User Support**: Isolated data per user with Pinecone namespaces

## 🔧 Configuration

This app requires the following environment variables in Hugging Face Space Settings:

### Required

- `NVIDIA_KEYS`: JSON array of NVIDIA NIM API keys `["nvapi-xxx", "nvapi-yyy"]`
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_INDEX_NAME`: Pinecone index name (default: `framesift`)

### Optional

- `MONGODB_URI`: MongoDB connection string for user tracking
- `REDIS_URL`: Redis URL for caching (format: `redis://user:pass@host:port`)
- `ENSEMBLE_MODE`: Enable multi-model ensemble (default: `true`)

## 📖 API Documentation

Once deployed, visit:

- **Swagger UI**: `https://your-space.hf.space/docs`
- **ReDoc**: `https://your-space.hf.space/redoc`

### Core Endpoints

#### `POST /upload`

Upload and process a video

- Header: `X-User-Id: your-user-id`
- Body: `multipart/form-data` with video file

#### `POST /search`

Search for moments in videos

- Header: `X-User-Id: your-user-id`
- Body: `{"query": "when does the lightning strike?", "top_k": 5}`

#### `POST /clear-database`

Clear user data on logout

- Header: `X-User-Id: your-user-id`

## 🏗️ Architecture

```
Local Filters (CPU) → Multi-Model Analysis (NVIDIA NIM) → Vector Search (Pinecone) → RAG Answer
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details
