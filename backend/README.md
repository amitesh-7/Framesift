---
title: FrameSift Backend
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# FrameSift Backend API

AI-Powered Semantic Video Search Engine - Backend Service

## 🔧 Environment Variables

Configure these in Space Settings → Variables:

**Required:**
- `NVIDIA_KEYS` - JSON array: `["nvapi-key1","nvapi-key2"]`
- `PINECONE_API_KEY` - Your Pinecone API key
- `PINECONE_INDEX_NAME` - Index name (default: `framesift`)

**Optional:**
- `MONGODB_URI` - MongoDB connection string
- `REDIS_URL` - Redis connection URL
- `ENSEMBLE_MODE` - Enable multi-model (default: `true`)

## 📖 API Documentation

Visit `/docs` after deployment for interactive API documentation.

## 🚀 Main Endpoints

- `POST /upload` - Upload and process videos
- `POST /search` - Search video frames semantically  
- `POST /clear-database` - Clear user data
- `GET /job/{job_id}` - Check processing status
