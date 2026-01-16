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

AI-Powered Semantic Video Search Engine

## 🔧 Setup Instructions

After deploying this Space, configure these environment variables in Settings → Variables:

**Required:**

- `NVIDIA_KEYS` = `["nvapi-xxx","nvapi-yyy"]`
- `PINECONE_API_KEY` = `your-pinecone-key`
- `PINECONE_INDEX_NAME` = `framesift`

**Optional:**

- `MONGODB_URI` = MongoDB connection string
- `REDIS_URL` = Redis connection string
- `ENSEMBLE_MODE` = `true` (default)

## 📖 API Documentation

Once running, visit `/docs` for Swagger UI documentation.

## 🚀 Endpoints

- `POST /upload` - Upload and process video
- `POST /search` - Search video frames
- `POST /clear-database` - Clear user data

See full documentation at https://github.com/aimbot7/framesift

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
