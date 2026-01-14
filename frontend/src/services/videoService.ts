import api from "./api";

export interface SearchResult {
  video_id: string;
  timestamp: number;
  description: string;
  score: number;
}

export interface SearchResponse {
  results: SearchResult[];
  query: string;
}

export interface DeepScanRequest {
  video_id: string;
  start_time: number;
  end_time: number;
  fps?: number;
}

export interface DeepScanResponse {
  status: string;
  message: string;
  frames_processed: number;
  frames_indexed: number;
}

export const videoService = {
  search: async (
    query: string,
    videoId?: string,
    topK = 10
  ): Promise<SearchResponse> => {
    const { data } = await api.post("/search", {
      query,
      video_id: videoId,
      top_k: topK,
    });
    // Backend returns array directly, wrap it for consistency
    return { results: data, query };
  },

  upload: async (
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<{ video_id: string }> => {
    const formData = new FormData();
    formData.append("file", file);

    const { data } = await api.post("/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: (e) => {
        if (e.total && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      },
    });
    return data;
  },

  getVideos: async (): Promise<{ videos: string[] }> => {
    const { data } = await api.get("/videos");
    return data;
  },

  deepScan: async (request: DeepScanRequest): Promise<DeepScanResponse> => {
    const { data } = await api.post("/deep-scan", {
      video_id: request.video_id,
      start_time: request.start_time,
      end_time: request.end_time,
      fps: request.fps || 1,
    });
    return data;
  },
};
