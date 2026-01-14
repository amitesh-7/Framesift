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
};
