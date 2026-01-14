import { useState, useRef, ChangeEvent, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  Upload,
  FileVideo,
  Loader2,
  CheckCircle,
  AlertCircle,
  Film,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui";
import { videoService } from "@/services";
import { formatFileSize } from "@/lib/utils";
import { useNavigate } from "react-router-dom";
import api from "@/services/api";

interface UploadModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess?: (videoId: string) => void;
}

interface JobStatus {
  job_id: string;
  video_id: string;
  status: string;
  message?: string;
  progress?: number;
  frames_processed?: number;
}

export function UploadModal({ open, onClose, onSuccess }: UploadModalProps) {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<
    "idle" | "uploading" | "processing" | "success" | "error"
  >("idle");
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [processingMessage, setProcessingMessage] =
    useState<string>("Analyzing video...");
  const [framesProcessed, setFramesProcessed] = useState<number>(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected && selected.type.startsWith("video/")) {
      setFile(selected);
      setStatus("idle");
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setStatus("uploading");
    setProgress(0);

    try {
      // Upload the file
      const result = await videoService.upload(file, setProgress);
      setJobId(result.job_id);
      setVideoId(result.video_id);

      // Switch to processing status
      setStatus("processing");
      setProcessingMessage("Extracting frames...");
    } catch (err) {
      setStatus("error");
      setError(
        err instanceof Error ? err.message : "Upload failed. Please try again."
      );
      setUploading(false);
    }
  };

  // Poll job status
  useEffect(() => {
    if (!jobId || status !== "processing") return;

    let pollCount = 0;
    const maxPolls = 120; // 2 minutes max (1s intervals)

    const pollInterval = setInterval(async () => {
      try {
        const { data } = await api.get<JobStatus>(`/job/${jobId}`);

        // Update processing message
        if (data.message) {
          setProcessingMessage(data.message);
        }
        if (data.frames_processed) {
          setFramesProcessed(data.frames_processed);
        }

        // Check if completed
        if (data.status === "completed") {
          clearInterval(pollInterval);
          setStatus("success");
          setUploading(false);

          // Redirect to search page with video ID
          setTimeout(() => {
            if (videoId) {
              navigate(`/search?video=${videoId}`);
            }
            handleClose();
          }, 2000);
        } else if (data.status === "failed") {
          clearInterval(pollInterval);
          setStatus("error");
          setError(data.message || "Processing failed");
          setUploading(false);
        }

        pollCount++;
        if (pollCount >= maxPolls) {
          clearInterval(pollInterval);
          setStatus("error");
          setError("Processing timeout. Please check back later.");
          setUploading(false);
        }
      } catch (err) {
        console.error("Error polling job status:", err);
        // Don't stop polling on temporary errors
      }
    }, 1000);

    return () => clearInterval(pollInterval);
  }, [jobId, status, videoId, navigate]);

  const handleClose = () => {
    if (!uploading) {
      onClose();
      setFile(null);
      setStatus("idle");
      setProgress(0);
      setError(null);
      setJobId(null);
      setVideoId(null);
      setProcessingMessage("Analyzing video...");
      setFramesProcessed(0);
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
          onClick={handleClose}
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-md rounded-xl border border-zinc-800 bg-zinc-900 p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-medium text-white">Upload Video</h2>
              <button
                onClick={handleClose}
                disabled={uploading}
                className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors disabled:opacity-50"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              className="hidden"
            />

            {!file ? (
              <button
                onClick={() => inputRef.current?.click()}
                className="w-full py-12 rounded-lg border-2 border-dashed border-zinc-700 bg-zinc-800/50 hover:border-zinc-600 hover:bg-zinc-800 transition-colors flex flex-col items-center justify-center"
              >
                <Upload className="w-8 h-8 text-zinc-500 mb-3" />
                <p className="text-sm font-medium text-white mb-1">
                  Click to upload
                </p>
                <p className="text-xs text-zinc-500">
                  MP4, WebM, MOV up to 500MB
                </p>
              </button>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center gap-3 p-3 rounded-lg bg-zinc-800">
                  <FileVideo className="w-8 h-8 text-violet-400" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white truncate">
                      {file.name}
                    </p>
                    <p className="text-xs text-zinc-500">
                      {formatFileSize(file.size)}
                    </p>
                  </div>
                  {status === "success" && (
                    <CheckCircle className="w-5 h-5 text-emerald-500" />
                  )}
                  {status === "error" && (
                    <AlertCircle className="w-5 h-5 text-red-500" />
                  )}
                </div>

                {status === "uploading" && (
                  <div className="space-y-2">
                    <div className="h-2 bg-zinc-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-violet-500 transition-all"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                    <p className="text-xs text-zinc-500 text-center">
                      Uploading: {progress}%
                    </p>
                  </div>
                )}

                {status === "processing" && (
                  <div className="space-y-3 p-4 rounded-lg bg-violet-500/10 border border-violet-500/20">
                    <div className="flex items-center gap-3">
                      <div className="relative">
                        <Sparkles className="w-5 h-5 text-violet-400 animate-pulse" />
                        <div className="absolute inset-0 blur-md bg-violet-400/50 animate-pulse" />
                      </div>
                      <div className="flex-1">
                        <p className="text-sm font-medium text-white">
                          Processing Video
                        </p>
                        <p className="text-xs text-violet-300/80">
                          {processingMessage}
                        </p>
                      </div>
                    </div>
                    {framesProcessed > 0 && (
                      <div className="flex items-center gap-2 text-xs text-zinc-400">
                        <Film className="w-3.5 h-3.5" />
                        <span>{framesProcessed} frames analyzed</span>
                      </div>
                    )}
                    <div className="h-1 bg-zinc-700/50 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-violet-500 via-fuchsia-500 to-violet-500 animate-[shimmer_2s_infinite] w-1/3" />
                    </div>
                  </div>
                )}

                {status === "success" && (
                  <div className="p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                    <div className="flex items-center gap-3 text-emerald-400">
                      <CheckCircle className="w-5 h-5" />
                      <div>
                        <p className="text-sm font-medium">
                          Processing Complete!
                        </p>
                        <p className="text-xs text-emerald-300/80">
                          Redirecting to search...
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {error && <p className="text-sm text-red-400">{error}</p>}

                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    className="flex-1"
                    onClick={() => {
                      setFile(null);
                      setStatus("idle");
                      setError(null);
                    }}
                    disabled={uploading}
                  >
                    Change
                  </Button>
                  <Button
                    className="flex-1"
                    onClick={handleUpload}
                    disabled={
                      uploading ||
                      status === "success" ||
                      status === "processing"
                    }
                  >
                    {status === "uploading" ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Uploading
                      </>
                    ) : status === "processing" ? (
                      <>
                        <Sparkles className="w-4 h-4 mr-2 animate-pulse" />
                        Processing
                      </>
                    ) : status === "success" ? (
                      <>
                        <CheckCircle className="w-4 h-4 mr-2" />
                        Done!
                      </>
                    ) : (
                      "Upload"
                    )}
                  </Button>
                </div>
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
