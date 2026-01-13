"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useRouter } from "next/navigation";
import {
  Upload,
  Loader2,
  CheckCircle,
  AlertCircle,
  X,
  Film,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
}

type UploadStatus = "idle" | "uploading" | "processing" | "complete" | "error";

interface JobStatus {
  job_id: string;
  status: string;
  progress: number;
  frames_processed: number;
  frames_total: number;
  error?: string;
}

export function UploadModal({ isOpen, onClose }: UploadModalProps) {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const pollJobStatus = useCallback(
    async (id: string) => {
      try {
        const response = await fetch(`${API_BASE_URL}/job/${id}`);
        if (!response.ok) throw new Error("Failed to get job status");

        const data: JobStatus = await response.json();

        if (data.status === "completed") {
          setStatus("complete");
          setProgress(100);
          // Navigate to search page after a short delay
          setTimeout(() => {
            router.push(`/search?video=${id}`);
          }, 1500);
        } else if (data.status === "failed") {
          setStatus("error");
          setError(data.error || "Processing failed");
        } else {
          // Still processing
          setProgress(data.progress * 100);
          // Continue polling
          setTimeout(() => pollJobStatus(id), 2000);
        }
      } catch (err) {
        setStatus("error");
        setError(err instanceof Error ? err.message : "Failed to check status");
      }
    },
    [router]
  );

  const handleUpload = useCallback(async () => {
    if (!file) return;

    setStatus("uploading");
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Upload failed");
      }

      const data = await response.json();
      setJobId(data.job_id);
      setStatus("processing");

      // Start polling for job status
      pollJobStatus(data.job_id);
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Upload failed");
    }
  }, [file, pollJobStatus]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith("video/")) {
      setFile(droppedFile);
      setError(null);
    } else {
      setError("Please drop a valid video file");
    }
  };

  const resetUpload = () => {
    setFile(null);
    setStatus("idle");
    setJobId(null);
    setProgress(0);
    setError(null);
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
        >
          <Card className="w-full max-w-lg bg-white/95 dark:bg-slate-900/95 backdrop-blur-xl">
            <CardContent className="p-6">
              {/* Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-purple-100 dark:bg-purple-900/30">
                    <Film className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                  </div>
                  <h2 className="text-xl font-semibold text-slate-900 dark:text-white">
                    Upload Video
                  </h2>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                >
                  <X className="w-5 h-5 text-slate-400" />
                </button>
              </div>

              {/* Upload Area */}
              {status === "idle" && (
                <div
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  className={cn(
                    "border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200",
                    file
                      ? "border-purple-500 bg-purple-50 dark:bg-purple-900/10"
                      : "border-slate-200 dark:border-slate-700 hover:border-purple-400 dark:hover:border-purple-500"
                  )}
                >
                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleFileChange}
                    className="hidden"
                    id="video-upload"
                  />
                  <label
                    htmlFor="video-upload"
                    className="cursor-pointer flex flex-col items-center"
                  >
                    <div className="p-4 rounded-full bg-slate-100 dark:bg-slate-800 mb-4">
                      <Upload className="w-8 h-8 text-slate-400" />
                    </div>
                    {file ? (
                      <>
                        <p className="text-sm font-medium text-slate-900 dark:text-white mb-1">
                          {file.name}
                        </p>
                        <p className="text-xs text-slate-500">
                          {(file.size / (1024 * 1024)).toFixed(2)} MB
                        </p>
                      </>
                    ) : (
                      <>
                        <p className="text-sm font-medium text-slate-900 dark:text-white mb-1">
                          Drop your video here or click to browse
                        </p>
                        <p className="text-xs text-slate-500">
                          Supports MP4, AVI, MOV, WebM
                        </p>
                      </>
                    )}
                  </label>
                </div>
              )}

              {/* Processing State */}
              {(status === "uploading" || status === "processing") && (
                <div className="text-center py-8">
                  <Loader2 className="w-12 h-12 mx-auto mb-4 animate-spin text-purple-500" />
                  <p className="text-sm font-medium text-slate-900 dark:text-white mb-2">
                    {status === "uploading"
                      ? "Uploading video..."
                      : "Processing frames..."}
                  </p>
                  <div className="w-full bg-slate-100 dark:bg-slate-800 rounded-full h-2 overflow-hidden">
                    <motion.div
                      className="h-full bg-linear-to-r from-purple-500 to-pink-500"
                      initial={{ width: 0 }}
                      animate={{ width: `${progress}%` }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>
                  <p className="text-xs text-slate-500 mt-2">
                    {Math.round(progress)}% complete
                  </p>
                  {jobId && (
                    <p className="text-xs text-slate-400 mt-2 font-mono">
                      Job ID: {jobId.slice(0, 8)}...
                    </p>
                  )}
                </div>
              )}

              {/* Complete State */}
              {status === "complete" && (
                <div className="text-center py-8">
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="w-16 h-16 mx-auto mb-4 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center"
                  >
                    <CheckCircle className="w-8 h-8 text-green-500" />
                  </motion.div>
                  <p className="text-sm font-medium text-slate-900 dark:text-white">
                    Processing complete!
                  </p>
                  <p className="text-xs text-slate-500 mt-1">
                    Redirecting to search...
                  </p>
                </div>
              )}

              {/* Error State */}
              {status === "error" && (
                <div className="text-center py-8">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
                    <AlertCircle className="w-8 h-8 text-red-500" />
                  </div>
                  <p className="text-sm font-medium text-slate-900 dark:text-white mb-1">
                    Upload Failed
                  </p>
                  <p className="text-xs text-red-500 mb-4">{error}</p>
                  <Button onClick={resetUpload} variant="outline" size="sm">
                    Try Again
                  </Button>
                </div>
              )}

              {/* Actions */}
              {status === "idle" && (
                <div className="flex gap-3 mt-6">
                  <Button
                    variant="outline"
                    onClick={onClose}
                    className="flex-1"
                  >
                    Cancel
                  </Button>
                  <Button
                    variant="gradient"
                    onClick={handleUpload}
                    disabled={!file}
                    className="flex-1"
                  >
                    <Upload className="w-4 h-4 mr-2" />
                    Upload & Process
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

export default UploadModal;
