import { useState, useRef, ChangeEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  Upload,
  FileVideo,
  Loader2,
  CheckCircle,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui";
import { videoService } from "@/services";
import { formatFileSize } from "@/lib/utils";

interface UploadModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess?: (videoId: string) => void;
}

export function UploadModal({ open, onClose, onSuccess }: UploadModalProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<
    "idle" | "uploading" | "success" | "error"
  >("idle");
  const [error, setError] = useState<string | null>(null);
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

    try {
      const result = await videoService.upload(file, setProgress);
      setStatus("success");
      onSuccess?.(result.video_id);
      setTimeout(() => {
        handleClose();
      }, 1500);
    } catch {
      setStatus("error");
      setError("Upload failed. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  const handleClose = () => {
    if (!uploading) {
      onClose();
      setFile(null);
      setStatus("idle");
      setProgress(0);
      setError(null);
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
                  <div className="space-y-1">
                    <div className="h-2 bg-zinc-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-violet-500 transition-all"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                    <p className="text-xs text-zinc-500 text-center">
                      {progress}%
                    </p>
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
                    disabled={uploading || status === "success"}
                  >
                    {uploading ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Uploading
                      </>
                    ) : status === "success" ? (
                      "Done!"
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
