import { useState, FormEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Zap, Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { Button, Input } from "@/components/ui";
import { videoService } from "@/services";

interface DeepScanModalProps {
  open: boolean;
  onClose: () => void;
  videoId: string;
  onComplete?: () => void;
}

export function DeepScanModal({
  open,
  onClose,
  videoId,
  onComplete,
}: DeepScanModalProps) {
  const [startTime, setStartTime] = useState("0");
  const [endTime, setEndTime] = useState("60");
  const [isScanning, setIsScanning] = useState(false);
  const [result, setResult] = useState<{
    status: "success" | "error";
    message: string;
  } | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    const start = parseFloat(startTime);
    const end = parseFloat(endTime);

    if (isNaN(start) || isNaN(end) || start < 0 || end <= start) {
      setResult({
        status: "error",
        message:
          "Invalid time range. End time must be greater than start time.",
      });
      return;
    }

    setIsScanning(true);
    setResult(null);

    try {
      const response = await videoService.deepScan({
        video_id: videoId,
        start_time: start,
        end_time: end,
        fps: 1,
      });

      setResult({
        status: "success",
        message: `Deep scan complete! Processed ${response.frames_processed} frames, indexed ${response.frames_indexed}.`,
      });

      // Wait 2 seconds before closing and triggering refresh
      setTimeout(() => {
        onComplete?.();
        onClose();
      }, 2000);
    } catch (error) {
      setResult({
        status: "error",
        message:
          error instanceof Error
            ? error.message
            : "Deep scan failed. Please try again.",
      });
    } finally {
      setIsScanning(false);
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md z-50"
          >
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl shadow-2xl overflow-hidden">
              {/* Header */}
              <div className="p-6 border-b border-zinc-800 bg-gradient-to-r from-violet-500/10 to-fuchsia-500/10">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
                      <Zap className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h2 className="text-lg font-semibold text-white">
                        Deep Scan Mode
                      </h2>
                      <p className="text-sm text-zinc-400">
                        Process every frame in a specific range
                      </p>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onClose}
                    disabled={isScanning}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
              </div>

              {/* Body */}
              <form onSubmit={handleSubmit} className="p-6 space-y-4">
                <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
                  <p className="text-sm text-blue-400">
                    💡 <strong>Tip:</strong> Use Deep Scan when the AI missed
                    important frames in a specific segment. This bypasses all
                    filters and processes every frame.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-zinc-300 mb-2">
                      Start Time (seconds)
                    </label>
                    <Input
                      type="number"
                      step="0.1"
                      min="0"
                      value={startTime}
                      onChange={(e) => setStartTime(e.target.value)}
                      placeholder="0"
                      disabled={isScanning}
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-zinc-300 mb-2">
                      End Time (seconds)
                    </label>
                    <Input
                      type="number"
                      step="0.1"
                      min="0"
                      value={endTime}
                      onChange={(e) => setEndTime(e.target.value)}
                      placeholder="60"
                      disabled={isScanning}
                      required
                    />
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-zinc-800/50 border border-zinc-700">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-zinc-400">Duration:</span>
                    <span className="text-white font-medium">
                      {(parseFloat(endTime) - parseFloat(startTime)).toFixed(1)}
                      s
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm mt-2">
                    <span className="text-zinc-400">Estimated frames:</span>
                    <span className="text-white font-medium">
                      ~{Math.ceil(parseFloat(endTime) - parseFloat(startTime))}
                    </span>
                  </div>
                </div>

                {/* Result Message */}
                {result && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`p-3 rounded-lg flex items-start gap-2 ${
                      result.status === "success"
                        ? "bg-green-500/10 border border-green-500/20 text-green-400"
                        : "bg-red-500/10 border border-red-500/20 text-red-400"
                    }`}
                  >
                    {result.status === "success" ? (
                      <CheckCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                    ) : (
                      <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                    )}
                    <p className="text-sm">{result.message}</p>
                  </motion.div>
                )}

                {/* Actions */}
                <div className="flex gap-3 pt-2">
                  <Button
                    type="button"
                    variant="outline"
                    className="flex-1"
                    onClick={onClose}
                    disabled={isScanning}
                  >
                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    className="flex-1 bg-gradient-to-r from-violet-500 to-fuchsia-500 hover:from-violet-600 hover:to-fuchsia-600 text-white"
                    disabled={isScanning}
                  >
                    {isScanning ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Scanning...
                      </>
                    ) : (
                      <>
                        <Zap className="w-4 h-4 mr-2" />
                        Start Deep Scan
                      </>
                    )}
                  </Button>
                </div>
              </form>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
