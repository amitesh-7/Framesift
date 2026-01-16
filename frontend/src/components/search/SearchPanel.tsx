import { useState, FormEvent } from "react";
import { createPortal } from "react-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  Upload,
  Clock,
  ChevronRight,
  Loader2,
  AlertCircle,
  Zap,
  MessageSquare,
} from "lucide-react";
import { Button, Input } from "@/components/ui";
import { videoService, SearchResult } from "@/services";
import { formatTime } from "@/lib/utils";
import { DeepScanModal } from "./DeepScanModal";

interface SearchPanelProps {
  videoId?: string;
  onResultClick?: (timestamp: number) => void;
  onUploadClick?: () => void;
}

export function SearchPanel({
  videoId,
  onResultClick,
  onUploadClick,
}: SearchPanelProps) {
  const [query, setQuery] = useState("");
  const [lastQuery, setLastQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [aiAnswer, setAiAnswer] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deepScanOpen, setDeepScanOpen] = useState(false);

  const handleSearch = async (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setAiAnswer(null);

    try {
      const trimmedQuery = query.trim();
      const data = await videoService.search(trimmedQuery, videoId, 5);
      setResults(data.results || []);
      setLastQuery(trimmedQuery);
      setAiAnswer(data.ai_answer || null);
    } catch {
      setError("Search failed. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeepScanComplete = () => {
    // Refresh search results after deep scan
    if (lastQuery) {
      handleSearch({ preventDefault: () => {} } as FormEvent);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-zinc-800">
        <h2 className="font-medium text-white mb-3">Search</h2>

        <form onSubmit={handleSearch}>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={
                videoId
                  ? "Describe what you're looking for..."
                  : "Upload a video to search..."
              }
              className="pl-10"
              disabled={!videoId}
            />
          </div>
        </form>

        {videoId && (
          <Button
            variant="outline"
            size="sm"
            onClick={onUploadClick}
            className="w-full mt-3"
          >
            <Upload className="w-4 h-4 mr-1.5" />
            Upload Another Video
          </Button>
        )}
      </div>

      {/* Results */}
      <div className="flex-1 overflow-y-auto p-4">
        {isLoading && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-6 h-6 animate-spin text-violet-500" />
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 text-red-400 text-sm">
            <AlertCircle className="w-4 h-4" />
            {error}
          </div>
        )}

        {!isLoading && !error && results.length === 0 && lastQuery && (
          <div className="space-y-4">
            <div className="text-center py-8">
              <Search className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
              <p className="text-sm text-zinc-400 mb-2">
                No results found for "{lastQuery}"
              </p>
            </div>

            {/* Deep Scan CTA */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-4 rounded-lg border-2 border-dashed border-violet-500/30 bg-gradient-to-br from-violet-500/5 to-fuchsia-500/5"
            >
              <div className="flex items-start gap-3 mb-3">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center flex-shrink-0">
                  <Zap className="w-4 h-4 text-white" />
                </div>
                <div className="flex-1">
                  <h3 className="text-sm font-semibold text-white mb-1">
                    Try Deep Scan Mode
                  </h3>
                  <p className="text-xs text-zinc-400 leading-relaxed">
                    Not finding the moment? Deep Scan processes every frame in a
                    specific segment, bypassing all filters for maximum
                    accuracy.
                  </p>
                </div>
              </div>
              <Button
                onClick={() => setDeepScanOpen(true)}
                className="w-full bg-gradient-to-r from-violet-500 to-fuchsia-500 hover:from-violet-600 hover:to-fuchsia-600 text-white"
                size="sm"
              >
                <Zap className="w-3.5 h-3.5 mr-1.5" />
                Open Deep Scan
              </Button>
            </motion.div>
          </div>
        )}

        {!isLoading && !error && results.length === 0 && !lastQuery && (
          <div className="text-center py-12">
            <Search className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
            <p className="text-sm text-zinc-500">Enter a search query</p>
          </div>
        )}

        {!isLoading && !error && results.length > 0 && lastQuery && (
          <>
            {/* Search Query Info */}
            <div className="mb-3 p-3 rounded-lg bg-violet-500/10 border border-violet-500/20">
              <p className="text-xs text-zinc-400 mb-1">Search query:</p>
              <p className="text-sm text-white font-medium">"{lastQuery}"</p>
              <p className="text-xs text-zinc-500 mt-1">
                {results.length} result{results.length !== 1 ? "s" : ""} found
              </p>
            </div>

            {/* AI Answer Card */}
            {aiAnswer && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-4 p-4 rounded-xl bg-gradient-to-br from-emerald-500/10 to-cyan-500/10 border border-emerald-500/20"
              >
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center flex-shrink-0">
                    <MessageSquare className="w-4 h-4 text-white" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-emerald-400 mb-1.5">
                      AI Answer
                    </p>
                    <p className="text-sm text-zinc-200 leading-relaxed">
                      {aiAnswer}
                    </p>
                  </div>
                </div>
              </motion.div>
            )}
          </>
        )}

        <AnimatePresence>
          <div className="space-y-2">
            {results.map((result, i) => (
              <motion.button
                key={`${result.video_id}-${result.timestamp}-${i}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                onClick={() => onResultClick?.(result.timestamp)}
                className="w-full p-3 rounded-lg border border-zinc-800 bg-zinc-900/50 text-left hover:border-zinc-700 hover:bg-zinc-900 transition-colors group"
              >
                <div className="flex items-center gap-2 mb-1">
                  <Clock className="w-3.5 h-3.5 text-violet-400" />
                  <span className="text-sm font-medium text-violet-400">
                    {formatTime(result.timestamp)}
                  </span>
                  <span className="ml-auto text-xs text-zinc-600">
                    {Math.round(result.score * 100)}%
                  </span>
                </div>
                <p className="text-sm text-zinc-400 whitespace-pre-wrap break-words">
                  {result.description}
                </p>
                <ChevronRight className="w-4 h-4 text-zinc-600 mt-2 group-hover:translate-x-1 transition-transform" />
              </motion.button>
            ))}
          </div>
        </AnimatePresence>
      </div>

      {/* Deep Scan Modal - rendered via portal to escape overflow containers */}
      {videoId &&
        createPortal(
          <DeepScanModal
            open={deepScanOpen}
            onClose={() => setDeepScanOpen(false)}
            videoId={videoId}
            onComplete={handleDeepScanComplete}
          />,
          document.body
        )}
    </div>
  );
}
