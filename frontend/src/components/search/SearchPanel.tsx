import { useState, FormEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  Upload,
  Clock,
  ChevronRight,
  Loader2,
  AlertCircle,
} from "lucide-react";
import { Button, Input } from "@/components/ui";
import { videoService, SearchResult } from "@/services";
import { formatTime } from "@/lib/utils";

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
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const data = await videoService.search(query.trim(), videoId, 5);
      setResults(data.results || []);
    } catch {
      setError("Search failed. Please try again.");
    } finally {
      setIsLoading(false);
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

        {!isLoading && !error && results.length === 0 && (
          <div className="text-center py-12">
            <Search className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
            <p className="text-sm text-zinc-500">
              {query ? "No results found" : "Enter a search query"}
            </p>
          </div>
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
    </div>
  );
}
