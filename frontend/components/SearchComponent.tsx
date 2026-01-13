"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Loader2, Clock, AlertCircle, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface SearchResult {
  timestamp: number;
  score: number;
  description: string;
  frame_id: string;
}

interface SearchComponentProps {
  videoId?: string;
  onResultClick?: (timestamp: number) => void;
  className?: string;
}

export function SearchComponent({
  videoId,
  onResultClick,
  className,
}: SearchComponentProps) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query.trim(),
          video_id: videoId,
          top_k: 10,
        }),
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  }, [query, videoId]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  const formatTimestamp = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Header */}
      <div className="p-4 border-b border-slate-200 dark:border-slate-800">
        <div className="flex items-center gap-2 mb-3">
          <Sparkles className="w-5 h-5 text-purple-500" />
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
            Semantic Search
          </h2>
        </div>
        <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
          Describe what you&apos;re looking for in the video
        </p>

        {/* Search Input */}
        <div className="flex gap-2">
          <Input
            placeholder="e.g., 'person walking in the park'"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            className="flex-1"
            disabled={isLoading}
          />
          <Button
            onClick={handleSearch}
            disabled={isLoading || !query.trim()}
            variant="gradient"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Search className="w-4 h-4" />
            )}
          </Button>
        </div>
      </div>

      {/* Results */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {/* Error State */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800"
            >
              <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
                <AlertCircle className="w-4 h-4" />
                <span className="text-sm">{error}</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Empty State */}
        {!isLoading && results.length === 0 && !error && (
          <div className="flex-1 flex items-center justify-center py-12">
            <div className="text-center text-slate-400">
              <Search className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p className="text-sm">Enter a search query to find moments</p>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <Loader2 className="w-8 h-8 mx-auto mb-4 animate-spin text-purple-500" />
              <p className="text-sm text-slate-400">Searching...</p>
            </div>
          </div>
        )}

        {/* Results List */}
        <AnimatePresence>
          {results.map((result, index) => (
            <motion.div
              key={result.frame_id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ delay: index * 0.05 }}
            >
              <Card
                className="cursor-pointer hover:shadow-lg hover:scale-[1.02] transition-all duration-200 overflow-hidden"
                onClick={() => onResultClick?.(result.timestamp)}
              >
                <CardContent className="p-4">
                  <div className="flex items-start gap-3">
                    {/* Timestamp Badge */}
                    <div className="shrink-0 flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300">
                      <Clock className="w-3.5 h-3.5" />
                      <span className="text-sm font-mono font-medium">
                        {formatTimestamp(result.timestamp)}
                      </span>
                    </div>

                    {/* Description */}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-slate-700 dark:text-slate-300 line-clamp-2">
                        {result.description}
                      </p>
                      <div className="mt-2 flex items-center gap-2">
                        <div className="h-1.5 flex-1 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-linear-to-r from-purple-500 to-pink-500 rounded-full"
                            style={{ width: `${result.score * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-slate-400 tabular-nums">
                          {Math.round(result.score * 100)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}

export default SearchComponent;
