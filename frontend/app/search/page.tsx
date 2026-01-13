"use client";

import { useRef, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import { Home, Upload, Settings, Menu, X } from "lucide-react";
import { VideoPlayer, VideoPlayerHandle } from "@/components/VideoPlayer";
import { SearchComponent } from "@/components/SearchComponent";
import { UploadModal } from "@/components/UploadModal";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

function SearchPageContent() {
  const searchParams = useSearchParams();
  const videoId = searchParams.get("video");

  const videoRef = useRef<VideoPlayerHandle>(null);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  const handleResultClick = (timestamp: number) => {
    videoRef.current?.seekTo(timestamp);
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
      {/* Top Navigation */}
      <header className="sticky top-0 z-40 w-full border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg">
        <div className="flex items-center justify-between h-14 px-4">
          {/* Logo */}
          <Link
            href="/"
            className="flex items-center gap-2 font-bold text-xl text-slate-900 dark:text-white"
          >
            <div className="w-8 h-8 rounded-lg bg-linear-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white text-sm font-bold">
              F
            </div>
            FrameSift
          </Link>

          {/* Actions */}
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" asChild>
              <Link href="/">
                <Home className="w-5 h-5" />
              </Link>
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsUploadOpen(true)}
            >
              <Upload className="w-5 h-5" />
            </Button>
            <Button variant="ghost" size="icon">
              <Settings className="w-5 h-5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="md:hidden"
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            >
              {isSidebarOpen ? (
                <X className="w-5 h-5" />
              ) : (
                <Menu className="w-5 h-5" />
              )}
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex h-[calc(100vh-3.5rem)]">
        {/* Video Panel */}
        <motion.main
          className={cn(
            "flex-1 p-4 md:p-6 overflow-y-auto",
            isSidebarOpen ? "md:mr-96" : ""
          )}
          layout
        >
          <div className="max-w-5xl mx-auto">
            {/* Video Player */}
            <VideoPlayer
              ref={videoRef}
              src={videoUrl || undefined}
              className="mb-6"
            />

            {/* Video Info */}
            {videoId && (
              <div className="p-4 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
                  Video Analysis Complete
                </h2>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  Video ID: <span className="font-mono">{videoId}</span>
                </p>
                <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
                  Use the search panel on the right to find specific moments in
                  this video.
                </p>
              </div>
            )}

            {!videoId && !videoUrl && (
              <div className="p-8 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-center">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center">
                  <Upload className="w-8 h-8 text-purple-500" />
                </div>
                <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
                  Get Started
                </h2>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
                  Upload a video to analyze it with AI and enable semantic
                  search.
                </p>
                <Button
                  variant="gradient"
                  onClick={() => setIsUploadOpen(true)}
                >
                  <Upload className="w-4 h-4 mr-2" />
                  Upload Video
                </Button>
              </div>
            )}

            {/* Local File Upload */}
            <div className="mt-6 p-4 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
              <h3 className="text-sm font-medium text-slate-900 dark:text-white mb-3">
                Or load a local video file
              </h3>
              <input
                type="file"
                accept="video/*"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    setVideoUrl(URL.createObjectURL(file));
                  }
                }}
                className="w-full text-sm text-slate-500 dark:text-slate-400
                  file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0
                  file:text-sm file:font-medium file:bg-purple-50 file:text-purple-700
                  dark:file:bg-purple-900/30 dark:file:text-purple-300
                  hover:file:bg-purple-100 dark:hover:file:bg-purple-900/50
                  file:cursor-pointer cursor-pointer"
              />
            </div>
          </div>
        </motion.main>

        {/* Search Sidebar */}
        <motion.aside
          initial={false}
          animate={{
            x: isSidebarOpen ? 0 : "100%",
            opacity: isSidebarOpen ? 1 : 0,
          }}
          transition={{ duration: 0.2 }}
          className={cn(
            "fixed right-0 top-14 bottom-0 w-full md:w-96 bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-800 shadow-xl",
            isSidebarOpen ? "pointer-events-auto" : "pointer-events-none"
          )}
        >
          <SearchComponent
            videoId={videoId || undefined}
            onResultClick={handleResultClick}
          />
        </motion.aside>
      </div>

      {/* Upload Modal */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
      />
    </div>
  );
}

export default function SearchPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-slate-950">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
        </div>
      }
    >
      <SearchPageContent />
    </Suspense>
  );
}
