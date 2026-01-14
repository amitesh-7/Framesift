import { useState, useRef } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Home, Menu, X, Video, Search as SearchIcon } from "lucide-react";
import {
  VideoPlayer,
  VideoPlayerHandle,
  SearchPanel,
  UploadModal,
} from "@/components/search";
import { Button } from "@/components/ui";
import { useAuthStore } from "@/store";
import { cn } from "@/lib/utils";

export function SearchPage() {
  const [searchParams] = useSearchParams();
  const videoId = searchParams.get("video") || undefined;
  const { isAuthenticated } = useAuthStore();

  const videoRef = useRef<VideoPlayerHandle>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [uploadOpen, setUploadOpen] = useState(false);

  const handleResultClick = (timestamp: number) => {
    videoRef.current?.seekTo(timestamp);
  };

  return (
    <div className="min-h-screen bg-black">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 h-14 border-b border-zinc-800 bg-zinc-950 flex items-center justify-between px-4">
        <Link to="/" className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
            <Video className="w-4 h-4 text-white" />
          </div>
          <span className="font-semibold text-white hidden sm:inline">
            FrameSift
          </span>
        </Link>

        <div className="flex items-center gap-2">
          <Link to="/">
            <Button variant="ghost" size="sm">
              <Home className="w-4 h-4" />
            </Button>
          </Link>
          <Button
            variant="ghost"
            size="sm"
            className="md:hidden"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            {sidebarOpen ? (
              <X className="w-4 h-4" />
            ) : (
              <Menu className="w-4 h-4" />
            )}
          </Button>
        </div>
      </header>

      {/* Main */}
      <div className="flex pt-14 min-h-screen">
        {/* Video Area */}
        <main
          className={cn("flex-1 p-4 transition-all", sidebarOpen && "md:mr-80")}
        >
          <div className="max-w-4xl mx-auto">
            <VideoPlayer ref={videoRef} className="mb-4" />

            {!videoId && (
              <div className="p-8 rounded-xl border border-zinc-800 bg-zinc-900/50 text-center">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center mx-auto mb-4">
                  <SearchIcon className="w-6 h-6 text-white" />
                </div>
                <h2 className="text-lg font-medium text-white mb-2">
                  No Video Selected
                </h2>
                <p className="text-sm text-zinc-500 mb-4">
                  Upload a video or use the search panel to find content.
                </p>
                {!isAuthenticated && (
                  <p className="text-xs text-zinc-600">
                    Sign in to upload videos
                  </p>
                )}
              </div>
            )}
          </div>
        </main>

        {/* Sidebar */}
        <aside
          className={cn(
            "fixed top-14 right-0 bottom-0 w-full md:w-80 border-l border-zinc-800 bg-zinc-950 transition-transform",
            sidebarOpen ? "translate-x-0" : "translate-x-full"
          )}
        >
          <SearchPanel
            videoId={videoId}
            onResultClick={handleResultClick}
            onUploadClick={() => setUploadOpen(true)}
          />
        </aside>
      </div>

      <UploadModal open={uploadOpen} onClose={() => setUploadOpen(false)} />
    </div>
  );
}
