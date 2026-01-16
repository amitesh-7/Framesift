import { useState, useRef } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Home, Menu, X, Video, Upload } from "lucide-react";
import {
  VideoPlayer,
  VideoPlayerHandle,
  SearchPanel,
  UploadModal,
} from "@/components/search";
import { Button, BackgroundPathsWrapper } from "@/components/ui";
import { cn } from "@/lib/utils";

export function SearchPage() {
  const [searchParams] = useSearchParams();
  const videoId = searchParams.get("video") || undefined;

  const videoRef = useRef<VideoPlayerHandle>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [uploadOpen, setUploadOpen] = useState(false);

  // Generate video source URL from video ID
  const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
  const videoSrc = videoId ? `${apiUrl}/videos/${videoId}` : undefined;

  const handleResultClick = (timestamp: number) => {
    videoRef.current?.seekTo(timestamp);
  };

  return (
    <BackgroundPathsWrapper>
      <div className="min-h-screen">
        {/* Header */}
        <header className="fixed top-0 left-0 right-0 z-50 h-14 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-sm flex items-center justify-between px-4">
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
            className={cn(
              "flex-1 p-4 transition-all",
              sidebarOpen && "md:mr-80"
            )}
          >
            <div className="max-w-4xl mx-auto">
              <VideoPlayer ref={videoRef} src={videoSrc} className="mb-4" />

              {!videoId && (
                <div className="p-12 rounded-xl border border-zinc-800 bg-zinc-900/50 backdrop-blur-sm flex items-center justify-center">
                  <Button
                    onClick={() => setUploadOpen(true)}
                    size="lg"
                    className="bg-gradient-to-r from-violet-500 to-fuchsia-500 hover:from-violet-600 hover:to-fuchsia-600 text-white px-8 py-6 text-lg"
                  >
                    <Upload className="w-5 h-5 mr-2" />
                    Upload Video
                  </Button>
                </div>
              )}
            </div>
          </main>

          {/* Sidebar */}
          <aside
            className={cn(
              "fixed top-14 right-0 bottom-0 w-full md:w-80 border-l border-zinc-800 bg-zinc-950/90 backdrop-blur-sm transition-transform",
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
    </BackgroundPathsWrapper>
  );
}
