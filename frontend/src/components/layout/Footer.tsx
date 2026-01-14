import { Video } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-zinc-800 py-8">
      <div className="max-w-6xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-md bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
            <Video className="w-3.5 h-3.5 text-white" />
          </div>
          <span className="font-medium text-white">FrameSift</span>
        </div>
        <p className="text-sm text-zinc-600">
          Built with <span className="text-fuchsia-500">♥</span> by Amitesh
          Vishwakarma
        </p>
      </div>
    </footer>
  );
}
