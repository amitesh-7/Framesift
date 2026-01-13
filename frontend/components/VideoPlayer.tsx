"use client";

import React, { useRef, forwardRef, useImperativeHandle } from "react";
import { cn } from "@/lib/utils";

export interface VideoPlayerHandle {
  seekTo: (seconds: number) => void;
  play: () => void;
  pause: () => void;
  getCurrentTime: () => number;
}

interface VideoPlayerProps {
  src?: string;
  className?: string;
  onTimeUpdate?: (time: number) => void;
}

export const VideoPlayer = forwardRef<VideoPlayerHandle, VideoPlayerProps>(
  ({ src, className, onTimeUpdate }, ref) => {
    const videoRef = useRef<HTMLVideoElement>(null);

    useImperativeHandle(ref, () => ({
      seekTo: (seconds: number) => {
        if (videoRef.current) {
          videoRef.current.currentTime = seconds;
          videoRef.current.play();
        }
      },
      play: () => {
        videoRef.current?.play();
      },
      pause: () => {
        videoRef.current?.pause();
      },
      getCurrentTime: () => {
        return videoRef.current?.currentTime ?? 0;
      },
    }));

    const handleTimeUpdate = () => {
      if (videoRef.current && onTimeUpdate) {
        onTimeUpdate(videoRef.current.currentTime);
      }
    };

    if (!src) {
      return (
        <div
          className={cn(
            "w-full aspect-video bg-slate-900 rounded-xl flex items-center justify-center",
            className
          )}
        >
          <div className="text-center text-slate-400">
            <svg
              className="w-16 h-16 mx-auto mb-4 opacity-50"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
            <p className="text-sm">Upload a video to get started</p>
          </div>
        </div>
      );
    }

    return (
      <div
        className={cn(
          "w-full rounded-xl overflow-hidden shadow-2xl",
          className
        )}
      >
        <video
          ref={videoRef}
          src={src}
          controls
          className="w-full aspect-video bg-black"
          onTimeUpdate={handleTimeUpdate}
        >
          <track kind="captions" />
        </video>
      </div>
    );
  }
);

VideoPlayer.displayName = "VideoPlayer";
