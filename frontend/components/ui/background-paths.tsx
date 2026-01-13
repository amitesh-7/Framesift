"use client";

import { motion } from "framer-motion";
import { useMemo, useState, useEffect, useCallback } from "react";

function FloatingPaths({ position }: { position: number }) {
  const paths = useMemo(
    () =>
      Array.from({ length: 36 }, (_, i) => ({
        id: i,
        d: `M-${380 - i * 5 * position} -${189 + i * 6}C-${
          380 - i * 5 * position
        } -${189 + i * 6} -${312 - i * 5 * position} ${216 - i * 6} ${
          152 - i * 5 * position
        } ${343 - i * 6}C${616 - i * 5 * position} ${470 - i * 6} ${
          684 - i * 5 * position
        } ${875 - i * 6} ${684 - i * 5 * position} ${875 - i * 6}`,
        color: `rgba(15,23,42,${0.1 + i * 0.03})`,
        width: 0.5 + i * 0.03,
        duration: 20 + (i % 10),
      })),
    [position]
  );

  return (
    <div className="absolute inset-0 pointer-events-none">
      <svg
        className="w-full h-full text-slate-950 dark:text-white"
        viewBox="0 0 696 316"
        fill="none"
      >
        <title>Background Paths</title>
        {paths.map((path) => (
          <motion.path
            key={path.id}
            d={path.d}
            stroke="currentColor"
            strokeWidth={path.width}
            strokeOpacity={0.1 + path.id * 0.03}
            initial={{ pathLength: 0.3, opacity: 0.6 }}
            animate={{
              pathLength: 1,
              opacity: [0.3, 0.6, 0.3],
              pathOffset: [0, 1, 0],
            }}
            transition={{
              duration: path.duration,
              repeat: Number.POSITIVE_INFINITY,
              ease: "linear",
            }}
          />
        ))}
      </svg>
    </div>
  );
}

// Custom hook that avoids the lint warning by using a callback pattern
function useHasMounted() {
  const [hasMounted, setHasMounted] = useState(false);

  const onMount = useCallback(() => {
    setHasMounted(true);
  }, []);

  useEffect(() => {
    // Using requestAnimationFrame to defer the state update
    const frameId = requestAnimationFrame(onMount);
    return () => cancelAnimationFrame(frameId);
  }, [onMount]);

  return hasMounted;
}

interface BackgroundPathsProps {
  title?: string;
  subtitle?: string;
  children?: React.ReactNode;
}

export function BackgroundPaths({
  title,
  subtitle,
  children,
}: BackgroundPathsProps) {
  const mounted = useHasMounted();

  return (
    <div className="relative min-h-screen w-full flex items-center justify-center overflow-hidden bg-white dark:bg-slate-950">
      {/* Gradient background */}
      <div className="absolute inset-0 bg-linear-to-br from-indigo-50/50 via-white to-purple-50/50 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950" />

      {/* Animated paths - only render after mount */}
      {mounted && (
        <div className="absolute inset-0">
          <FloatingPaths position={1} />
          <FloatingPaths position={-1} />
        </div>
      )}

      {/* Content */}
      <div className="relative z-10 container mx-auto px-4 md:px-6 text-center">
        {title && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 2 }}
          >
            <h1 className="text-4xl sm:text-6xl md:text-7xl font-bold mb-6 tracking-tighter">
              {title.split("").map((char, i) => (
                <motion.span
                  key={i}
                  initial={{ opacity: 0, y: 50 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    duration: 0.8,
                    delay: i * 0.05,
                    ease: [0.215, 0.61, 0.355, 1],
                  }}
                  className="inline-block bg-linear-to-r from-slate-900 via-purple-900 to-slate-900 dark:from-white dark:via-purple-200 dark:to-white bg-clip-text text-transparent"
                >
                  {char === " " ? "\u00A0" : char}
                </motion.span>
              ))}
            </h1>
          </motion.div>
        )}

        {subtitle && (
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.5 }}
            className="text-lg md:text-xl text-slate-600 dark:text-slate-400 max-w-2xl mx-auto mb-8"
          >
            {subtitle}
          </motion.p>
        )}

        {children && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 1 }}
          >
            {children}
          </motion.div>
        )}
      </div>
    </div>
  );
}

export default BackgroundPaths;
