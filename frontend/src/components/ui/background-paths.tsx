import { motion } from "framer-motion";

function FloatingPaths({ position }: { position: number }) {
  const paths = Array.from({ length: 36 }, (_, i) => ({
    id: i,
    d: `M-${380 - i * 5 * position} -${189 + i * 6}C-${
      380 - i * 5 * position
    } -${189 + i * 6} -${312 - i * 5 * position} ${216 - i * 6} ${
      152 - i * 5 * position
    } ${343 - i * 6}C${616 - i * 5 * position} ${470 - i * 6} ${
      684 - i * 5 * position
    } ${875 - i * 6} ${684 - i * 5 * position} ${875 - i * 6}`,
    width: 0.5 + i * 0.03,
    duration: 20 + i * 0.5,
  }));

  return (
    <div className="absolute inset-0">
      <svg
        className="absolute w-full h-full min-w-full min-h-full"
        viewBox="-100 -100 896 516"
        fill="none"
        preserveAspectRatio="xMidYMid slice"
        style={{
          transform: "scale(1.5)",
          transformOrigin: "center center",
        }}
      >
        <defs>
          <linearGradient
            id={`pathGradient-${position}`}
            x1="0%"
            y1="0%"
            x2="100%"
            y2="100%"
          >
            <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.4" />
            <stop offset="50%" stopColor="#d946ef" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.4" />
          </linearGradient>
        </defs>
        {paths.map((path) => (
          <motion.path
            key={path.id}
            d={path.d}
            stroke={`url(#pathGradient-${position})`}
            strokeWidth={path.width}
            strokeOpacity={0.2 + path.id * 0.015}
            fill="none"
            initial={{ pathLength: 0.3 }}
            animate={{
              pathLength: [0.3, 1, 0.3],
              pathOffset: [0, 1, 0],
            }}
            transition={{
              duration: path.duration,
              repeat: Infinity,
              ease: "linear",
            }}
          />
        ))}
      </svg>
    </div>
  );
}

// Background wrapper component for pages (not hero)
export function BackgroundPathsWrapper({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="relative min-h-screen bg-black">
      {/* Fixed animated background */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <FloatingPaths position={1} />
        <FloatingPaths position={-1} />
      </div>
      {/* Content */}
      <div className="relative z-10">{children}</div>
    </div>
  );
}

// Original BackgroundPaths hero component (kept for compatibility)
export function BackgroundPaths() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      <FloatingPaths position={1} />
      <FloatingPaths position={-1} />
    </div>
  );
}

export { FloatingPaths };
