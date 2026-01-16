import { useState, useEffect, useCallback, useRef } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowRight, Play, Search } from "lucide-react";
import { useGoogleLogin } from "@react-oauth/google";
import { Button } from "@/components/ui";
import { useAuthStore } from "@/store";
import { config } from "@/config";

// Character type for raining letters
interface Character {
  char: string;
  x: number;
  y: number;
  speed: number;
}

// Text scramble class for the title effect
class TextScramble {
  el: HTMLElement;
  chars: string;
  queue: Array<{
    from: string;
    to: string;
    start: number;
    end: number;
    char?: string;
  }>;
  frame: number;
  frameRequest: number;
  resolve: (value: void | PromiseLike<void>) => void;

  constructor(el: HTMLElement) {
    this.el = el;
    this.chars = "!<>-_\\/[]{}—=+*^?#";
    this.queue = [];
    this.frame = 0;
    this.frameRequest = 0;
    this.resolve = () => {};
    this.update = this.update.bind(this);
  }

  setText(newText: string) {
    const oldText = this.el.innerText;
    const length = Math.max(oldText.length, newText.length);
    const promise = new Promise<void>((resolve) => (this.resolve = resolve));
    this.queue = [];

    for (let i = 0; i < length; i++) {
      const from = oldText[i] || "";
      const to = newText[i] || "";
      const start = Math.floor(Math.random() * 40);
      const end = start + Math.floor(Math.random() * 40);
      this.queue.push({ from, to, start, end });
    }

    cancelAnimationFrame(this.frameRequest);
    this.frame = 0;
    this.update();
    return promise;
  }

  update() {
    let output = "";
    let complete = 0;

    for (let i = 0, n = this.queue.length; i < n; i++) {
      const { from, to, start, end } = this.queue[i];
      let { char } = this.queue[i];
      if (this.frame >= end) {
        complete++;
        output += to;
      } else if (this.frame >= start) {
        if (!char || Math.random() < 0.28) {
          char = this.chars[Math.floor(Math.random() * this.chars.length)];
          this.queue[i].char = char;
        }
        output += `<span class="text-violet-400">${char}</span>`;
      } else {
        output += from;
      }
    }

    this.el.innerHTML = output;
    if (complete === this.queue.length) {
      this.resolve();
    } else {
      this.frameRequest = requestAnimationFrame(this.update);
      this.frame++;
    }
  }
}

// Scrambled title component
function ScrambledTitle() {
  const elementRef = useRef<HTMLHeadingElement>(null);
  const scramblerRef = useRef<TextScramble | null>(null);
  const [mounted, setMounted] = useState(false);

  const phrases = [
    "Find Any Moment",
    "In Your Videos",
    "Semantic Search",
    "AI Powered",
    "FrameSift",
  ];

  useEffect(() => {
    if (elementRef.current && !scramblerRef.current) {
      scramblerRef.current = new TextScramble(elementRef.current);
      setTimeout(() => setMounted(true), 0);
    }
  }, []);

  useEffect(() => {
    if (mounted && scramblerRef.current) {
      let counter = 0;
      const phraseList = [
        "Find Any Moment",
        "In Your Videos",
        "Semantic Search",
        "AI Powered",
        "FrameSift",
      ];
      const next = () => {
        if (scramblerRef.current) {
          scramblerRef.current.setText(phraseList[counter]).then(() => {
            setTimeout(next, 2500);
          });
          counter = (counter + 1) % phraseList.length;
        }
      };
      next();
    }
  }, [mounted]);

  return (
    <h1
      ref={elementRef}
      className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight text-white font-mono"
    >
      {phrases[0]}
    </h1>
  );
}

export function Hero() {
  const { isAuthenticated, login } = useAuthStore();
  const [characters, setCharacters] = useState<Character[]>([]);
  const [activeIndices, setActiveIndices] = useState<Set<number>>(new Set());

  const googleLogin = useGoogleLogin({
    onSuccess: async (response) => {
      try {
        const res = await fetch(
          "https://www.googleapis.com/oauth2/v3/userinfo",
          {
            headers: { Authorization: `Bearer ${response.access_token}` },
          }
        );
        const data = await res.json();
        login(
          {
            id: data.sub,
            email: data.email,
            name: data.name,
            picture: data.picture,
          },
          response.access_token
        );
      } catch (error) {
        console.error("Login failed:", error);
      }
    },
  });

  // Create raining characters
  const createCharacters = useCallback(() => {
    const allChars =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?";
    const charCount = 300;
    const newCharacters: Character[] = [];

    for (let i = 0; i < charCount; i++) {
      newCharacters.push({
        char: allChars[Math.floor(Math.random() * allChars.length)],
        x: Math.random() * 100,
        y: Math.random() * 100,
        speed: 0.1 + Math.random() * 0.3,
      });
    }
    return newCharacters;
  }, []);

  useEffect(() => {
    setCharacters(createCharacters());
  }, [createCharacters]);

  // Flicker active characters
  useEffect(() => {
    const updateActiveIndices = () => {
      const newActiveIndices = new Set<number>();
      const numActive = Math.floor(Math.random() * 3) + 3;
      for (let i = 0; i < numActive; i++) {
        newActiveIndices.add(Math.floor(Math.random() * characters.length));
      }
      setActiveIndices(newActiveIndices);
    };

    const flickerInterval = setInterval(updateActiveIndices, 50);
    return () => clearInterval(flickerInterval);
  }, [characters.length]);

  // Animate character positions
  useEffect(() => {
    let animationFrameId: number;

    const updatePositions = () => {
      setCharacters((prevChars) =>
        prevChars.map((char) => ({
          ...char,
          y: char.y + char.speed,
          ...(char.y >= 100 && {
            y: -5,
            x: Math.random() * 100,
            char: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?"[
              Math.floor(
                Math.random() *
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?"
                    .length
              )
            ],
          }),
        }))
      );
      animationFrameId = requestAnimationFrame(updatePositions);
    };

    animationFrameId = requestAnimationFrame(updatePositions);
    return () => cancelAnimationFrame(animationFrameId);
  }, []);

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden bg-black">
      {/* Raining Characters Background - with higher z-index to show in front */}
      <div className="absolute inset-0 z-10">
        {characters.map((char, index) => (
          <span
            key={index}
            className={`absolute transition-colors duration-100 select-none pointer-events-none ${
              activeIndices.has(index)
                ? "text-green-500 font-bold scale-125 animate-pulse"
                : "text-slate-700 font-light"
            }`}
            style={{
              left: `${char.x}%`,
              top: `${char.y}%`,
              transform: `translate(-50%, -50%) ${
                activeIndices.has(index) ? "scale(1.25)" : "scale(1)"
              }`,
              textShadow: activeIndices.has(index)
                ? "0 0 8px rgba(34, 197, 94, 0.8), 0 0 12px rgba(34, 197, 94, 0.4)"
                : "none",
              opacity: activeIndices.has(index) ? 1 : 0.4,
              fontSize: "1.8rem",
              willChange: "transform, top",
            }}
          >
            {char.char}
          </span>
        ))}
      </div>

      {/* Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-black/30 via-transparent to-black/60 pointer-events-none z-[5]" />

      {/* Content */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="relative z-20 max-w-5xl mx-auto px-4 py-20 text-center"
      >
        {/* Badge */}
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-violet-500/30 bg-violet-500/10 text-violet-300 text-sm mb-10">
          <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
          Powered by NVIDIA NIM + Llama 3.2 Vision
        </div>

        {/* Scrambled Title */}
        <div className="mb-8">
          <ScrambledTitle />
        </div>

        {/* Subtitle */}
        <p className="text-xl md:text-2xl bg-gradient-to-r from-violet-400 to-fuchsia-400 bg-clip-text text-transparent font-semibold mb-10">
          AI-Powered Video Search
        </p>

        {/* Description */}
        <p className="max-w-2xl mx-auto text-zinc-400 text-base md:text-lg mb-12 leading-relaxed">
          Describe what you're looking for and jump directly to the exact
          moment. Semantic search powered by multimodal AI.
        </p>

        {/* CTAs */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-6 mb-16">
          {isAuthenticated ? (
            <Link to="/search">
              <Button size="lg" className="min-w-[200px]">
                <Search className="w-4 h-4 mr-2" />
                Start Searching
                <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
            </Link>
          ) : (
            <Button
              size="lg"
              onClick={() => googleLogin()}
              className="min-w-[200px]"
            >
              <svg className="w-4 h-4 mr-2" viewBox="0 0 24 24">
                <path
                  fill="currentColor"
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                />
                <path
                  fill="currentColor"
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                />
                <path
                  fill="currentColor"
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                />
                <path
                  fill="currentColor"
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                />
              </svg>
              Get Started with Google
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          )}
        </div>

        {/* Stats */}
        <div className="flex flex-wrap justify-center gap-12 pt-8">
          {[
            {
              value: config.features.embedDimensions,
              label: "Embeddings",
            },
            { value: config.features.modelSize, label: "Vision Model" },
            { value: config.features.searchLatency, label: "Latency" },
          ].map((stat) => (
            <div key={stat.label} className="text-center">
              <div className="text-2xl font-bold text-white">{stat.value}</div>
              <div className="text-xs text-zinc-500">{stat.label}</div>
            </div>
          ))}
        </div>
      </motion.div>
    </section>
  );
}

export default Hero;
