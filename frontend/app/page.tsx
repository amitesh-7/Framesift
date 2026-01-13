"use client";

import { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Upload,
  ArrowRight,
  Sparkles,
  Zap,
  Search,
  Video,
  Brain,
  Database,
  Cpu,
  Cloud,
  MessageSquare,
} from "lucide-react";
import { BackgroundPaths } from "@/components/ui/background-paths";
import { Button } from "@/components/ui/button";
import { ExpandableTabs } from "@/components/ui/expandable-tabs";
import { UploadModal } from "@/components/UploadModal";

export default function Home() {
  const [isUploadOpen, setIsUploadOpen] = useState(false);

  const featureTabs = [
    { title: "Upload", icon: Upload },
    { title: "Analyze", icon: Brain },
    { type: "separator" as const },
    { title: "Search", icon: Search },
    { title: "Results", icon: MessageSquare },
  ];

  return (
    <>
      <BackgroundPaths
        title="FrameSift"
        subtitle="Semantic Video Search powered by Hybrid AI. Find any moment in your videos using natural language."
      >
        {/* Spacer */}
        <div className="h-8" />

        {/* Main CTA Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.2 }}
          className="flex flex-col sm:flex-row items-center justify-center gap-6 mb-16"
        >
          <Button
            variant="gradient"
            size="xl"
            onClick={() => setIsUploadOpen(true)}
            className="group min-w-[200px] shadow-2xl shadow-purple-500/25"
          >
            <Upload className="w-5 h-5 mr-3 transition-transform group-hover:-translate-y-0.5" />
            Upload Video
            <ArrowRight className="w-4 h-4 ml-3 transition-transform group-hover:translate-x-1" />
          </Button>

          <Button
            variant="outline"
            size="xl"
            asChild
            className="min-w-[200px] backdrop-blur-lg bg-white/10 border-white/20 hover:bg-white/20"
          >
            <Link href="/search">
              <Search className="w-5 h-5 mr-3" />
              Search Existing
            </Link>
          </Button>
        </motion.div>

        {/* Expandable Tabs - Workflow Preview */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 1.4 }}
          className="flex justify-center mb-16"
        >
          <ExpandableTabs
            tabs={featureTabs}
            activeColor="text-purple-500"
            className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-xl border-slate-200/50 dark:border-slate-700/50 shadow-xl"
          />
        </motion.div>

        {/* Feature Pills */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.6 }}
          className="flex flex-wrap items-center justify-center gap-4 mb-16"
        >
          {[
            { icon: Sparkles, text: "CLIP + NVIDIA NIM", color: "purple" },
            { icon: Zap, text: "Local + Cloud Hybrid", color: "amber" },
            { icon: Search, text: "Natural Language", color: "teal" },
          ].map((feature, i) => (
            <div
              key={i}
              className="flex items-center gap-3 px-5 py-3 rounded-full bg-white/80 dark:bg-slate-800/80 backdrop-blur-md border border-slate-200/50 dark:border-slate-700/50 shadow-lg hover:shadow-xl transition-shadow"
            >
              <feature.icon
                className={`w-5 h-5 ${
                  feature.color === "purple"
                    ? "text-purple-500"
                    : feature.color === "amber"
                    ? "text-amber-500"
                    : "text-teal-500"
                }`}
              />
              <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">
                {feature.text}
              </span>
            </div>
          ))}
        </motion.div>

        {/* AI Models Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.8 }}
          className="grid grid-cols-1 sm:grid-cols-3 gap-6 max-w-3xl mx-auto"
        >
          {[
            {
              icon: Video,
              title: "Llama 3.2",
              subtitle: "Vision Analysis",
              description: "90B multimodal model",
            },
            {
              icon: Brain,
              title: "NV-Embed",
              subtitle: "Text Embeddings",
              description: "4096-dim vectors",
            },
            {
              icon: Database,
              title: "Pinecone",
              subtitle: "Vector Search",
              description: "Serverless Index",
            },
          ].map((model, i) => (
            <div
              key={i}
              className="group relative p-6 rounded-2xl bg-white/70 dark:bg-slate-900/70 backdrop-blur-md border border-slate-200/50 dark:border-slate-700/50 shadow-lg hover:shadow-2xl transition-all hover:-translate-y-1"
            >
              <div className="flex flex-col items-center text-center space-y-3">
                <div className="p-3 rounded-xl bg-linear-to-br from-purple-500/20 to-pink-500/20 group-hover:from-purple-500/30 group-hover:to-pink-500/30 transition-colors">
                  <model.icon className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-slate-900 dark:text-white">
                    {model.title}
                  </h3>
                  <p className="text-sm font-medium text-purple-600 dark:text-purple-400">
                    {model.subtitle}
                  </p>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  {model.description}
                </p>
              </div>
            </div>
          ))}
        </motion.div>

        {/* Spacer at bottom */}
        <div className="h-12" />

        {/* Architecture Pills at Bottom */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 2 }}
          className="flex items-center justify-center gap-4 text-xs text-slate-500 dark:text-slate-400"
        >
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4" />
            <span>Local Scout (CPU)</span>
          </div>
          <span>+</span>
          <div className="flex items-center gap-2">
            <Cloud className="w-4 h-4" />
            <span>Cloud Intelligence (GPU)</span>
          </div>
        </motion.div>
      </BackgroundPaths>

      {/* Upload Modal */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
      />
    </>
  );
}
