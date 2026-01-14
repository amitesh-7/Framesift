import { motion } from "framer-motion";
import { Brain, Zap, Search, Clock, Shield, Layers } from "lucide-react";
import { cn } from "@/lib/utils";

const features = [
  {
    icon: Brain,
    title: "Multimodal AI",
    description: "Vision + language understanding",
    color: "text-violet-400 bg-violet-500/20",
  },
  {
    icon: Zap,
    title: "Hybrid Processing",
    description: "Local + cloud architecture",
    color: "text-amber-400 bg-amber-500/20",
  },
  {
    icon: Search,
    title: "Semantic Search",
    description: "Natural language queries",
    color: "text-cyan-400 bg-cyan-500/20",
  },
  {
    icon: Clock,
    title: "Instant Results",
    description: "Sub-second latency",
    color: "text-blue-400 bg-blue-500/20",
  },
  {
    icon: Shield,
    title: "Privacy First",
    description: "Local processing option",
    color: "text-emerald-400 bg-emerald-500/20",
  },
  {
    icon: Layers,
    title: "Scalable",
    description: "Personal to enterprise",
    color: "text-rose-400 bg-rose-500/20",
  },
];

export function Features() {
  return (
    <section id="features" className="py-20 border-t border-zinc-900">
      <div className="max-w-6xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-3">
            Powerful Features
          </h2>
          <p className="text-zinc-500">Enterprise-grade video intelligence</p>
        </motion.div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {features.map((feature, i) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.05 }}
              className="p-5 rounded-xl border border-zinc-800 bg-zinc-900/30"
            >
              <div
                className={cn(
                  "w-10 h-10 rounded-lg flex items-center justify-center mb-3",
                  feature.color
                )}
              >
                <feature.icon className="w-5 h-5" />
              </div>
              <h3 className="font-medium text-white mb-1">{feature.title}</h3>
              <p className="text-sm text-zinc-500">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
