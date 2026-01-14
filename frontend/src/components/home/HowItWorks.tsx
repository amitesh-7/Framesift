import { motion } from "framer-motion";
import { ArrowRight, Brain, Search } from "lucide-react";

const steps = [
  {
    step: "01",
    icon: ArrowRight,
    title: "Sign In",
    description: "Connect with Google",
  },
  {
    step: "02",
    icon: Brain,
    title: "Upload",
    description: "Add videos for AI processing",
  },
  {
    step: "03",
    icon: Search,
    title: "Search",
    description: "Find moments naturally",
  },
];

export function HowItWorks() {
  return (
    <section id="how-it-works" className="py-20">
      <div className="max-w-6xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-3">
            How It Works
          </h2>
          <p className="text-zinc-500">
            Three simple steps to searchable videos
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-6">
          {steps.map((item, i) => (
            <motion.div
              key={item.step}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="relative p-6 rounded-xl border border-zinc-800 bg-zinc-900/30"
            >
              <span className="text-5xl font-black text-zinc-800">
                {item.step}
              </span>
              <div className="mt-4">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center mb-3">
                  <item.icon className="w-5 h-5 text-white" />
                </div>
                <h3 className="text-lg font-medium text-white mb-1">
                  {item.title}
                </h3>
                <p className="text-sm text-zinc-500">{item.description}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
