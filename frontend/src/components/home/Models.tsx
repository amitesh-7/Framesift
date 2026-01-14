import { motion } from "framer-motion";
import { Brain, Database, Zap } from "lucide-react";

const models = [
  {
    icon: Brain,
    name: "Llama 3.2 Vision",
    description: "90B multimodal model for frame analysis",
    gradient: "from-violet-500 to-purple-500",
  },
  {
    icon: Database,
    name: "NV-Embed v2",
    description: "4096-dimensional semantic embeddings",
    gradient: "from-blue-500 to-cyan-500",
  },
  {
    icon: Zap,
    name: "Pinecone",
    description: "Serverless vector similarity search",
    gradient: "from-amber-500 to-orange-500",
  },
];

export function Models() {
  return (
    <section className="py-20 border-t border-zinc-900">
      <div className="max-w-6xl mx-auto px-4">
        <div className="grid md:grid-cols-3 gap-4">
          {models.map((model, i) => (
            <motion.div
              key={model.name}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="p-6 rounded-xl border border-zinc-800 bg-zinc-900/30 hover:border-zinc-700 transition-colors"
            >
              <div
                className={`w-10 h-10 rounded-lg bg-gradient-to-br ${model.gradient} flex items-center justify-center mb-4`}
              >
                <model.icon className="w-5 h-5 text-white" />
              </div>
              <h3 className="text-lg font-medium text-white mb-1">
                {model.name}
              </h3>
              <p className="text-sm text-zinc-500">{model.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
