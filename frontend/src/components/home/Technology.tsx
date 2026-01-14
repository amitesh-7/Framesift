import { motion } from "framer-motion";

const tech = ["NVIDIA NIM", "Llama 3.2", "NV-Embed v2", "Pinecone"];

export function Technology() {
  return (
    <section id="tech" className="py-20">
      <div className="max-w-6xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center"
        >
          <p className="text-sm text-zinc-500 mb-6">
            Powered by industry leaders
          </p>
          <div className="flex flex-wrap justify-center gap-3">
            {tech.map((name) => (
              <span
                key={name}
                className="px-4 py-2 rounded-full border border-zinc-800 bg-zinc-900/50 text-sm text-zinc-300"
              >
                {name}
              </span>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
