/**
 * Default source catalog. Ported/expanded from the Python repo's
 * ingestion-service/src/config/sources.yaml and the challenge brief
 * (HackerNews, AI newsletters, subreddits, X accounts).
 *
 * `kind` selects the ingest adapter in lib/pipeline/ingest.
 * `weight` is source reputation (0..1) used by the ranker.
 */
export type SeedSource = {
  sourceId: string;
  name: string;
  kind: "rss" | "hackernews" | "reddit" | "x" | "newsletter";
  url: string;
  topics: string[];
  weight: number;
  enabled: boolean;
};

export const SEED_SOURCES: SeedSource[] = [
  // ---- HackerNews (real points + comments for popularity) ----
  {
    sourceId: "hackernews",
    name: "Hacker News",
    kind: "hackernews",
    url: "topstories",
    topics: ["programming", "startups", "ai", "security"],
    weight: 0.9,
    enabled: true,
  },

  // ---- AI newsletters (RSS) ----
  {
    sourceId: "tldr-ai",
    name: "TLDR AI",
    kind: "newsletter",
    url: "https://tldr.tech/api/rss/ai",
    topics: ["ai", "llm", "ml"],
    weight: 0.85,
    enabled: true,
  },
  {
    sourceId: "ai-news-smol",
    name: "AI News (smol.ai)",
    kind: "newsletter",
    url: "https://news.smol.ai/rss.xml",
    topics: ["ai", "llm", "agents"],
    weight: 0.85,
    enabled: true,
  },
  {
    sourceId: "latent-space",
    name: "Latent Space",
    kind: "newsletter",
    url: "https://www.latent.space/feed",
    topics: ["ai", "llm", "agents", "startups"],
    weight: 0.8,
    enabled: true,
  },
  {
    sourceId: "the-rundown-ai",
    name: "The Rundown AI",
    kind: "newsletter",
    url: "https://rss.beehiiv.com/feeds/2R3C6Bt5wj.xml",
    topics: ["ai", "business"],
    weight: 0.75,
    enabled: true,
  },

  // ---- Subreddits (RSS) ----
  {
    sourceId: "r-machinelearning",
    name: "r/MachineLearning",
    kind: "reddit",
    url: "https://www.reddit.com/r/MachineLearning/.rss",
    topics: ["ml", "ai", "data"],
    weight: 0.7,
    enabled: true,
  },
  {
    sourceId: "r-localllama",
    name: "r/LocalLLaMA",
    kind: "reddit",
    url: "https://www.reddit.com/r/LocalLLaMA/.rss",
    topics: ["llm", "open-source", "ai"],
    weight: 0.7,
    enabled: true,
  },
  {
    sourceId: "r-artificial",
    name: "r/artificial",
    kind: "reddit",
    url: "https://www.reddit.com/r/artificial/.rss",
    topics: ["ai", "policy"],
    weight: 0.6,
    enabled: true,
  },
  {
    sourceId: "r-programming",
    name: "r/programming",
    kind: "reddit",
    url: "https://www.reddit.com/r/programming/.rss",
    topics: ["programming", "open-source"],
    weight: 0.6,
    enabled: true,
  },
  {
    sourceId: "r-startups",
    name: "r/startups",
    kind: "reddit",
    url: "https://www.reddit.com/r/startups/.rss",
    topics: ["startups", "business"],
    weight: 0.55,
    enabled: true,
  },

  // ---- Lab / company blogs (RSS) ----
  {
    sourceId: "openai-blog",
    name: "OpenAI Blog",
    kind: "rss",
    url: "https://openai.com/blog/rss.xml",
    topics: ["ai", "llm", "agents"],
    weight: 0.9,
    enabled: true,
  },
  {
    sourceId: "huggingface-blog",
    name: "Hugging Face Blog",
    kind: "rss",
    url: "https://huggingface.co/blog/feed.xml",
    topics: ["ai", "open-source", "ml"],
    weight: 0.8,
    enabled: true,
  },
  {
    sourceId: "deepmind-blog",
    name: "Google DeepMind",
    kind: "rss",
    url: "https://deepmind.google/blog/rss.xml",
    topics: ["ai", "science", "ml"],
    weight: 0.85,
    enabled: true,
  },

  // ---- General tech press (RSS) ----
  {
    sourceId: "techcrunch",
    name: "TechCrunch",
    kind: "rss",
    url: "https://techcrunch.com/feed/",
    topics: ["startups", "business", "ai"],
    weight: 0.7,
    enabled: true,
  },
  {
    sourceId: "the-verge",
    name: "The Verge",
    kind: "rss",
    url: "https://www.theverge.com/rss/index.xml",
    topics: ["hardware", "ai", "business"],
    weight: 0.65,
    enabled: true,
  },
  {
    sourceId: "ars-technica",
    name: "Ars Technica",
    kind: "rss",
    url: "https://feeds.arstechnica.com/arstechnica/index",
    topics: ["science", "security", "hardware"],
    weight: 0.7,
    enabled: true,
  },
  {
    sourceId: "venturebeat-ai",
    name: "VentureBeat AI",
    kind: "rss",
    url: "https://venturebeat.com/category/ai/feed/",
    topics: ["ai", "business"],
    weight: 0.6,
    enabled: true,
  },

  // ---- X / Twitter accounts (no free API: routed via RSS bridge, off by
  //      default; user supplies a working bridge URL in the dashboard) ----
  {
    sourceId: "x-openai",
    name: "X · @OpenAI",
    kind: "x",
    url: "https://nitter.net/OpenAI/rss",
    topics: ["ai", "llm"],
    weight: 0.6,
    enabled: false,
  },
  {
    sourceId: "x-karpathy",
    name: "X · @karpathy",
    kind: "x",
    url: "https://nitter.net/karpathy/rss",
    topics: ["ai", "ml", "llm"],
    weight: 0.7,
    enabled: false,
  },
  {
    sourceId: "x-sama",
    name: "X · @sama",
    kind: "x",
    url: "https://nitter.net/sama/rss",
    topics: ["ai", "startups", "business"],
    weight: 0.6,
    enabled: false,
  },
];
