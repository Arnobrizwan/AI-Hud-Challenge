import { cronJobs } from "convex/server";
import { internal } from "./_generated/api";

const crons = cronJobs();

// Pull + rank fresh content every 20 minutes (well under HN/Reddit rate limits).
crons.interval(
  "ingest pipeline",
  { minutes: 20 },
  internal.pipeline.runPipeline,
  { trigger: "cron" },
);

export default crons;
