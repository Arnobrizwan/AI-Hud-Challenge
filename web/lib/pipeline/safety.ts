/**
 * Stage 12 — content safety. Lightweight profanity/NSFW keyword filter applied
 * to titles + summaries so flagged items can be down-ranked / hidden. Not a
 * full classifier — a deterministic guard with no external calls.
 */
const BLOCK = [
  "porn", "nsfw", "xxx", "nude", "naked", "sex tape", "onlyfans", "escort",
  "viagra", "casino", "betting odds", "crypto giveaway", "free bitcoin",
  "miracle cure", "get rich quick",
];

const PROFANITY = ["fuck", "shit", "bitch", "asshole", "cunt"];

export function isFlagged(title: string, text: string): boolean {
  const hay = (title + " " + (text || "")).toLowerCase();
  for (const b of BLOCK) if (hay.includes(b)) return true;
  // profanity only flags if it dominates (spammy), not incidental
  let prof = 0;
  for (const p of PROFANITY) if (hay.includes(p)) prof++;
  return prof >= 2;
}
