/**
 * Versioned, type-specific, hot-swappable prompt registry.
 *
 * (Production-AI-app pattern: prompts/templates + registry.) Every LLM prompt
 * lives here keyed by a stable id, with multiple versions retained for audit and
 * A/B; an ACTIVE map selects which version is live so prompts can be swapped
 * without touching call sites. `renderPrompt` does {{var}} interpolation.
 */

export interface PromptTemplate {
  id: string;
  version: number;
  text: string;
  note?: string;
}

const REGISTRY: Record<string, PromptTemplate[]> = {
  "summarize.system": [
    {
      id: "summarize.system",
      version: 1,
      note: "initial",
      text:
        "Summarize the news item in one short factual sentence. Use only the " +
        "provided text. Output only the sentence.",
    },
    {
      id: "summarize.system",
      version: 2,
      note: "tightened grounding + length",
      text:
        "You compress a news item into a single, factual teaser of 18-32 words. " +
        "Rules: use ONLY facts present in the provided text; never speculate or add " +
        "details; no opinions; no hashtags; no clickbait; preserve named entities and " +
        "numbers exactly. Output only the teaser sentence.",
    },
  ],
  "summarize.user": [
    {
      id: "summarize.user",
      version: 1,
      text: "TITLE: {{title}}\n\nTEXT:\n{{text}}",
    },
  ],
};

/** Live version per prompt id (the "hot-swappable" selector). */
const ACTIVE: Record<string, number> = {
  "summarize.system": 2,
  "summarize.user": 1,
};

/** Fetch a prompt template; defaults to the ACTIVE version for its id. */
export function getPrompt(id: string, version?: number): PromptTemplate {
  const versions = REGISTRY[id];
  if (!versions || versions.length === 0) throw new Error(`unknown prompt: ${id}`);
  const v = version ?? ACTIVE[id] ?? versions[versions.length - 1].version;
  const found = versions.find((t) => t.version === v);
  if (!found) throw new Error(`prompt ${id} has no version ${v}`);
  return found;
}

/** All retained versions of a prompt id (for audit / diffing). */
export function listPromptVersions(id: string): PromptTemplate[] {
  return REGISTRY[id] ?? [];
}

/** Render a template with {{var}} substitution; missing vars become "". */
export function renderPrompt(
  id: string,
  vars: Record<string, string> = {},
  version?: number,
): string {
  return getPrompt(id, version).text.replace(/\{\{\s*(\w+)\s*\}\}/g, (_, k) => vars[k] ?? "");
}
