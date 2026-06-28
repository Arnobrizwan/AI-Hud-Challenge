import { extractiveSummary } from "./text";

/**
 * Stage 6 — summarization.
 *  - Extractive teaser: always available, zero-cost, no key required.
 *  - Abstractive (optional): uses a BYO OpenAI/Anthropic key with guardrails:
 *      source-grounding, length control, no-speculation instruction, and a
 *      post-check that the summary doesn't invent named entities.
 * If anything fails we fall back to extractive — the app never hard-depends on
 * an LLM key.
 */

export type Provider = "openai" | "anthropic";

const DEFAULT_MODELS: Record<Provider, string> = {
  openai: "gpt-4o-mini",
  anthropic: "claude-3-5-haiku-latest",
};

const SYSTEM_PROMPT =
  "You compress a news item into a single, factual teaser of 18-32 words. " +
  "Rules: use ONLY facts present in the provided text; never speculate or add " +
  "details; no opinions; no hashtags; no clickbait; preserve named entities and " +
  "numbers exactly. Output only the teaser sentence.";

export function buildExtractive(title: string, text: string): string {
  const body = text && text.length > 40 ? text : title;
  return extractiveSummary(body, 180);
}

/** Reject a summary that introduces capitalized entities not in the source. */
export function isGrounded(summary: string, sourceText: string): boolean {
  const src = sourceText.toLowerCase();
  const caps = summary.match(/\b[A-Z][a-zA-Z]{3,}\b/g) ?? [];
  let invented = 0;
  for (const c of caps) {
    const w = c.toLowerCase();
    if (["this", "that", "these", "with", "from", "after", "while"].includes(w)) continue;
    if (!src.includes(w)) invented++;
  }
  return invented <= 1; // tolerate 1 (e.g. sentence-initial word)
}

async function callOpenAI(key: string, model: string, prompt: string): Promise<string> {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
    body: JSON.stringify({
      model,
      temperature: 0.2,
      max_tokens: 90,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: prompt },
      ],
    }),
  });
  if (!res.ok) throw new Error(`openai ${res.status}`);
  const data = (await res.json()) as { choices?: { message?: { content?: string } }[] };
  return (data.choices?.[0]?.message?.content ?? "").trim();
}

async function callAnthropic(key: string, model: string, prompt: string): Promise<string> {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": key,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model,
      max_tokens: 120,
      temperature: 0.2,
      system: SYSTEM_PROMPT,
      messages: [{ role: "user", content: prompt }],
    }),
  });
  if (!res.ok) throw new Error(`anthropic ${res.status}`);
  const data = (await res.json()) as { content?: { text?: string }[] };
  return (data.content?.[0]?.text ?? "").trim();
}

export interface AbstractiveInput {
  provider: Provider;
  key: string;
  model?: string | null;
  title: string;
  text: string;
}

/** Returns a grounded abstractive summary, or null on failure/ungrounded. */
export async function abstractiveSummary(input: AbstractiveInput): Promise<string | null> {
  const model = input.model || DEFAULT_MODELS[input.provider];
  const source = (input.text || "").slice(0, 2000);
  const prompt = `TITLE: ${input.title}\n\nTEXT:\n${source || input.title}`;
  try {
    const out =
      input.provider === "openai"
        ? await callOpenAI(input.key, model, prompt)
        : await callAnthropic(input.key, model, prompt);
    const clean = out.replace(/^["']|["']$/g, "").trim();
    if (!clean || clean.length < 10) return null;
    if (!isGrounded(clean, (source || input.title) + " " + input.title)) return null;
    return clean.length > 280 ? extractiveSummary(clean, 240) : clean;
  } catch {
    return null;
  }
}
