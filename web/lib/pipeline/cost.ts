/**
 * LLM / embedding cost estimation (observability: cost tracking).
 *
 * Pure pricing table + estimators so the dashboard can show roughly what the
 * BYO-key AI calls cost. Prices are USD per 1,000,000 tokens and are ESTIMATES
 * (provider list prices, configurable). Token counts are approximated as
 * chars/4 when exact usage isn't available.
 */

export interface Price {
  inputPerM: number; // USD per 1M input tokens
  outputPerM: number; // USD per 1M output tokens
}

/** Keyed by `${provider}:${model}`. Values are list-price estimates. */
export const PRICING: Record<string, Price> = {
  "openai:gpt-4o-mini": { inputPerM: 0.15, outputPerM: 0.6 },
  "openai:text-embedding-3-small": { inputPerM: 0.02, outputPerM: 0 },
  "anthropic:claude-3-5-haiku-latest": { inputPerM: 0.8, outputPerM: 4 },
};

/** Rough token estimate from text (≈ 4 chars/token). */
export function estimateTokens(text: string | undefined | null): number {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

export interface TokenUsage {
  inputTokens: number;
  outputTokens?: number;
}

/**
 * Estimated USD for one call. Unknown (provider, model) returns 0 — callers can
 * treat 0 as "unpriced" rather than free if they need to flag it.
 */
export function estimateCostUSD(provider: string, model: string, usage: TokenUsage): number {
  const price = PRICING[`${provider}:${model}`];
  if (!price) return 0;
  const input = (usage.inputTokens / 1_000_000) * price.inputPerM;
  const output = ((usage.outputTokens ?? 0) / 1_000_000) * price.outputPerM;
  return input + output;
}

/** Is a (provider, model) pair in the pricing table? */
export function isPriced(provider: string, model: string): boolean {
  return !!PRICING[`${provider}:${model}`];
}
