import { describe, it, expect } from "vitest";
import { setPRF, evaluateNerTopics, NER_TOPIC_GOLD } from "../nerEval";

describe("NER / topic measured evaluation", () => {
  it("setPRF counts tp/fp/fn case-insensitively", () => {
    expect(setPRF(["OpenAI", "Google"], ["openai", "nvidia"])).toEqual({ tp: 1, fp: 1, fn: 1 });
    expect(setPRF([], ["a"])).toEqual({ tp: 0, fp: 0, fn: 1 });
    expect(setPRF(["a"], [])).toEqual({ tp: 0, fp: 1, fn: 0 });
  });

  it("evaluates the labeled set with non-trivial precision/recall", () => {
    const r = evaluateNerTopics(NER_TOPIC_GOLD);
    expect(r.sampleSize).toBe(NER_TOPIC_GOLD.length);
    // The known-org dictionary should recover the labeled entities reasonably.
    expect(r.entity.recall).toBeGreaterThanOrEqual(0.7);
    expect(r.entity.precision).toBeGreaterThan(0.4);
    // Topic classifier should hit most labeled topics.
    expect(r.topic.recall).toBeGreaterThanOrEqual(0.6);
    expect(r.topic.f1).toBeGreaterThan(0.5);
  });

  it("returns perfect scores for an exactly-matching doc", () => {
    // The title-caps heuristic also surfaces "GPT", so it's part of the match.
    const r = evaluateNerTopics([
      { title: "OpenAI ships GPT model", text: "OpenAI ships a new gpt language model", entities: ["OpenAI", "GPT"], topics: ["ai", "llm"] },
    ]);
    expect(r.entity.f1).toBe(1);
  });
});
