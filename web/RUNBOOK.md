# Runbook & ADRs

## On-call runbook (section 15)

Alerts surface in the dashboard (`alerts` table, `mlops.listAlerts`) and from `pipelineRuns.status="error"`.

| Symptom | Likely cause | Remediation |
|---|---|---|
| Pipeline run `error` | adapter/parse failure | Dashboard → Recent runs → inspect failing stage; check `sources` health for the offending source; disable it from the dashboard. |
| A source shows red / high errorCount | feed moved / 429 rate-limit | Fix URL via dashboard, or it auto-downgrades (`autoDowngradeSources`). Reddit/X 429 = datacenter IP; needs OAuth/bridge. |
| `data_quality` alert | empty titles / no topics / non-en spike | Inspect recent items; tighten ingest filters; check a parser regression. |
| `drift` alert (JS > 0.15) | topic distribution shifted (event flood) | Expected during big news events; verify not a classifier bug; diversity cap prevents single-event domination. |
| Feed empty | cron not running / all items aged out | `npx convex run pipeline:runPipeline '{"trigger":"manual"}'`; check `crons.ts` is deployed. |
| Ranking looks off | bad config edit | Dashboard → roll back via `mlops.rollbackTo({version})` to a known-good `configVersions` snapshot. |
| Bad summary | LLM hallucination | Grounding check (`isGrounded`) already filters; report via labeling UI (`summary_factual`). |

**Rollback (one-click):** `mlops.rollbackTo({ version })` copies a prior `configVersions` snapshot into the live config — no redeploy.

## ADRs (Architecture Decision Records, section 16)

- **ADR-1 · Convex over the Python microservices.** Brief requires React/Next.js on Vercel. Rebuilt the pipeline in TypeScript on Convex (reactive DB + actions + cron) for one Vercel-native deploy, live queries, and per-user isolation. Python services kept as reference.
- **ADR-2 · Read-time ranking.** Personalized scores computed when the feed is read, so pref/weight/mix changes apply instantly. Batch-global features (popularity, velocity) precomputed once per ingest.
- **ADR-3 · MinHash/LSH + SimHash + incremental.** MinHash/LSH for in-batch near-dup; SimHash Hamming for cross-batch incremental clustering into persistent events.
- **ADR-4 · BYO keys, AES-GCM.** Summaries are extractive by default; abstractive uses the user's encrypted key. App never hard-depends on an LLM.
- **ADR-5 · Hashing-trick vectors over an embedding service.** A 64-dim hashing vector gives a dependency-free semantic signal in the V8 runtime; swap for real embeddings (BYO key) later without changing callers.
- **ADR-6 · Advisory reference-CI.** The Python project's workflows are disabled; `HUD Web · CI/CD` is the strict gate (typecheck + build + deploy) for the deliverable.

## Privacy / usage policy (section 0)

- Stored: article **title, summary, URL, metadata, topics, entities** (no full paywalled bodies). BYO keys stored **AES-GCM ciphertext only**.
- Retention: items + clusters pruned after **5 days** (`pruneOld`). Per-user erasure: `account.deleteMyAccount` (GDPR/CCPA) removes all user data + account.
- Sources: respect robots/terms via conditional GET, rate-limit + backoff, per-source budgets; only feeds/APIs the publisher exposes.
