# Code style & architecture rules (HUD)

Context for AI coding agents working in `web/`. See also `web/AGENTS.md` (Next.js 16 has breaking changes — read `node_modules/next/dist/docs/` before writing Next code).

## Architecture
- **Pipeline stages are pure TypeScript** in `lib/pipeline/*` with typed input/output contracts (`types.ts`). They MUST stay decoupled from Convex/the app — no Convex imports inside `lib/pipeline/`. Convex actions in `convex/*` orchestrate and persist; the Next app only *reads* reactively.
- Each stage is a pure function: `ingest → normalize → enrich → dedup → rank → summarize`. Add a new stage as a pure function + call it from `convex/pipeline.ts`.
- **Prompts** live in the versioned registry `lib/pipeline/prompts.ts` (never hardcode prompts at call sites). Add a new version; flip `ACTIVE` to hot-swap.
- **Cost** estimation goes through `lib/pipeline/cost.ts` (pricing table). Don't sprinkle pricing constants elsewhere.
- **Security guards** (production-AI-app pattern): input = `ingest` robots + `normalize`; content = `safety.isFlagged`; output = `summarize.isGrounded`.

## Conventions
- TypeScript strict; prefer pure, side-effect-free functions; explicit return types on exported fns.
- Convex: gate every global-state operator mutation/action with `requireAdmin` (`convex/authz.ts`). Internal-only functions use `internalQuery`/`internalMutation`. `convex/_generated` is committed.
- Runtime: Convex actions run in a V8 isolate — `fetch` + Web Crypto OK; no Node APIs unless `"use node"`.
- Keep comments at the density of the surrounding file; explain *why*, not *what*.
- After code changes: push to GitHub and keep README/docs current.

## kluster
- kluster review tooling may be unavailable (connection error). When unavailable, rely on `npm test` + `npm run build` + `npx convex dev --once` + a live check instead, and say so.
