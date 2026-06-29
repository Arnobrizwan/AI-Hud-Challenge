# Testing rules (HUD)

## Runner
- Vitest. `npm test` runs everything; tests live in `lib/**/__tests__/*.test.ts`.
- Tests import pipeline modules with **relative paths** (e.g. `../text`, `../../health`) — the `@/` tsconfig alias is NOT resolved by Vitest, so don't use it in tests.
- Don't import Convex `_generated` or hit the network in unit tests. Test the **pure** functions; keep Convex functions thin so the logic under test is pure.

## What to test
- Every new pure pipeline stage / helper gets a unit test.
- Pure logic extracted from route handlers / scripts (e.g. `lib/api.ts`, `lib/health.ts`) is tested directly; the thin I/O wrapper isn't.

## Quality gates (must stay green)
- `npm test` — full suite.
- `npm run eval:gate` — **enforced metric gate**: Precision@10 / nDCG@10 / DupF1 over the gold fixture via the real ranking + dedup code. Thresholds in `lib/pipeline/evalMetrics.ts`. CI fails the build if any metric regresses — keep fixtures comfortably above thresholds.
- `npm run build` — Next production build / typecheck.
- `npm run healthcheck` — post-deploy smoke test against the live `/api/feed`.

## CI/CD
- `.github/workflows/hud-web.yml`: install → lint → test → eval:gate → build → deploy Convex + Vercel on push to `main`. All four deploy secrets are set, so pushes auto-deploy. Keep CI green.
