"use client";

import { useEffect, useState } from "react";
import { useQuery, useMutation, useAction } from "convex/react";
import { api } from "@/convex/_generated/api";
import { cn, timeAgo } from "@/lib/utils";
import { Gauge } from "./Gauge";
import { OpsPanels } from "./OpsPanels";
import {
  Play, RefreshCw, Activity, Database, Layers, GitMerge, Loader2,
  CheckCircle2, AlertTriangle, FlaskConical, Lock,
} from "lucide-react";

const WEIGHT_KEYS = [
  "recency", "sourceWeight", "topicalMatch", "novelty", "velocity", "popularity",
] as const;

/** Admin gate: the operator console mutates global state, so it's admins-only. */
export function DashboardView() {
  const me = useQuery(api.users.currentUser);
  if (me === undefined) {
    return <div className="flex items-center justify-center h-full hud-label">checking access…</div>;
  }
  if (!me?.isAdmin) return <Restricted />;
  return <Console />;
}

function Restricted() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center gap-3 px-6">
      <Lock className="w-8 h-8 text-ink-faint" />
      <h1 className="hud-title text-xl text-ink">Operator console — restricted</h1>
      <p className="text-ink-dim text-sm max-w-sm">
        The pipeline console tunes global ranking, sources, and experiments, so
        it&apos;s limited to admins. Your feed, bookmarks, and config are all on
        the other tabs.
      </p>
    </div>
  );
}

function Console() {
  const overview = useQuery(api.dashboard.overview);
  const config = useQuery(api.config.getConfig);
  const sources = useQuery(api.sources.listSources);
  const evals = useQuery(api.evaluation.listEvals);
  const updateConfig = useMutation(api.config.updateConfig);
  const toggleSource = useMutation(api.sources.toggleSource);
  const triggerRun = useAction(api.pipeline.triggerRun);
  const runEval = useMutation(api.evaluation.runEval);

  const [running, setRunning] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [weights, setWeights] = useState<Record<string, number>>({});
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    if (config) setWeights({ ...config.weights });
  }, [config]);

  async function onRun() {
    setRunning(true);
    try {
      await triggerRun();
    } finally {
      setRunning(false);
    }
  }
  async function onEval() {
    setEvaluating(true);
    try {
      await runEval({ k: 10 });
    } finally {
      setEvaluating(false);
    }
  }
  async function applyWeights() {
    await updateConfig({
      weights: {
        recency: weights.recency, sourceWeight: weights.sourceWeight,
        topicalMatch: weights.topicalMatch, novelty: weights.novelty,
        velocity: weights.velocity, popularity: weights.popularity,
      },
    });
    setDirty(false);
  }

  const latestEval = evals?.[0];

  return (
    <div className="h-full overflow-y-auto hud-scroll px-4 sm:px-8 py-6">
      <div className="max-w-5xl mx-auto space-y-6">
        <header className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h1 className="hud-title text-2xl text-ink">Pipeline console</h1>
            <p className="text-ink-dim text-sm mt-1">Ingest → enrich → dedup → rank → summarize — observed and tunable.</p>
          </div>
          <div className="flex gap-2">
            <button onClick={onEval} disabled={evaluating} className="hud-btn !border-[rgba(160,107,255,0.5)] !text-violet-soft !bg-[rgba(160,107,255,0.1)]">
              {evaluating ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <FlaskConical className="w-3.5 h-3.5" />}
              Run eval
            </button>
            <button onClick={onRun} disabled={running} className="hud-btn">
              {running ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
              Run pipeline
            </button>
          </div>
        </header>

        {/* stat tiles */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatTile icon={Database} label="Signals (48h)" value={overview?.itemCount ?? "—"} />
          <StatTile icon={Layers} label="Event clusters" value={overview?.clusterCount ?? "—"} />
          <StatTile icon={GitMerge} label="Multi-source" value={overview?.multiSourceClusters ?? "—"} sub="deduped events" />
          <StatTile icon={Activity} label="Sources live" value={overview ? `${overview.sourcesEnabled}/${overview.sourcesTotal}` : "—"} />
        </div>

        {/* eval metrics */}
        <section className="hud-panel p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="hud-label">Evaluation · P@10 / nDCG / coverage</h2>
            {latestEval && <span className="text-[10px] text-ink-faint">last run {timeAgo(latestEval.createdAt)} ago · n={latestEval.sampleSize}</span>}
          </div>
          {latestEval ? (
            <div className="flex flex-wrap gap-6 justify-around">
              <Gauge value={latestEval.metrics.precisionAtK} label="P@10" size={62} />
              <Gauge value={latestEval.metrics.ndcgAtK} label="nDCG" size={62} color="var(--violet)" />
              <Gauge value={latestEval.metrics.coverage} label="Coverage" size={62} />
              <Gauge value={latestEval.metrics.novelty} label="Novelty" size={62} color="var(--violet)" />
              <Gauge value={latestEval.metrics.diversity} label="Diversity" size={62} />
              <Gauge value={latestEval.metrics.dupF1} label="Dup F1" size={62} color="var(--violet)" />
            </div>
          ) : (
            <p className="text-ink-dim text-sm">No eval runs yet — hit <span className="text-violet-soft">Run eval</span>.</p>
          )}
        </section>

        {/* ranking weights */}
        <section className="hud-panel p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="hud-label">Ranking weights · hot-reload</h2>
            <button onClick={applyWeights} disabled={!dirty} className="hud-btn !py-1.5 !px-3">
              <RefreshCw className="w-3.5 h-3.5" /> Apply
            </button>
          </div>
          <div className="grid sm:grid-cols-2 gap-x-8 gap-y-3">
            {WEIGHT_KEYS.map((k) => (
              <div key={k} className="flex items-center gap-3">
                <span className="hud-label !text-[9px] w-24 shrink-0">{k}</span>
                <input
                  type="range" min={0} max={0.5} step={0.01}
                  value={weights[k] ?? 0}
                  onChange={(e) => {
                    setWeights((w) => ({ ...w, [k]: parseFloat(e.target.value) }));
                    setDirty(true);
                  }}
                  className="hud-range flex-1"
                />
                <span className="text-[11px] text-cyan-soft w-9 text-right tabular-nums">
                  {(weights[k] ?? 0).toFixed(2)}
                </span>
              </div>
            ))}
          </div>
          <p className="text-ink-faint text-[10px] mt-3">
            Changes take effect on the next feed read — no redeploy. The feed blends these by your focus↔trending mix.
          </p>
        </section>

        {/* ops: alerts, drift, config registry, A/B canary, labeling */}
        <OpsPanels />

        {/* analytics: top sources + topics */}
        <div className="grid md:grid-cols-2 gap-4">
          <section className="hud-panel p-5">
            <h2 className="hud-label mb-3">Top sources (48h)</h2>
            <BarList items={(overview?.topSources ?? []).map((s) => ({ label: s.name, value: s.count }))} color="var(--cyan)" />
          </section>
          <section className="hud-panel p-5">
            <h2 className="hud-label mb-3">Topic distribution</h2>
            <BarList items={(overview?.topicDist ?? []).map((s) => ({ label: s.topic, value: s.count }))} color="var(--violet)" />
          </section>
        </div>

        {/* sources health */}
        <section className="hud-panel p-5">
          <h2 className="hud-label mb-3">Sources</h2>
          <div className="space-y-1.5">
            {(sources ?? []).map((s) => (
              <div key={s.sourceId} className="flex items-center gap-3 py-1.5 border-b border-[var(--line)] last:border-0">
                <button
                  onClick={() => toggleSource({ sourceId: s.sourceId, enabled: !s.enabled })}
                  className={cn(
                    "w-9 h-5 rounded-full relative transition-colors shrink-0",
                    s.enabled ? "bg-[rgba(46,230,230,0.4)]" : "bg-[var(--line)]",
                  )}
                >
                  <span className={cn("absolute top-0.5 w-4 h-4 rounded-full bg-ink transition-all", s.enabled ? "left-[18px]" : "left-0.5")} />
                </button>
                <span className="text-xs text-ink font-medium w-44 truncate">{s.name}</span>
                <span className="text-[9px] text-ink-faint uppercase tracking-wide w-20">{s.kind}</span>
                <div className="flex items-center gap-2 ml-auto text-[10px]">
                  {s.lastSuccessAt ? (
                    <span className="flex items-center gap-1 text-green"><CheckCircle2 className="w-3 h-3" /> {s.successCount}</span>
                  ) : s.errorCount > 0 ? (
                    <span className="flex items-center gap-1 text-rose" title={s.lastError}><AlertTriangle className="w-3 h-3" /> {s.lastError ?? "err"}</span>
                  ) : (
                    <span className="text-ink-faint">idle</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* pipeline runs */}
        <section className="hud-panel p-5">
          <h2 className="hud-label mb-3">Recent runs</h2>
          <div className="space-y-2">
            {(overview?.runs ?? []).slice(0, 8).map((r) => (
              <div key={r._id} className="flex items-center gap-3 text-xs py-1.5 border-b border-[var(--line)] last:border-0">
                <span className={cn("hud-chip", r.status === "ok" ? "" : r.status === "error" ? "hud-chip-rose" : "hud-chip-amber")}>
                  {r.status}
                </span>
                <span className="text-ink-faint">{r.trigger}</span>
                <span className="text-ink-dim">{timeAgo(r.startedAt)} ago</span>
                {r.durationMs != null && <span className="text-ink-faint">{(r.durationMs / 1000).toFixed(1)}s</span>}
                <span className="ml-auto text-ink-dim tabular-nums">
                  +{r.counts.inserted} new · {r.counts.duplicates} dup · {r.counts.clusters} clusters
                </span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

function StatTile({ icon: Icon, label, value, sub }: { icon: React.ElementType; label: string; value: React.ReactNode; sub?: string }) {
  return (
    <div className="hud-panel hud-clip p-4">
      <Icon className="w-4 h-4 text-cyan mb-2" />
      <div className="hud-title text-2xl text-ink">{value}</div>
      <div className="hud-label !text-[8px] mt-1">{label}</div>
      {sub && <div className="text-[9px] text-ink-faint mt-0.5">{sub}</div>}
    </div>
  );
}

function BarList({ items, color }: { items: { label: string; value: number }[]; color: string }) {
  const max = Math.max(1, ...items.map((i) => i.value));
  if (items.length === 0) return <p className="text-ink-faint text-xs">no data yet</p>;
  return (
    <div className="space-y-1.5">
      {items.map((it) => (
        <div key={it.label} className="flex items-center gap-2">
          <span className="text-[10px] text-ink-dim w-28 truncate shrink-0">{it.label}</span>
          <div className="flex-1 h-2 rounded-full bg-[var(--line)] overflow-hidden">
            <div className="h-full rounded-full" style={{ width: `${(it.value / max) * 100}%`, background: color }} />
          </div>
          <span className="text-[10px] text-ink-faint w-6 text-right tabular-nums">{it.value}</span>
        </div>
      ))}
    </div>
  );
}
