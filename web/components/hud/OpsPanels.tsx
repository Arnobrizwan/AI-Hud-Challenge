"use client";

import { useState } from "react";
import { useQuery, useMutation } from "convex/react";
import { api } from "@/convex/_generated/api";
import type { Id } from "@/convex/_generated/dataModel";
import { cn, timeAgo } from "@/lib/utils";
import { AlertTriangle, GitBranch, Activity, Tag, FlaskConical, RotateCcw, Camera, Check, X } from "lucide-react";

export function OpsPanels() {
  return (
    <div className="grid md:grid-cols-2 gap-4">
      <AlertsPanel />
      <DriftPanel />
      <ConfigRegistryPanel />
      <ExperimentsPanel />
      <LabelingPanel />
    </div>
  );
}

function Panel({ icon: Icon, title, children }: { icon: React.ElementType; title: string; children: React.ReactNode }) {
  return (
    <section className="hud-panel p-5">
      <div className="flex items-center gap-2 mb-3">
        <Icon className="w-4 h-4 text-cyan" />
        <h2 className="hud-label">{title}</h2>
      </div>
      {children}
    </section>
  );
}

function AlertsPanel() {
  const alerts = useQuery(api.mlops.listAlerts);
  const resolve = useMutation(api.mlops.resolveAlert);
  const open = (alerts ?? []).filter((a) => !a.resolved);
  return (
    <Panel icon={AlertTriangle} title="Alerts">
      {open.length === 0 ? (
        <p className="text-ink-faint text-xs">no open alerts</p>
      ) : (
        <div className="space-y-1.5">
          {open.slice(0, 6).map((a) => (
            <div key={a._id} className="flex items-center gap-2 text-xs">
              <span className={cn("hud-chip", a.severity === "critical" ? "hud-chip-rose" : a.severity === "warn" ? "hud-chip-amber" : "")}>{a.type}</span>
              <span className="text-ink-dim truncate flex-1">{a.message}</span>
              <span className="text-ink-faint">{timeAgo(a.createdAt)}</span>
              <button onClick={() => resolve({ id: a._id as Id<"alerts"> })} className="text-ink-faint hover:text-green" title="Resolve"><Check className="w-3.5 h-3.5" /></button>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

function DriftPanel() {
  const drift = useQuery(api.mlops.driftHistory);
  const latest = drift?.[0];
  const max = Math.max(0.0001, ...(drift ?? []).map((d) => d.divergence));
  return (
    <Panel icon={Activity} title="Topic drift (JS divergence)">
      {!latest ? (
        <p className="text-ink-faint text-xs">no snapshots yet</p>
      ) : (
        <>
          <div className="flex items-end gap-1 h-12 mb-2">
            {[...(drift ?? [])].reverse().map((d) => (
              <div key={d._id} className="flex-1 rounded-t" style={{ height: `${(d.divergence / max) * 100}%`, minHeight: 2, background: d.divergence > 0.15 ? "var(--rose)" : "var(--cyan)" }} title={d.divergence.toFixed(3)} />
            ))}
          </div>
          <p className="text-[11px] text-ink-dim">latest JS={latest.divergence.toFixed(3)} {latest.divergence > 0.15 ? "· shifted" : "· stable"}</p>
        </>
      )}
    </Panel>
  );
}

function ConfigRegistryPanel() {
  const versions = useQuery(api.mlops.listVersions);
  const snapshot = useMutation(api.mlops.snapshotVersion);
  const rollback = useMutation(api.mlops.rollbackTo);
  return (
    <Panel icon={GitBranch} title="Config registry">
      <button onClick={() => snapshot({ note: "manual snapshot" })} className="hud-btn !py-1.5 !px-3 mb-3">
        <Camera className="w-3.5 h-3.5" /> Snapshot live config
      </button>
      {(versions ?? []).length === 0 ? (
        <p className="text-ink-faint text-xs">no versions — snapshot to create v1</p>
      ) : (
        <div className="space-y-1">
          {(versions ?? []).slice(0, 6).map((v) => (
            <div key={v._id} className="flex items-center gap-2 text-xs py-1 border-b border-[var(--line)] last:border-0">
              <span className="hud-chip">v{v.version}</span>
              <span className="text-ink-faint truncate flex-1">{v.note ?? ""} · {timeAgo(v.createdAt)}</span>
              <button onClick={() => rollback({ version: v.version })} className="text-ink-faint hover:text-cyan flex items-center gap-1" title="Rollback to this version">
                <RotateCcw className="w-3 h-3" /> rollback
              </button>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

function ExperimentsPanel() {
  const exps = useQuery(api.mlops.listExperiments);
  const versions = useQuery(api.mlops.listVersions);
  const start = useMutation(api.mlops.startCanary);
  const stop = useMutation(api.mlops.stopCanary);
  const [pct, setPct] = useState(20);
  return (
    <Panel icon={FlaskConical} title="A/B canary">
      <div className="flex items-center gap-2 mb-3">
        <input type="range" min={5} max={50} step={5} value={pct} onChange={(e) => setPct(parseInt(e.target.value, 10))} className="hud-range flex-1" />
        <span className="text-[11px] text-cyan-soft w-10">{pct}%</span>
        <button
          onClick={() => start({ name: `canary-${pct}`, variantVersion: versions?.[0]?.version ?? 1, trafficPct: pct })}
          disabled={!versions?.length}
          className="hud-btn !py-1.5 !px-3"
        >
          Start
        </button>
      </div>
      {(exps ?? []).length === 0 ? (
        <p className="text-ink-faint text-xs">no experiments (snapshot a config first)</p>
      ) : (
        <div className="space-y-1">
          {(exps ?? []).slice(0, 4).map((e) => (
            <div key={e._id} className="flex items-center gap-2 text-xs py-1">
              <span className={cn("hud-chip", e.status === "running" ? "" : "hud-chip-amber")}>{e.status}</span>
              <span className="text-ink-dim flex-1 truncate">{e.name} · {e.trafficPct}% → v{e.variantVersion}</span>
              {e.status === "running" && <button onClick={() => stop({ id: e._id as Id<"experiments"> })} className="text-ink-faint hover:text-rose"><X className="w-3.5 h-3.5" /></button>}
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

function LabelingPanel() {
  const pair = useQuery(api.labels.nextDupPair);
  const stats = useQuery(api.labels.trainingSet);
  const submit = useMutation(api.labels.submit);
  return (
    <Panel icon={Tag} title="Labeling · training set">
      {stats && <p className="text-[11px] text-ink-faint mb-2">{stats.total} labels collected</p>}
      {!pair ? (
        <p className="text-ink-faint text-xs">no dup-pair to label right now</p>
      ) : (
        <div className="space-y-2">
          <p className="text-[11px] text-ink-dim">Same event? <span className="text-ink">A:</span> {pair.a.title.slice(0, 50)} · <span className="text-ink">B:</span> {pair.b.title.slice(0, 50)}</p>
          <div className="flex gap-2">
            {["yes", "no", "unsure"].map((label) => (
              <button key={label} onClick={() => submit({ kind: "dup_pair", itemId: pair.a.id as Id<"items">, otherItemId: pair.b.id as Id<"items">, label })} className="hud-btn !py-1.5 !px-3 capitalize">
                {label}
              </button>
            ))}
          </div>
        </div>
      )}
    </Panel>
  );
}
