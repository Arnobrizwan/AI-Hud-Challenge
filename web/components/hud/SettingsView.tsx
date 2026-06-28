"use client";

import { useEffect, useState } from "react";
import { useQuery, useMutation, useAction } from "convex/react";
import { api } from "@/convex/_generated/api";
import { cn } from "@/lib/utils";
import { KeyRound, Check, X, Loader2, Trash2, Sparkles } from "lucide-react";

const TOPICS = [
  "ai", "llm", "ml", "agents", "startups", "programming", "open-source",
  "security", "crypto", "science", "hardware", "robotics", "data", "business",
  "design", "policy",
];

export function SettingsView() {
  const prefs = useQuery(api.prefs.getPrefs);
  const updatePrefs = useMutation(api.prefs.updatePrefs);

  const [topics, setTopics] = useState<string[]>([]);
  const [resurfaceHours, setResurfaceHours] = useState(48);
  const [savedAt, setSavedAt] = useState(0);

  useEffect(() => {
    if (prefs) {
      setTopics(prefs.focusTopics);
      setResurfaceHours(prefs.bookmarkResurfaceHours);
    }
  }, [prefs]);

  function toggleTopic(t: string) {
    const next = topics.includes(t) ? topics.filter((x) => x !== t) : [...topics, t];
    setTopics(next);
    updatePrefs({ focusTopics: next, onboarded: true });
  }

  return (
    <div className="h-full overflow-y-auto hud-scroll px-4 sm:px-8 py-6">
      <div className="max-w-3xl mx-auto space-y-6">
        <header>
          <h1 className="hud-title text-2xl text-ink">Configuration</h1>
          <p className="text-ink-dim text-sm mt-1">
            Tune what the HUD prioritizes and plug in your own AI keys.
          </p>
        </header>

        {/* focus topics */}
        <section className="hud-panel p-5">
          <h2 className="hud-label mb-1">Focus topics</h2>
          <p className="text-ink-dim text-xs mb-3">
            Stories matching these get the <span className="text-cyan">Focus</span> lane and higher rank.
          </p>
          <div className="flex flex-wrap gap-2">
            {TOPICS.map((t) => (
              <button
                key={t}
                onClick={() => toggleTopic(t)}
                className={cn(
                  "px-3 py-1.5 rounded-lg text-xs font-semibold uppercase tracking-wider border transition-all",
                  topics.includes(t)
                    ? "text-cyan border-[var(--line-bright)] bg-[rgba(46,230,230,0.12)]"
                    : "text-ink-faint border-[var(--line)] hover:text-ink-dim",
                )}
              >
                {t}
              </button>
            ))}
          </div>
        </section>

        {/* BYO keys */}
        <ApiKeysSection />

        {/* bookmark resurfacing */}
        <section className="hud-panel p-5">
          <h2 className="hud-label mb-1">Bookmark resurfacing</h2>
          <p className="text-ink-dim text-xs mb-3">
            Saved items re-enter the stream after this many hours.
          </p>
          <div className="flex items-center gap-4">
            <input
              type="range" min={6} max={168} step={6} value={resurfaceHours}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10);
                setResurfaceHours(v);
              }}
              onMouseUp={() => {
                updatePrefs({ bookmarkResurfaceHours: resurfaceHours });
                setSavedAt(Date.now());
              }}
              className="hud-range flex-1 max-w-xs"
            />
            <span className="hud-title text-cyan text-sm">{resurfaceHours}h</span>
            {savedAt > 0 && <span className="text-green text-[10px]">saved</span>}
          </div>
        </section>

        <p className="text-ink-faint text-[11px] text-center pb-6">
          Stream speed and the focus↔trending mix live on the Feed control deck.
        </p>
      </div>
    </div>
  );
}

function ApiKeysSection() {
  const keys = useQuery(api.apiKeys.listKeys);
  const saveKey = useAction(api.apiKeys.saveKey);
  const deleteKey = useMutation(api.apiKeys.deleteKey);

  return (
    <section className="hud-panel hud-panel-glow p-5">
      <div className="flex items-center gap-2 mb-1">
        <KeyRound className="w-4 h-4 text-violet" />
        <h2 className="hud-label !text-violet-soft">Bring your own AI key</h2>
      </div>
      <p className="text-ink-dim text-xs mb-4">
        Optional. Adds <Sparkles className="inline w-3 h-3 text-violet" /> abstractive
        summaries. Keys are AES-GCM encrypted at rest and never leave the server.
        Without a key the HUD uses extractive summaries.
      </p>
      <div className="space-y-3">
        <KeyRow provider="openai" label="OpenAI" placeholder="sk-…" stored={keys?.find((k) => k.provider === "openai")} onSave={saveKey} onDelete={deleteKey} />
        <KeyRow provider="anthropic" label="Anthropic" placeholder="sk-ant-…" stored={keys?.find((k) => k.provider === "anthropic")} onSave={saveKey} onDelete={deleteKey} />
      </div>
    </section>
  );
}

type StoredKey = { provider: string; last4: string; valid: boolean; model: string | null } | undefined;

function KeyRow({
  provider,
  label,
  placeholder,
  stored,
  onSave,
  onDelete,
}: {
  provider: "openai" | "anthropic";
  label: string;
  placeholder: string;
  stored: StoredKey;
  onSave: (a: { provider: "openai" | "anthropic"; key: string }) => Promise<{ valid: boolean }>;
  onDelete: (a: { provider: "openai" | "anthropic" }) => Promise<unknown>;
}) {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<null | "ok" | "bad">(null);

  async function save() {
    if (!value.trim()) return;
    setBusy(true);
    setResult(null);
    try {
      const r = await onSave({ provider, key: value.trim() });
      setResult(r.valid ? "ok" : "bad");
      if (r.valid) setValue("");
    } catch {
      setResult("bad");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="w-20 text-xs text-ink-dim font-medium">{label}</span>
      {stored ? (
        <div className="flex items-center gap-2 flex-1">
          <span className={cn("hud-chip", stored.valid ? "" : "hud-chip-rose")}>
            {stored.valid ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
            ••••{stored.last4}
          </span>
          <span className="text-[10px] text-ink-faint">{stored.valid ? "validated" : "invalid"}</span>
          <button
            onClick={() => onDelete({ provider })}
            className="ml-auto p-1.5 rounded text-ink-faint hover:text-rose transition-colors"
            title="Remove key"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      ) : (
        <>
          <input
            type="password"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={placeholder}
            className="hud-input flex-1 min-w-[180px]"
          />
          <button onClick={save} disabled={busy || !value.trim()} className="hud-btn">
            {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : "Validate & save"}
          </button>
          {result === "ok" && <Check className="w-4 h-4 text-green" />}
          {result === "bad" && <X className="w-4 h-4 text-rose" />}
        </>
      )}
    </div>
  );
}
