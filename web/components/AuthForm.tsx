"use client";

import { useState } from "react";
import { useAuthActions } from "@convex-dev/auth/react";
import { useRouter } from "next/navigation";
import { Loader2, ScanLine } from "lucide-react";

export function AuthForm() {
  const { signIn } = useAuthActions();
  const router = useRouter();
  const [flow, setFlow] = useState<"signIn" | "signUp">("signIn");
  const [loading, setLoading] = useState<null | "password" | "guest">(null);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setLoading("password");
    const form = new FormData(e.currentTarget);
    form.set("flow", flow);
    try {
      await signIn("password", form);
      router.push("/feed");
    } catch {
      setError(
        flow === "signIn"
          ? "Invalid credentials. New here? Switch to Create account."
          : "Could not create account. Try a stronger password (8+ chars).",
      );
      setLoading(null);
    }
  }

  async function asGuest() {
    setError(null);
    setLoading("guest");
    try {
      await signIn("anonymous");
      router.push("/feed");
    } catch {
      setError("Guest sign-in failed.");
      setLoading(null);
    }
  }

  return (
    <div className="hud-panel hud-panel-glow w-full max-w-sm p-7">
      <div className="flex items-center gap-2 mb-1">
        <ScanLine className="w-4 h-4 text-cyan" />
        <span className="hud-label">Access terminal</span>
      </div>
      <h1 className="hud-title text-2xl font-semibold text-ink mb-5">
        {flow === "signIn" ? "Sign in" : "Create account"}
      </h1>

      <form onSubmit={onSubmit} className="space-y-3">
        <input
          name="email"
          type="email"
          required
          placeholder="you@signal.dev"
          className="hud-input"
          autoComplete="email"
        />
        <input
          name="password"
          type="password"
          required
          placeholder="password"
          className="hud-input"
          autoComplete={flow === "signIn" ? "current-password" : "new-password"}
        />
        {error && <p className="text-xs text-rose">{error}</p>}
        <button type="submit" disabled={loading !== null} className="hud-btn w-full">
          {loading === "password" ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : flow === "signIn" ? (
            "Enter"
          ) : (
            "Create"
          )}
        </button>
      </form>

      <button
        onClick={asGuest}
        disabled={loading !== null}
        className="mt-3 w-full text-center text-xs text-ink-dim hover:text-cyan-soft transition-colors py-2"
      >
        {loading === "guest" ? "Initializing…" : "› Enter as guest (instant demo)"}
      </button>

      <div className="hud-divider my-4" />
      <button
        onClick={() => {
          setError(null);
          setFlow(flow === "signIn" ? "signUp" : "signIn");
        }}
        className="w-full text-center text-xs text-ink-faint hover:text-ink-dim transition-colors"
      >
        {flow === "signIn"
          ? "No account? Create one →"
          : "Already have an account? Sign in →"}
      </button>
    </div>
  );
}
