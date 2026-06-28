import { AuthForm } from "@/components/AuthForm";
import Link from "next/link";

export default function SignInPage() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center px-6 py-16">
      <Link
        href="/"
        className="hud-title text-cyan-soft text-sm tracking-[0.3em] mb-8 hover:hud-glow-text transition-all"
      >
        ◈ HUD
      </Link>
      <AuthForm />
      <p className="text-[11px] text-ink-faint mt-6 max-w-sm text-center leading-relaxed">
        Your data is yours. BYO AI keys are encrypted at rest; nothing is shared
        between accounts.
      </p>
    </main>
  );
}
