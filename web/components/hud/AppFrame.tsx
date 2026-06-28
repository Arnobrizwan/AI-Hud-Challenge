"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useQuery } from "convex/react";
import { useAuthActions } from "@convex-dev/auth/react";
import { api } from "@/convex/_generated/api";
import { cn } from "@/lib/utils";
import { Radar, Bookmark, SlidersHorizontal, Gauge, LogOut, Activity, ShieldCheck } from "lucide-react";

const NAV = [
  { href: "/feed", label: "Feed", icon: Radar, adminOnly: false },
  { href: "/bookmarks", label: "Saved", icon: Bookmark, adminOnly: false },
  { href: "/dashboard", label: "Pipeline", icon: Gauge, adminOnly: true },
  { href: "/settings", label: "Config", icon: SlidersHorizontal, adminOnly: false },
];

export function AppFrame({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { signOut } = useAuthActions();
  const user = useQuery(api.users.currentUser);
  const stats = useQuery(api.feed.getFeedStats);

  return (
    <div className="flex flex-1 min-h-0">
      {/* nav rail */}
      <nav className="hidden sm:flex flex-col items-stretch w-[78px] shrink-0 border-r border-[var(--line)] py-5 px-2 gap-1">
        <Link href="/feed" className="flex flex-col items-center mb-5 group">
          <span className="hud-title text-cyan text-lg group-hover:hud-glow-text transition-all">
            ◈
          </span>
          <span className="hud-label !text-[8px] mt-1">HUD</span>
        </Link>
        {NAV.filter((n) => !n.adminOnly || user?.isAdmin).map((n) => {
          const active = pathname.startsWith(n.href);
          return (
            <Link
              key={n.href}
              href={n.href}
              className={cn(
                "flex flex-col items-center gap-1 py-2.5 rounded-lg transition-all",
                active
                  ? "bg-[rgba(46,230,230,0.1)] text-cyan border border-[var(--line-bright)]"
                  : "text-ink-faint hover:text-ink-dim border border-transparent",
              )}
            >
              <n.icon className="w-4.5 h-4.5" style={{ width: 18, height: 18 }} />
              <span className="text-[9px] font-semibold tracking-wide">{n.label}</span>
            </Link>
          );
        })}
        <div className="flex-1" />
        <button
          onClick={async () => {
            await signOut();
            router.push("/");
          }}
          className="flex flex-col items-center gap-1 py-2.5 rounded-lg text-ink-faint hover:text-rose transition-all"
        >
          <LogOut style={{ width: 18, height: 18 }} />
          <span className="text-[9px] font-semibold tracking-wide">Exit</span>
        </button>
      </nav>

      {/* main column */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* top status strip */}
        <div className="flex items-center justify-between px-4 sm:px-6 h-12 border-b border-[var(--line)] shrink-0">
          <div className="flex items-center gap-2 text-ink-faint">
            <Activity className="w-3.5 h-3.5 text-green hud-pulse" />
            <span className="hud-label">
              {stats
                ? `${stats.itemCount} signals · ${stats.sourceCount} sources live`
                : "syncing…"}
            </span>
          </div>
          <div className="flex items-center gap-3">
            {stats?.lastRunStatus && (
              <span
                className={cn(
                  "hud-chip",
                  stats.lastRunStatus === "ok" ? "" : "hud-chip-rose",
                )}
              >
                pipeline {stats.lastRunStatus}
              </span>
            )}
            {user?.isAdmin && (
              <span className="hud-chip flex items-center gap-1 !text-cyan" title="You have operator access">
                <ShieldCheck style={{ width: 11, height: 11 }} /> Admin
              </span>
            )}
            <span className="text-[11px] text-ink-dim truncate max-w-[160px]">
              {user?.isAnonymous ? "guest" : user?.email ?? user?.name ?? "…"}
            </span>
          </div>
        </div>

        <div className="flex-1 min-h-0">{children}</div>
      </div>
    </div>
  );
}
