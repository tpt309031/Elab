"use client";

import { AlertTriangle, RefreshCw } from "lucide-react";

export default function ErrorPage({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <main className="grid min-h-screen place-items-center bg-background p-6 text-foreground">
      <section className="w-full max-w-md border border-border bg-card p-6">
        <AlertTriangle className="mb-4 size-6 text-primary" aria-hidden="true" />
        <h1 className="text-xl font-semibold">The research console could not load.</h1>
        <p className="mt-2 text-sm text-muted-foreground">The last committed research artifact remains intact. Retry the client view.</p>
        <button className="mt-5 inline-flex items-center gap-2 border border-primary bg-primary px-4 py-2 text-sm font-medium text-primary-foreground" onClick={reset}>
          <RefreshCw className="size-4" aria-hidden="true" /> Retry
        </button>
      </section>
    </main>
  );
}
