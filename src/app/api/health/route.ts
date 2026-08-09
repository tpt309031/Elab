import { readFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

import type { DeepResearchArtifact, ResearchArtifact, SystemHealthResponse } from "@/lib/types";

export const dynamic = "force-dynamic";
export const revalidate = 0;
export const runtime = "nodejs";

async function readArtifact<T>(filename: string): Promise<T> {
  const absolutePath = path.join(process.cwd(), "public", "data", filename);
  return JSON.parse(await readFile(absolutePath, "utf8")) as T;
}

function utcDate(daysAgo = 0): string {
  const now = new Date();
  now.setUTCDate(now.getUTCDate() - daysAgo);
  return now.toISOString().slice(0, 10);
}

export async function GET() {
  const checkedAt = new Date().toISOString();
  const expectedClosedUtc = utcDate(1);
  const response: SystemHealthResponse = {
    status: "healthy",
    checkedAt,
    expectedClosedUtc,
    artifact: {
      latestClosedUtc: null,
      generatedAt: null,
      stale: true,
      marketHealth: null,
    },
    deep: {
      available: false,
      generatedAt: null,
      stale: false,
      architectures: 0,
    },
  };

  try {
    let artifact: ResearchArtifact;
    try {
      artifact = await readArtifact<ResearchArtifact>("hybrid_research_core.json");
    } catch {
      artifact = await readArtifact<ResearchArtifact>("hybrid_research.json");
    }
    response.artifact.latestClosedUtc = artifact.meta.latest_closed_utc;
    response.artifact.generatedAt = artifact.meta.generated_at;
    response.artifact.marketHealth = artifact.health?.market.status ?? "unknown";
    response.artifact.stale = artifact.meta.latest_closed_utc < expectedClosedUtc || Boolean(artifact.health?.market.stale);
    if (response.artifact.stale) response.status = "unhealthy";
  } catch (error) {
    response.status = "unhealthy";
    response.artifact.error = error instanceof Error ? error.message : "Hybrid artifact could not be read";
  }

  try {
    const deep = await readArtifact<DeepResearchArtifact>("deep_research.json");
    response.deep.available = true;
    response.deep.generatedAt = deep.meta.generated_at;
    response.deep.architectures = deep.models.rankings.length;
    const ageMilliseconds = Date.now() - new Date(deep.meta.generated_at).getTime();
    response.deep.stale = !Number.isFinite(ageMilliseconds) || ageMilliseconds > 10 * 24 * 60 * 60 * 1000;
    if (response.deep.stale && response.status === "healthy") response.status = "degraded";
  } catch (error) {
    response.deep.error = error instanceof Error ? error.message : "Deep artifact could not be read";
  }

  return NextResponse.json(response, {
    status: response.status === "unhealthy" ? 503 : 200,
    headers: { "Cache-Control": "no-store, max-age=0" },
  });
}
