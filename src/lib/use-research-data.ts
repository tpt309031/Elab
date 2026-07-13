"use client";

import useSWR from "swr";

import type { DeepResearchArtifact, LiveMarketResponse, ResearchArtifact, SystemHealthResponse } from "@/lib/types";

async function jsonFetcher<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}

async function optionalJsonFetcher<T>(url: string): Promise<T | undefined> {
  const response = await fetch(url, { cache: "no-store" });
  if (response.status === 404) return undefined;
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}

async function healthFetcher(url: string): Promise<SystemHealthResponse> {
  const response = await fetch(url, { cache: "no-store" });
  const payload = await response.json() as SystemHealthResponse;
  if (!payload.status) throw new Error(`${response.status} ${response.statusText}`);
  return payload;
}

export function useResearchData() {
  const research = useSWR<ResearchArtifact>("/data/hybrid_research.json", jsonFetcher, {
    revalidateOnFocus: false,
    dedupingInterval: 60_000,
  });
  const live = useSWR<LiveMarketResponse>("/api/market?timeframe=5m", jsonFetcher, {
    refreshInterval: 300_000,
    revalidateOnFocus: true,
    dedupingInterval: 60_000,
  });
  const deep = useSWR<DeepResearchArtifact | undefined>("/data/deep_research.json", optionalJsonFetcher, {
    revalidateOnFocus: false,
    dedupingInterval: 300_000,
    shouldRetryOnError: false,
  });
  const health = useSWR<SystemHealthResponse>("/api/health", healthFetcher, {
    refreshInterval: 300_000,
    revalidateOnFocus: true,
    dedupingInterval: 60_000,
  });
  return { research, live, deep, health };
}
