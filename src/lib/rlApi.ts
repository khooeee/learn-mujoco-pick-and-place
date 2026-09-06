const PREFIX = "/rl";

export type TrainStatus = {
  state: string;
  run_id: string | null;
  episode: number;
  episodes_target: number;
  success_rate: number;
  reward: number;
  r_reach?: number;
  r_close?: number;
  r_lift?: number;
  log: string;
  alive?: boolean;
  log_tail?: string[];
};

export type MintObject = {
  id: string;
  prompt: string;
  created?: string;
  w?: number;
  h?: number;
  scale?: number;
  has_glb?: boolean;
};

export type Health = {
  ok: boolean;
  mint: boolean;
  mint_busy?: boolean;
  mint_prompt?: string;
  mint_error?: string | null;
  mint_last?: MintObject | null;
};

export type RunRow = {
  id: string;
  state: string;
  episode: number;
  episodes_target: number;
  success_rate: number;
  reward: number;
  episodes: number;
  videos: number;
  has_policy: boolean;
  mtime: number;
};

export type EpisodeRow = {
  index: number;
  success: boolean;
  reward: number;
  r_reach?: number;
  r_close?: number;
  r_lift?: number;
  lifted_z: number;
  has_video: boolean;
};

async function req(path: string, init?: RequestInit) {
  const res = await fetch(PREFIX + path, init);
  if (!res.ok) {
    const t = await res.text();
    let msg = t || res.statusText;
    try {
      const j = JSON.parse(t) as { detail?: string };
      if (typeof j.detail === "string") msg = j.detail;
    } catch {
      /* keep raw body */
    }
    throw new Error(msg);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res;
}

export const rl = {
  health: (): Promise<Health | null> => req("/health").catch(() => null),
  objects: async () => {
    const raw = await req("/objects");
    if (Array.isArray(raw)) {
      return { selected: null as string | null, items: [] as MintObject[] };
    }
    const row = raw as { selected?: string | null; items?: MintObject[] };
    return { selected: row.selected ?? null, items: row.items ?? [] };
  },
  selectObject: (id: string) =>
    req("/objects/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    }) as Promise<{ selected: string | null; items: MintObject[] }>,
  mintGenerate: (prompt: string) =>
    req("/mint/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    }) as Promise<{ ok: boolean; busy: boolean; prompt: string }>,
  importGlb: (file: File, name?: string) => {
    const body = new FormData();
    body.append("file", file);
    if (name) body.append("name", name);
    return req("/objects/import", { method: "POST", body }) as Promise<MintObject>;
  },
  status: (): Promise<TrainStatus> => req("/status"),
  start: (episodes: number, resume = false, runId?: string | null) =>
    req("/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        episodes,
        resume,
        ...(resume && runId ? { run_id: runId } : {}),
      }),
    }),
  stop: () => req("/train/stop", { method: "POST" }),
  runs: () => req("/runs") as Promise<RunRow[]>,
  openRun: (runId: string) =>
    req(`/runs/${encodeURIComponent(runId)}/open`, { method: "POST" }),
  episodes: (runId: string): Promise<EpisodeRow[]> => req(`/runs/${runId}/episodes`),
  metrics: (
    runId: string,
  ): Promise<{ episode: number; rate: number; reward: number; r_reach?: number; r_close?: number; r_lift?: number }[]> =>
    req(`/runs/${runId}/metrics`),
  render: (runId: string, episode: number) =>
    req("/runs/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: runId, episode }),
    }),
  openEpisode: (runId: string, episode: number) =>
    req(`/runs/${encodeURIComponent(runId)}/episodes/${episode}/open`, { method: "POST" }),
  videoUrl: (runId: string, episode: number) => `${PREFIX}/runs/${runId}/videos/${episode}`,
  objectMeshUrl: (id: string) => `${PREFIX}/objects/${encodeURIComponent(id)}/mesh`,
  objectGlbUrl: (id: string) => `${PREFIX}/objects/${encodeURIComponent(id)}/source.glb`,
};
