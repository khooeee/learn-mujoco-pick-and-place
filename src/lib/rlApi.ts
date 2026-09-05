const PREFIX = "/rl";

export type TrainStatus = {
  state: string;
  run_id: string | null;
  episode: number;
  episodes_target: number;
  success_rate: number;
  reward: number;
  log: string;
  alive?: boolean;
  log_tail?: string[];
};

export type EpisodeRow = {
  index: number;
  success: boolean;
  reward: number;
  lifted_z: number;
  has_video: boolean;
};

async function req(path: string, init?: RequestInit) {
  const res = await fetch(PREFIX + path, init);
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || res.statusText);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res;
}

export const rl = {
  health: () => req("/health").catch(() => null),
  status: (): Promise<TrainStatus> => req("/status"),
  start: (episodes: number, resume = false) =>
    req("/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ episodes, resume }),
    }),
  stop: () => req("/train/stop", { method: "POST" }),
  runs: () => req("/runs") as Promise<{ id: string }[]>,
  episodes: (runId: string): Promise<EpisodeRow[]> => req(`/runs/${runId}/episodes`),
  metrics: (runId: string): Promise<{ episode: number; rate: number; reward: number }[]> =>
    req(`/runs/${runId}/metrics`),
  render: (runId: string, episode: number) =>
    req("/runs/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: runId, episode }),
    }),
  videoUrl: (runId: string, episode: number) => `${PREFIX}/runs/${runId}/videos/${episode}`,
};
