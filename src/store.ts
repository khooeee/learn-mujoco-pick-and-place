import { create } from "zustand";
import type { EpisodeRow, TrainStatus } from "./lib/rlApi";

type LogItem = { t: number; text: string };

type Cloud = {
  splatUrl?: string;
  marbleUrl?: string;
  objectGlbUrl?: string;
  colliderUrl?: string;
  worldStatus: string;
  objectStatus: string;
  mintStatus: string;
};

type Store = {
  tablePhoto: string | null;
  objectPhoto: string | null;
  logs: LogItem[];
  cloud: Cloud;
  rlOnline: boolean;
  status: TrainStatus | null;
  runId: string | null;
  episodes: EpisodeRow[];
  metrics: { episode: number; rate: number; reward: number }[];
  videoUrl: string | null;
  renderBusy: number | null;
  episodesTarget: number;
  setTablePhoto: (url: string | null) => void;
  setObjectPhoto: (url: string | null) => void;
  log: (text: string) => void;
  setCloud: (patch: Partial<Cloud>) => void;
  setRlOnline: (v: boolean) => void;
  setStatus: (s: TrainStatus | null) => void;
  setRunId: (id: string | null) => void;
  setEpisodes: (e: EpisodeRow[]) => void;
  setMetrics: (m: Store["metrics"]) => void;
  setVideoUrl: (u: string | null) => void;
  setRenderBusy: (n: number | null) => void;
  setEpisodesTarget: (n: number) => void;
};

export const useTwin = create<Store>((set) => ({
  tablePhoto: null,
  objectPhoto: null,
  logs: [],
  cloud: {
    worldStatus: "idle",
    objectStatus: "idle",
    mintStatus: "idle",
  },
  rlOnline: false,
  status: null,
  runId: null,
  episodes: [],
  metrics: [],
  videoUrl: null,
  renderBusy: null,
  episodesTarget: 200,
  setTablePhoto: (url) => set({ tablePhoto: url }),
  setObjectPhoto: (url) => set({ objectPhoto: url }),
  log: (text) =>
    set((s) => ({
      logs: [{ t: Date.now(), text }, ...s.logs].slice(0, 24),
    })),
  setCloud: (patch) => set((s) => ({ cloud: { ...s.cloud, ...patch } })),
  setRlOnline: (rlOnline) => set({ rlOnline }),
  setStatus: (status) => set({ status }),
  setRunId: (runId) => set({ runId }),
  setEpisodes: (episodes) => set({ episodes }),
  setMetrics: (metrics) => set({ metrics }),
  setVideoUrl: (videoUrl) => set({ videoUrl }),
  setRenderBusy: (renderBusy) => set({ renderBusy }),
  setEpisodesTarget: (episodesTarget) => set({ episodesTarget }),
}));
