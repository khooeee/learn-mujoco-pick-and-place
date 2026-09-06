import { create } from "zustand";
import type { EpisodeRow, MintObject, RunRow, TrainStatus } from "./lib/rlApi";

type Store = {
  rlOnline: boolean;
  status: TrainStatus | null;
  runId: string | null;
  episodes: EpisodeRow[];
  metrics: { episode: number; rate: number; reward: number }[];
  videoUrl: string | null;
  renderBusy: number | null;
  episodesTarget: number;
  mintPrompt: string;
  mintBusy: boolean;
  mintJobPrompt: string;
  mintConfigured: boolean;
  objects: MintObject[];
  selectedId: string | null;
  previewId: string | null;
  previewPrompt: string | null;
  runs: RunRow[];
  followLive: boolean;
  log: (text: string) => void;
  setRlOnline: (v: boolean) => void;
  setStatus: (s: TrainStatus | null) => void;
  setRunId: (id: string | null) => void;
  setEpisodes: (e: EpisodeRow[]) => void;
  setMetrics: (m: Store["metrics"]) => void;
  setVideoUrl: (u: string | null) => void;
  setRenderBusy: (n: number | null) => void;
  setEpisodesTarget: (n: number) => void;
  setMintPrompt: (v: string) => void;
  setMintBusy: (v: boolean) => void;
  setMintJobPrompt: (v: string) => void;
  setMintConfigured: (v: boolean) => void;
  setObjects: (items: MintObject[], selected: string | null) => void;
  setPreview: (id: string | null, prompt?: string | null) => void;
  setRuns: (runs: RunRow[]) => void;
  selectRun: (id: string, followLive?: boolean) => void;
};

export const useTwin = create<Store>((set) => ({
  rlOnline: false,
  status: null,
  runId: null,
  episodes: [],
  metrics: [],
  videoUrl: null,
  renderBusy: null,
  episodesTarget: 30000,
  mintPrompt: "small ceramic mug",
  mintBusy: false,
  mintJobPrompt: "",
  mintConfigured: false,
  objects: [],
  selectedId: null,
  previewId: null,
  previewPrompt: null,
  runs: [],
  followLive: false,
  log: (text) => {
    console.log(text);
  },
  setRlOnline: (rlOnline) => set({ rlOnline }),
  setStatus: (status) => set({ status }),
  setRunId: (runId) =>
    set((s) =>
      s.runId === runId
        ? { runId, followLive: true }
        : { runId, followLive: true, episodes: [], metrics: [], videoUrl: null },
    ),
  setEpisodes: (episodes) => set({ episodes }),
  setMetrics: (metrics) => set({ metrics }),
  setVideoUrl: (videoUrl) =>
    set(videoUrl ? { videoUrl, previewId: null, previewPrompt: null } : { videoUrl }),
  setRenderBusy: (renderBusy) => set({ renderBusy }),
  setEpisodesTarget: (episodesTarget) => set({ episodesTarget }),
  setMintPrompt: (mintPrompt) => set({ mintPrompt }),
  setMintBusy: (mintBusy) => set({ mintBusy }),
  setMintJobPrompt: (mintJobPrompt) => set({ mintJobPrompt }),
  setMintConfigured: (mintConfigured) => set({ mintConfigured }),
  setObjects: (objects, selectedId) => set({ objects: objects ?? [], selectedId }),
  setPreview: (previewId, previewPrompt = null) =>
    set(
      previewId
        ? { previewId, previewPrompt: previewPrompt ?? null, videoUrl: null }
        : { previewId: null, previewPrompt: null },
    ),
  setRuns: (runs) => set({ runs }),
  selectRun: (runId, followLive = false) =>
    set({ runId, followLive, videoUrl: null, episodes: [], metrics: [] }),
}));
