import { create } from "zustand";
import type { EpisodeRow, MintObject, TrainStatus } from "./lib/rlApi";

type LogItem = { t: number; text: string };

type Store = {
  logs: LogItem[];
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
  mintConfigured: boolean;
  objects: MintObject[];
  selectedId: string | null;
  previewId: string | null;
  previewPrompt: string | null;
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
  setMintConfigured: (v: boolean) => void;
  setObjects: (items: MintObject[], selected: string | null) => void;
  setPreview: (id: string | null, prompt?: string | null) => void;
};

export const useTwin = create<Store>((set) => ({
  logs: [],
  rlOnline: false,
  status: null,
  runId: null,
  episodes: [],
  metrics: [],
  videoUrl: null,
  renderBusy: null,
  episodesTarget: 200,
  mintPrompt: "small ceramic mug",
  mintBusy: false,
  mintConfigured: false,
  objects: [],
  selectedId: null,
  previewId: null,
  previewPrompt: null,
  log: (text) =>
    set((s) => ({
      logs: [{ t: Date.now(), text }, ...s.logs].slice(0, 24),
    })),
  setRlOnline: (rlOnline) => set({ rlOnline }),
  setStatus: (status) => set({ status }),
  setRunId: (runId) => set({ runId }),
  setEpisodes: (episodes) => set({ episodes }),
  setMetrics: (metrics) => set({ metrics }),
  setVideoUrl: (videoUrl) =>
    set(videoUrl ? { videoUrl, previewId: null, previewPrompt: null } : { videoUrl }),
  setRenderBusy: (renderBusy) => set({ renderBusy }),
  setEpisodesTarget: (episodesTarget) => set({ episodesTarget }),
  setMintPrompt: (mintPrompt) => set({ mintPrompt }),
  setMintBusy: (mintBusy) => set({ mintBusy }),
  setMintConfigured: (mintConfigured) => set({ mintConfigured }),
  setObjects: (objects, selectedId) => set({ objects: objects ?? [], selectedId }),
  setPreview: (previewId, previewPrompt = null) =>
    set(
      previewId
        ? { previewId, previewPrompt: previewPrompt ?? null, videoUrl: null }
        : { previewId: null, previewPrompt: null },
    ),
}));
