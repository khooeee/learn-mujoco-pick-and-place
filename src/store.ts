import { create } from "zustand";
import type { MLP } from "./lib/mlp";
import { HOME, type ArmStep, type Joints } from "./lib/ik";
import { PLACE, randomCupPose, type Vec3 } from "./lib/policy";

export type Phase =
  | "idle"
  | "reach"
  | "grasp"
  | "lift"
  | "carry"
  | "place"
  | "retreat";

export type Brain = "expert" | "learned";

type LogItem = { t: number; text: string };

type Cloud = {
  splatUrl?: string;
  marbleUrl?: string;
  cupGlbUrl?: string;
  boxGlbUrl?: string;
  colliderUrl?: string;
  worldStatus: string;
  objectStatus: string;
  mintStatus: string;
};

type Store = {
  tablePhoto: string | null;
  objectPhoto: string | null;
  cup: Vec3;
  holding: boolean;
  joints: Joints;
  phase: Phase;
  brain: Brain;
  net: MLP | null;
  losses: number[];
  lastLoss: number | null;
  logs: LogItem[];
  cloud: Cloud;
  pickBusy: boolean;
  pickSteps: ArmStep[] | null;
  setTablePhoto: (url: string | null) => void;
  setObjectPhoto: (url: string | null) => void;
  scatterCup: () => void;
  setJoints: (j: Joints) => void;
  setPhase: (p: Phase) => void;
  setHolding: (h: boolean) => void;
  setNet: (net: MLP, losses: number[]) => void;
  log: (text: string) => void;
  setCloud: (patch: Partial<Cloud>) => void;
  setPickBusy: (b: boolean) => void;
  setBrain: (b: Brain) => void;
};

export const useTwin = create<Store>((set) => ({
  tablePhoto: null,
  objectPhoto: null,
  cup: { x: 0.12, y: 0.145, z: 0.04 },
  holding: false,
  joints: HOME,
  phase: "idle",
  brain: "expert",
  net: null,
  losses: [],
  lastLoss: null,
  logs: [],
  cloud: {
    worldStatus: "idle",
    objectStatus: "idle",
    mintStatus: "idle",
  },
  pickBusy: false,
  pickSteps: null,
  setTablePhoto: (url) => set({ tablePhoto: url }),
  setObjectPhoto: (url) => set({ objectPhoto: url }),
  scatterCup: () => set({ cup: randomCupPose(), holding: false }),
  setJoints: (joints) => set({ joints }),
  setPhase: (phase) => set({ phase }),
  setHolding: (holding) => set({ holding }),
  setNet: (net, losses) =>
    set({
      net,
      losses,
      lastLoss: losses[losses.length - 1] ?? null,
    }),
  log: (text) =>
    set((s) => ({
      logs: [{ t: Date.now(), text }, ...s.logs].slice(0, 24),
    })),
  setCloud: (patch) => set((s) => ({ cloud: { ...s.cloud, ...patch } })),
  setPickBusy: (pickBusy) => set({ pickBusy }),
  setBrain: (brain) => set({ brain }),
}));

export { PLACE };
