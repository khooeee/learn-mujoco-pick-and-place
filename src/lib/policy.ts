import { BASE, HOME, ik, type Joints } from "./ik";
import { forward, initMLP, paramCount, trainEpoch, type MLP } from "./mlp";

export type Vec3 = { x: number; y: number; z: number };

/** shape 0 = box, 1 = cylinder. Physics and the net only see a bounding box + this flag. */
export type PickObject = Vec3 & { h: number; w: number; shape: 0 | 1 };

export const TABLE_TOP = 0.102;

const TABLE = {
  x0: -0.16,
  x1: 0.36,
  z0: -0.2,
  z1: 0.2,
};

export function randomObject(): PickObject {
  const h = 0.05 + Math.random() * 0.12;
  const w = 0.035 + Math.random() * 0.07;
  const shape: 0 | 1 = Math.random() < 0.5 ? 0 : 1;
  return {
    x: TABLE.x0 + Math.random() * (TABLE.x1 - TABLE.x0),
    y: TABLE_TOP + h / 2,
    z: TABLE.z0 + Math.random() * (TABLE.z1 - TABLE.z0),
    h,
    w,
    shape,
  };
}

export function encodeObject(o: PickObject): number[] {
  return [
    (o.x - BASE.x) / 0.9,
    o.y / 0.5,
    (o.z - BASE.z) / 0.7,
    o.h / 0.2,
    o.w / 0.12,
    o.shape,
  ];
}

export function encodeGrasp(g: Vec3): number[] {
  return [(g.x - BASE.x) / 0.9, g.y / 0.5, (g.z - BASE.z) / 0.7];
}

export function decodeGrasp(y: number[]): Vec3 {
  return {
    x: y[0] * 0.9 + BASE.x,
    y: y[1] * 0.5,
    z: y[2] * 0.7 + BASE.z,
  };
}

export function graspPoint(o: PickObject): Vec3 {
  return { x: o.x, y: o.y, z: o.z };
}

/** Net predicts a 3D grasp point. Arm IK is separate. */
export function policyGrasp(net: MLP, o: PickObject): Vec3 {
  return decodeGrasp(forward(net, encodeObject(o)));
}

export function policyJoints(net: MLP | null, o: PickObject): Joints {
  if (!net) return HOME;
  return ik(policyGrasp(net, o));
}

/** Fast IK-only trainer (no physics). Prefer trainOnPhysics for the real loop. */
export function trainPolicy(epochs = 60, lr = 0.05) {
  const net = initMLP(6, 64, 64, 3);
  const xs: number[][] = [];
  const ys: number[][] = [];
  for (let i = 0; i < 900; i++) {
    const o = randomObject();
    xs.push(encodeObject(o));
    ys.push(encodeGrasp(graspPoint(o)));
  }
  const losses: number[] = [];
  for (let e = 0; e < epochs; e++) {
    const loss = trainEpoch(net, xs, ys, lr);
    if (e % 3 === 0 || e === epochs - 1) losses.push(loss);
  }
  return { net, losses, params: paramCount(net) };
}

export type SavedPolicy = {
  version: 1;
  kind: "twinpick-grasp";
  architecture: [number, number, number, number];
  net: MLP;
  meta: {
    trainedAt: string;
    episodes?: number;
    successes?: number;
    holdOut?: number;
    params: number;
  };
};

export function serializePolicy(
  net: MLP,
  meta: Partial<SavedPolicy["meta"]>,
): SavedPolicy {
  return {
    version: 1,
    kind: "twinpick-grasp",
    architecture: [net.inSize, net.h1, net.h2, net.outSize],
    net,
    meta: {
      trainedAt: new Date().toISOString(),
      params: paramCount(net),
      ...meta,
    },
  };
}

export function parsePolicy(raw: unknown): SavedPolicy {
  const p = raw as SavedPolicy;
  if (!p || p.kind !== "twinpick-grasp" || p.version !== 1 || !p.net?.W1) {
    throw new Error("Not a TwinPick grasp policy file");
  }
  return p;
}

export function downloadPolicy(saved: SavedPolicy) {
  const blob = new Blob([JSON.stringify(saved, null, 2)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `twinpick-grasp-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}
