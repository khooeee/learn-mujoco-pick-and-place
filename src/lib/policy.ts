import { BASE, HOME, ik, type Joints } from "./ik";
import { forward, initMLP, trainEpoch, type MLP } from "./mlp";

export type Vec3 = { x: number; y: number; z: number };

const TABLE = {
  x0: -0.18,
  x1: 0.38,
  z0: -0.22,
  z1: 0.22,
  y: 0.145,
};

export function randomCupPose(): Vec3 {
  return {
    x: TABLE.x0 + Math.random() * (TABLE.x1 - TABLE.x0),
    y: TABLE.y,
    z: TABLE.z0 + Math.random() * (TABLE.z1 - TABLE.z0),
  };
}

export const PLACE: Vec3 = { x: 0.28, y: 0.16, z: -0.16 };

function encodePos(p: Vec3): number[] {
  return [(p.x - BASE.x) / 0.9, p.y / 0.5, (p.z - BASE.z) / 0.7];
}

function encodeJoints(j: Joints): number[] {
  return [j.yaw / Math.PI, j.shoulder / Math.PI, j.elbow / Math.PI];
}

function decodeJoints(y: number[]): Joints {
  return {
    yaw: y[0] * Math.PI,
    shoulder: y[1] * Math.PI,
    elbow: y[2] * Math.PI,
  };
}

export function makeDataset(n = 400) {
  const xs: number[][] = [];
  const ys: number[][] = [];
  for (let i = 0; i < n; i++) {
    const p = randomCupPose();
    p.y = 0.1 + Math.random() * 0.18;
    const j = ik(p);
    xs.push(encodePos(p));
    ys.push(encodeJoints(j));
  }
  return { xs, ys };
}

export function trainPolicy(epochs = 80, lr = 0.08) {
  const net = initMLP();
  const { xs, ys } = makeDataset(500);
  const losses: number[] = [];
  for (let e = 0; e < epochs; e++) {
    const loss = trainEpoch(net, xs, ys, lr);
    if (e % 4 === 0 || e === epochs - 1) losses.push(loss);
  }
  return { net, losses };
}

export function policyJoints(net: MLP | null, target: Vec3): Joints {
  if (!net) return HOME;
  return decodeJoints(forward(net, encodePos(target)));
}

export { encodePos };

