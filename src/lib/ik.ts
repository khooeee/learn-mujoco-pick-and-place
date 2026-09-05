export const L1 = 0.36;
export const L2 = 0.32;
export const BASE = { x: -0.62, y: 0.04, z: 0 } as const;

export type Joints = { yaw: number; shoulder: number; elbow: number };

function clamp(x: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, x));
}

/** Analytic IK for a yaw + 2-link arm. This is the expert the net clones. */
export function ik(target: { x: number; y: number; z: number }): Joints {
  const dx = target.x - BASE.x;
  const dy = target.y - BASE.y;
  const dz = target.z - BASE.z;
  const yaw = Math.atan2(dx, dz);
  const r = Math.hypot(dx, dz);
  let d = Math.hypot(r, dy);
  d = clamp(d, Math.abs(L1 - L2) + 0.01, L1 + L2 - 0.01);

  const cosElbow = clamp((L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2), -1, 1);
  const elbow = Math.PI - Math.acos(cosElbow);

  const cosA = clamp((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1);
  const a = Math.acos(cosA);
  const shoulder = Math.atan2(dy, r) + a;

  return { yaw, shoulder, elbow };
}

export function fk(j: Joints) {
  const r = L1 * Math.cos(j.shoulder) + L2 * Math.cos(j.shoulder - j.elbow);
  const y = BASE.y + L1 * Math.sin(j.shoulder) + L2 * Math.sin(j.shoulder - j.elbow);
  return {
    x: BASE.x + r * Math.sin(j.yaw),
    y,
    z: BASE.z + r * Math.cos(j.yaw),
  };
}

export function lerpJoints(a: Joints, b: Joints, t: number): Joints {
  const u = clamp(t, 0, 1);
  return {
    yaw: a.yaw + (b.yaw - a.yaw) * u,
    shoulder: a.shoulder + (b.shoulder - a.shoulder) * u,
    elbow: a.elbow + (b.elbow - a.elbow) * u,
  };
}

export const HOME: Joints = { yaw: 0.4, shoulder: 1.15, elbow: 1.35 };

export type ArmStep = { joints: Joints; hold?: boolean; ms: number };
