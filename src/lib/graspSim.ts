import RAPIER from "@dimforge/rapier3d-compat";
import { forward, initMLP, paramCount, trainEpoch } from "./mlp";
import {
  decodeGrasp,
  encodeGrasp,
  encodeObject,
  randomObject,
  type PickObject,
  type Vec3,
} from "./policy";

const IDENTITY = { x: 0, y: 0, z: 0, w: 1 };
const TABLE_Y = 0.05;
const TABLE_HX = 0.575;
const TABLE_HY = 0.05;
const TABLE_HZ = 0.35;

let ready = false;

export async function ensureRapier() {
  if (ready) return;
  await RAPIER.init();
  ready = true;
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

/** Noisy expert grasp. Physics decides if it still works — that's the dataset. */
export function noisyExpertGrasp(o: PickObject): Vec3 {
  return {
    x: o.x + (Math.random() - 0.5) * 0.05,
    y: o.y + (Math.random() - 0.5) * 0.02,
    z: o.z + (Math.random() - 0.5) * 0.05,
  };
}

export type EpisodeResult = {
  success: boolean;
  object: PickObject;
  grasp: Vec3;
};

/**
 * One pick in a throwaway Rapier world (no renderer).
 * Kinematic gripper flies to `grasp`, welds if close, lifts. Gravity is on.
 */
export function runPickEpisode(object: PickObject, grasp: Vec3): EpisodeResult {
  const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });

  const table = world.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(0.08, TABLE_Y, 0));
  world.createCollider(
    RAPIER.ColliderDesc.cuboid(TABLE_HX, TABLE_HY, TABLE_HZ).setFriction(0.95),
    table,
  );

  const obj = world.createRigidBody(
    RAPIER.RigidBodyDesc.dynamic().setTranslation(object.x, object.y, object.z),
  );
  const hx = object.w / 2;
  const hy = object.h / 2;
  if (object.shape === 1) {
    world.createCollider(
      RAPIER.ColliderDesc.cylinder(hy, Math.max(0.015, hx)).setMass(0.35).setFriction(0.85),
      obj,
    );
  } else {
    world.createCollider(
      RAPIER.ColliderDesc.cuboid(hx, hy, hx).setMass(0.35).setFriction(0.85),
      obj,
    );
  }

  const home = { x: -0.35, y: 0.38, z: 0.05 };
  const grip = world.createRigidBody(
    RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(home.x, home.y, home.z),
  );
  world.createCollider(RAPIER.ColliderDesc.cuboid(0.03, 0.02, 0.03).setFriction(0.9), grip);

  const settle = 20;
  const reachN = 45;
  const liftN = 40;
  const graspTol = Math.max(0.045, object.w * 0.7 + 0.02);

  for (let i = 0; i < settle; i++) world.step();

  let welded = false;
  for (let i = 1; i <= reachN; i++) {
    const t = i / reachN;
    grip.setNextKinematicTranslation({
      x: lerp(home.x, grasp.x, t),
      y: lerp(home.y, grasp.y, t),
      z: lerp(home.z, grasp.z, t),
    });
    world.step();
  }

  const gp = grip.translation();
  const op = obj.translation();
  const dist = Math.hypot(gp.x - op.x, gp.y - op.y, gp.z - op.z);
  if (dist < graspTol) {
    const params = RAPIER.JointData.fixed(
      { x: 0, y: 0, z: 0 },
      IDENTITY,
      { x: 0, y: 0, z: 0 },
      IDENTITY,
    );
    world.createImpulseJoint(params, grip, obj, true);
    welded = true;
  }

  const liftTo = { x: grasp.x, y: grasp.y + 0.16, z: grasp.z };
  for (let i = 1; i <= liftN; i++) {
    const t = i / liftN;
    grip.setNextKinematicTranslation({
      x: lerp(grasp.x, liftTo.x, t),
      y: lerp(grasp.y, liftTo.y, t),
      z: lerp(grasp.z, liftTo.z, t),
    });
    world.step();
  }

  const lifted = obj.translation();
  const success = welded && lifted.y > object.y + 0.08;

  world.free();
  return { success, object, grasp };
}

export type TrainProgress = {
  i: number;
  n: number;
  successes: number;
  phase: "sim" | "fit" | "holdout";
};

export async function trainOnPhysics(
  n = 300,
  onProgress?: (p: TrainProgress) => void,
) {
  await ensureRapier();
  const xs: number[][] = [];
  const ys: number[][] = [];
  let successes = 0;

  for (let i = 0; i < n; i++) {
    const o = randomObject();
    const g = noisyExpertGrasp(o);
    const ep = runPickEpisode(o, g);
    if (ep.success) {
      successes += 1;
      xs.push(encodeObject(o));
      ys.push(encodeGrasp(g));
    }
    if (i % 12 === 0 || i === n - 1) {
      onProgress?.({ i: i + 1, n, successes, phase: "sim" });
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  if (xs.length < 20) {
    throw new Error(`Only ${xs.length} successful grasps — need more. Try training again.`);
  }

  onProgress?.({ i: n, n, successes, phase: "fit" });
  const net = initMLP(6, 64, 64, 3);
  const losses: number[] = [];
  for (let e = 0; e < 80; e++) {
    losses.push(trainEpoch(net, xs, ys, 0.04));
    if (e % 10 === 0) await new Promise((r) => setTimeout(r, 0));
  }

  onProgress?.({ i: n, n, successes, phase: "holdout" });
  let holdOk = 0;
  const holdN = 40;
  for (let i = 0; i < holdN; i++) {
    const o = randomObject();
    const g = decodeGrasp(forward(net, encodeObject(o)));
    if (runPickEpisode(o, g).success) holdOk += 1;
    if (i % 8 === 0) await new Promise((r) => setTimeout(r, 0));
  }

  return {
    net,
    losses,
    params: paramCount(net),
    episodes: n,
    successes,
    holdOut: holdOk / holdN,
  };
}
