import { Suspense, useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { ContactShadows, Environment, OrbitControls } from "@react-three/drei";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { BASE, HOME, L1, L2, fk, ik, lerpJoints, type Joints } from "../lib/ik";
import { policyJoints } from "../lib/policy";
import { PLACE, useTwin } from "../store";

export function queuePick() {
  const st = useTwin.getState();
  if (st.pickBusy) return;
  if (st.brain === "learned" && !st.net) {
    st.log("Train the pick brain first");
    return;
  }
  const aim = (p: { x: number; y: number; z: number }) =>
    st.brain === "learned" && st.net ? policyJoints(st.net, p) : ik(p);

  const grasp = aim({ ...st.cup, y: st.cup.y + 0.02 });
  const lift = aim({ ...st.cup, y: st.cup.y + 0.16 });
  const hover = aim({ ...PLACE, y: PLACE.y + 0.16 });
  const down = aim({ ...PLACE, y: PLACE.y + 0.05 });

  st.log(st.brain === "learned" ? "Running learned pick" : "Running expert IK pick");
  useTwin.setState({
    pickBusy: true,
    pickSteps: [
      { joints: grasp, ms: 900 },
      { joints: grasp, hold: true, ms: 220 },
      { joints: lift, ms: 500 },
      { joints: hover, ms: 800 },
      { joints: down, hold: false, ms: 480 },
      { joints: HOME, ms: 700 },
    ],
  });
}

function SparkHost() {
  const { gl, scene } = useThree();
  useEffect(() => {
    const spark = new SparkRenderer({ renderer: gl });
    scene.add(spark);
    return () => {
      scene.remove(spark);
    };
  }, [gl, scene]);
  return null;
}

function SplatWorld({ url }: { url: string }) {
  const { scene } = useThree();
  useEffect(() => {
    const mesh = new SplatMesh({ url });
    mesh.quaternion.set(1, 0, 0, 0);
    scene.add(mesh);
    return () => {
      scene.remove(mesh);
    };
  }, [scene, url]);
  return null;
}

function Table() {
  const photo = useTwin((s) => s.tablePhoto);
  const splat = useTwin((s) => s.cloud.splatUrl);
  const tex = useMemo(() => {
    if (!photo) return null;
    const t = new THREE.TextureLoader().load(photo);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }, [photo]);

  if (splat) return null;

  return (
    <group>
      <mesh position={[0.08, 0.05, 0]} receiveShadow>
        <boxGeometry args={[1.15, 0.1, 0.7]} />
        <meshStandardMaterial color="#3d2a1c" roughness={0.85} />
      </mesh>
      <mesh
        position={[0.08, 0.102, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        receiveShadow
      >
        <planeGeometry args={[1.12, 0.66]} />
        <meshStandardMaterial
          map={tex ?? undefined}
          color={tex ? "#ffffff" : "#6b5344"}
          roughness={0.7}
        />
      </mesh>
      {[-0.42, 0.42].flatMap((x) =>
        [-0.28, 0.28].map((z) => (
          <mesh key={`${x}:${z}`} position={[x + 0.08, -0.2, z]}>
            <boxGeometry args={[0.06, 0.5, 0.06]} />
            <meshStandardMaterial color="#2a1c14" />
          </mesh>
        )),
      )}
    </group>
  );
}

function Glb({ url, scale = 0.12 }: { url: string; scale?: number }) {
  const gltf = useLoader(GLTFLoader, url);
  return <primitive object={gltf.scene.clone()} scale={scale} />;
}

function Cup() {
  const cup = useTwin((s) => s.cup);
  const holding = useTwin((s) => s.holding);
  const joints = useTwin((s) => s.joints);
  const photo = useTwin((s) => s.objectPhoto);
  const glb = useTwin((s) => s.cloud.cupGlbUrl);
  const tex = useMemo(() => {
    if (!photo) return null;
    const t = new THREE.TextureLoader().load(photo);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }, [photo]);

  const grip = fk(joints);
  const pos = holding
    ? ([grip.x, grip.y - 0.04, grip.z] as const)
    : ([cup.x, cup.y, cup.z] as const);

  return (
    <group position={pos}>
      {glb ? (
        <Glb url={glb} />
      ) : (
        <>
          <mesh castShadow>
            <cylinderGeometry args={[0.045, 0.04, 0.09, 24]} />
            <meshStandardMaterial
              map={tex ?? undefined}
              color={tex ? "#fff" : "#c45c38"}
              roughness={0.45}
            />
          </mesh>
          <mesh position={[0.05, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
            <torusGeometry args={[0.028, 0.007, 8, 16, Math.PI]} />
            <meshStandardMaterial color="#a64b2e" />
          </mesh>
        </>
      )}
    </group>
  );
}

function PlaceBox() {
  const glb = useTwin((s) => s.cloud.boxGlbUrl);
  return (
    <group position={[PLACE.x, 0.12, PLACE.z]}>
      {glb ? (
        <Glb url={glb} scale={0.18} />
      ) : (
        <mesh castShadow>
          <boxGeometry args={[0.12, 0.06, 0.12]} />
          <meshStandardMaterial color="#c4a574" roughness={0.9} />
        </mesh>
      )}
    </group>
  );
}

function Link({ length, color }: { length: number; color: string }) {
  return (
    <mesh position={[0, 0, length / 2]} rotation={[Math.PI / 2, 0, 0]} castShadow>
      <capsuleGeometry args={[0.028, Math.max(0.01, length - 0.056), 6, 12]} />
      <meshStandardMaterial color={color} metalness={0.35} roughness={0.35} />
    </mesh>
  );
}

function Arm() {
  const j = useTwin((s) => s.joints);
  return (
    <group position={[BASE.x, BASE.y, BASE.z]}>
      <mesh position={[0, -0.04, 0]}>
        <cylinderGeometry args={[0.08, 0.1, 0.08, 20]} />
        <meshStandardMaterial color="#1a1a1a" />
      </mesh>
      <group rotation={[0, j.yaw, 0]}>
        <mesh position={[0, 0.02, 0]}>
          <boxGeometry args={[0.08, 0.08, 0.08]} />
          <meshStandardMaterial color="#e85d04" />
        </mesh>
        <group rotation={[j.shoulder, 0, 0]}>
          <Link length={L1} color="#2b2b2b" />
          <group position={[0, 0, L1]} rotation={[-j.elbow, 0, 0]}>
            <Link length={L2} color="#3a3a3a" />
            <group position={[0, 0, L2]}>
              <mesh>
                <boxGeometry args={[0.07, 0.04, 0.05]} />
                <meshStandardMaterial color="#e85d04" />
              </mesh>
              <mesh position={[0.03, -0.03, 0.02]}>
                <boxGeometry args={[0.012, 0.05, 0.03]} />
                <meshStandardMaterial color="#111" />
              </mesh>
              <mesh position={[-0.03, -0.03, 0.02]}>
                <boxGeometry args={[0.012, 0.05, 0.03]} />
                <meshStandardMaterial color="#111" />
              </mesh>
            </group>
          </group>
        </group>
      </group>
    </group>
  );
}

function PickController() {
  const t0 = useRef(0);
  const from = useRef<Joints>(HOME);

  useFrame((_, dt) => {
    const s = useTwin.getState();
    const steps = s.pickSteps;
    if (!s.pickBusy || !steps || steps.length === 0) return;
    const step = steps[0];
    t0.current += dt * 1000;
    const u = Math.min(1, t0.current / step.ms);
    const ease = u * u * (3 - 2 * u);
    s.setJoints(lerpJoints(from.current, step.joints, ease));
    if (u < 1) return;

    if (step.hold === true) {
      const g = fk(step.joints);
      useTwin.setState({ holding: true, cup: { x: g.x, y: g.y - 0.04, z: g.z } });
    }
    if (step.hold === false) {
      useTwin.setState({ holding: false, cup: { ...PLACE } });
    }
    from.current = step.joints;
    t0.current = 0;
    const rest = steps.slice(1);
    if (rest.length === 0) {
      useTwin.setState({ pickBusy: false, pickSteps: null, phase: "idle" });
      useTwin.getState().log("Pick finished");
    } else {
      useTwin.setState({ pickSteps: rest });
    }
  });

  const pickBusy = useTwin((s) => s.pickBusy);
  useEffect(() => {
    if (pickBusy) {
      from.current = useTwin.getState().joints;
      t0.current = 0;
    }
  }, [pickBusy]);

  return null;
}

export function Sim() {
  const splatUrl = useTwin((s) => s.cloud.splatUrl);

  return (
    <Canvas
      shadows
      camera={{ position: [1.15, 0.95, 1.35], fov: 42 }}
      gl={{ antialias: true }}
    >
      <color attach="background" args={["#141210"]} />
      <fog attach="fog" args={["#141210", 6, 14]} />
      <SparkHost />
      {splatUrl && <SplatWorld url={splatUrl} />}
      <ambientLight intensity={0.45} />
      <directionalLight
        position={[2.2, 3.4, 1.4]}
        intensity={1.4}
        castShadow
        shadow-mapSize={1024}
      />
      <Environment preset="warehouse" />
      <Table />
      <Arm />
      <Suspense fallback={null}>
        <Cup />
        <PlaceBox />
      </Suspense>
      <ContactShadows position={[0, -0.44, 0]} opacity={0.45} scale={4} blur={2} />
      <OrbitControls makeDefault target={[0.05, 0.2, 0]} />
      <PickController />
    </Canvas>
  );
}
