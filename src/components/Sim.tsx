import { Suspense, useEffect, useMemo } from "react";
import { Canvas, useLoader, useThree } from "@react-three/fiber";
import { Environment, OrbitControls } from "@react-three/drei";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { useTwin } from "../store";

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

  return (
    <group position={[0.08, 0.05, 0]}>
      <mesh receiveShadow>
        <boxGeometry args={[1.15, 0.1, 0.7]} />
        <meshStandardMaterial
          color="#3d2a1c"
          roughness={0.85}
          transparent={Boolean(splat)}
          opacity={splat ? 0.0 : 1}
        />
      </mesh>
      {!splat && (
        <mesh position={[0, 0.052, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
          <planeGeometry args={[1.12, 0.66]} />
          <meshStandardMaterial map={tex ?? undefined} color={tex ? "#ffffff" : "#6b5344"} />
        </mesh>
      )}
    </group>
  );
}

function Glb({ url }: { url: string }) {
  const gltf = useLoader(GLTFLoader, url);
  return <primitive object={gltf.scene.clone()} scale={0.12} position={[0.12, 0.14, 0]} />;
}

function ObjectPreview() {
  const photo = useTwin((s) => s.objectPhoto);
  const glb = useTwin((s) => s.cloud.objectGlbUrl);
  const tex = useMemo(() => {
    if (!photo) return null;
    const t = new THREE.TextureLoader().load(photo);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }, [photo]);
  if (glb) return <Glb url={glb} />;
  return (
    <mesh position={[0.12, 0.14, 0]} castShadow>
      <boxGeometry args={[0.08, 0.1, 0.08]} />
      <meshStandardMaterial map={tex ?? undefined} color={tex ? "#fff" : "#d97746"} />
    </mesh>
  );
}

export function Sim() {
  const splatUrl = useTwin((s) => s.cloud.splatUrl);
  const videoUrl = useTwin((s) => s.videoUrl);

  return (
    <div className="stage-inner">
      {videoUrl ? (
        <video className="replay" src={videoUrl} controls autoPlay loop />
      ) : (
        <Canvas shadows camera={{ position: [1.15, 0.95, 1.35], fov: 42 }} gl={{ antialias: true }}>
          <color attach="background" args={["#141210"]} />
          <SparkHost />
          {splatUrl && <SplatWorld url={splatUrl} />}
          <ambientLight intensity={0.45} />
          <directionalLight position={[2.2, 3.4, 1.4]} intensity={1.4} castShadow />
          <Environment preset="warehouse" />
          <Table />
          <Suspense fallback={null}>
            <ObjectPreview />
          </Suspense>
          <OrbitControls makeDefault target={[0.05, 0.2, 0]} />
        </Canvas>
      )}
    </div>
  );
}
