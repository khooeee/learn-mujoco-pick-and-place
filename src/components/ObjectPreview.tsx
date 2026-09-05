import { Center, OrbitControls, useGLTF } from "@react-three/drei";
import { Canvas, useLoader } from "@react-three/fiber";
import { Suspense, useEffect, useMemo } from "react";
import * as THREE from "three";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { rl } from "../lib/rlApi";
import { useTwin } from "../store";

function StlMesh({ url }: { url: string }) {
  const geom = useLoader(STLLoader, url);
  useEffect(() => {
    geom.computeVertexNormals();
  }, [geom]);
  return (
    <Center>
      <mesh geometry={geom}>
        <meshStandardMaterial color="#e07030" roughness={0.42} metalness={0.08} />
      </mesh>
    </Center>
  );
}

function GlbMesh({ url, scale }: { url: string; scale: number }) {
  const { scene } = useGLTF(url);
  const cloned = useMemo(() => {
    const root = scene.clone(true);
    root.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) return;
      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const mat of mats) {
        const std = mat as THREE.MeshStandardMaterial;
        if (std.map) std.map.colorSpace = THREE.SRGBColorSpace;
        if (std.emissiveMap) std.emissiveMap.colorSpace = THREE.SRGBColorSpace;
      }
    });
    return root;
  }, [scene]);
  return (
    <Center>
      <group scale={scale}>
        <primitive object={cloned} />
      </group>
    </Center>
  );
}

export function ObjectPreview({ id, prompt }: { id: string; prompt: string }) {
  const item = useTwin((s) => (s.objects ?? []).find((o) => o.id === id));
  const textured = Boolean(item?.has_glb);
  const scale = item?.scale && item.scale > 0 ? item.scale : 1;
  const url = textured
    ? `${rl.objectGlbUrl(id)}?t=${id}`
    : `${rl.objectMeshUrl(id)}?t=${id}`;

  return (
    <div className="stage-inner preview">
      <Canvas
        camera={{ position: [0.14, 0.1, 0.14], fov: 40 }}
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
      >
        <color attach="background" args={["#141210"]} />
        <hemisphereLight args={["#fff4e8", "#1a1612", 0.85]} />
        <directionalLight position={[0.3, 0.45, 0.25]} intensity={1.35} />
        <directionalLight position={[-0.25, 0.2, -0.15]} intensity={0.4} />
        <gridHelper args={[0.24, 12, "#3a3128", "#2a241e"]} />
        <Suspense fallback={null}>
          {textured ? <GlbMesh url={url} scale={scale} /> : <StlMesh url={url} />}
        </Suspense>
        <OrbitControls makeDefault enableDamping />
      </Canvas>
      <div className="preview-bar">
        <p>
          {prompt} · drag to orbit
          {textured ? "" : " · no GLB textures"}
        </p>
        <button onClick={() => useTwin.getState().setPreview(null)}>Close</button>
      </div>
    </div>
  );
}
