import { Center, OrbitControls } from "@react-three/drei";
import { Canvas, useLoader } from "@react-three/fiber";
import { Suspense, useEffect } from "react";
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

export function ObjectPreview({ id, prompt }: { id: string; prompt: string }) {
  const url = `${rl.objectMeshUrl(id)}?t=${id}`;
  return (
    <div className="stage-inner preview">
      <Canvas camera={{ position: [0.14, 0.1, 0.14], fov: 40 }} gl={{ antialias: true }}>
        <color attach="background" args={["#141210"]} />
        <ambientLight intensity={0.55} />
        <directionalLight position={[0.25, 0.4, 0.2]} intensity={1.15} />
        <directionalLight position={[-0.2, 0.15, -0.1]} intensity={0.35} />
        <gridHelper args={[0.24, 12, "#3a3128", "#2a241e"]} />
        <Suspense fallback={null}>
          <StlMesh url={url} />
        </Suspense>
        <OrbitControls makeDefault enableDamping />
      </Canvas>
      <div className="preview-bar">
        <p>{prompt} · drag to orbit</p>
        <button onClick={() => useTwin.getState().setPreview(null)}>Close</button>
      </div>
    </div>
  );
}
