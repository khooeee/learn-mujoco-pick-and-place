/**
 * Talks to Convex actions when VITE_CONVEX_URL is set.
 * Generated `api` appears after `npx convex dev`.
 */

import { useTwin } from "../store";

async function callAction(name: string, args: Record<string, unknown>) {
  const url = import.meta.env.VITE_CONVEX_URL;
  if (!url) throw new Error("Convex is not connected");
  const { ConvexHttpClient } = await import("convex/browser");
  const mod = await import("../../convex/_generated/api");
  const client = new ConvexHttpClient(url);
  const parts = name.split(".");
  let ref: unknown = (mod as { api: Record<string, unknown> }).api;
  for (const p of parts) ref = (ref as Record<string, unknown>)[p];
  return client.action(ref as never, args as never);
}

async function blobFromObjectUrl(objectUrl: string, filename: string) {
  const res = await fetch(objectUrl);
  const blob = await res.blob();
  const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
  return file;
}

async function uploadToConvex(file: File): Promise<string> {
  const url = import.meta.env.VITE_CONVEX_URL!;
  const { ConvexHttpClient } = await import("convex/browser");
  const { api } = await import("../../convex/_generated/api");
  const client = new ConvexHttpClient(url);
  const uploadUrl = await client.mutation(api.files.generateUploadUrl as never, {});
  const result = await fetch(uploadUrl, {
    method: "POST",
    headers: { "Content-Type": file.type },
    body: file,
  });
  const json = (await result.json()) as { storageId: string };
  return json.storageId;
}

export async function generateObjectWithTripo() {
  const st = useTwin.getState();
  if (!st.objectPhoto) return;
  st.setCloud({ objectStatus: "uploading…" });
  try {
    const file = await blobFromObjectUrl(st.objectPhoto, "object.jpg");
    const storageId = await uploadToConvex(file);
    st.setCloud({ objectStatus: "Tripo generating…" });
    const out = (await callAction("generate.objectFromPhoto", { storageId })) as {
      glbUrl?: string;
    };
    st.setCloud({ objectStatus: "ready", objectGlbUrl: out.glbUrl });
    st.log("Tripo object is in the scene");
  } catch (e) {
    st.setCloud({ objectStatus: "error" });
    st.log(`Tripo: ${e instanceof Error ? e.message : "failed"}`);
  }
}

export async function generateBoxWithMint() {
  const st = useTwin.getState();
  st.setCloud({ mintStatus: "mint generating…" });
  try {
    const out = (await callAction("generate.boxFromMint", {})) as { glbUrl?: string };
    st.setCloud({ mintStatus: "ready", objectGlbUrl: out.glbUrl });
    st.log("mint object is in the scene — this is what the arm will pick");
  } catch (e) {
    st.setCloud({ mintStatus: "error" });
    st.log(`mint: ${e instanceof Error ? e.message : "failed"}`);
  }
}
