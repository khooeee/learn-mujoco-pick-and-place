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

export async function generateBoxWithMint() {
  const st = useTwin.getState();
  st.setCloud({ mintStatus: "mint generating…" });
  try {
    const out = (await callAction("generate.boxFromMint", {})) as { glbUrl?: string };
    st.setCloud({ mintStatus: "ready", objectGlbUrl: out.glbUrl });
    st.log("mint object generated");
  } catch (e) {
    st.setCloud({ mintStatus: "error" });
    st.log(`mint: ${e instanceof Error ? e.message : "failed"}`);
  }
}
