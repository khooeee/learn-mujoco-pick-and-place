import { v } from "convex/values";
import { action } from "./_generated/server";

async function sleep(ms: number) {
  await new Promise((r) => setTimeout(r, ms));
}

async function fileFromStorage(
  ctx: { storage: { getUrl: (id: string) => Promise<string | null> } },
  storageId: string,
) {
  const url = await ctx.storage.getUrl(storageId as never);
  if (!url) throw new Error("Photo is not in Convex storage");
  const res = await fetch(url);
  if (!res.ok) throw new Error("Could not read photo from storage");
  const bytes = new Uint8Array(await res.arrayBuffer());
  const contentType = res.headers.get("content-type") ?? "image/jpeg";
  return { bytes, contentType, url };
}

export const objectFromPhoto = action({
  args: { storageId: v.id("_storage") },
  handler: async (ctx, { storageId }) => {
    const key = process.env.TRIPO_API_KEY;
    if (!key) throw new Error("Set TRIPO_API_KEY with npx convex env set");
    const file = await fileFromStorage(ctx, storageId);

    const form = new FormData();
    form.append(
      "file",
      new Blob([file.bytes], { type: file.contentType }),
      "object.jpg",
    );
    const up = await fetch("https://api.tripo3d.ai/v2/openapi/upload/sts", {
      method: "POST",
      headers: { Authorization: `Bearer ${key}` },
      body: form,
    });
    if (!up.ok) throw new Error(`Tripo upload ${up.status}: ${await up.text()}`);
    const upJson = (await up.json()) as { data?: { image_token?: string } };
    const token = upJson.data?.image_token;
    if (!token) throw new Error("Tripo upload returned no token");

    const gen = await fetch("https://openapi.tripo3d.ai/v3/generation/image-to-model", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        file_token: token,
        model: "v3.1-20260211",
      }),
    });
    if (!gen.ok) throw new Error(`Tripo generate ${gen.status}: ${await gen.text()}`);
    const genJson = (await gen.json()) as { data?: { task_id?: string } };
    const taskId = genJson.data?.task_id;
    if (!taskId) throw new Error("Tripo did not return a task id");

    for (let i = 0; i < 60; i++) {
      await sleep(2500);
      const poll = await fetch(`https://openapi.tripo3d.ai/v3/tasks/${taskId}`, {
        headers: { Authorization: `Bearer ${key}` },
      });
      const body = (await poll.json()) as {
        data?: { status?: string; output?: { model_url?: string } };
      };
      const status = body.data?.status;
      if (status === "success") {
        const modelUrl = body.data?.output?.model_url;
        if (!modelUrl) throw new Error("Tripo success without model_url");
        const glb = await fetch(modelUrl);
        const blob = await glb.blob();
        const stored = await ctx.storage.store(blob);
        const glbUrl = await ctx.storage.getUrl(stored);
        return { glbUrl };
      }
      if (status === "failed" || status === "cancelled") {
        throw new Error("Tripo generation failed");
      }
    }
    throw new Error("Tripo timed out");
  },
});

export const boxFromMint = action({
  args: {},
  handler: async (ctx) => {
    const key = process.env.MINT_API_KEY;
    if (!key) throw new Error("Set MINT_API_KEY with npx convex env set");

    const gen = await fetch("https://api.mint.gg/v1/models:generate", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt: "A small plain cardboard shipping box, game-ready, no logos",
      }),
    });
    if (!gen.ok) throw new Error(`mint generate ${gen.status}: ${await gen.text()}`);
    const started = (await gen.json()) as { id?: string; operation?: { id?: string } };
    const opId = started.id ?? started.operation?.id;
    if (!opId) throw new Error("mint did not return an operation id");

    let resourceId: string | undefined;
    for (let i = 0; i < 50; i++) {
      await sleep(3000);
      const poll = await fetch(`https://api.mint.gg/v1/operations/${opId}`, {
        headers: { Authorization: `Bearer ${key}` },
      });
      const body = (await poll.json()) as {
        status?: string;
        resource?: { id?: string };
      };
      if (body.status === "failed") throw new Error("mint generation failed");
      if (body.status === "succeeded" || body.resource?.id) {
        resourceId = body.resource?.id;
        if (resourceId) break;
      }
    }
    if (!resourceId) throw new Error("mint timed out");

    const model = await fetch(`https://api.mint.gg/v1/models/${resourceId}`, {
      headers: { Authorization: `Bearer ${key}` },
    });
    const modelJson = (await model.json()) as {
      files?: { url?: string; type?: string; format?: string }[];
    };
    const glbFile = modelJson.files?.find(
      (f) => f.format === "glb" || f.url?.includes(".glb"),
    );
    const fileUrl = glbFile?.url ?? modelJson.files?.[0]?.url;
    if (!fileUrl) throw new Error("mint model has no file url yet");
    const glb = await fetch(fileUrl);
    const blob = await glb.blob();
    const stored = await ctx.storage.store(blob);
    const glbUrl = await ctx.storage.getUrl(stored);
    return { glbUrl };
  },
});
