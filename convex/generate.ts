import { action } from "./_generated/server";

async function sleep(ms: number) {
  await new Promise((r) => setTimeout(r, ms));
}

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
