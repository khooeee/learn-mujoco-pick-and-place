import { useEffect } from "react";
import { generateBoxWithMint } from "../lib/cloud";
import { rl } from "../lib/rlApi";
import { useTwin } from "../store";

function RateChart({ points }: { points: { episode: number; rate: number }[] }) {
  if (points.length < 2) return <p className="hint">Success curve appears after a few episodes.</p>;
  const w = 320;
  const h = 72;
  const maxX = Math.max(...points.map((p) => p.episode), 1);
  const d = points
    .map((p, i) => {
      const x = (p.episode / maxX) * w;
      const y = h - p.rate * (h - 6) - 3;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} width="100%" height="72">
      <path d={d} fill="none" stroke="#e85d04" strokeWidth="2" />
    </svg>
  );
}

export function Panel() {
  const logs = useTwin((s) => s.logs);
  const cloud = useTwin((s) => s.cloud);
  const rlOnline = useTwin((s) => s.rlOnline);
  const status = useTwin((s) => s.status);
  const runId = useTwin((s) => s.runId);
  const episodes = useTwin((s) => s.episodes);
  const metrics = useTwin((s) => s.metrics);
  const renderBusy = useTwin((s) => s.renderBusy);
  const videoUrl = useTwin((s) => s.videoUrl);
  const episodesTarget = useTwin((s) => s.episodesTarget);
  const convexOn = Boolean(import.meta.env.VITE_CONVEX_URL);

  useEffect(() => {
    let stop = false;
    const tick = async () => {
      const s = useTwin.getState();
      try {
        const health = await rl.health();
        if (stop) return;
        s.setRlOnline(Boolean(health));
        if (!health) return;
        const st = await rl.status();
        if (stop) return;
        s.setStatus(st);
        if (st.run_id) s.setRunId(st.run_id);
        const id = st.run_id || s.runId;
        if (id) {
          const [eps, met] = await Promise.all([rl.episodes(id), rl.metrics(id)]);
          if (stop) return;
          s.setEpisodes(eps);
          s.setMetrics(met);
        }
      } catch {
        if (!stop) s.setRlOnline(false);
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), 1500);
    return () => {
      stop = true;
      window.clearInterval(id);
    };
  }, []);

  const running = status?.state === "running" || status?.alive;

  return (
    <aside className="panel">
      <header>
        <p className="kicker">Physical AI · SO-101 · PPO</p>
        <h1>TwinPick</h1>
        <p className="lede">
          Training is vision RL in MuJoCo on a real SO-101: cameras in, joints
          out, lift reward. No IK. Replay any episode as an mp4.
        </p>
        <p className="metric">
          trainer {rlOnline ? "connected" : "offline — run python ml/server.py"}
        </p>
      </header>

      <section>
        <h2>1. Object (optional)</h2>
        <p className="hint">
          {convexOn
            ? "Convex is connected. Mint can generate a box GLB (not used in v1 train yet)."
            : "Train works without keys. For Mint: npx convex dev."}
        </p>
        <div className="row">
          <button disabled={!convexOn} onClick={() => void generateBoxWithMint()}>
            Make object · mint
          </button>
          <span className="status">{cloud.mintStatus}</span>
        </div>
      </section>

      <section>
        <h2>2. Train (headless MuJoCo)</h2>
        <label className="field">
          <span>Episodes</span>
          <input
            type="number"
            min={2}
            max={20000}
            value={episodesTarget}
            onChange={(e) => useTwin.getState().setEpisodesTarget(Number(e.target.value) || 200)}
          />
        </label>
        <div className="row">
          <button
            className="primary"
            disabled={!rlOnline || running}
            onClick={() => {
              void (async () => {
                const s = useTwin.getState();
                try {
                  const out = (await rl.start(s.episodesTarget)) as { run_id: string };
                  s.setRunId(out.run_id);
                  s.setVideoUrl(null);
                  s.log(`Training ${out.run_id}`);
                } catch (e) {
                  s.log(e instanceof Error ? e.message : "start failed");
                }
              })();
            }}
          >
            Train
          </button>
          <button
            disabled={!rlOnline || !running}
            onClick={() => {
              void rl.stop().then(() => useTwin.getState().log("Stop requested"));
            }}
          >
            Stop
          </button>
          <button
            disabled={!rlOnline || running || !runId}
            onClick={() => {
              void (async () => {
                const s = useTwin.getState();
                try {
                  await rl.start(s.episodesTarget, true);
                  s.log("Resuming from last checkpoint");
                } catch (e) {
                  s.log(e instanceof Error ? e.message : "resume failed");
                }
              })();
            }}
          >
            Resume
          </button>
        </div>
        {status && (
          <p className="metric">
            {status.run_id ?? "—"} · {status.state} · ep {status.episode}/{status.episodes_target} ·
            success {(status.success_rate * 100).toFixed(0)}% · R {status.reward.toFixed(2)}
          </p>
        )}
        <RateChart points={metrics} />
        <pre className="logtail">{(status?.log_tail ?? []).slice(-12).join("\n") || " "}</pre>
      </section>

      <section>
        <h2>3. Episodes</h2>
        <p className="hint">Training stays headless. Click render, wait, then watch the mp4.</p>
        {videoUrl && (
          <button onClick={() => useTwin.getState().setVideoUrl(null)}>Clear video</button>
        )}
        <div className="eps">
          {episodes.length === 0 && <p className="hint">No episodes yet.</p>}
          {episodes
            .slice()
            .reverse()
            .slice(0, 80)
            .map((e) => (
              <div className="ep" key={e.index}>
                <span>
                  #{e.index} {e.success ? "lift" : "miss"} R {e.reward.toFixed(1)}
                </span>
                <button
                  disabled={!runId || renderBusy !== null}
                  onClick={() => {
                    const id = useTwin.getState().runId;
                    if (!id) return;
                    useTwin.getState().setRenderBusy(e.index);
                    useTwin.getState().log(`Rendering episode ${e.index}…`);
                    void (async () => {
                      try {
                        await rl.render(id, e.index);
                        useTwin.getState().setVideoUrl(rl.videoUrl(id, e.index) + `?t=${Date.now()}`);
                        useTwin.getState().log(`Episode ${e.index} ready`);
                      } catch (err) {
                        useTwin.getState().log(err instanceof Error ? err.message : "render failed");
                      } finally {
                        useTwin.getState().setRenderBusy(null);
                      }
                    })();
                  }}
                >
                  {renderBusy === e.index ? "rendering…" : e.has_video ? "view mp4" : "render mp4"}
                </button>
              </div>
            ))}
        </div>
      </section>

      <section>
        <h2>Log</h2>
        <ul className="log">
          {logs.map((l) => (
            <li key={l.t + l.text}>{l.text}</li>
          ))}
        </ul>
      </section>
    </aside>
  );
}
