import { useEffect, useRef, useState } from "react";
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
  const rlOnline = useTwin((s) => s.rlOnline);
  const status = useTwin((s) => s.status);
  const runId = useTwin((s) => s.runId);
  const episodes = useTwin((s) => s.episodes);
  const metrics = useTwin((s) => s.metrics);
  const renderBusy = useTwin((s) => s.renderBusy);
  const videoUrl = useTwin((s) => s.videoUrl);
  const episodesTarget = useTwin((s) => s.episodesTarget);
  const mintPrompt = useTwin((s) => s.mintPrompt);
  const mintBusy = useTwin((s) => s.mintBusy);
  const mintJobPrompt = useTwin((s) => s.mintJobPrompt);
  const mintConfigured = useTwin((s) => s.mintConfigured);
  const objects = useTwin((s) => s.objects);
  const selectedId = useTwin((s) => s.selectedId);
  const previewId = useTwin((s) => s.previewId);
  const runs = useTwin((s) => s.runs);
  const runRows = [...runs].sort((a, b) => b.mtime - a.mtime || b.id.localeCompare(a.id));
  const selectedRun = runRows.find((r) => r.id === runId);
  const [epWindow, setEpWindow] = useState<{ runId: string; from: number } | null>(null);
  const [epFilter, setEpFilter] = useState<"latest" | "all" | "viewable">("latest");
  const epFrom =
    epWindow && epWindow.runId === runId
      ? epWindow.from
      : episodes.length
        ? Math.max(0, Math.max(...episodes.map((e) => e.index)) - 49)
        : 0;
  const epRows = [...episodes]
    .sort((a, b) => a.index - b.index)
    .filter((e) => {
      if (epFilter === "viewable") return e.has_video;
      if (epFilter === "latest") return e.index >= epFrom;
      return true;
    });
  const glbInput = useRef<HTMLInputElement>(null);
  const [importBusy, setImportBusy] = useState(false);

  const mintWasBusy = useRef(false);
  const mintPost = useRef(false);

  useEffect(() => {
    let stop = false;
    const tick = async () => {
      const s = useTwin.getState();
      try {
        const health = await rl.health();
        if (stop) return;
        s.setRlOnline(Boolean(health));
        if (!health) return;
        s.setMintConfigured(Boolean(health.mint));
        const busy = Boolean(health.mint_busy);
        if (busy) {
          s.setMintBusy(true);
          s.setMintJobPrompt(health.mint_prompt ?? "");
        } else if (!mintPost.current) {
          s.setMintBusy(false);
        }
        if (busy && !mintWasBusy.current) {
          s.log(`Mint generating… ${health.mint_prompt || ""}`.trim());
        }
        if (!busy && mintWasBusy.current && !mintPost.current) {
          if (health.mint_error) s.log(health.mint_error);
          else if (health.mint_last) s.log(`Saved ${health.mint_last.prompt} (${health.mint_last.id})`);
        }
        mintWasBusy.current = busy || mintPost.current;
        const [st, objs, runList] = await Promise.all([rl.status(), rl.objects(), rl.runs()]);
        if (stop) return;
        s.setObjects(objs.items, objs.selected);
        s.setStatus(st);
        s.setRuns(runList);
        const sorted = [...runList].sort((a, b) => b.mtime - a.mtime || b.id.localeCompare(a.id));
        const live = Boolean((st.state === "running" || st.alive) && st.run_id);
        const current = useTwin.getState().runId;
        if (live && (s.followLive || !current)) {
          s.setRunId(st.run_id);
        } else if (!current && sorted[0]) {
          s.selectRun(sorted[0].id, false);
        }
        const id = useTwin.getState().runId;
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

  useEffect(() => {
    if (epFilter !== "latest") return;
    if (!runId || episodes.length === 0) {
      setEpWindow(null);
      return;
    }
    setEpWindow((w) => {
      if (w && w.runId === runId) return w;
      const maxIdx = Math.max(...episodes.map((e) => e.index));
      return { runId, from: Math.max(0, maxIdx - 49) };
    });
  }, [runId, episodes, epFilter]);

  const running = status?.state === "running" || status?.alive;

  return (
    <aside className="panel">
      <header>
        <p className="kicker">Physical AI · SO-101 · PPO</p>
        <h1>TwinPick</h1>
        <p className="lede">
          Training is vision RL in MuJoCo on a real SO-101: cameras in, joints
          out, lift reward. Replay any episode as an mp4.
        </p>
        <p className="metric">
          trainer {rlOnline ? "connected" : "offline — run python ml/server.py"}
          {rlOnline ? ` · mint ${mintConfigured ? "ready" : "off (set MINT_API_KEY)"}` : ""}
        </p>
      </header>

      <section>
        <h2>1. Object</h2>
        <p className="hint">
          Mint or import a GLB. Either way we rescale to 4.5 cm and save it. Use
          any saved object; the next episode picks it up.
        </p>
        <label className="field">
          <span>Prompt</span>
          <textarea
            rows={2}
            value={mintPrompt}
            onChange={(e) => useTwin.getState().setMintPrompt(e.target.value)}
          />
        </label>
        <div className="row">
          <button
            className="primary"
            disabled={!rlOnline || mintBusy || !mintConfigured}
            onClick={() => {
              void (async () => {
                const s = useTwin.getState();
                mintPost.current = true;
                mintWasBusy.current = true;
                s.setMintBusy(true);
                s.setMintJobPrompt(s.mintPrompt);
                s.log(`Mint generating… ${s.mintPrompt}`);
                try {
                  await rl.mintGenerate(s.mintPrompt);
                } catch (e) {
                  mintWasBusy.current = false;
                  s.setMintBusy(false);
                  s.setMintJobPrompt("");
                  s.log(e instanceof Error ? e.message : "mint failed");
                } finally {
                  mintPost.current = false;
                }
              })();
            }}
          >
            {mintBusy ? "Generating…" : "Generate"}
          </button>
        </div>
        {mintBusy && (
          <p className="metric">
            generating{mintJobPrompt ? ` “${mintJobPrompt}”` : ""} · this can take a few minutes
          </p>
        )}
        <label className="field">
          <span>Import GLB</span>
          <input ref={glbInput} type="file" accept=".glb,model/gltf-binary" />
        </label>
        <div className="row">
          <button
            disabled={!rlOnline || importBusy || mintBusy}
            onClick={() => {
              const file = glbInput.current?.files?.[0];
              if (!file) {
                useTwin.getState().log("Choose a .glb file first");
                return;
              }
              void (async () => {
                const s = useTwin.getState();
                setImportBusy(true);
                s.log(`Importing ${file.name}…`);
                try {
                  const meta = await rl.importGlb(file, file.name.replace(/\.glb$/i, ""));
                  const lib = await rl.objects();
                  s.setObjects(lib.items, lib.selected);
                  s.setPreview(meta.id, meta.prompt);
                  s.log(`Imported ${meta.prompt} (${(meta.w ?? 0).toFixed(3)} × ${(meta.h ?? 0).toFixed(3)} m)`);
                  if (glbInput.current) glbInput.current.value = "";
                } catch (e) {
                  s.log(e instanceof Error ? e.message : "import failed");
                } finally {
                  setImportBusy(false);
                }
              })();
            }}
          >
            {importBusy ? "Importing…" : "Import"}
          </button>
        </div>
        <div className="olist">
          {(objects ?? []).length === 0 && <p className="hint">No saved objects yet.</p>}
          {(objects ?? []).map((o) => (
            <div className={o.id === selectedId ? "ob sel" : "ob"} key={o.id}>
              <span>
                {o.prompt}
                <p>
                  {(o.w ?? 0).toFixed(3)} × {(o.h ?? 0).toFixed(3)} m · {o.created ?? o.id}
                </p>
              </span>
              <div className="ob-actions">
                <button
                  className={previewId === o.id ? "on" : ""}
                  disabled={!rlOnline}
                  onClick={() => {
                    useTwin.getState().setPreview(o.id, o.prompt);
                    useTwin.getState().log(`Showing ${o.prompt}`);
                  }}
                >
                  {previewId === o.id ? "showing" : "display"}
                </button>
                <button
                  className={o.id === selectedId ? "on" : ""}
                  disabled={!rlOnline}
                  onClick={() => {
                    void (async () => {
                      const s = useTwin.getState();
                      try {
                        const lib = await rl.selectObject(o.id);
                        s.setObjects(lib.items, lib.selected);
                        s.log(`Using ${o.prompt}`);
                      } catch (e) {
                        s.log(e instanceof Error ? e.message : "select failed");
                      }
                    })();
                  }}
                >
                  {o.id === selectedId ? "selected" : "use"}
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2>2. Runs</h2>
        <p className="hint">Click a run to load its episodes. Resume continues the highlighted run.</p>
        <div className="runs-wrap">
          {(runRows ?? []).length === 0 && <p className="hint">No runs yet.</p>}
          {(runRows ?? []).length > 0 && (
            <table className="runs">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>State</th>
                  <th>Ep</th>
                  <th>Success</th>
                  <th>Videos</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {runRows.map((r) => (
                  <tr
                    key={r.id}
                    className={r.id === runId ? "on" : ""}
                    onClick={() => {
                      const s = useTwin.getState();
                      s.selectRun(r.id);
                      s.log(`Viewing ${r.id}`);
                      void Promise.all([rl.episodes(r.id), rl.metrics(r.id)]).then(([eps, met]) => {
                        s.setEpisodes(eps);
                        s.setMetrics(met);
                      });
                    }}
                  >
                    <td className="run-id">{r.id}</td>
                    <td>{r.state}</td>
                    <td>
                      {r.episodes}
                      {r.episodes_target ? `/${r.episodes_target}` : ""}
                    </td>
                    <td>{(r.success_rate * 100).toFixed(0)}%</td>
                    <td>{r.videos}</td>
                    <td>
                      <button
                        disabled={!rlOnline}
                        onClick={(ev) => {
                          ev.stopPropagation();
                          void rl.openRun(r.id).catch((e) => {
                            useTwin.getState().log(e instanceof Error ? e.message : "open failed");
                          });
                        }}
                      >
                        folder
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>

      <section>
        <h2>3. Train</h2>
        <label className="field">
          <span>Episodes</span>
          <input
            type="number"
            min={2}
            max={100000}
            value={episodesTarget}
            onChange={(e) => useTwin.getState().setEpisodesTarget(Number(e.target.value) || 30000)}
          />
        </label>
        <div className="row">
          <button
            className="primary"
            disabled={!rlOnline || running || !runId || selectedRun?.has_policy === false}
            onClick={() => {
              void (async () => {
                const s = useTwin.getState();
                const id = s.runId;
                if (!id) {
                  s.log("Click a run first");
                  return;
                }
                try {
                  const out = (await rl.start(s.episodesTarget, true, id)) as { run_id: string };
                  s.setRunId(out.run_id);
                  s.log(`Resuming ${out.run_id}`);
                } catch (e) {
                  s.log(e instanceof Error ? e.message : "resume failed");
                }
              })();
            }}
          >
            Resume
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
            disabled={!rlOnline || running || (objects ?? []).length === 0}
            onClick={() => {
              void (async () => {
                const s = useTwin.getState();
                try {
                  const out = (await rl.start(s.episodesTarget)) as { run_id: string };
                  const now = Date.now() / 1000;
                  s.setRunId(out.run_id);
                  s.setVideoUrl(null);
                  s.setRuns([
                    {
                      id: out.run_id,
                      state: "running",
                      episode: 0,
                      episodes_target: s.episodesTarget,
                      success_rate: 0,
                      reward: 0,
                      episodes: 0,
                      videos: 0,
                      has_policy: false,
                      mtime: now,
                    },
                    ...s.runs.filter((r) => r.id !== out.run_id),
                  ]);
                  s.log(`Training ${out.run_id}`);
                } catch (e) {
                  s.log(e instanceof Error ? e.message : "start failed");
                }
              })();
            }}
          >
            Train New
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
        <h2>4. Episodes</h2>
        <label className="field">
          <span>Show</span>
          <select
            value={epFilter}
            onChange={(e) => {
              const next = e.target.value as "latest" | "all" | "viewable";
              setEpFilter(next);
              if (next === "latest") setEpWindow(null);
            }}
          >
            <option value="latest">Latest 50</option>
            <option value="all">All</option>
            <option value="viewable">Viewable</option>
          </select>
        </label>
        {videoUrl && (
          <button onClick={() => useTwin.getState().setVideoUrl(null)}>Stop video</button>
        )}
        <div className="eps">
          {epRows.length === 0 && <p className="hint">No episodes yet.</p>}
          {epRows.map((e) => (
              <div className="ep" key={e.index}>
                <span>
                  #{e.index} {e.success ? "lift" : "miss"} R {e.reward.toFixed(1)}
                </span>
                <div className="ep-actions">
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
                          const s = useTwin.getState();
                          s.setEpisodes(
                            s.episodes.map((row) =>
                              row.index === e.index ? { ...row, has_video: true } : row,
                            ),
                          );
                          s.setVideoUrl(rl.videoUrl(id, e.index) + `?t=${Date.now()}`);
                          s.log(`Episode ${e.index} ready`);
                        } catch (err) {
                          useTwin.getState().log(err instanceof Error ? err.message : "render failed");
                        } finally {
                          useTwin.getState().setRenderBusy(null);
                        }
                      })();
                    }}
                  >
                    {renderBusy === e.index ? "rendering…" : "render"}
                  </button>
                  <button
                    disabled={!runId || !e.has_video}
                    onClick={() => {
                      const id = useTwin.getState().runId;
                      if (!id) return;
                      useTwin.getState().setVideoUrl(rl.videoUrl(id, e.index) + `?t=${Date.now()}`);
                    }}
                  >
                    view
                  </button>
                </div>
              </div>
            ))}
        </div>
      </section>
    </aside>
  );
}
