import { trainPolicy } from "../lib/policy";
import { queuePick } from "./Sim";
import { useTwin } from "../store";
import { generateBoxWithMint, generateObjectWithTripo, generateWorldWithMarble } from "../lib/cloud";

function PhotoField({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint: string;
  value: string | null;
  onChange: (url: string | null) => void;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <p className="hint">{hint}</p>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (!f) return;
          onChange(URL.createObjectURL(f));
        }}
      />
      {value && <img src={value} alt="" className="thumb" />}
    </label>
  );
}

export function Panel() {
  const tablePhoto = useTwin((s) => s.tablePhoto);
  const objectPhoto = useTwin((s) => s.objectPhoto);
  const brain = useTwin((s) => s.brain);
  const net = useTwin((s) => s.net);
  const lastLoss = useTwin((s) => s.lastLoss);
  const losses = useTwin((s) => s.losses);
  const logs = useTwin((s) => s.logs);
  const cloud = useTwin((s) => s.cloud);
  const pickBusy = useTwin((s) => s.pickBusy);
  const convexOn = Boolean(import.meta.env.VITE_CONVEX_URL);

  return (
    <aside className="panel">
      <header>
        <p className="kicker">Physical AI · photo twin</p>
        <h1>TwinPick</h1>
        <p className="lede">
          Photo of a real table becomes the scene. A small network learns how to
          bend the arm (copying classic IK). Cloud tools make the 3D room and
          objects — they are not the robot brain.
        </p>
      </header>

      <section>
        <h2>1. Reality</h2>
        <PhotoField
          label="Photo of the table"
          hint="This is the room. World Labs turns it into a 3D world."
          value={tablePhoto}
          onChange={(url) => {
            useTwin.getState().setTablePhoto(url);
            useTwin.getState().log("Table photo loaded (shown on the fake table until Marble finishes)");
          }}
        />
        <PhotoField
          label="Photo of the object to pick"
          hint="Cup, can, snack — Tripo turns this into a 3D object."
          value={objectPhoto}
          onChange={(url) => {
            useTwin.getState().setObjectPhoto(url);
            useTwin.getState().log("Object photo loaded (wrapped on the stand-in cup)");
          }}
        />
      </section>

      <section>
        <h2>2. Cloud 3D (optional until keys)</h2>
        <p className="hint">
          {convexOn
            ? "Convex is connected. Keys live on the Convex dashboard, not in the browser."
            : "Sim works now. For World Labs / Tripo / Mint: run npx convex dev, then set API keys with npx convex env set."}
        </p>
        <div className="row">
          <button
            disabled={!convexOn || !tablePhoto}
            onClick={() => void generateWorldWithMarble()}
          >
            Make room · World Labs
          </button>
          <span className="status">{cloud.worldStatus}</span>
        </div>
        <div className="row">
          <button
            disabled={!convexOn || !objectPhoto}
            onClick={() => void generateObjectWithTripo()}
          >
            Make object · Tripo
          </button>
          <span className="status">{cloud.objectStatus}</span>
        </div>
        <div className="row">
          <button disabled={!convexOn} onClick={() => void generateBoxWithMint()}>
            Make drop box · mint
          </button>
          <span className="status">{cloud.mintStatus}</span>
        </div>
        {cloud.marbleUrl && (
          <a className="marble" href={cloud.marbleUrl} target="_blank" rel="noreferrer">
            Open World Labs viewer
          </a>
        )}
      </section>

      <section>
        <h2>3. Robot brain (on this Mac)</h2>
        <p className="hint">
          Expert = math IK. Learned = a 3→32→3 net trained to copy IK for random
          pick points above the table.
        </p>
        <button
          onClick={() => {
            useTwin.getState().log("Training pick policy…");
            const { net: trained, losses: ls } = trainPolicy();
            useTwin.getState().setNet(trained, ls);
            useTwin.getState().log(`Trained. Final loss ${ls[ls.length - 1]?.toFixed(4)}`);
          }}
        >
          Train pick policy
        </button>
        {lastLoss !== null && (
          <p className="metric">
            loss {lastLoss.toFixed(4)} · {losses.length} checkpoints
            {net ? " · ready" : ""}
          </p>
        )}
        <div className="row">
          <button
            className={brain === "expert" ? "on" : ""}
            onClick={() => useTwin.getState().setBrain("expert")}
          >
            Expert IK
          </button>
          <button
            className={brain === "learned" ? "on" : ""}
            disabled={!net}
            onClick={() => useTwin.getState().setBrain("learned")}
          >
            Learned
          </button>
        </div>
        <div className="row">
          <button
            onClick={() => {
              useTwin.getState().scatterCup();
              useTwin.getState().log("Moved the cup to a new spot");
            }}
          >
            New cup spot
          </button>
          <button className="primary" disabled={pickBusy} onClick={() => queuePick()}>
            Pick it up
          </button>
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
