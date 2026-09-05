import { ObjectPreview } from "./ObjectPreview";
import { useTwin } from "../store";

export function Sim() {
  const videoUrl = useTwin((s) => s.videoUrl);
  const previewId = useTwin((s) => s.previewId);
  const previewPrompt = useTwin((s) => s.previewPrompt);
  const status = useTwin((s) => s.status);
  const running = status?.state === "running" || status?.alive;

  if (previewId) {
    return <ObjectPreview id={previewId} prompt={previewPrompt || previewId} />;
  }

  return (
    <div className="stage-inner">
      {videoUrl ? (
        <video className="replay" src={videoUrl} controls autoPlay loop />
      ) : (
        <div className="empty-stage">
          <p className="kicker">MuJoCo SO-101</p>
          <h2>Episode replay</h2>
          <p>
            {running
              ? "Training is headless. Render an episode from the list when you want to watch."
              : "Display an object, train, then render an episode mp4 (overhead + wrist)."}
          </p>
        </div>
      )}
    </div>
  );
}
