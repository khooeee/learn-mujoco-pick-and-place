import { useTwin } from "../store";

export function Sim() {
  const videoUrl = useTwin((s) => s.videoUrl);
  const status = useTwin((s) => s.status);
  const running = status?.state === "running" || status?.alive;

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
              : "Train, then click render mp4 on an episode. The robot only appears in those videos."}
          </p>
        </div>
      )}
    </div>
  );
}
