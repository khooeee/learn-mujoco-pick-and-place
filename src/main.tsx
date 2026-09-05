import { ConvexProvider, ConvexReactClient } from "convex/react";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Panel } from "./components/Panel";
import { Sim } from "./components/Sim";
import "./index.css";

const convexUrl = import.meta.env.VITE_CONVEX_URL;
const convex = convexUrl ? new ConvexReactClient(convexUrl) : null;

function Shell() {
  return (
    <div className="app">
      <div className="stage">
        <Sim />
      </div>
      <Panel />
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {convex ? (
      <ConvexProvider client={convex}>
        <Shell />
      </ConvexProvider>
    ) : (
      <Shell />
    )}
  </StrictMode>,
);
