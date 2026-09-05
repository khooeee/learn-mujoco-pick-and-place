import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Panel } from "./components/Panel";
import { Sim } from "./components/Sim";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <div className="app">
      <div className="stage">
        <Sim />
      </div>
      <Panel />
    </div>
  </StrictMode>,
);
