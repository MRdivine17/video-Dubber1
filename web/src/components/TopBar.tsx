import { AudioLines, Cpu, Sparkles, TriangleAlert } from "lucide-react";
import type { SystemInfo } from "../api";
import { providerName } from "./Composer";

export function TopBar({
  system,
  running,
  onHome,
  onOpenJob,
}: {
  system: SystemInfo | null;
  running: { progress: number } | null;
  onHome: () => void;
  onOpenJob: () => void;
}) {
  const gpu = system?.gpu;
  const modelsReady = system ? Object.entries(system.models).filter(([k]) => k !== "indictrans2").every(([, v]) => v) : false;

  return (
    <header className="topbar">
      <button className="brand" onClick={onHome} aria-label="Home">
        <span className="brand-mark"><AudioLines size={17} /></span>
        <span className="brand-name">Dub <em>Studio</em></span>
      </button>
      <div className="topbar-spacer" />
      <div className="chips">
        {running ? (
          <button className="chip keep" onClick={onOpenJob} style={{ cursor: "pointer" }}>
            <span className="dot busy" />
            Dubbing <span className="mono">{Math.round(running.progress * 100)}%</span>
          </button>
        ) : null}
        {system?.llm ? (
          <span className={`chip ${system.llm.state === "ok" ? "ok" : "warn"}`}
            title={system.llm.detail || "LLM for the translation polish pass"}>
            <Sparkles size={14} />
            {system.llm.state === "ok"
              ? <>{providerName(system.llm.provider)} <span className="mono">{system.llm.model}</span>{system.llm.local ? " · local" : ""}</>
              : system.llm.state === "checking" ? "Detecting LLM"
                : system.llm.state === "invalid" ? `${providerName(system.llm.provider)} unavailable` : "No LLM"}
          </span>
        ) : null}
        <span className={`chip ${modelsReady ? "ok" : "warn"}`} title="Model weights on disk">
          <span className={`dot ${modelsReady ? "ok" : ""}`} />
          {modelsReady ? "Models ready" : "Models downloading"}
        </span>
        {gpu ? (
          <span className="chip ok keep" title={`CUDA ${gpu.cuda} · PyTorch ${gpu.torch}`}>
            <Cpu size={14} />
            {gpu.name.replace("NVIDIA GeForce ", "")}
            <span className="mono">{gpu.vram_gb} GB</span>
          </span>
        ) : system ? (
          <span className="chip err keep"><TriangleAlert size={14} /> No CUDA GPU</span>
        ) : null}
      </div>
    </header>
  );
}
