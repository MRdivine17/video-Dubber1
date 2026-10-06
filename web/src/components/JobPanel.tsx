import { useEffect, useRef } from "react";
import {
  Ban, CircleCheck, CircleDashed, CircleX, Clock, LoaderCircle, RotateCcw, Square, Terminal, Timer, TriangleAlert, Workflow,
} from "lucide-react";
import type { JobSnap, StageSnap, StageStatus } from "../api";
import { fmtClock, fmtDuration, pct } from "../format";
import { isTerminal, useTicker } from "../hooks";
import { ProgressBar, Thumb } from "./ui";

const STATUS_LABEL: Record<JobSnap["status"], string> = {
  queued: "Queued", running: "Dubbing", done: "Finished", failed: "Failed", cancelled: "Cancelled",
};

function StepIcon({ status }: { status: StageStatus }) {
  const icon = {
    pending: <CircleDashed size={17} />,
    running: <LoaderCircle size={17} className="spin" />,
    done: <CircleCheck size={18} />,
    cached: <CircleCheck size={18} />,
    failed: <CircleX size={18} />,
    cancelled: <Ban size={17} />,
  }[status];
  return <span className={`step-icon ${status}`}>{icon}</span>;
}

function Step({ stage, index, now }: { stage: StageSnap; index: number; now: number }) {
  const live = stage.status === "running";
  // Live clock for the running step; server time is only refreshed with each snapshot.
  const seconds = live && stage.seconds != null ? stage.seconds + now : stage.seconds;
  return (
    <li className={`step ${stage.status}`}>
      <StepIcon status={stage.status} />
      <div className="step-body">
        <div className="step-title">
          <span className="step-num">{String(index + 1).padStart(2, "0")}</span>
          <span className="step-name">{stage.title}</span>
          {stage.status === "cached" ? <span className="tag">cached</span> : null}
        </div>
        <div className="step-tool">{stage.tool}</div>
        {live && stage.tasks.length ? (
          <div className="tasks">
            {stage.tasks.map((task, i) => (
              <div key={i}>
                <div className="task-top">
                  <span>{task.description}</span>
                  <span className="mono">
                    {pct(task.fraction)}
                    {task.eta_seconds != null && task.fraction != null && task.fraction < 1 ? ` · ${fmtDuration(task.eta_seconds)} left` : ""}
                  </span>
                </div>
                <ProgressBar fraction={task.fraction} running={task.fraction != null && task.fraction < 1} />
              </div>
            ))}
          </div>
        ) : null}
      </div>
      <span className="step-time">{seconds != null ? fmtDuration(seconds) : ""}</span>
    </li>
  );
}

function LogConsole({ job }: { job: JobSnap }) {
  const ref = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [job.logs.length]);
  const t0 = job.started ?? job.created;
  return (
    <div className="card console">
      <h3 className="card-title"><Terminal size={15} /> Live log <span className="right mono muted">{job.logs.length} lines</span></h3>
      <div className="console-body" ref={ref}
        onScroll={(e) => {
          const el = e.currentTarget;
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
        }}>
        {job.logs.length === 0 ? <div className="console-empty">Waiting for output…</div> : null}
        {job.logs.map((line, i) => (
          <div key={i} className={`log-line ${line.level}`}>
            <span className="t">{fmtClock(Math.max(0, line.t - t0))}</span>
            <span className="x">{line.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function JobPanel({
  job,
  onCancel,
  onRetry,
}: {
  job: JobSnap;
  onCancel: () => void;
  onRetry: () => void;
}) {
  const active = !isTerminal(job.status);
  useTicker(active);

  // Seconds since the last snapshot, so running clocks keep ticking between updates.
  const receivedAt = useRef({ version: job.version, at: Date.now() });
  if (receivedAt.current.version !== job.version) receivedAt.current = { version: job.version, at: Date.now() };
  const drift = active ? (Date.now() - receivedAt.current.at) / 1000 : 0;

  const elapsed = job.elapsed + drift;
  const progress = job.progress;
  const eta = job.status === "running" && progress > 0.04 ? (elapsed * (1 - progress)) / progress : null;
  const current = job.stages.find((s) => s.key === job.current);
  const video = job.video;

  return (
    <>
      <section className="card">
        <div className="job-head">
          <Thumb src={video?.thumbnail} alt={video?.title ?? "video"} duration={video?.duration ? fmtDuration(video.duration) : undefined} />
          <div style={{ minWidth: 0 }}>
            <span className={`status-pill ${job.status}`}>
              {active ? <span className="dot busy" /> : null}
              {STATUS_LABEL[job.status]}
            </span>
            <p className="job-title" style={{ marginTop: 10 }}>{video?.title ?? job.url}</p>
            <div className="meta-row">
              {video?.uploader ? <span>{video.uploader}</span> : null}
              <span>{job.options.max_minutes ? `First ${job.options.max_minutes} min` : "Full video"}</span>
              <span>{job.options.voice === "clone" ? "Cloned voices" : "Neural voices"}</span>
            </div>
          </div>
          <div>
            {active ? (
              <button className="btn danger sm" onClick={onCancel}><Square size={13} /> Cancel</button>
            ) : job.status !== "done" ? (
              <button className="btn sm" onClick={onRetry}><RotateCcw size={14} /> Resume</button>
            ) : null}
          </div>
        </div>

        <div className="overall">
          <div className="overall-top">
            <span className="overall-pct">{Math.round(progress * 100)}%</span>
            <span className="overall-label">
              {job.status === "queued" ? "Waiting for the GPU"
                : job.status === "running" && current ? current.title
                  : job.status === "done" ? "Dub complete" : STATUS_LABEL[job.status]}
            </span>
            <span className="overall-right">
              <span><Clock size={13} />{fmtClock(elapsed)}</span>
              {eta != null ? <span><Timer size={13} />~{fmtDuration(eta)} left</span> : null}
            </span>
          </div>
          <ProgressBar fraction={progress} size="lg" tone={job.status === "done" ? "ok" : undefined} running={job.status === "running"} />
        </div>
      </section>

      {job.status === "failed" && job.error ? (
        <section className="card">
          <div className="error-card">
            <TriangleAlert size={20} />
            <div>
              <strong>The dub stopped with an error.</strong>
              <pre>{job.error}</pre>
              <p className="muted" style={{ margin: "8px 0 0", fontSize: 13 }}>
                Finished steps are cached. Press Resume to continue from where it stopped.
              </p>
            </div>
          </div>
        </section>
      ) : null}

      <div className="job-grid">
        <section className="card">
          <h3 className="card-title"><Workflow size={15} /> Pipeline <span className="right mono muted">8 steps</span></h3>
          <ol className="steps">
            {job.stages.map((stage, i) => <Step key={stage.key} stage={stage} index={i} now={drift} />)}
          </ol>
        </section>
        <LogConsole job={job} />
      </div>
    </>
  );
}
