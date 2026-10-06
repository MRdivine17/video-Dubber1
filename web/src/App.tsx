import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, LoaderCircle, Plus } from "lucide-react";
import { api, type JobRequest, type Report, type SystemInfo } from "./api";
import { Composer } from "./components/Composer";
import { JobPanel } from "./components/JobPanel";
import { Library } from "./components/Library";
import { ResultView } from "./components/ResultView";
import { TopBar } from "./components/TopBar";
import { isTerminal, useJobStream, useStoredState } from "./hooks";

type View = { kind: "home" } | { kind: "job" } | { kind: "report"; report: Report };

export default function App() {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [systemError, setSystemError] = useState<string | null>(null);
  const [library, setLibrary] = useState<Report[] | null>(null);
  const [jobId, setJobId] = useStoredState<string | null>("dub.activeJob", null);
  const [view, setView] = useState<View>({ kind: jobId ? "job" : "home" });
  const { job, lost } = useJobStream(jobId);

  // The open dub lives in the URL (#report/<name>) so refresh and bookmarks keep it.
  const [linkedReport, setLinkedReport] = useState<string | null>(() => {
    const match = window.location.hash.match(/^#report\/(.+)$/);
    return match ? decodeURIComponent(match[1]) : null;
  });

  useEffect(() => {
    if (!linkedReport || !library) return;
    const report = library.find((r) => r.media.report_name === linkedReport);
    if (report) setView({ kind: "report", report });
    setLinkedReport(null);
  }, [linkedReport, library]);

  useEffect(() => {
    if (linkedReport) return;   // wait until the linked dub is resolved
    const name = view.kind === "report" ? view.report.media.report_name : null;
    const hash = name ? `#report/${encodeURIComponent(name)}` : "";
    if (window.location.hash !== hash) window.history.replaceState(null, "", hash || window.location.pathname);
  }, [view, linkedReport]);

  // System status (GPU, models, detected LLM) refreshes in the background.
  const loadSystem = useCallback(() => {
    api.system()
      .then((s) => { setSystem(s); setSystemError(null); })
      .catch((e: Error) => setSystemError(e.message));
  }, []);
  useEffect(() => {
    loadSystem();
    const id = window.setInterval(loadSystem, 8000);
    return () => window.clearInterval(id);
  }, [loadSystem]);

  // A tab opened while a dub is already running attaches to it.
  useEffect(() => {
    if (jobId) return;
    api.jobs()
      .then((jobs) => {
        const live = jobs.find((j) => j.status === "running" || j.status === "queued");
        if (live) setJobId(live.id);
      })
      .catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadLibrary = useCallback(() => {
    api.library().then(setLibrary).catch(() => setLibrary([]));
  }, []);
  useEffect(loadLibrary, [loadLibrary]);

  // Refresh the library when the active job finishes.
  useEffect(() => {
    if (job?.status === "done") loadLibrary();
  }, [job?.status, loadLibrary]);

  // The server restarted and no longer knows this job: attach to whatever it is running now.
  useEffect(() => {
    if (!lost) return;
    setJobId(null);
    api.jobs()
      .then((jobs) => {
        const live = jobs.find((j) => j.status === "running" || j.status === "queued");
        if (live) setJobId(live.id);
        else setView({ kind: "home" });
      })
      .catch(() => setView({ kind: "home" }));
  }, [lost, setJobId]);

  const start = async (req: JobRequest) => {
    const snap = await api.createJob(req);
    setJobId(snap.id);
    setView({ kind: "job" });
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const retry = async () => {
    if (!job) return;
    await start({
      url: job.url,
      max_minutes: job.options.max_minutes,
      voice: job.options.voice as JobRequest["voice"],
      speakers: job.options.speakers,
      translator: job.options.translator ?? "auto",
      polish: job.options.polish,
      llm_model: job.options.llm_model,
    });
  };

  const goHome = () => {
    setView({ kind: "home" });
    window.scrollTo({ top: 0 });
  };

  const running = job && !isTerminal(job.status) ? { progress: job.progress } : null;

  return (
    <div className="app">
      <TopBar system={system} running={running} onHome={goHome} onOpenJob={() => setView({ kind: "job" })} />
      <main className="main">
        {systemError ? (
          <div className="banner err">Can't reach the dubbing server ({systemError}). Start it with <span className="mono">python serve.py</span>.</div>
        ) : null}

        {view.kind === "home" ? (
          <>
            <div className="hero">
              <span className="eyebrow">YouTube → English dubbing</span>
              <h1>Any language in. <em>English</em> out.</h1>
              <p>Same video, same voices, same energy. Speech is separated from the music, transcribed, translated for
                meaning, re-voiced in a clone of each speaker and placed back in sync.</p>
            </div>
            {running ? (
              <div className="banner info">
                <LoaderCircle size={16} className="spin" />
                A dub is in progress ({Math.round(running.progress * 100)}%).
                <button className="btn sm" onClick={() => setView({ kind: "job" })}>Open progress</button>
              </div>
            ) : null}
            <Composer system={system} onStart={start} onSystemChanged={loadSystem} />
            <Library items={library} onOpen={(report) => setView({ kind: "report", report })} />
          </>
        ) : null}

        {view.kind === "job" ? (
          job ? (
            <>
              <div style={{ display: "flex", gap: 10 }}>
                <button className="btn ghost back" onClick={goHome}><ArrowLeft size={16} /> Home</button>
                {isTerminal(job.status) ? (
                  <button className="btn ghost" onClick={() => { setJobId(null); goHome(); }}><Plus size={16} /> New dub</button>
                ) : null}
              </div>
              <JobPanel job={job} onCancel={() => void api.cancel(job.id)} onRetry={() => void retry()} />
              {job.status === "done" && job.report ? <ResultView report={job.report} /> : null}
            </>
          ) : (
            <div className="empty"><LoaderCircle size={18} className="spin" /> Connecting to the job…</div>
          )
        ) : null}

        {view.kind === "report" ? (
          <>
            <button className="btn ghost back" onClick={goHome}><ArrowLeft size={16} /> All dubs</button>
            <ResultView report={view.report} />
          </>
        ) : null}
      </main>
    </div>
  );
}
