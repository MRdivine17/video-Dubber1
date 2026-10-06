import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowRight, AudioLines, Captions, Clock, Download, FileBraces, Film, Gauge, Languages, MicVocal, Rows3, Search,
  ShieldCheck, Timer, Users, Zap,
} from "lucide-react";
import { api, type Report, type TranscriptRow } from "../api";
import { fmtClock, fmtDuration, languageName } from "../format";
import { ProgressBar, Segmented } from "./ui";

const STAGE_NAMES: Record<string, string> = {
  download: "Download", separate: "Speech / music split", transcribe: "Transcribe", speakers: "Speakers",
  translate: "Translate", synthesize: "Voice synthesis", mix: "Align & mix", mux: "Save video",
};

function Stat({ icon, label, value, sub, highlight }: {
  icon: React.ReactNode; label: string; value: string; sub?: string; highlight?: boolean;
}) {
  return (
    <div className={`stat ${highlight ? "hl" : ""}`}>
      <div className="stat-label">{icon}{label}</div>
      <div className="stat-value" title={value}>{value}</div>
      {sub ? <div className="stat-sub">{sub}</div> : null}
    </div>
  );
}

function Player({ report, videoRef }: { report: Report; videoRef: React.RefObject<HTMLVideoElement | null> }) {
  const [track, setTrack] = useState<"dub" | "original">("dub");
  const [subs, setSubs] = useState<"on" | "off">("on");
  const pendingSeek = useRef<{ at: number; playing: boolean } | null>(null);
  const src = track === "dub" ? report.media.video : report.media.original;

  // Switching between dubbed and original keeps the playhead and play state.
  const switchTrack = (next: "dub" | "original") => {
    const video = videoRef.current;
    pendingSeek.current = { at: video?.currentTime ?? 0, playing: video ? !video.paused : false };
    setTrack(next);
  };
  const restorePosition = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const pending = pendingSeek.current;
    if (!pending) return;
    e.currentTarget.currentTime = pending.at;
    if (pending.playing) void e.currentTarget.play();
    pendingSeek.current = null;
  };

  useEffect(() => {
    const textTrack = videoRef.current?.textTracks?.[0];
    if (textTrack) textTrack.mode = subs === "on" && track === "dub" ? "showing" : "hidden";
  }, [subs, track, src, videoRef]);

  return (
    <div>
      <div className="player-wrap">
        <video ref={videoRef} key={src ?? "none"} src={src ?? undefined} controls preload="metadata"
          onLoadedMetadata={restorePosition}>
          {report.media.vtt ? <track kind="subtitles" src={report.media.vtt} srcLang="en" label="English" default /> : null}
        </video>
      </div>
      <div className="player-bar">
        <Segmented value={track} onChange={switchTrack}
          options={[
            { value: "dub", label: <><AudioLines size={14} /> English dub</> },
            { value: "original", label: <><Film size={14} /> Original</>, disabled: !report.media.original },
          ]} />
        <Segmented value={subs} onChange={setSubs}
          options={[{ value: "on", label: <><Captions size={14} /> Subtitles</> }, { value: "off", label: "Off" }]} />
      </div>
    </div>
  );
}

function Transcript({ reportName, videoRef }: { reportName: string; videoRef: React.RefObject<HTMLVideoElement | null> }) {
  const [rows, setRows] = useState<TranscriptRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(150);
  const [active, setActive] = useState<number | null>(null);

  useEffect(() => {
    api.transcript(reportName).then(setRows).catch((e: Error) => setError(e.message));
  }, [reportName]);

  const filtered = useMemo(() => {
    if (!rows) return [];
    const q = query.trim().toLowerCase();
    return q ? rows.filter((r) => r.english.toLowerCase().includes(q) || r.source.toLowerCase().includes(q)) : rows;
  }, [rows, query]);

  const seek = (row: TranscriptRow) => {
    const video = videoRef.current;
    setActive(row.id);
    if (!video) return;
    video.currentTime = row.dub_start ?? row.start;
    void video.play();
    video.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  return (
    <section className="card">
      <h3 className="card-title"><Rows3 size={15} /> Transcript & translation
        <span className="right mono muted">{rows ? `${rows.length} lines` : ""}</span></h3>
      <div className="transcript-tools">
        <label className="search"><Search size={15} />
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search source or English" />
        </label>
        <span className="muted" style={{ fontSize: 13 }}>Click a line to play it</span>
      </div>
      {error ? <div className="banner err">{error}</div> : null}
      {!rows && !error ? <div className="sk" style={{ height: 120 }} /> : null}
      {rows ? (
        <div className="table-wrap">
          <table className="transcript">
            <thead><tr><th>Time</th><th>Spk</th><th>Original</th><th>English</th><th style={{ textAlign: "right" }}>Speed</th></tr></thead>
            <tbody>
              {filtered.slice(0, limit).map((row) => (
                <tr key={row.id} className={active === row.id ? "active" : ""} onClick={() => seek(row)}>
                  <td className="time">{fmtClock(row.start)}</td>
                  <td className="spk">{row.speaker}</td>
                  <td className="src">{row.source}</td>
                  <td>{row.english}</td>
                  <td className={`speed ${row.speed && row.speed > 1.15 ? "fast" : ""}`}>{row.speed ? `${row.speed.toFixed(2)}x` : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      {filtered.length > limit ? (
        <button className="btn sm" style={{ marginTop: 12 }} onClick={() => setLimit(filtered.length)}>
          Show all {filtered.length} lines
        </button>
      ) : null}
    </section>
  );
}

export function ResultView({ report }: { report: Report }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const speakers = Object.entries(report.speakers);
  const maxStage = Math.max(...report.stages.map((s) => s.seconds), 1);
  const t = report.timing;

  return (
    <>
      <section className="card">
        <div className="result-head">
          <div style={{ minWidth: 0, flex: 1 }}>
            <span className="status-pill done">English dub ready</span>
            <h2>{report.video.title}</h2>
            <span className="lang-flow">
              {languageName(report.source_language)} <ArrowRight size={15} /> English
              <span className="muted">· {report.run === "full" ? "full video" : report.run.replace("first-", "first ").replace("min", " min")}</span>
            </span>
          </div>
          <div className="actions">
            {report.media.video ? <a className="btn primary" href={report.media.video} download><Download size={16} /> MP4</a> : null}
            {report.media.srt ? <a className="btn" href={report.media.srt} download><Captions size={16} /> SRT</a> : null}
            {report.media.report ? <a className="btn" href={report.media.report} download><FileBraces size={16} /> Report</a> : null}
          </div>
        </div>
        <div style={{ marginTop: 18 }}><Player report={report} videoRef={videoRef} /></div>
      </section>

      <div className="stats">
        <Stat highlight icon={<Timer size={14} />} label="Processing time" value={report.processing_time}
          sub={`${report.realtime_factor.toFixed(2)}x real time`} />
        <Stat icon={<Clock size={14} />} label="Video length" value={fmtDuration(report.video_seconds)} sub={`${report.lines} spoken lines`} />
        <Stat icon={<Languages size={14} />} label="Translation" value={report.translation.engine ?? "MT"}
          sub={report.translation.polish_model ? `polished by ${report.translation.polish_model}` : "no LLM polish"} />
        <Stat icon={<MicVocal size={14} />} label="Voice" value={report.voice_engine === "clone" ? "XTTS v2 clone" : "edge-tts"}
          sub={`${speakers.length} speaker${speakers.length === 1 ? "" : "s"} · ${report.gpu?.name?.replace("NVIDIA GeForce ", "") ?? "GPU"}`} />
      </div>

      <div className="two-col">
        <section className="card">
          <h3 className="card-title"><Gauge size={15} /> Timing vs original</h3>
          <div className="kv"><span className="k">Lines sped up</span><span className="v">{t.sped_up_pct}%</span></div>
          <div style={{ margin: "4px 0 10px" }}><ProgressBar fraction={t.sped_up_pct / 100} /></div>
          <div className="kv"><span className="k">Average speed</span><span className="v">{t.avg_speed.toFixed(2)}x</span></div>
          <div className="kv"><span className="k">Fastest line</span><span className="v">{t.max_speed.toFixed(2)}x</span></div>
          <div className="kv"><span className="k">Median delay vs original</span><span className="v">{t.median_late_s.toFixed(2)} s</span></div>
          <div className="kv"><span className="k">Largest delay</span><span className="v">{t.max_late_s.toFixed(2)} s</span></div>
          <div className="kv"><span className="k">Lines running into the next</span><span className="v">{t.lines_with_overflow}</span></div>
        </section>

        <section className="card">
          <h3 className="card-title"><Users size={15} /> Speakers & cloned voices</h3>
          {speakers.map(([name, spk]) => (
            <div className="speaker" key={name}>
              <span className="avatar">{name}</span>
              <div>
                <div style={{ fontSize: 14 }}>{spk.gender === "female" ? "Female" : "Male"} voice
                  <span className="muted mono" style={{ fontSize: 12 }}> · {Math.round(spk.pitch_hz)} Hz · {spk.lines} lines · {spk.reference_seconds}s reference</span>
                </div>
                {report.media.speaker_refs?.[name] ? (
                  <audio controls preload="none" src={report.media.speaker_refs[name] ?? undefined} aria-label={`${name} voice reference`} />
                ) : null}
              </div>
            </div>
          ))}
        </section>
      </div>

      <div className="two-col">
        <section className="card">
          <h3 className="card-title"><Zap size={15} /> Where the time went</h3>
          <div className="timebars">
            {report.stages.map((s) => (
              <div className="timebar" key={s.key}>
                <span className="name">{STAGE_NAMES[s.key] ?? s.key}{s.cached ? " (cached)" : ""}</span>
                <ProgressBar fraction={s.seconds / maxStage} />
                <span className="val">{fmtDuration(s.seconds)}</span>
              </div>
            ))}
          </div>
        </section>
        <section className="card">
          <h3 className="card-title"><ShieldCheck size={15} /> Output checks</h3>
          <ul className="checks">
            {report.checks.map((c) => <li key={c}><ShieldCheck size={16} />{c}</li>)}
          </ul>
          <div className="kv" style={{ marginTop: 14 }}><span className="k">Speech recognition</span><span className="v">Whisper {report.whisper_model}</span></div>
          <div className="kv"><span className="k">Finished</span><span className="v">{new Date(report.finished_at).toLocaleString()}</span></div>
        </section>
      </div>

      {report.media.report_name ? <Transcript reportName={report.media.report_name} videoRef={videoRef} /> : null}
    </>
  );
}
