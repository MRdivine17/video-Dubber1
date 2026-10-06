import { useEffect, useRef, useState } from "react";
import {
  ArrowRight, Clock, ClipboardPaste, Info, Languages, Link2, LoaderCircle, MicVocal, RotateCcw, Sparkles, TriangleAlert, User,
  Users, X,
} from "lucide-react";
import { api, cleanUrl, isYoutubeUrl, type JobRequest, type SystemInfo, type Translator, type VideoMeta } from "../api";
import { fmtDuration } from "../format";
import { Segmented, Thumb, Toggle } from "./ui";

type PreviewState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ok"; meta: VideoMeta }
  | { kind: "error"; message: string };

const PROVIDER_NAMES: Record<string, string> = {
  anthropic: "Claude", openai: "OpenAI", gemini: "Gemini", ollama: "Ollama", "openai-compatible": "OpenAI-compatible",
};
export const providerName = (provider: string | null | undefined) =>
  provider ? PROVIDER_NAMES[provider] ?? provider : "LLM";

export function Composer({ system, onStart, onSystemChanged }: {
  system: SystemInfo | null;
  onStart: (req: JobRequest) => Promise<void>;
  onSystemChanged: () => void;
}) {
  // ?url=<link> pre-fills the box (handy for bookmarks and sharing).
  const [url, setUrl] = useState(() => new URLSearchParams(window.location.search).get("url") ?? "");
  const [preview, setPreview] = useState<PreviewState>({ kind: "idle" });
  const [previewNonce, setPreviewNonce] = useState(0);   // bump to re-fetch the preview
  const [length, setLength] = useState<"full" | "clip">("full");
  const [minutes, setMinutes] = useState(2);
  const [voice, setVoice] = useState<"clone" | "edge">("clone");
  const [speakers, setSpeakers] = useState<number | null>(null);
  const [translator, setTranslator] = useState<Translator>("auto");
  const [polish, setPolish] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const [llmModel, setLlmModel] = useState<string | null>(null);   // null = auto-detected model
  const [redetecting, setRedetecting] = useState(false);

  const valid = isYoutubeUrl(url);
  const llm = system?.llm;
  const llmOk = llm?.state === "ok";

  const redetect = async () => {
    setRedetecting(true);
    try {
      await api.refreshLlm();
      setLlmModel(null);
      onSystemChanged();
    } finally {
      setRedetecting(false);
    }
  };

  // Fetch title / thumbnail / length shortly after a valid link is entered.
  useEffect(() => {
    if (!valid) {
      setPreview({ kind: "idle" });
      return;
    }
    const ctrl = new AbortController();
    setPreview({ kind: "loading" });
    const timer = window.setTimeout(() => {
      api.preview(cleanUrl(url), ctrl.signal)
        .then((meta) => setPreview({ kind: "ok", meta }))
        .catch((err: Error) => {
          if (!ctrl.signal.aborted) setPreview({ kind: "error", message: err.message });
        });
    }, 350);
    return () => {
      ctrl.abort();
      window.clearTimeout(timer);
    };
  }, [url, valid, previewNonce]);

  const paste = async () => {
    try {
      setUrl((await navigator.clipboard.readText()).trim());
    } catch {
      inputRef.current?.focus();
    }
  };

  const submit = async () => {
    if (!valid || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await onStart({
        url: cleanUrl(url),
        max_minutes: length === "clip" ? minutes : null,
        voice,
        speakers,
        translator,
        polish: polish && llmOk,
        llm_model: polish && llmOk ? llmModel : null,
      });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  const meta = preview.kind === "ok" ? preview.meta : null;
  const dubSeconds = meta?.duration ? (length === "clip" ? Math.min(meta.duration, minutes * 60) : meta.duration) : null;
  const gpuMissing = system && !system.gpu;

  return (
    <section className="card">
      <form onSubmit={(e) => { e.preventDefault(); submit(); }}>
        <div className="url-row">
          <label className={`url-field ${url && !valid ? "invalid" : ""}`}>
            <Link2 size={19} />
            <input
              ref={inputRef}
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Paste a YouTube link: youtube.com/watch?v=… or youtu.be/…"
              spellCheck={false}
              autoFocus
              aria-label="YouTube video link"
            />
            {url ? (
              <button type="button" className="icon-btn" onClick={() => setUrl("")} aria-label="Clear"><X size={16} /></button>
            ) : (
              <button type="button" className="icon-btn" onClick={paste} aria-label="Paste from clipboard" title="Paste">
                <ClipboardPaste size={16} />
              </button>
            )}
          </label>
          <button className="btn primary lg" type="submit" disabled={!valid || submitting || !!gpuMissing}>
            {submitting ? <LoaderCircle size={18} className="spin" /> : <Languages size={18} />}
            Dub to English
            <ArrowRight size={17} />
          </button>
        </div>

        {url && !valid ? (
          <div className="preview-error"><TriangleAlert size={16} /> That isn't a YouTube video link.</div>
        ) : null}

        {preview.kind === "loading" ? (
          <div className="preview skeleton" aria-busy="true">
            <div className="thumb" />
            <div><div className="sk" style={{ width: "80%" }} /><div className="sk" style={{ width: "45%" }} /></div>
          </div>
        ) : null}
        {preview.kind === "error" ? (
          <div className="preview-error">
            <TriangleAlert size={16} /> {preview.message}
            <button type="button" className="btn sm" onClick={() => setPreviewNonce((n) => n + 1)}>
              <RotateCcw size={13} /> Retry
            </button>
          </div>
        ) : null}
        {meta ? (
          <div className="preview">
            <Thumb src={meta.thumbnail} alt={meta.title} duration={fmtDuration(meta.duration)} />
            <div>
              <p className="preview-title">{meta.title}</p>
              <div className="meta-row">
                {meta.uploader ? <span><User size={14} />{meta.uploader}</span> : null}
                <span><Clock size={14} />{fmtDuration(meta.duration)}</span>
                {dubSeconds ? <span><MicVocal size={14} />Dubbing {length === "clip" ? `first ${fmtDuration(dubSeconds)}` : "full video"}</span> : null}
              </div>
            </div>
          </div>
        ) : null}

        <div className="options">
          <div className="option">
            <span className="option-label"><Clock size={13} /> Length</span>
            <Segmented value={length} onChange={setLength}
              options={[{ value: "full", label: "Full video" }, { value: "clip", label: "First minutes" }]} />
            {length === "clip" ? (
              <div className="number-input">
                <input type="number" min={1} max={600} value={minutes}
                  onChange={(e) => setMinutes(Math.max(1, Math.min(600, Number(e.target.value) || 1)))} aria-label="Minutes" />
                <span className="option-hint">minutes, quick test run</span>
              </div>
            ) : <span className="option-hint">Whole video, start to end</span>}
          </div>

          <div className="option">
            <span className="option-label"><MicVocal size={13} /> Voice</span>
            <Segmented value={voice} onChange={setVoice}
              options={[{ value: "clone", label: "Clone speaker" }, { value: "edge", label: "Neural voice" }]} />
            <span className="option-hint">
              {voice === "clone" ? "XTTS v2 clones each original speaker" : "edge-tts voices matched to gender"}
            </span>
          </div>

          <div className="option">
            <span className="option-label"><Users size={13} /> Speakers</span>
            <Segmented value={speakers} onChange={setSpeakers}
              options={[{ value: null, label: "Auto" }, { value: 1, label: "1" }, { value: 2, label: "2" }, { value: 3, label: "3" }, { value: 4, label: "4" }]} />
            <span className="option-hint">{speakers ? `Force ${speakers} distinct voice${speakers > 1 ? "s" : ""}` : "Detected from the audio"}</span>
          </div>

          <div className="option">
            <span className="option-label"><Languages size={13} /> Translate from</span>
            <Segmented value={translator} onChange={setTranslator}
              options={[{ value: "auto", label: "Auto" }, { value: "speech", label: "Audio" }, { value: "text", label: "Text" }]} />
            <span className="option-hint">
              {translator === "auto" ? "Audio for Indian languages, text for others"
                : translator === "speech" ? "Whisper translates the speech itself"
                  : "IndicTrans2 / NLLB translate the transcript"}
            </span>
            <span className="option-label" style={{ marginTop: 6 }}>
              <Sparkles size={13} /> Translation polish
              <button type="button" className="icon-btn mini" onClick={redetect} disabled={redetecting}
                title="Re-detect API keys and local model servers" aria-label="Re-detect LLM">
                <RotateCcw size={12} className={redetecting ? "spin" : ""} />
              </button>
            </span>
            <Toggle on={polish && llmOk} onChange={setPolish} disabled={!llmOk}
              label={llmOk ? `${providerName(llm?.provider)}${llm?.local ? " (local)" : ""}` : "No LLM available"} />
            {llmOk && llm && llm.models.length > 1 ? (
              <select className="select" value={llmModel ?? llm.model ?? ""} disabled={!polish}
                onChange={(e) => setLlmModel(e.target.value === llm.model ? null : e.target.value)} aria-label="LLM model">
                {llm.models.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            ) : null}
            <span className={`option-hint ${llmOk ? "" : "warn"}`}>
              {llmOk ? `${llmModel ?? llm?.model} refines the translation`
                : llm?.state === "checking" ? "Detecting API keys and local models…"
                  : llm?.state === "invalid" ? `Found but not usable: ${llm.detail}`
                    : "Add a key to .env (see .env.example) or start a local model server"}
            </span>
          </div>
        </div>

        <div className="composer-foot">
          <span className="note">
            <Info size={15} />
            Runs entirely on the GPU. The original video stream is copied untouched; only the speech is replaced.
          </span>
        </div>
        {error ? <div className="banner err" style={{ marginTop: 14 }}><TriangleAlert size={16} />{error}</div> : null}
        {gpuMissing ? <div className="banner err" style={{ marginTop: 14 }}><TriangleAlert size={16} />{system?.gpu_error}</div> : null}
      </form>
    </section>
  );
}
