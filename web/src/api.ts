// Types and calls for the Python API (server/app.py).

/** auto = from audio for Indian languages, from text otherwise. */
export type Translator = "auto" | "speech" | "text";
export type StageStatus = "pending" | "running" | "done" | "cached" | "failed" | "cancelled";
export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";
export type LogLevel = "info" | "warn" | "error" | "success";

export interface VideoMeta {
  id: string;
  title: string;
  duration: number | null;
  uploader?: string | null;
  thumbnail?: string | null;
  url: string;
  language?: string | null;
}

export interface Gpu {
  name: string;
  vram_gb: number;
  free_gb: number;
  cuda: string;
  torch: string;
}

/** The LLM auto-detected on the server machine (dubber/llm.py) for the translation polish pass. */
export interface LLMStatus {
  state: "checking" | "ok" | "invalid" | "missing";
  detail: string;
  provider: string | null;
  model: string | null;
  local: boolean;
  models: string[];
}

export interface SystemInfo {
  gpu: Gpu | null;
  gpu_error: string | null;
  models: Record<"whisper" | "demucs" | "xtts" | "nllb" | "indictrans2", boolean>;
  ffmpeg: boolean;
  rubberband: boolean;
  llm: LLMStatus;
  hf_token: boolean;
  stages: { key: string; title: string; tool: string }[];
  busy: boolean;
}

export interface TaskSnap {
  description: string;
  fraction: number | null;
  eta_seconds: number | null;
}

export interface StageSnap {
  key: string;
  title: string;
  tool: string;
  status: StageStatus;
  seconds: number | null;
  fraction: number;
  tasks: TaskSnap[];
}

export interface LogLine {
  t: number;
  level: LogLevel;
  text: string;
}

export interface Speaker {
  reference: string;
  reference_seconds: number;
  pitch_hz: number;
  gender: "male" | "female";
  lines: number;
}

export interface Report {
  video: VideoMeta;
  run: string;
  started_at: string;
  finished_at: string;
  gpu: Gpu;
  video_seconds: number;
  processing_seconds: number;
  processing_time: string;
  realtime_factor: number;
  stages: { key: string; seconds: number; cached: boolean }[];
  source_language: string;
  lines: number;
  speakers: Record<string, Speaker>;
  voice_engine: string;
  whisper_model: string;
  translation: { engine?: string; polish_model: string | null };
  timing: {
    sped_up_pct: number;
    avg_speed: number;
    max_speed: number;
    max_late_s: number;
    median_late_s: number;
    lines_with_overflow: number;
  };
  checks: string[];
  work_dir: string;
  source_video: string;
  output: string;
  subtitles: string;
  subtitles_vtt: string;
  report: string;
  media: {
    video: string | null;
    original: string | null;
    srt: string | null;
    vtt: string | null;
    report: string | null;
    report_name: string | null;
    speaker_refs: Record<string, string | null>;
  };
}

export interface JobSnap {
  id: string;
  status: JobStatus;
  url: string;
  video: VideoMeta | null;
  created: number;
  max_minutes: number | null;
  progress: number;
  current: string | null;
  options: {
    max_minutes: number | null;
    voice: string;
    speakers: number | null;
    translator: Translator;
    polish: boolean;
    llm_model: string | null;
    language: string | null;
  };
  started: number | null;
  finished: number | null;
  elapsed: number;
  error: string | null;
  stages: StageSnap[];
  logs: LogLine[];
  report: Report | null;
  version: number;
}

export interface TranscriptRow {
  id: number;
  start: number;
  end: number;
  speaker: string;
  source: string;
  draft: string;
  english: string;
  dub_start: number | null;
  dub_end: number | null;
  speed: number | null;
}

export interface JobRequest {
  url: string;
  max_minutes: number | null;
  voice: "clone" | "edge";
  speakers: number | null;
  translator: Translator;
  polish: boolean;
  llm_model: string | null;
}

const SERVER_DOWN = "Can't reach the dubbing server. Make sure serve.py is running, then try again.";

/** fetch() that rides out a server restart or a network blip: retries connection failures with backoff. */
async function fetchWithRetry(path: string, init: RequestInit, attempts = 4): Promise<Response> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await fetch(path, init);
    } catch (err) {
      if ((err as Error).name === "AbortError") throw err;
      if (attempt >= attempts) throw new Error(SERVER_DOWN);
      await new Promise((r) => setTimeout(r, 600 * attempt));
    }
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetchWithRetry(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* non-JSON error body */
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export const api = {
  system: () => call<SystemInfo>("/api/system"),
  preview: (url: string, signal?: AbortSignal) =>
    call<VideoMeta>(`/api/preview?url=${encodeURIComponent(url)}`, { signal }),
  createJob: (body: JobRequest) => call<JobSnap>("/api/jobs", { method: "POST", body: JSON.stringify(body) }),
  job: (id: string) => call<JobSnap>(`/api/jobs/${id}`),
  jobs: () => call<Pick<JobSnap, "id" | "status" | "progress">[]>("/api/jobs"),
  cancel: (id: string) => call<JobSnap>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  library: () => call<Report[]>("/api/library"),
  refreshLlm: () => call<LLMStatus>("/api/llm/refresh", { method: "POST" }),
  transcript: (reportName: string) =>
    call<TranscriptRow[]>(`/api/library/${encodeURIComponent(reportName)}/transcript`),
};

const YOUTUBE = /^(https?:\/\/)?(www\.|m\.|music\.)?(youtube\.com\/(watch\?.*v=|shorts\/|live\/|embed\/)|youtu\.be\/)[\w-]{6,}/i;

/** Pull the YouTube link out of whatever was pasted (extra text, spaces, line breaks). */
export function cleanUrl(text: string): string {
  const match = text.match(/(https?:\/\/)?(www\.|m\.|music\.)?(youtube\.com|youtu\.be)\/\S+/i);
  return (match ? match[0] : text).trim();
}
export const isYoutubeUrl = (url: string) => YOUTUBE.test(cleanUrl(url));
