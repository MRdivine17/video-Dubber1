import { useEffect, useState } from "react";
import { api, type JobSnap, type JobStatus } from "./api";

export const TERMINAL: JobStatus[] = ["done", "failed", "cancelled"];
export const isTerminal = (status: JobStatus) => TERMINAL.includes(status);

/**
 * Live job state over Server-Sent Events. Falls back to a fetch + reconnect if the
 * stream drops; reports `lost` if the server no longer knows the job (restarted).
 */
export function useJobStream(jobId: string | null) {
  const [job, setJob] = useState<JobSnap | null>(null);
  const [lost, setLost] = useState(false);

  useEffect(() => {
    setJob(null);
    setLost(false);
    if (!jobId) return;
    let closed = false;
    let source: EventSource | null = null;
    let timer: number | undefined;

    const open = () => {
      source = new EventSource(`/api/jobs/${jobId}/events`);
      source.onmessage = (event) => {
        const snap = JSON.parse(event.data) as JobSnap;
        setJob(snap);
        if (isTerminal(snap.status)) source?.close();
      };
      source.onerror = () => {
        source?.close();
        if (closed) return;
        api
          .job(jobId)
          .then((snap) => {
            setJob(snap);
            if (!isTerminal(snap.status)) timer = window.setTimeout(open, 1500);
          })
          .catch((err: Error) => {
            if (/not found/i.test(err.message)) setLost(true);
            else timer = window.setTimeout(open, 3000);
          });
      };
    };
    open();
    return () => {
      closed = true;
      source?.close();
      window.clearTimeout(timer);
    };
  }, [jobId]);

  return { job, lost };
}

/** Re-render every `ms` while `active` (for ticking clocks). */
export function useTicker(active: boolean, ms = 1000) {
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => setTick((t) => t + 1), ms);
    return () => window.clearInterval(id);
  }, [active, ms]);
}

export function useStoredState<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = window.localStorage.getItem(key);
      return raw ? (JSON.parse(raw) as T) : initial;
    } catch {
      return initial;
    }
  });
  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* storage unavailable (private mode) */
    }
  }, [key, value]);
  return [value, setValue] as const;
}
