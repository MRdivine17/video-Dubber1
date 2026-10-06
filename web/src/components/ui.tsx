import type { ReactNode } from "react";

export function ProgressBar({
  fraction,
  size,
  tone,
  running,
}: {
  fraction: number | null;
  size?: "lg";
  tone?: "ok";
  running?: boolean;
}) {
  const indeterminate = fraction == null;
  const classes = ["bar", size, tone, indeterminate ? "indeterminate" : "", running ? "running" : ""]
    .filter(Boolean)
    .join(" ");
  return (
    <div className={classes} role="progressbar" aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : Math.round(fraction * 100)}>
      <i style={{ width: `${Math.max(0, Math.min(1, fraction ?? 0)) * 100}%` }} />
    </div>
  );
}

export function Segmented<T extends string | number | null>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: ReactNode; disabled?: boolean }[];
  onChange: (value: T) => void;
}) {
  return (
    <div className="segmented" role="radiogroup">
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          className={o.value === value ? "on" : ""}
          disabled={o.disabled}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({
  on,
  onChange,
  label,
  disabled,
}: {
  on: boolean;
  onChange: (on: boolean) => void;
  label: ReactNode;
  disabled?: boolean;
}) {
  return (
    <button type="button" role="switch" aria-checked={on} className={`toggle ${on ? "on" : ""}`}
      disabled={disabled} onClick={() => onChange(!on)}>
      <span className="track" />
      <span>{label}</span>
    </button>
  );
}

export function Thumb({ src, duration, alt }: { src?: string | null; duration?: string; alt: string }) {
  return (
    <div className="thumb">
      {src ? <img src={src} alt={alt} loading="lazy" referrerPolicy="no-referrer" /> : null}
      {duration ? <span className="badge">{duration}</span> : null}
    </div>
  );
}
