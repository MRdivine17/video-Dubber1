import { Clock, ListVideo, Timer } from "lucide-react";
import type { Report } from "../api";
import { fmtDuration, languageName, timeAgo } from "../format";
import { Thumb } from "./ui";

export function Library({ items, onOpen }: { items: Report[] | null; onOpen: (report: Report) => void }) {
  return (
    <section>
      <h3 className="card-title"><ListVideo size={15} /> Finished dubs
        <span className="right mono muted">{items ? items.length : ""}</span></h3>
      {items && items.length === 0 ? (
        <div className="empty">No dubs yet. Paste a YouTube link above to make the first one.</div>
      ) : null}
      <div className="lib-grid">
        {(items ?? []).map((r) => (
          <button key={r.report} className="lib-card" onClick={() => onOpen(r)}>
            <Thumb src={r.video.thumbnail} alt={r.video.title} duration={fmtDuration(r.video_seconds)} />
            <div className="lib-body">
              <p className="lib-title">{r.video.title}</p>
              <div className="lib-meta">
                <span>{languageName(r.source_language)} → EN</span>
                <span><Timer size={12} />{r.processing_time}</span>
                <span><Clock size={12} />{timeAgo(r.finished_at)}</span>
                {r.run !== "full" ? <span>{r.run.replace("first-", "first ").replace("min", " min")}</span> : null}
              </div>
            </div>
          </button>
        ))}
      </div>
    </section>
  );
}
