const pct = n => `${Math.round(n * 100)}%`;
const nice = s => (s || 'unknown').replace(/_/g, ' ');

export default function Results({ results }) {
  return (
    <section className="full">
      <h2>Analysis results</h2>
      {results.length === 0 && <div className="empty">Nothing analysed yet.</div>}
      {results.map(r =>
        r.error ? (
          <div key={r.id} className="result alert">
            Could not analyse {r.name}: {r.error}. Check the backend URL or switch on demo mode.
          </div>
        ) : (
          <div key={r.id} className={`result${r.alert ? ' alert' : ''}`}>
            <div className="top">
              <span>{r.alert ? 'Alert: ' : ''}{nice(r.label)}</span>
              <span>{pct(r.confidence)}</span>
            </div>
            <div className="bar"><i style={{ width: pct(r.confidence) }} /></div>
            <div className="meta">{r.source} · {r.name} · {new Date(r.time).toLocaleTimeString()}</div>
            {r.events?.length > 0 && (
              <ul className="events">
                {r.events.map((ev, i) => (
                  <li key={i}>{nice(ev.label)} ({pct(ev.confidence)}) at {ev.start}s–{ev.end}s</li>
                ))}
              </ul>
            )}
          </div>
        )
      )}
    </section>
  );
}
