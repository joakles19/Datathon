const pct = n => `${Math.round(n * 100)}%`;
const nice = s => (s || 'unknown').replace(/_/g, ' ');
const topScores = scores =>
  Object.entries(scores || {}).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${nice(k)} ${pct(v)}`).join(' · ');

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
          <div key={r.id} className={`result ${r.action === 'alert' ? 'alert' : r.action === 'suppressed' ? 'suppressed' : ''}`}>
            <div className="top">
              <span>
                {r.action === 'alert' ? 'Alert: ' : r.action === 'suppressed' ? 'Suppressed: ' : ''}
                {nice(r.label)}
              </span>
              <span>score {pct(r.confidence)}</span>
            </div>
            {r.message && <div>{r.message}{r.priority ? ` · ${r.priority} priority` : ''}</div>}
            <div className="bar"><i style={{ width: pct(r.confidence) }} /></div>
            <div className="meta">
              {r.source} · {r.name}{r.window ? ` · ${r.window}` : ''} · {new Date(r.time).toLocaleTimeString()}
              {r.processing_ms ? ` · ${Math.round(r.processing_ms)} ms` : ''}
            </div>
            {r.scores && Object.keys(r.scores).length > 0 && <div className="meta">Top scores: {topScores(r.scores)}</div>}
          </div>
        )
      )}
    </section>
  );
}