const pct = n => `${Math.round(n * 100)}%`;

export default function AlertLog({ history, onClear }) {
  return (
    <section>
      <h2>
        Alert history
        <button className="mini" onClick={onClear} disabled={!history.length}>Clear</button>
      </h2>
      {history.length === 0 && <div className="empty">No events yet.</div>}
      {history.slice(0, 12).map(h => {
        const kind = h.action === 'alert' || h.action === 'suppressed' ? h.action : h.action ? 'clear' : h.alert ? 'alert' : 'clear';
        return (
        <div key={h.id} className="logrow">
          <span>
            {new Date(h.time).toLocaleTimeString()} · {(h.label || '').replace(/_/g, ' ')} ({pct(h.confidence)})
          </span>
          <span className={`tag ${kind}`}>{kind}</span>
        </div>
        );
      })}
    </section>
  );
}