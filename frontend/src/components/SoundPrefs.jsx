import { useCallback, useEffect, useState } from 'react';
import { fetchPrefs, snoozeSound, updatePref } from '../lib/api.js';

const nice = s => s.replace(/_/g, ' ');

/** Per-sound alert settings, read from and saved to the backend. */
export default function SoundPrefs({ api, demo }) {
  const [prefs, setPrefs] = useState(null);
  const [msg, setMsg] = useState('');

  const load = useCallback(async () => {
    setMsg('');
    try {
      setPrefs(await fetchPrefs(api));
    } catch {
      setPrefs(null);
      setMsg('Could not load sound settings from the backend.');
    }
  }, [api]);

  useEffect(() => {
    if (demo) {
      setPrefs(null);
      setMsg('');
    } else {
      load();
    }
  }, [demo, load]);

  const save = async (label, patch) => {
    try {
      const next = await updatePref(api, label, patch);
      setPrefs(p => ({ ...p, [label]: next }));
      setMsg('');
    } catch (err) {
      setMsg(err.message);
    }
  };
  const editLocal = (label, patch) => setPrefs(p => ({ ...p, [label]: { ...p[label], ...patch } }));
  const snooze = async (label, minutes) => {
    try {
      await snoozeSound(api, label, minutes);
      setMsg(`${nice(label)} snoozed for ${minutes} minutes.`);
    } catch (err) {
      setMsg(err.message);
    }
  };

  return (
    <section>
      <h2>Sound alerts</h2>
      {demo && <div className="empty">Turn off demo mode to edit the backend's alert settings.</div>}
      {!demo && !prefs && !msg && <div className="empty">Loading…</div>}
      {prefs && Object.entries(prefs).map(([label, p]) => (
        <div key={label} className="result">
          <label className="check" style={{ margin: 0 }}>
            <input type="checkbox" checked={p.enabled} onChange={e => save(label, { enabled: e.target.checked })} />
            <b style={{ textTransform: 'capitalize' }}>{nice(label)}</b>
          </label>
          <label htmlFor={`thr-${label}`} style={{ marginTop: 8 }}>
            Alert when score is at least <b>{Math.round(p.threshold * 100)}%</b>
          </label>
          <input id={`thr-${label}`} type="range" min="0" max="100" value={Math.round(p.threshold * 100)}
            onChange={e => editLocal(label, { threshold: +e.target.value / 100 })}
            onPointerUp={() => save(label, { threshold: p.threshold })}
            onKeyUp={() => save(label, { threshold: p.threshold })} />
          <div className="row" style={{ marginTop: 8 }}>
            <label htmlFor={`cd-${label}`} style={{ margin: 0 }}>Quiet for</label>
            <input id={`cd-${label}`} type="number" min="0" value={p.cooldown} style={{ width: 80 }}
              onChange={e => editLocal(label, { cooldown: Math.max(0, +e.target.value || 0) })}
              onBlur={() => save(label, { cooldown: p.cooldown })} />
            <span className="meta">seconds after an alert</span>
            <select value={p.priority} onChange={e => save(label, { priority: e.target.value })}>
              <option value="high">high priority</option>
              <option value="normal">normal priority</option>
            </select>
            <button className="mini" onClick={() => snooze(label, 15)}>Snooze 15 min</button>
          </div>
        </div>
      ))}
      <p className="meta" style={{ margin: '10px 0 0' }}>
        Starting values were chosen on validation data. Lowering a threshold catches more sounds but gives more false alerts.
      </p>
      {msg && <p className="meta" style={{ margin: '6px 0 0' }}>{msg}</p>}
    </section>
  );
}