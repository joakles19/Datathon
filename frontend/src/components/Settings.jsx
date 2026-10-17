import { useState } from 'react';
import { checkHealth } from '../lib/api.js';

export default function Settings({ settings, setSettings, onTestAlert }) {
  const [msg, setMsg] = useState('');
  const set = patch => setSettings(s => ({ ...s, ...patch }));

  const toggleNotify = async checked => {
    if (checked && 'Notification' in window && Notification.permission !== 'granted') {
      const perm = await Notification.requestPermission();
      if (perm !== 'granted') return set({ notify: false });
    }
    set({ notify: checked });
  };

  const testConnection = async () => {
    if (settings.demo) return setMsg('Demo mode is on – no backend is called.');
    setMsg('Checking…');
    try {
      setMsg((await checkHealth(settings.api)) ? 'Backend reachable.' : 'Backend responded with an error.');
    } catch {
      setMsg('Could not reach backend (check the URL and that CORS is enabled).');
    }
  };

  return (
    <section>
      <h2>Settings</h2>
      <label className="check">
        <input type="checkbox" checked={settings.demo} onChange={e => set({ demo: e.target.checked })} />
        Demo mode (simulated results, no backend)
      </label>

      <label htmlFor="api">Backend URL</label>
      <input id="api" type="text" value={settings.api} placeholder="http://localhost:8000"
        onChange={e => set({ api: e.target.value.trim() })} />

      <div style={{ marginTop: 12 }}>
        <div>
          <label htmlFor="chunk">Monitor chunk (2–5 s)</label>
          <input id="chunk" type="number" min="2" max="5" value={settings.chunk}
            onChange={e => set({ chunk: Math.min(5, Math.max(2, +e.target.value || 5)) })} />
        </div>
      </div>

      <label className="check" style={{ marginTop: 12 }}>
        <input type="checkbox" checked={settings.sound} onChange={e => set({ sound: e.target.checked })} />
        Play alarm sound
      </label>
      <label className="check">
        <input type="checkbox" checked={settings.notify} onChange={e => toggleNotify(e.target.checked)} />
        Browser notifications
      </label>

      <div className="row" style={{ marginTop: 8 }}>
        <button onClick={onTestAlert}>Test alert</button>
        <button onClick={testConnection}>Test connection</button>
      </div>
      {msg && <p className="meta" style={{ margin: '8px 0 0' }}>{msg}</p>}
    </section>
  );
}