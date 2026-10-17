// All screens and panels of the app. Each section below is one component.

import { useCallback, useEffect, useRef, useState } from 'react';
import { APP_NAME, CATALOG, checkHealth, levelOf, nice, reportText, runDiagnostics, toWav } from './lib.js';
import { useDevices } from './hooks.js';

const pct = n => `${Math.round(n * 100)}%`;

// ======================================================================
// Logo
// ======================================================================

// Replace public/logo.svg with your own file (or point this at a different file, e.g. 'logo.png').
const LOGO_FILE = 'logo.svg';

function Logo() {
  const [missing, setMissing] = useState(false);
  if (missing) return <div className="logo placeholder">LOGO</div>;
  return (
    <img
      className="logo"
      src={`${import.meta.env.BASE_URL}${LOGO_FILE}`}
      alt="Logo"
      onError={() => setMissing(true)}
    />
  );
}


// ======================================================================
// Clock
// ======================================================================

function Clock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 5000);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="clock">
      <div className="date">
        {now.toLocaleDateString(undefined, { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })}
      </div>
      <div className="time">{now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
    </div>
  );
}


// ======================================================================
// AlertBanner
// ======================================================================

function AlertBanner({ alert, onAck }) {
  if (!alert) return null;
  const urgent = alert.level === 'urgent';
  return (
    <div className={`banner${urgent ? '' : ' notice'}`} role={urgent ? 'alert' : 'status'}>
      <span className="bannertext">
        <b>{urgent ? 'Urgent: ' : ''}{alert.title}</b>
        <span className="bannermeta">
          {alert.detail ? `${alert.detail} · ` : ''}
          {new Date(alert.time).toLocaleTimeString()} · {Math.round(alert.confidence * 100)}% sure
        </span>
      </span>
      <button onClick={onAck}>{urgent ? 'Acknowledge' : 'Dismiss'}</button>
    </div>
  );
}


// ======================================================================
// Sidebar
// ======================================================================

const ICONS = {
  all: 'M4 4h7v7H4z M13 4h7v7h-7z M4 13h7v7H4z M13 13h7v7h-7z',
  mic: 'M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z M19 11a7 7 0 0 1-14 0 M12 18v3',
  upload: 'M12 16V4 M7 9l5-5 5 5 M5 20h14',
  results: 'M5 20V10 M12 20V4 M19 20v-7',
  sounds: 'M3 12h2 M7 8v8 M11 4v16 M15 8v8 M19 10v4',
  history: 'M18 16v-5a6 6 0 1 0-12 0v5l-2 2h16z M10 21h4',
  settings: 'M4 7h10 M18 7h2 M4 17h2 M10 17h10 M14 4v6 M6 14v6',
  devices: 'M9 2v6 M15 2v6 M6 8h12v4a6 6 0 0 1-12 0z M12 18v4',
  troubleshooting: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z M9.1 9a3 3 0 0 1 5.8 1c0 2-3 2.5-3 4 M12 17.5h.01',
};

const NAV_GROUPS = [
  { title: null, items: [{ id: 'all', label: 'Overview' }] },
  {
    title: 'Monitor',
    items: [
      { id: 'mic', label: 'Live microphone' },
      { id: 'upload', label: 'Upload audio' },
    ],
  },
  {
    title: 'Review',
    items: [
      { id: 'results', label: 'Analysis results' },
      { id: 'history', label: 'Alert history' },
    ],
  },
  {
    title: 'Configure',
    items: [
      { id: 'devices', label: 'Devices' },
      { id: 'sounds', label: 'Sounds to detect' },
      { id: 'settings', label: 'Settings' },
    ],
  },
  { title: 'Help', items: [{ id: 'troubleshooting', label: 'Troubleshooting' }] },
];

function Icon({ name }) {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={ICONS[name]} />
    </svg>
  );
}

function Sidebar({ view, onSelect, badges }) {
  return (
    <aside className="sidebar">
      <nav className="sidenav" aria-label="Sections">
        {NAV_GROUPS.map((g, i) => (
          <div key={i} className="navgroup">
            {g.title && <div className="navhead">{g.title}</div>}
            {g.items.map(item => {
              const badge = badges[item.id];
              return (
                <button
                  key={item.id}
                  className={`navbtn${view === item.id ? ' active' : ''}`}
                  onClick={() => onSelect(item.id)}
                  aria-current={view === item.id ? 'page' : undefined}
                >
                  <Icon name={item.id} />
                  <span className="navlabel">{item.label}</span>
                  {badge?.n > 0 && <span className={`badge${badge.danger ? ' danger' : ''}`}>{badge.n}</span>}
                </button>
              );
            })}
          </div>
        ))}
      </nav>
    </aside>
  );
}


// ======================================================================
// Recorder
// ======================================================================

const mmss = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

function Recorder({ recorder, lastClip }) {
  const { mode, seconds, analyser, error, startRecording, stopRecording, startMonitoring, stopMonitoring } = recorder;
  const [url, setUrl] = useState(null);
  const [playErr, setPlayErr] = useState('');

  // Play a WAV conversion of the clip: raw MediaRecorder blobs (webm/mp4/ogg) are the least
  // portable thing to hand to <audio> (no duration in Chrome webm, codec gaps elsewhere).
  useEffect(() => {
    if (!lastClip) return;
    let cancelled = false;
    let u = null;
    setPlayErr('');
    setUrl(null);
    toWav(lastClip, { sampleRate: 44100, channels: 1 })
      .then(wav => {
        if (cancelled) return;
        u = URL.createObjectURL(wav);
        setUrl(u);
      })
      .catch(e => !cancelled && setPlayErr(`Could not prepare playback: ${e.message} (${lastClip.type}, ${lastClip.size} bytes)`));
    return () => {
      cancelled = true;
      if (u) URL.revokeObjectURL(u);
    };
  }, [lastClip]);

  return (
    <section>
      <h2>Live microphone</h2>
      <LevelMeter analyser={analyser} />
      <div className="row" style={{ marginTop: 12 }}>
        {mode === 'recording' ? (
          <button className="danger" onClick={stopRecording}>Stop &amp; analyse</button>
        ) : (
          <button className="primary" onClick={startRecording} disabled={mode === 'monitoring'}>Record</button>
        )}
        {mode === 'monitoring' ? (
          <button className="danger" onClick={stopMonitoring}>Stop monitoring</button>
        ) : (
          <button onClick={startMonitoring} disabled={mode === 'recording'}>Start monitoring</button>
        )}
        <span className="timer">{mmss(seconds)}</span>
      </div>
      {error && <p className="meta err">{error}</p>}
      <p className="meta" style={{ margin: '10px 0 0' }}>
        <b>Record</b> captures one clip and analyses it when you stop. <b>Monitoring</b> keeps the mic open and
        analyses it in short chunks.
      </p>
      {playErr && <p className="meta err">{playErr}</p>}
      {url && (
        <audio
          src={url}
          controls
          onError={e => setPlayErr(`Audio element error code ${e.currentTarget.error?.code ?? '?'}`)}
        />
      )}
    </section>
  );
}


// ======================================================================
// LevelMeter
// ======================================================================

function LevelMeter({ analyser }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const g = canvas.getContext('2d');
    let raf;

    if (!analyser) {
      g.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const data = new Uint8Array(analyser.frequencyBinCount);
    const loop = () => {
      analyser.getByteFrequencyData(data);
      g.clearRect(0, 0, canvas.width, canvas.height);
      const w = canvas.width / data.length;
      g.fillStyle = '#2563eb';
      data.forEach((v, i) => {
        const h = (v / 255) * canvas.height;
        g.fillRect(i * w, canvas.height - h, w - 1, h);
      });
      raf = requestAnimationFrame(loop);
    };
    loop();
    return () => cancelAnimationFrame(raf);
  }, [analyser]);

  return <canvas ref={canvasRef} width={600} height={70} aria-label="Microphone level" />;
}


// ======================================================================
// Uploader
// ======================================================================

const AUDIO_EXT = /\.(wav|mp3|m4a|ogg|webm|flac|aac)$/i;

function Uploader({ onFile, onReject }) {
  const inputRef = useRef(null);
  const [over, setOver] = useState(false);
  const [preview, setPreview] = useState(null);
  const [previewName, setPreviewName] = useState('');

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  const handle = files => {
    for (const f of files) {
      if (!f.type.startsWith('audio/') && !AUDIO_EXT.test(f.name)) {
        onReject(f.name);
        continue;
      }
      setPreview(f ? URL.createObjectURL(f) : null);
      setPreviewName(f.name);
      onFile(f);
    }
  };

  return (
    <section>
      <h2>Upload audio</h2>
      <div
        className={`drop${over ? ' over' : ''}`}
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current.click()}
        onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && inputRef.current.click()}
        onDragEnter={e => { e.preventDefault(); setOver(true); }}
        onDragOver={e => { e.preventDefault(); setOver(true); }}
        onDragLeave={e => { e.preventDefault(); setOver(false); }}
        onDrop={e => { e.preventDefault(); setOver(false); handle(Array.from(e.dataTransfer.files)); }}
      >
        Drop audio files here or click to browse
        <br />
        <span className="meta">wav, mp3, m4a, ogg, webm, flac</span>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        multiple
        hidden
        onChange={e => { handle(Array.from(e.target.files)); e.target.value = ''; }}
      />
      {preview && (
        <>
          <p className="meta" style={{ margin: '10px 0 0' }}>{previewName}</p>
          <audio src={preview} controls />
        </>
      )}
    </section>
  );
}


// ======================================================================
// Results
// ======================================================================

function Results({ results, onClear }) {
  return (
    <section className="full">
      <h2>
        Analysis results
        <button className="mini" onClick={onClear} disabled={!results.length}>Clear</button>
      </h2>
      {results.length === 0 && <div className="empty">Nothing analysed yet.</div>}
      {results.map(r =>
        r.error ? (
          <div key={r.id} className="result err">
            Could not analyse {r.name}: {r.error}. Check the backend URL or switch on demo mode.
          </div>
        ) : (
          <div key={r.id} className={`result${r.level === 'urgent' ? ' alert' : r.level === 'notice' ? ' notice' : ''}`}>
            <div className="top">
              <span>
                {r.level === 'urgent' ? 'Urgent: ' : ''}{r.title || nice(r.label)}
              </span>
              <span>{pct(r.confidence)}</span>
            </div>
            <div className="bar"><i style={{ width: pct(r.confidence) }} /></div>
            <div className="meta">
              {r.detail ? `${r.detail} · ` : ''}{r.source} · {r.name} · {new Date(r.time).toLocaleTimeString()}
              {r.level === 'urgent' ? ' · kept until cleared' : ''}
            </div>
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


// ======================================================================
// SoundList
// ======================================================================

const SOUND_GROUPS = [
  { level: 'urgent', title: 'Urgent', hint: 'Full-screen alert, kept until you clear it.' },
  { level: 'notice', title: 'Everyday', hint: 'Short banner and flash.' },
];

function SoundList({ muted, onChange }) {
  const isOn = id => !muted.includes(id);
  const toggle = id => onChange(isOn(id) ? [...muted, id] : muted.filter(m => m !== id));
  const setGroup = (level, on) => {
    const ids = CATALOG.filter(c => c.level === level).map(c => c.id);
    onChange(on ? muted.filter(m => !ids.includes(m)) : [...new Set([...muted, ...ids])]);
  };

  return (
    <section className="full">
      <h2>Sounds to detect</h2>
      <p className="meta" style={{ margin: '0 0 8px' }}>
        Choose which sounds raise an alert. The backend model has to be able to recognise a sound for it to show up here.
      </p>
      {SOUND_GROUPS.map(g => (
        <div key={g.level} className="soundgroup">
          <div className="grouphead">
            <span>
              <b>{g.title}</b> <span className="meta">{g.hint}</span>
            </span>
            <span className="row" style={{ gap: 6 }}>
              <button className="mini" onClick={() => setGroup(g.level, true)}>All</button>
              <button className="mini" onClick={() => setGroup(g.level, false)}>None</button>
            </span>
          </div>
          <div className="soundgrid">
            {CATALOG.filter(c => c.level === g.level).map(c => (
              <label key={c.id} className={`soundopt${isOn(c.id) ? ' on' : ''}`}>
                <input type="checkbox" checked={isOn(c.id)} onChange={() => toggle(c.id)} />
                <span className="soundtext">
                  <span>{c.title}</span>
                  <small>{c.examples.map(nice).join(', ')}</small>
                </span>
              </label>
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}


// ======================================================================
// AlertLog
// ======================================================================

const TAG = { urgent: 'urgent', notice: 'notice', none: 'clear' };

function AlertLog({ history, onClear }) {
  return (
    <section>
      <h2>
        Alert history
        <button className="mini" onClick={onClear} disabled={!history.length}>Clear</button>
      </h2>
      {history.length === 0 && <div className="empty">No events yet.</div>}
      <div className="logscroll">
        {history.map(h => {
          const level = levelOf(h);
          return (
            <div key={h.id} className="logrow">
              <span>
                {new Date(h.time).toLocaleTimeString()} · {h.title || (h.label || '').replace(/_/g, ' ')}{h.detail ? ` – ${h.detail}` : ''} ({pct(h.confidence)})
              </span>
              <span className={`tag ${level}`}>{TAG[level]}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}


// ======================================================================
// Settings
// ======================================================================

function Settings({ settings, setSettings, onTestAlert }) {
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
        Demo mode (simulated detections, no backend)
      </label>

      <label htmlFor="api">Backend URL</label>
      <input id="api" type="text" value={settings.api} placeholder="http://localhost:8000"
        onChange={e => set({ api: e.target.value.trim() })} />

      <div className="grid2" style={{ marginTop: 12 }}>
        <div>
          <label htmlFor="thr">Alert threshold: <b>{settings.threshold}%</b></label>
          <input id="thr" type="range" min="10" max="99" value={settings.threshold}
            onChange={e => set({ threshold: +e.target.value })} />
        </div>
        <div>
          <label htmlFor="chunk">Monitor chunk (seconds)</label>
          <input id="chunk" type="number" min="2" max="30" value={settings.chunk}
            onChange={e => set({ chunk: Math.min(30, Math.max(2, +e.target.value || 5)) })} />
        </div>
      </div>

      <div style={{ marginTop: 12 }}>
        <label htmlFor="ad">Auto-delete non-urgent events after (seconds, 0 = never)</label>
        <input id="ad" type="number" min="0" max="3600" value={settings.autoDelete}
          onChange={e => set({ autoDelete: Math.min(3600, Math.max(0, Math.floor(+e.target.value) || 0)) })} />
        <p className="meta" style={{ margin: '4px 0 0' }}>Urgent alerts are always kept until you clear them.</p>
      </div>

      <label className="check" style={{ marginTop: 12 }}>
        <input type="checkbox" checked={settings.flash} onChange={e => set({ flash: e.target.checked })} />
        Flash the screen on alerts
      </label>
      <label className="check">
        <input type="checkbox" checked={settings.vibrate} onChange={e => set({ vibrate: e.target.checked })} />
        Vibrate on alerts (phones that support it)
      </label>
      <label className="check">
        <input type="checkbox" checked={settings.sound} onChange={e => set({ sound: e.target.checked })} />
        Play alert sound
      </label>
      <label className="check">
        <input type="checkbox" checked={settings.notify} onChange={e => toggleNotify(e.target.checked)} />
        Browser notifications
      </label>

      <div className="row" style={{ marginTop: 8 }}>
        <button onClick={() => onTestAlert('urgent')}>Test urgent alert</button>
        <button onClick={() => onTestAlert('notice')}>Test everyday alert</button>
        <button onClick={testConnection}>Test connection</button>
      </div>
      {msg && <p className="meta" style={{ margin: '8px 0 0' }}>{msg}</p>}
    </section>
  );
}


// ======================================================================
// Devices
// ======================================================================

const Tag = ({ tone = '', children }) => <span className={`tag ${tone}`}>{children}</span>;

function Row({ name, meta, children }) {
  return (
    <div className="devrow">
      <div>
        <div className="devname">{name}</div>
        {meta && <div className="meta">{meta}</div>}
      </div>
      <div className="devright">{children}</div>
    </div>
  );
}

function Devices({ settings, setSettings, recorder, onTestSound, onOpenHelp }) {
  const { mics, named, permission, error: listError, refresh, requestAccess } = useDevices();
  const set = patch => setSettings(s => ({ ...s, ...patch }));
  const idle = recorder.mode === 'idle';
  const micOpen = recorder.activeDeviceId !== null;

  // ---- microphones ----
  const rows = [
    { id: '', label: 'System default' },
    ...mics.filter(m => m.id !== 'default' && m.id !== 'communications'),
  ];

  const connect = async id => {
    set({ micId: id });
    await recorder.connectMic(id);
    refresh();
  };

  // ---- backend ----
  const [backend, setBackend] = useState('checking'); // checking | demo | ok | error | down
  const checkBackend = useCallback(async () => {
    if (settings.demo) return setBackend('demo');
    setBackend('checking');
    try {
      setBackend((await checkHealth(settings.api)) ? 'ok' : 'error');
    } catch {
      setBackend('down');
    }
  }, [settings.api, settings.demo]);
  useEffect(() => {
    checkBackend();
  }, [checkBackend]);

  const backendTag = {
    checking: <Tag>Checking</Tag>,
    demo: <Tag tone="notice">Demo mode</Tag>,
    ok: <Tag tone="clear">Connected</Tag>,
    error: <Tag tone="notice">Answered with an error</Tag>,
    down: <Tag tone="urgent">Not reachable</Tag>,
  }[backend];

  // ---- alert outputs ----
  const canVibrate = 'vibrate' in navigator;
  const notifSupported = 'Notification' in window;
  const notifPerm = notifSupported ? Notification.permission : 'unsupported';
  const enableNotifications = async () => {
    const perm = await Notification.requestPermission();
    set({ notify: perm === 'granted' });
  };

  return (
    <section className="full">
      <h2>
        Devices
        <button className="mini" onClick={onOpenHelp}>Troubleshooting</button>
      </h2>

      <h3 className="subhead">
        Microphones
        <span className="row" style={{ gap: 6 }}>
          {!named && mics.length > 0 && <button className="mini" onClick={requestAccess}>Allow access</button>}
          <button className="mini" onClick={refresh}>Refresh</button>
        </span>
      </h3>
      {!named && mics.length > 0 && (
        <p className="meta" style={{ margin: '0 0 4px' }}>Device names are hidden until you allow microphone access.</p>
      )}
      {permission === 'denied' && (
        <p className="meta err" style={{ margin: '0 0 4px' }}>
          Microphone access is blocked for this site. Allow it in the browser's site settings, then reload.
        </p>
      )}
      {mics.length === 0 && (
        <p className="meta" style={{ margin: '0 0 4px' }}>No microphones found. Plug one in, then press Refresh.</p>
      )}
      {rows.map(m => {
        const selected = (settings.micId || '') === m.id;
        const connected = micOpen && recorder.activeDeviceId === m.id;
        return (
          <Row key={m.id || 'default'} name={m.label} meta={m.id === '' ? 'Uses whichever microphone the system picks' : ''}>
            {connected ? (
              <>
                <Tag tone="clear">Connected</Tag>
                <button className="mini" onClick={recorder.releaseMic} disabled={!idle}>Disconnect</button>
              </>
            ) : (
              <>
                {selected && <Tag>Selected</Tag>}
                <button className="mini" onClick={() => connect(m.id)} disabled={!idle}>
                  {selected ? 'Connect' : 'Use this'}
                </button>
              </>
            )}
          </Row>
        );
      })}
      {!idle && <p className="meta" style={{ margin: '6px 0 0' }}>Stop recording or monitoring to change microphone.</p>}
      {(recorder.error || listError) && <p className="meta err" style={{ margin: '6px 0 0' }}>{recorder.error || listError}</p>}

      <h3 className="subhead">Backend</h3>
      <Row name="Analysis server" meta={settings.demo ? 'Demo mode: detections are simulated' : settings.api}>
        {backendTag}
        <button className="mini" onClick={checkBackend}>Check</button>
      </Row>

      <h3 className="subhead">Alert outputs</h3>
      <Row name="Screen flash" meta="Flashes the whole screen when an alert fires">
        <Tag tone={settings.flash ? 'clear' : ''}>{settings.flash ? 'On' : 'Off'}</Tag>
      </Row>
      <Row name="Vibration" meta={canVibrate ? 'Vibrates phones and tablets that support it' : 'Not available on this device'}>
        <Tag tone={canVibrate && settings.vibrate ? 'clear' : ''}>{!canVibrate ? 'Not available' : settings.vibrate ? 'On' : 'Off'}</Tag>
        {canVibrate && <button className="mini" onClick={() => navigator.vibrate([200])}>Test</button>}
      </Row>
      <Row name="Notifications" meta="Pop-ups even when this tab is in the background">
        {!notifSupported && <Tag>Not available</Tag>}
        {notifPerm === 'denied' && <Tag tone="urgent">Blocked</Tag>}
        {notifPerm === 'granted' && <Tag tone={settings.notify ? 'clear' : ''}>{settings.notify ? 'On' : 'Off'}</Tag>}
        {notifPerm === 'default' && (
          <>
            <Tag>Not enabled</Tag>
            <button className="mini" onClick={enableNotifications}>Enable</button>
          </>
        )}
      </Row>
      <Row name="Alert sound" meta="Plays a tone through this device's speakers">
        <Tag tone={settings.sound ? 'clear' : ''}>{settings.sound ? 'On' : 'Off'}</Tag>
        <button className="mini" onClick={onTestSound}>Play test sound</button>
      </Row>
    </section>
  );
}


// ======================================================================
// Troubleshooting
// ======================================================================

const TONE = { pass: 'clear', warn: 'notice', fail: 'urgent', skip: '' };
const LABEL = { pass: 'Pass', warn: 'Warning', fail: 'Problem', skip: 'Skipped' };

const FAQ = [
  {
    q: 'The level meter does not move',
    a: 'The wrong microphone may be selected, or it may be muted. Pick the right one in Devices, check for a mute switch or key on the microphone, and make sure the input volume is up in your system sound settings. Another app (a video call, for example) can also hold the microphone; close it and try again.',
  },
  {
    q: 'The browser says microphone access is blocked',
    a: 'Click the padlock or site-settings icon next to the address, set Microphone to Allow, and reload. In Safari on iPhone or iPad use the aA menu, then Website Settings, then Microphone. In Safari on a Mac use Safari, Settings, Websites, Microphone.',
  },
  {
    q: 'My USB or Bluetooth microphone is not listed',
    a: 'Plug it in or connect it, then press Refresh in Devices. If it is still missing, check that your system sees it in its sound settings. Some browsers hide device names until you press Allow access.',
  },
  {
    q: 'I am not getting alerts for a sound',
    a: 'Check that its category is switched on in Sounds to detect, that the alert threshold in Settings is not too high, and that the backend model can recognise the sound. In demo mode detections are random and do not react to real sound.',
  },
  {
    q: 'I get too many false alerts',
    a: 'Raise the alert threshold in Settings, or switch off the categories you do not need in Sounds to detect. Moving the microphone away from the television or speakers also helps.',
  },
  {
    q: 'Alerts stop when the screen locks or I switch tabs',
    a: 'Browsers pause or slow down pages that are in the background, and phones stop the microphone when the screen locks. Keep this tab in front and the screen on, for example with the phone on a charger.',
  },
  {
    q: 'There is no vibration or screen flash',
    a: 'Vibration is not available on desktops or iPhones. Screen flash can be switched off in Settings. If your system is set to reduce motion, the flash is shown as a steady tint instead of pulsing.',
  },
  {
    q: 'The backend cannot be reached',
    a: 'Check the URL in Settings and that the server is running. The server must allow requests from this page (CORS), and if this page is on https the backend must be https as well. Use Test connection in Settings, then run the checks above.',
  },
  {
    q: 'An uploaded file fails to analyse',
    a: 'The browser has to be able to decode the file. WAV, MP3, M4A, OGG and FLAC normally work; some other formats and very large files do not. Try converting the file to WAV or MP3.',
  },
];

function Troubleshooting({ settings }) {
  const [results, setResults] = useState([]);
  const [running, setRunning] = useState(false);
  const [note, setNote] = useState('');

  const run = async () => {
    setRunning(true);
    setNote('');
    setResults([]);
    try {
      await runDiagnostics(settings, setResults);
    } finally {
      setRunning(false);
    }
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(reportText(APP_NAME, results));
      setNote('Report copied.');
    } catch {
      setNote('Could not copy. Select the results and copy them by hand.');
    }
  };

  const count = s => results.filter(r => r.status === s).length;

  return (
    <section className="full">
      <h2>Troubleshooting</h2>

      <h3 className="subhead">
        Automatic checks
        <span className="row" style={{ gap: 6 }}>
          <button className="mini" onClick={copy} disabled={running || !results.length}>Copy report</button>
          <button className="primary mini" onClick={run} disabled={running}>{running ? 'Running…' : 'Run checks'}</button>
        </span>
      </h3>
      {!results.length && !running && (
        <p className="meta" style={{ margin: 0 }}>
          Tests the browser, microphone, backend and alert outputs, and tells you how to fix anything that is wrong.
        </p>
      )}
      {results.length > 0 && (
        <p className="meta" style={{ margin: '0 0 4px' }}>
          {count('pass')} passed, {count('warn')} warning{count('warn') === 1 ? '' : 's'}, {count('fail')} problem{count('fail') === 1 ? '' : 's'}
          {running ? ' so far' : ''}
        </p>
      )}
      {note && <p className="meta" style={{ margin: '0 0 4px' }}>{note}</p>}
      {results.map(r => (
        <div key={r.id} className="devrow">
          <div>
            <div className="devname">{r.title}</div>
            <div className="meta">{r.detail}</div>
            {r.fix && (r.status === 'fail' || r.status === 'warn') && <div className="fix">{r.fix}</div>}
          </div>
          <div className="devright">
            <span className={`tag ${TONE[r.status]}`}>{LABEL[r.status]}</span>
          </div>
        </div>
      ))}

      <h3 className="subhead">Common problems</h3>
      <div className="faq">
        {FAQ.map(item => (
          <details key={item.q}>
            <summary>{item.q}</summary>
            <p>{item.a}</p>
          </details>
        ))}
      </div>
    </section>
  );
}

