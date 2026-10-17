import { useCallback, useEffect, useRef, useState } from 'react';
import { analyzeAudio, APP_NAME, classifyResult, DEFAULT_API, levelOf } from './lib.js';
import { useAlarm, usePersistentState, useRecorder } from './hooks.js';
import {
  AlertBanner, AlertLog, Clock, Devices, Logo, Recorder, Results, Settings, Sidebar, SoundList, Troubleshooting, Uploader,
} from './components.jsx';

const DEFAULT_SETTINGS = {
  api: DEFAULT_API, demo: true, threshold: 70, chunk: 5,
  sound: true, flash: true, vibrate: true, notify: false,
  autoDelete: 30, muted: [], micId: '',
};
const NOTICE_BANNER_MS = 12000;
let nextId = 1;

// Urgent entries are kept until the user clears them: they are never aged out and are not pushed
// out by other entries. Everything else (everyday sounds, clear results, errors) is capped and
// auto-deleted.
const keepUrgent = (list, maxOther, maxUrgent = 100) => {
  let other = 0;
  let urgent = 0;
  return list.filter(x => (levelOf(x) === 'urgent' ? urgent++ < maxUrgent : other++ < maxOther));
};

export default function App() {
  const [settings, setSettings] = usePersistentState('bim.settings', DEFAULT_SETTINGS);
  const [history, setHistory] = usePersistentState('bim.history', []);
  const [results, setResults] = useState([]);
  const [activeAlert, setActiveAlert] = useState(null);
  const [pending, setPending] = useState(0);
  const [lastClip, setLastClip] = useState(null);
  const [view, setView] = useState('all');
  const playAlarm = useAlarm();

  // Always read the latest settings from inside long-lived callbacks (mic chunks etc.)
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  const raiseAlert = useCallback(
    entry => {
      // an urgent alert stays on screen until acknowledged; everyday ones never replace it
      setActiveAlert(cur => (cur?.level === 'urgent' && entry.level !== 'urgent' ? cur : entry));
      const s = settingsRef.current;
      if (s.sound) playAlarm(entry.level);
      if (s.vibrate && navigator.vibrate) {
        navigator.vibrate(entry.level === 'urgent' ? [400, 150, 400, 150, 400] : [250]);
      }
      if (s.notify && 'Notification' in window && Notification.permission === 'granted') {
        new Notification(`${entry.level === 'urgent' ? 'Urgent: ' : ''}${entry.title}`, {
          body: `${Math.round(entry.confidence * 100)}% sure`,
        });
      }
    },
    [playAlarm]
  );

  const recordResult = useCallback(
    (r, name, source, { force = false } = {}) => {
      const s = settingsRef.current;
      const c = classifyResult(r, force ? {} : { threshold: s.threshold, muted: s.muted });
      const entry = {
        id: nextId++, time: Date.now(), name, source, ...r,
        level: c.level, alert: c.level !== 'none', title: c.title, detail: c.detail, confidence: c.confidence,
      };
      setResults(prev => keepUrgent([entry, ...prev], 8));
      setHistory(prev =>
        keepUrgent(
          [{ id: entry.id, time: entry.time, label: r.label, title: c.title, detail: c.detail, confidence: c.confidence, level: c.level, alert: entry.alert }, ...prev],
          30, 200
        )
      );
      if (entry.alert) raiseAlert(entry);
    },
    [raiseAlert, setHistory]
  );

  const processAudio = useCallback(
    async (blob, name, source) => {
      setPending(n => n + 1);
      try {
        const { api, demo } = settingsRef.current;
        const r = await analyzeAudio(blob, name, { api, demo });
        recordResult(r, name, source);
      } catch (err) {
        setResults(prev => keepUrgent([{ id: nextId++, time: Date.now(), name, source, error: err.message }, ...prev], 8));
      } finally {
        setPending(n => n - 1);
      }
    },
    [recordResult]
  );

  const recorder = useRecorder({
    chunkSeconds: settings.chunk,
    deviceId: settings.micId,
    onClip: (blob, name, source) => {
      if (source === 'microphone') setLastClip(blob);
      processAudio(blob, name, source);
    },
  });

  // Auto-delete anything that is not urgent once it is older than the configured time (0 = never).
  const autoDelete = Number(settings.autoDelete) || 0;
  useEffect(() => {
    if (autoDelete <= 0) return;
    const tick = () => {
      const cutoff = Date.now() - autoDelete * 1000;
      const fresh = x => levelOf(x) === 'urgent' || x.time > cutoff;
      setResults(prev => (prev.every(fresh) ? prev : prev.filter(fresh)));
      setHistory(prev => (prev.every(fresh) ? prev : prev.filter(fresh)));
    };
    tick();
    const t = setInterval(tick, 1000);
    return () => clearInterval(t);
  }, [autoDelete, setHistory]);

  // Everyday alerts hide their banner by themselves; urgent ones wait for Acknowledge.
  useEffect(() => {
    if (activeAlert?.level !== 'notice') return;
    const id = activeAlert.id;
    const t = setTimeout(() => setActiveAlert(cur => (cur?.id === id ? null : cur)), NOTICE_BANNER_MS);
    return () => clearTimeout(t);
  }, [activeAlert]);

  // Show the alert in the tab title while it is on screen
  useEffect(() => {
    document.title = activeAlert
      ? `${activeAlert.level === 'urgent' ? 'URGENT – ' : ''}${activeAlert.title} – ${APP_NAME}`
      : APP_NAME;
  }, [activeAlert]);

  const testAlert = kind =>
    recordResult(
      kind === 'urgent'
        ? { label: 'glass_breaking', confidence: 0.97, events: [] }
        : { label: 'microwave_ding', confidence: 0.95, events: [] },
      'test', 'test', { force: true }
    );

  const status =
    pending > 0 ? { text: 'analysing…', cls: 'warn' }
    : recorder.mode === 'recording' ? { text: 'recording', cls: 'bad' }
    : recorder.mode === 'monitoring' ? { text: 'monitoring', cls: 'warn' }
    : { text: 'idle', cls: 'ok' };

  const urgentCount = history.filter(h => levelOf(h) === 'urgent').length;
  const badges = {
    results: { n: results.length },
    history: { n: urgentCount, danger: true },
  };
  const show = id => view === 'all' || view === id;

  return (
    <>
      <AlertBanner alert={activeAlert} onAck={() => setActiveAlert(null)} />
      {settings.flash && activeAlert && <div key={activeAlert.id} className={`flash ${activeAlert.level}`} aria-hidden="true" />}
      <header>
        <div className="hleft">
          <Logo />
          <h1>
            {APP_NAME}
            <span className={`pill ${settings.demo ? 'warn' : 'ok'}`}>{settings.demo ? 'demo mode' : 'live backend'}</span>
          </h1>
        </div>
        <div className="hright">
          <Clock />
          <span className={`pill ${status.cls}`}>{status.text}</span>
        </div>
      </header>
      <div className="shell">
        <Sidebar view={view} onSelect={setView} badges={badges} />
        <main className={view === 'all' ? '' : 'single'}>
          <div className="pane" hidden={!show('mic')}>
            <Recorder recorder={recorder} lastClip={lastClip} />
          </div>
          <div className="pane" hidden={!show('upload')}>
            <Uploader
              onFile={f => processAudio(f, f.name, 'upload')}
              onReject={name => setResults(prev => keepUrgent([{ id: nextId++, time: Date.now(), name, source: 'upload', error: 'not an audio file' }, ...prev], 8))}
            />
          </div>
          <div className="pane" hidden={view !== 'devices'}>
            <Devices
              settings={settings}
              setSettings={setSettings}
              recorder={recorder}
              onTestSound={() => playAlarm('notice')}
              onOpenHelp={() => setView('troubleshooting')}
            />
          </div>
          <div className="pane" hidden={!show('results')}>
            <Results results={results} onClear={() => setResults([])} />
          </div>
          <div className="pane" hidden={!show('sounds')}>
            <SoundList muted={settings.muted} onChange={muted => setSettings(s => ({ ...s, muted }))} />
          </div>
          <div className="pane" hidden={!show('history')}>
            <AlertLog history={history} onClear={() => setHistory([])} />
          </div>
          <div className="pane" hidden={!show('settings')}>
            <Settings settings={settings} setSettings={setSettings} onTestAlert={testAlert} />
          </div>
          <div className="pane" hidden={view !== 'troubleshooting'}>
            <Troubleshooting settings={settings} />
          </div>
        </main>
      </div>
    </>
  );
}
