import { useCallback, useEffect, useRef, useState } from 'react';
import { analyzeAudio, DEFAULT_API, isAlert } from './lib/api.js';
import { usePersistentState } from './lib/storage.js';
import { useAlarm } from './hooks/useAlarm.js';
import { useRecorder } from './hooks/useRecorder.js';
import AlertBanner from './components/AlertBanner.jsx';
import Recorder from './components/Recorder.jsx';
import Uploader from './components/Uploader.jsx';
import Results from './components/Results.jsx';
import AlertLog from './components/AlertLog.jsx';
import Settings from './components/Settings.jsx';

const DEFAULT_SETTINGS = { api: DEFAULT_API, demo: true, threshold: 70, chunk: 5, sound: true, notify: false };
let nextId = 1;

export default function App() {
  const [settings, setSettings] = usePersistentState('bim.settings', DEFAULT_SETTINGS);
  const [history, setHistory] = usePersistentState('bim.history', []);
  const [results, setResults] = useState([]);
  const [activeAlert, setActiveAlert] = useState(null);
  const [pending, setPending] = useState(0);
  const [lastClip, setLastClip] = useState(null);
  const playAlarm = useAlarm();

  // Always read the latest settings from inside long-lived callbacks (mic chunks etc.)
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  const raiseAlert = useCallback(
    entry => {
      setActiveAlert(entry);
      const s = settingsRef.current;
      if (s.sound) playAlarm();
      if (s.notify && 'Notification' in window && Notification.permission === 'granted') {
        new Notification('Possible break-in detected', {
          body: `${entry.label} (${Math.round(entry.confidence * 100)}%)`,
        });
      }
    },
    [playAlarm]
  );

  const recordResult = useCallback(
    (r, name, source) => {
      const alert = isAlert(r, settingsRef.current.threshold);
      const entry = { id: nextId++, time: Date.now(), name, source, alert, ...r };
      setResults(prev => [entry, ...prev].slice(0, 8));
      setHistory(prev => [{ id: entry.id, time: entry.time, label: r.label, confidence: r.confidence, alert }, ...prev].slice(0, 50));
      if (alert) raiseAlert(entry);
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
        setResults(prev => [{ id: nextId++, time: Date.now(), name, source, error: err.message }, ...prev].slice(0, 8));
      } finally {
        setPending(n => n - 1);
      }
    },
    [recordResult]
  );

  const recorder = useRecorder({
    chunkSeconds: settings.chunk,
    onClip: (blob, name, source) => {
      if (source === 'microphone') setLastClip(blob);
      processAudio(blob, name, source);
    },
  });

  // Flash the tab title while an alert is unacknowledged
  useEffect(() => {
    document.title = activeAlert ? 'ALERT – Break-in Monitor' : 'Break-in Monitor';
  }, [activeAlert]);

  const testAlert = () => recordResult({ break_in: true, label: 'glass_breaking', confidence: 0.97, events: [] }, 'test', 'test');

  const status =
    pending > 0 ? { text: 'analysing…', cls: 'warn' }
    : recorder.mode === 'recording' ? { text: 'recording', cls: 'bad' }
    : recorder.mode === 'monitoring' ? { text: 'monitoring', cls: 'warn' }
    : { text: 'idle', cls: 'ok' };

  return (
    <>
      <AlertBanner alert={activeAlert} onAck={() => setActiveAlert(null)} />
      <header>
        <h1>
          Break-in Monitor
          <span className={`pill ${settings.demo ? 'warn' : 'ok'}`}>{settings.demo ? 'demo mode' : 'live backend'}</span>
        </h1>
        <span className={`pill ${status.cls}`}>{status.text}</span>
      </header>
      <main>
        <Recorder recorder={recorder} lastClip={lastClip} />
        <Uploader
          onFile={f => processAudio(f, f.name, 'upload')}
          onReject={name => setResults(prev => [{ id: nextId++, time: Date.now(), name, source: 'upload', error: 'not an audio file' }, ...prev].slice(0, 8))}
        />
        <Results results={results} />
        <AlertLog history={history} onClear={() => setHistory([])} />
        <Settings settings={settings} setSettings={setSettings} onTestAlert={testAlert} />
      </main>
    </>
  );
}
