import { useCallback, useEffect, useRef, useState } from 'react';
import { ackEvent, analyzeAudio, DEFAULT_API, isAlert } from './lib/api.js';
import { usePersistentState } from './lib/storage.js';
import { useAlarm } from './hooks/useAlarm.js';
import { useRecorder } from './hooks/useRecorder.js';
import AlertBanner from './components/AlertBanner.jsx';
import Recorder from './components/Recorder.jsx';
import Uploader from './components/Uploader.jsx';
import Results from './components/Results.jsx';
import AlertLog from './components/AlertLog.jsx';
import Settings from './components/Settings.jsx';
import SoundPrefs from './components/SoundPrefs.jsx';

const DEFAULT_SETTINGS = { api: DEFAULT_API, demo: true, chunk: 5, sound: true, notify: false };
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
        new Notification(entry.message || 'Sound alert', {
          body: `${entry.label.replace(/_/g, ' ')} (score ${Math.round(entry.confidence * 100)}%)`,
        });
      }
    },
    [playAlarm]
  );

  const recordResult = useCallback(
    (r, name, source) => {
      const alert = isAlert(r);
      const entry = { id: nextId++, time: Date.now(), name, source, alert, ...r };
      setResults(prev => [entry, ...prev].slice(0, 8));
      setHistory(prev => [{ id: entry.id, time: entry.time, label: r.label, confidence: r.confidence, action: r.action }, ...prev].slice(0, 50));
      if (alert) raiseAlert(entry);
    },
    [raiseAlert, setHistory]
  );

  const processAudio = useCallback(
    async (blob, name, source) => {
      setPending(n => n + 1);
      try {
        const { api, demo } = settingsRef.current;
        await analyzeAudio(blob, name, { api, demo, onResult: r => recordResult(r, name, source) });
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
    document.title = activeAlert ? 'ALERT – HearHere' : 'HearHere';
  }, [activeAlert]);

  const testAlert = () =>
    recordResult(
      { label: 'glass_breaking', predicted_label: 'glass_breaking', confidence: 0.97, scores: {}, action: 'alert',
        priority: 'high', message: 'Test alert', event_id: null },
      'test', 'test');

  // Tell the backend the user has seen the alert, then clear the banner
  const acknowledge = async () => {
    const a = activeAlert;
    setActiveAlert(null);
    const { api, demo } = settingsRef.current;
    if (!a?.event_id || demo) return;
    try {
      await ackEvent(api, a.event_id);
    } catch (err) {
      setResults(prev => [{ id: nextId++, time: Date.now(), name: 'acknowledge', source: 'app', error: `could not record acknowledgement (${err.message})` }, ...prev].slice(0, 8));
    }
  };

  const status =
    pending > 0 ? { text: 'analysing…', cls: 'warn' }
    : recorder.mode === 'recording' ? { text: 'recording', cls: 'bad' }
    : recorder.mode === 'monitoring' ? { text: 'monitoring', cls: 'warn' }
    : { text: 'idle', cls: 'ok' };

  return (
    <>
      <AlertBanner alert={activeAlert} onAck={acknowledge} />
      <header>
        <h1>
          HearHere
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
        <SoundPrefs api={settings.api} demo={settings.demo} />
        <Settings settings={settings} setSettings={setSettings} onTestAlert={testAlert} />
      </main>
    </>
  );
}