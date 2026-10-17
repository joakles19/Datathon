import { useEffect, useState } from 'react';
import LevelMeter from './LevelMeter.jsx';

const mmss = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

export default function Recorder({ recorder, lastClip }) {
  const { mode, seconds, analyser, error, startRecording, stopRecording, startMonitoring, stopMonitoring } = recorder;
  const [url, setUrl] = useState(null);

  useEffect(() => {
    if (!lastClip) return;
    const u = URL.createObjectURL(lastClip);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
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
      {url && <audio src={url} controls />}
    </section>
  );
}
