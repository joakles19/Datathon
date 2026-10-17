import { useCallback, useEffect, useRef, useState } from 'react';

function extFor(mime = '') {
  if (mime.includes('mp4')) return 'm4a';
  if (mime.includes('ogg')) return 'ogg';
  return 'webm';
}

/**
 * Microphone capture.
 *  - startRecording / stopRecording: one clip, delivered via onClip(blob, filename, 'microphone')
 *  - startMonitoring / stopMonitoring: back-to-back chunks of `chunkSeconds`, each delivered
 *    via onClip(blob, filename, 'monitor'). Each chunk is a complete, standalone file.
 * `analyser` is exposed for the level meter.
 */
export function useRecorder({ chunkSeconds, onClip }) {
  const [mode, setMode] = useState('idle'); // idle | recording | monitoring
  const [seconds, setSeconds] = useState(0);
  const [analyser, setAnalyser] = useState(null);
  const [error, setError] = useState('');

  const streamRef = useRef(null);
  const ctxRef = useRef(null);
  const recRef = useRef(null);
  const timerRef = useRef(null);
  const chunkTimerRef = useRef(null);
  const modeRef = useRef('idle');
  const onClipRef = useRef(onClip);
  const chunkRef = useRef(chunkSeconds);
  onClipRef.current = onClip;
  chunkRef.current = chunkSeconds;

  const setBothModes = m => {
    modeRef.current = m;
    setMode(m);
  };

  const ensureStream = useCallback(async () => {
    if (streamRef.current?.active) return streamRef.current;
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Microphone access needs HTTPS or localhost.');
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false },
    });
    streamRef.current = stream;

    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    const node = ctx.createAnalyser();
    node.fftSize = 256;
    ctx.createMediaStreamSource(stream).connect(node);
    ctxRef.current = ctx;
    setAnalyser(node);
    return stream;
  }, []);

  const startTimer = () => {
    const t0 = Date.now();
    setSeconds(0);
    timerRef.current = setInterval(() => setSeconds(Math.floor((Date.now() - t0) / 1000)), 250);
  };
  const stopTimer = () => {
    clearInterval(timerRef.current);
    setSeconds(0);
  };

  const beginRecorder = (stream, source) => {
    const parts = [];
    const rec = new MediaRecorder(stream);
    rec.ondataavailable = e => e.data.size && parts.push(e.data);
    rec.onstop = () => {
      const type = rec.mimeType || 'audio/webm';
      const blob = new Blob(parts, { type });
      if (blob.size > 1000) {
        onClipRef.current(blob, `${source}-${Date.now()}.${extFor(type)}`, source);
      }
    };
    rec.start();
    recRef.current = rec;
    return rec;
  };

  const startRecording = useCallback(async () => {
    setError('');
    try {
      const stream = await ensureStream();
      beginRecorder(stream, 'microphone');
      setBothModes('recording');
      startTimer();
    } catch (e) {
      setError(e.message);
    }
  }, [ensureStream]);

  const stopRecording = useCallback(() => {
    if (recRef.current?.state === 'recording') recRef.current.stop();
    stopTimer();
    setBothModes('idle');
  }, []);

  const runChunk = useCallback(stream => {
    const rec = beginRecorder(stream, 'monitor');
    chunkTimerRef.current = setTimeout(() => {
      if (rec.state === 'recording') rec.stop();
      if (modeRef.current === 'monitoring') runChunk(stream);
    }, chunkRef.current * 1000);
  }, []);

  const startMonitoring = useCallback(async () => {
    setError('');
    try {
      const stream = await ensureStream();
      setBothModes('monitoring');
      startTimer();
      runChunk(stream);
    } catch (e) {
      setError(e.message);
    }
  }, [ensureStream, runChunk]);

  const stopMonitoring = useCallback(() => {
    setBothModes('idle'); // set first so the chunk loop won't restart
    clearTimeout(chunkTimerRef.current);
    if (recRef.current?.state === 'recording') recRef.current.stop();
    stopTimer();
  }, []);

  // release the mic when the component unmounts
  useEffect(
    () => () => {
      modeRef.current = 'idle';
      clearTimeout(chunkTimerRef.current);
      clearInterval(timerRef.current);
      if (recRef.current?.state === 'recording') recRef.current.stop();
      streamRef.current?.getTracks().forEach(t => t.stop());
      ctxRef.current?.close();
    },
    []
  );

  return { mode, seconds, analyser, error, startRecording, stopRecording, startMonitoring, stopMonitoring };
}
