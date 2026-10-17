// React hooks: persistent state, alert sound, microphone list, microphone capture.

import { useCallback, useEffect, useRef, useState } from 'react';
import { micErrorMessage } from './lib.js';

// ======================================================================
// Persistent state
// ======================================================================

/** useState that mirrors to localStorage; silently degrades if storage is unavailable. */
function usePersistentState(key, initial) {
  const [value, setValue] = useState(() => {
    try {
      const raw = localStorage.getItem(key);
      if (raw == null) return initial;
      const parsed = JSON.parse(raw);
      return typeof initial === 'object' && !Array.isArray(initial) ? { ...initial, ...parsed } : parsed;
    } catch {
      return initial;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* ignore */
    }
  }, [key, value]);

  return [value, setValue];
}


// ======================================================================
// Alert sound
// ======================================================================

/**
 * Returns a function that plays an alert sound with WebAudio.
 * level 'urgent' = fast four-tone alarm, anything else = two soft beeps.
 */
function useAlarm() {
  const ctxRef = useRef(null);

  return useCallback((level = 'urgent') => {
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      ctxRef.current = ctxRef.current || new Ctx();
      const ctx = ctxRef.current;
      if (ctx.state === 'suspended') ctx.resume();
      const urgent = level === 'urgent';
      const count = urgent ? 4 : 2;
      for (let i = 0; i < count; i++) {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = urgent ? 'square' : 'sine';
        osc.frequency.value = urgent ? (i % 2 ? 660 : 880) : 740;
        gain.gain.value = urgent ? 0.08 : 0.06;
        osc.connect(gain).connect(ctx.destination);
        const t = ctx.currentTime + i * (urgent ? 0.25 : 0.35);
        osc.start(t);
        osc.stop(t + (urgent ? 0.2 : 0.18));
      }
    } catch {
      /* audio not available */
    }
  }, []);
}


// ======================================================================
// Microphone list
// ======================================================================

/** Lists microphones and keeps the list current when one is plugged in or removed. */
function useDevices() {
  const [mics, setMics] = useState([]);
  const [named, setNamed] = useState(false); // device names are hidden until mic permission is granted
  const [permission, setPermission] = useState('unknown'); // granted | prompt | denied | unknown
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setError('This browser cannot list audio devices (it needs HTTPS or localhost).');
      return;
    }
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      const inputs = all.filter(d => d.kind === 'audioinput');
      setMics(inputs.map((d, i) => ({ id: d.deviceId, label: d.label || `Microphone ${i + 1}` })));
      setNamed(inputs.some(d => d.label));
    } catch (e) {
      setError(e.message);
    }
    try {
      const p = await navigator.permissions?.query({ name: 'microphone' });
      if (p) {
        setPermission(p.state);
        p.onchange = () => setPermission(p.state);
      }
    } catch {
      /* permission state not available in this browser */
    }
  }, []);

  useEffect(() => {
    refresh();
    const md = navigator.mediaDevices;
    md?.addEventListener?.('devicechange', refresh);
    return () => md?.removeEventListener?.('devicechange', refresh);
  }, [refresh]);

  /** Ask for microphone permission so device names become visible. */
  const requestAccess = useCallback(async () => {
    setError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(t => t.stop());
    } catch (e) {
      setError(micErrorMessage(e));
    }
    await refresh();
  }, [refresh]);

  return { mics, named, permission, error, refresh, requestAccess };
}


// ======================================================================
// Microphone capture
// ======================================================================

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
 *  - connectMic(id) / releaseMic(): open or close a microphone without recording ('' = system default).
 * `deviceId` is the microphone to use ('' = system default).
 * `analyser` is exposed for the level meter; `activeDeviceId` is null while no microphone is open.
 */
function useRecorder({ chunkSeconds, deviceId = '', onClip }) {
  const [mode, setMode] = useState('idle'); // idle | recording | monitoring
  const [seconds, setSeconds] = useState(0);
  const [analyser, setAnalyser] = useState(null);
  const [activeDeviceId, setActiveDeviceId] = useState(null);
  const [error, setError] = useState('');

  const streamRef = useRef(null);
  const ctxRef = useRef(null);
  const recRef = useRef(null);
  const timerRef = useRef(null);
  const chunkTimerRef = useRef(null);
  const modeRef = useRef('idle');
  const openedForRef = useRef(null);
  const onClipRef = useRef(onClip);
  const chunkRef = useRef(chunkSeconds);
  const deviceRef = useRef(deviceId);
  onClipRef.current = onClip;
  chunkRef.current = chunkSeconds;
  deviceRef.current = deviceId;

  const setBothModes = m => {
    modeRef.current = m;
    setMode(m);
  };

  const closeStream = () => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    ctxRef.current?.close().catch(() => {});
    ctxRef.current = null;
    openedForRef.current = null;
    setAnalyser(null);
    setActiveDeviceId(null);
  };

  const ensureStream = useCallback(async () => {
    const wanted = deviceRef.current || '';
    if (streamRef.current?.active) {
      if (openedForRef.current === wanted) return streamRef.current;
      closeStream(); // a different microphone was chosen
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Microphone access needs HTTPS or localhost.');
    }
    const base = { echoCancellation: false, noiseSuppression: false };
    let stream;
    let fellBack = false;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: wanted ? { ...base, deviceId: { exact: wanted } } : base,
      });
    } catch (e) {
      if (!wanted || !['OverconstrainedError', 'NotFoundError'].includes(e.name)) throw e;
      fellBack = true; // the chosen microphone is gone: use the system default instead
      stream = await navigator.mediaDevices.getUserMedia({ audio: base });
    }
    streamRef.current = stream;
    openedForRef.current = fellBack ? '' : wanted;

    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    const node = ctx.createAnalyser();
    node.fftSize = 256;
    ctx.createMediaStreamSource(stream).connect(node);
    ctxRef.current = ctx;
    setAnalyser(node);
    setActiveDeviceId(openedForRef.current);
    if (fellBack) setError('The selected microphone is not connected, so the system default is being used.');

    // unplugged or revoked while in use
    stream.getAudioTracks()[0]?.addEventListener('ended', () => {
      if (streamRef.current !== stream) return;
      clearTimeout(chunkTimerRef.current);
      clearInterval(timerRef.current);
      setSeconds(0);
      if (recRef.current?.state === 'recording') recRef.current.stop();
      setBothModes('idle');
      closeStream();
      setError('The microphone was disconnected. Reconnect it or choose another in Devices.');
    });
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
      setError(micErrorMessage(e));
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
      setError(micErrorMessage(e));
    }
  }, [ensureStream, runChunk]);

  const stopMonitoring = useCallback(() => {
    setBothModes('idle'); // set first so the chunk loop won't restart
    clearTimeout(chunkTimerRef.current);
    if (recRef.current?.state === 'recording') recRef.current.stop();
    stopTimer();
  }, []);

  /** Open the given microphone ('' = system default) without recording. */
  const connectMic = useCallback(
    async id => {
      if (modeRef.current !== 'idle') return;
      setError('');
      deviceRef.current = id;
      try {
        await ensureStream();
      } catch (e) {
        setError(micErrorMessage(e));
      }
    },
    [ensureStream]
  );

  /** Close the microphone (only while not recording). */
  const releaseMic = useCallback(() => {
    if (modeRef.current === 'idle') closeStream();
  }, []);

  // release the mic when the component unmounts
  useEffect(
    () => () => {
      modeRef.current = 'idle';
      clearTimeout(chunkTimerRef.current);
      clearInterval(timerRef.current);
      if (recRef.current?.state === 'recording') recRef.current.stop();
      streamRef.current?.getTracks().forEach(t => t.stop());
      ctxRef.current?.close().catch(() => {});
    },
    []
  );

  return {
    mode, seconds, analyser, activeDeviceId, error,
    startRecording, stopRecording, startMonitoring, stopMonitoring, connectMic, releaseMic,
  };
}

