import { useCallback, useEffect, useRef, useState } from 'react';
import { encodeWav } from '../lib/wav.js';

function extFor(mime = '') {
  if (mime.includes('mp4')) return 'm4a';
  if (mime.includes('ogg')) return 'ogg';
  return 'webm';
}

/**
 * Microphone capture.
 *  - startRecording / stopRecording: one clip, delivered via onClip(blob, filename, 'microphone')
 *  - startMonitoring / stopMonitoring: rolling five-second WAV windows, emitted once per second.
 * `analyser` is exposed for the level meter.
 */
export function useRecorder({ onClip }) {
  const [mode, setMode] = useState('idle'); // idle | recording | monitoring
  const [seconds, setSeconds] = useState(0);
  const [analyser, setAnalyser] = useState(null);
  const [error, setError] = useState('');

  const streamRef = useRef(null);
  const ctxRef = useRef(null);
  const recRef = useRef(null);
  const timerRef = useRef(null);
  const modeRef = useRef('idle');
  const onClipRef = useRef(onClip);
  const sourceRef = useRef(null);
  const monitorGraphRef = useRef(null);
  onClipRef.current = onClip;

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
      audio: {
        channelCount: { ideal: 1 },
        echoCancellation: { ideal: true },
        noiseSuppression: { ideal: true },
        autoGainControl: { ideal: true },
      },
    });
    streamRef.current = stream;

    const Ctx = window.AudioContext || window.webkitAudioContext;
    let ctx;
    try {
      ctx = new Ctx({ sampleRate: 16000 });
    } catch {
      ctx = new Ctx();
    }
    const node = ctx.createAnalyser();
    node.fftSize = 256;
    const source = ctx.createMediaStreamSource(stream);
    source.connect(node);
    sourceRef.current = source;
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

  const startRollingCapture = useCallback(async () => {
    const ctx = ctxRef.current;
    const source = sourceRef.current;
    if (!ctx || !source || !ctx.createScriptProcessor) {
      throw new Error('This browser does not support rolling live audio analysis.');
    }
    await ctx.resume();
    if (modeRef.current !== 'monitoring') return;

    const sampleRate = ctx.sampleRate;
    const windowSize = sampleRate * 5;
    const hopSize = sampleRate;
    const ring = new Float32Array(windowSize);
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    const silentOutput = ctx.createGain();
    silentOutput.gain.value = 0;
    const meter = ctx.createAnalyser();
    meter.fftSize = 256;
    let writeAt = 0;
    let captured = 0;
    let lastEmittedAt = 0;
    let windowIndex = 0;
    const streamId = crypto.randomUUID().replaceAll('-', '');

    processor.onaudioprocess = event => {
      const input = event.inputBuffer.getChannelData(0);
      for (let i = 0; i < input.length; i++) {
        ring[writeAt] = input[i];
        writeAt = (writeAt + 1) % windowSize;
      }
      captured += input.length;
      if (captured < windowSize || captured - lastEmittedAt < hopSize) return;

      const window = new Float32Array(windowSize);
      window.set(ring.subarray(writeAt));
      window.set(ring.subarray(0, writeAt), windowSize - writeAt);
      lastEmittedAt = captured;
      const blob = encodeWav([window], sampleRate);
      onClipRef.current(blob, `monitor-${streamId}-${windowIndex++}.wav`, 'monitor');
    };

    source.connect(processor);
    source.connect(meter);
    processor.connect(silentOutput);
    silentOutput.connect(ctx.destination);
    monitorGraphRef.current = { processor, silentOutput, meter, source };
    setAnalyser(meter);
  }, []);

  const stopRollingCapture = () => {
    const graph = monitorGraphRef.current;
    if (!graph) return;
    graph.processor.onaudioprocess = null;
    graph.source.disconnect(graph.processor);
    graph.source.disconnect(graph.meter);
    graph.processor.disconnect();
    graph.silentOutput.disconnect();
    monitorGraphRef.current = null;
  };

  const startMonitoring = useCallback(async () => {
    setError('');
    try {
      await ensureStream();
      setBothModes('monitoring');
      startTimer();
      await startRollingCapture();
    } catch (e) {
      setBothModes('idle');
      stopTimer();
      setError(e.message);
    }
  }, [ensureStream, startRollingCapture]);

  const stopMonitoring = useCallback(() => {
    setBothModes('idle');
    stopRollingCapture();
    stopTimer();
  }, []);

  // release the mic when the component unmounts
  useEffect(
    () => () => {
      modeRef.current = 'idle';
      clearInterval(timerRef.current);
      if (monitorGraphRef.current) {
        monitorGraphRef.current.processor.onaudioprocess = null;
        monitorGraphRef.current.source.disconnect(monitorGraphRef.current.processor);
        monitorGraphRef.current.source.disconnect(monitorGraphRef.current.meter);
        monitorGraphRef.current.processor.disconnect();
        monitorGraphRef.current.silentOutput.disconnect();
      }
      if (recRef.current?.state === 'recording') recRef.current.stop();
      streamRef.current?.getTracks().forEach(t => t.stop());
      ctxRef.current?.close();
    },
    []
  );

  return { mode, seconds, analyser, error, startRecording, stopRecording, startMonitoring, stopMonitoring };
}
