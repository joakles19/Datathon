import { useCallback, useRef } from 'react';

/** Returns a function that plays a short two-tone alarm using WebAudio. */
export function useAlarm() {
  const ctxRef = useRef(null);

  return useCallback(() => {
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      ctxRef.current = ctxRef.current || new Ctx();
      const ctx = ctxRef.current;
      if (ctx.state === 'suspended') ctx.resume();
      for (let i = 0; i < 4; i++) {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'square';
        osc.frequency.value = i % 2 ? 660 : 880;
        gain.gain.value = 0.08;
        osc.connect(gain).connect(ctx.destination);
        const t = ctx.currentTime + i * 0.25;
        osc.start(t);
        osc.stop(t + 0.2);
      }
    } catch {
      /* audio not available */
    }
  }, []);
}
