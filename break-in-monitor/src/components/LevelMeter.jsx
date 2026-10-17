import { useEffect, useRef } from 'react';

export default function LevelMeter({ analyser }) {
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
