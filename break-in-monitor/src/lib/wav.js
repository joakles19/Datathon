/*
 * Browser-side audio -> WAV conversion.
 * Whatever the browser records or the user uploads (webm/opus, ogg, m4a, mp3, flac, wav...),
 * it is decoded, resampled, mixed to the requested channel count and written as 16-bit PCM WAV.
 */

export const DEFAULT_WAV = { sampleRate: 16000, channels: 1 };

/** Encode planar float samples (one Float32Array per channel) as a 16-bit PCM WAV Blob. */
export function encodeWav(channelData, sampleRate) {
  const channels = channelData.length;
  const frames = channelData[0].length;
  const bytesPerSample = 2;
  const dataSize = frames * channels * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * bytesPerSample, true); // byte rate
  view.setUint16(32, channels * bytesPerSample, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < frames; i++) {
    for (let c = 0; c < channels; c++) {
      const s = Math.max(-1, Math.min(1, channelData[c][i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

/** Average all source channels into one. */
function toMono(buf) {
  if (buf.numberOfChannels === 1) return buf.getChannelData(0);
  const out = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const data = buf.getChannelData(c);
    for (let i = 0; i < out.length; i++) out[i] += data[i] / buf.numberOfChannels;
  }
  return out;
}

/** Convert any browser-decodable audio Blob/File to a WAV Blob. */
export async function toWav(blob, { sampleRate, channels } = DEFAULT_WAV) {
  const Offline = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const encoded = await blob.arrayBuffer();

  // decodeAudioData resamples to the context's sample rate
  let buf;
  try {
    buf = await new Offline(1, 1, sampleRate).decodeAudioData(encoded);
  } catch {
    throw new Error('this browser could not decode the audio');
  }

  // Safety net for browsers that do not resample on decode
  if (buf.sampleRate !== sampleRate) {
    const ctx = new Offline(buf.numberOfChannels, Math.ceil(buf.duration * sampleRate), sampleRate);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.start();
    buf = await ctx.startRendering();
  }

  const planar =
    channels === 1
      ? [toMono(buf)]
      : Array.from({ length: channels }, (_, c) => buf.getChannelData(Math.min(c, buf.numberOfChannels - 1)));

  return encodeWav(planar, sampleRate);
}
