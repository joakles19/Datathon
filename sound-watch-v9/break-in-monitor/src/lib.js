// Shared helpers: app name, sound catalogue + classification, WAV conversion, backend calls, diagnostics.

// ======================================================================
// App name
// ======================================================================

// Change the app name here; it is used in the header, the tab title and notifications.
const APP_NAME = 'Sound Watch';


// ======================================================================
// Microphone errors
// ======================================================================

/** Turn a getUserMedia error into something a person can act on. */
function micErrorMessage(e) {
  switch (e?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Microphone access is blocked. Allow it in the browser site settings (see Troubleshooting), then try again.';
    case 'NotFoundError':
      return 'No microphone was found. Plug one in or check your system sound settings.';
    case 'NotReadableError':
    case 'AbortError':
      return 'The microphone could not be started. Another app may be using it.';
    case 'OverconstrainedError':
      return 'The selected microphone is not available.';
    default:
      return e?.message || 'Could not access the microphone.';
  }
}


// ======================================================================
// WAV conversion
// ======================================================================

/*
 * Browser-side audio -> WAV conversion.
 * Whatever the browser records or the user uploads (webm/opus, ogg, m4a, mp3, flac, wav...),
 * it is decoded, resampled, mixed to the requested channel count and written as 16-bit PCM WAV.
 */

const DEFAULT_WAV = { sampleRate: 16000, channels: 1 };

/** Encode planar float samples (one Float32Array per channel) as a 16-bit PCM WAV Blob. */
function encodeWav(channelData, sampleRate) {
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
async function toWav(blob, { sampleRate, channels } = DEFAULT_WAV) {
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


// ======================================================================
// Sound categories
// ======================================================================

/*
 * The sound categories the app can alert on, and how serious each one is.
 *
 *   urgent  safety / security: full-screen flashing alert, stays until acknowledged,
 *           and the history entry is kept until the user clears it.
 *   notice  everyday sounds: short banner, auto-deleted later.
 *   none    anything else (speech, footsteps, background noise): logged, no alert.
 *
 * The backend returns a `label`. It is matched (lower_snake_case, punctuation ignored) against a
 * category `id`, its `aliases` or its `examples`, so specific labels such as "microwave_ding" or
 * AudioSet names like "Smoke detector, smoke alarm" land in the right category.
 * The specific label is still shown on the alert as a detail line.
 * To add a category, or teach one a new label, edit CATALOG.
 */

const CATALOG = [
  // ---- urgent ----
  {
    id: 'alarm', level: 'urgent', title: 'Alarm',
    examples: ['smoke_alarm', 'fire_alarm', 'carbon_monoxide_alarm', 'security_alarm', 'siren'],
    aliases: ['alarm', 'smoke_detector_smoke_alarm', 'smoke_detector', 'co_alarm', 'burglar_alarm', 'alarm_tamper',
      'emergency_vehicle', 'police_car_siren', 'ambulance_siren', 'civil_defense_siren'],
  },
  {
    id: 'break_in', level: 'urgent', title: 'Break-in',
    examples: ['glass_breaking', 'door_forced', 'window_smash'],
    aliases: ['shatter', 'glass_break', 'breaking_glass'],
  },
  {
    id: 'shouting', level: 'urgent', title: 'Screaming or shouting',
    examples: ['scream', 'shouting'],
    aliases: ['screaming', 'shout', 'yell'],
  },

  // ---- everyday ----
  {
    id: 'door', level: 'notice', title: 'Doorbell or knock',
    examples: ['doorbell', 'door_knock'],
    aliases: ['ding_dong', 'knock', 'knocking'],
  },
  {
    id: 'appliance', level: 'notice', title: 'Timers and appliances',
    examples: ['microwave_ding', 'kitchen_timer', 'kettle_whistle', 'washing_machine'],
    aliases: ['microwave_oven', 'microwave', 'microwave_beep', 'oven_timer', 'timer', 'egg_timer',
      'whistling_kettle', 'steam_whistle', 'washer_finished', 'laundry_done'],
  },
  {
    id: 'phone', level: 'notice', title: 'Phone or alarm clock',
    examples: ['phone_ringing', 'alarm_clock'],
    aliases: ['telephone_bell_ringing', 'telephone', 'ringtone'],
  },
  {
    id: 'baby', level: 'notice', title: 'Baby crying',
    examples: ['baby_crying'],
    aliases: ['baby_cry_infant_cry', 'baby_cry'],
  },
  {
    id: 'animal', level: 'notice', title: 'Animal noises',
    examples: ['dog_bark', 'cat_meow'],
    aliases: ['bark', 'dog', 'barking', 'meow', 'cat'],
  },
  {
    id: 'vehicle', level: 'notice', title: 'Car horn',
    examples: ['car_horn'],
    aliases: ['vehicle_horn_car_horn_honking', 'honk'],
  },
];

const RANK = { none: 0, notice: 1, urgent: 2 };
const BY_KEY = new Map();
CATALOG.forEach(c => [c.id, ...c.aliases, ...c.examples].forEach(k => BY_KEY.set(k, c)));

// Labels that mean "someone is breaking in" even if they are not in the catalogue.
const THREAT_RE = /break|glass|forced|smash|tamper|kick|crowbar/i;

const normLabel = label =>
  String(label || '').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');

const nice = label => (label || 'unknown').replace(/_/g, ' ');
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);

/** Level of a stored entry; also understands older entries that only had `alert`. */
const levelOf = x => x.level || (x.alert ? 'urgent' : 'none');

/** Look up one label. `breakIn` is the legacy backend flag (only used for unknown labels). */
function describe(label, breakIn) {
  const key = normLabel(label);
  const known = BY_KEY.get(key);
  if (known) {
    return { id: known.id, title: known.title, level: known.level, detail: key === known.id ? '' : cap(nice(label)) };
  }
  const threat = breakIn ?? THREAT_RE.test(key);
  return { id: key, title: cap(nice(label)), level: threat ? 'urgent' : 'none', detail: '' };
}

/**
 * Decide what a backend result means. Looks at the top label and at every event,
 * so a doorbell inside a clip that is mostly speech is still caught.
 * Returns { level, id, title, detail, confidence }.
 */
function classifyResult(result, { threshold = 0, muted = [] } = {}) {
  const candidates = [
    { label: result.label, confidence: result.confidence, breakIn: result.break_in },
    ...(result.events || []).map(e => ({ label: e.label, confidence: e.confidence })),
  ];
  let best = null;
  for (const cand of candidates) {
    const d = describe(cand.label, cand.breakIn);
    let level = d.level;
    if (muted.includes(d.id) || cand.confidence * 100 < threshold) level = 'none';
    const scored = { ...d, level, confidence: cand.confidence };
    if (
      !best ||
      RANK[level] > RANK[best.level] ||
      (RANK[level] === RANK[best.level] && cand.confidence > best.confidence)
    ) {
      best = scored;
    }
  }
  return best;
}


// ======================================================================
// Backend / demo mode
// ======================================================================

/*
 * BACKEND CONTRACT — adjust analyzeAudio() if your API differs.
 *
 *   POST {api}/analyze   multipart/form-data, file field "audio"
 *                        the file is always 16-bit PCM WAV (see WAV_OPTIONS below),
 *                        converted in the browser from whatever was recorded/uploaded
 *   -> 200 JSON {
 *        "label": "microwave_ding",    // top class (see src/lib.js for known ids/aliases)
 *        "confidence": 0.93,           // 0..1
 *        "events": [{ "label": "doorbell", "confidence": 0.88, "start": 1.2, "end": 2.0 }],
 *        "break_in": true              // optional, legacy: only used for labels the catalogue does not know
 *      }
 *   GET  {api}/health    -> 200 when the service is up (optional)
 *
 * What counts as urgent / everyday / ignore is decided in the frontend, in src/lib.js.
 */


const DEFAULT_API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Format sent to the backend. Change these to match what the model expects.
const WAV_OPTIONS = { sampleRate: 16000, channels: 1 };

async function analyzeAudio(blob, filename, { api, demo }) {
  if (demo) return demoAnalyze();
  let wav;
  try {
    wav = await toWav(blob, WAV_OPTIONS);
  } catch (err) {
    throw new Error(`WAV conversion failed (${err.message})`);
  }
  const body = new FormData();
  body.append('audio', wav, filename.replace(/\.[^./\\]+$/, '') + '.wav');
  const res = await fetch(`${api.replace(/\/+$/, '')}/analyze`, { method: 'POST', body });
  if (!res.ok) throw new Error(`Server returned ${res.status}`);
  const data = await res.json();
  if (typeof data.confidence !== 'number') throw new Error('Unexpected response format');
  return data;
}

async function checkHealth(api, timeoutMs = 5000) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(`${api.replace(/\/+$/, '')}/health`, { signal: ctl.signal });
    return res.ok;
  } finally {
    clearTimeout(timer);
  }
}

/* ---------- demo mode: fake detections so the UI works without a backend ---------- */
const URGENT = CATALOG.filter(c => c.level === 'urgent');
const EVERYDAY = CATALOG.filter(c => c.level === 'notice');
const BENIGN = ['speech', 'footsteps', 'background_noise', 'music'];
const pick = arr => arr[Math.floor(Math.random() * arr.length)];

async function demoAnalyze() {
  await new Promise(r => setTimeout(r, 700 + Math.random() * 600));
  const roll = Math.random();
  if (roll < 0.5) {
    const label = pick((roll < 0.12 ? pick(URGENT) : pick(EVERYDAY)).examples);
    const confidence = 0.75 + Math.random() * 0.24;
    const start = +(Math.random() * 2).toFixed(1);
    return { label, confidence, events: [{ label, confidence, start, end: +(start + 1 + Math.random()).toFixed(1) }] };
  }
  return { label: pick(BENIGN), confidence: 0.5 + Math.random() * 0.4, events: [] };
}


// ======================================================================
// Troubleshooting checks
// ======================================================================

/*
 * Self-checks used by the Troubleshooting panel. Each check returns
 *   { id, title, status: 'pass' | 'warn' | 'fail' | 'skip', detail, fix }
 * `fix` is only shown for warn / fail.
 */

const wait = ms => new Promise(r => setTimeout(r, ms));
const out = (id, title, status, detail, fix = '') => ({ id, title, status, detail, fix });

/** Open a microphone for about a second and measure the loudest sample. */
async function measureSignal(micId) {
  const base = { echoCancellation: false, noiseSuppression: false };
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: micId ? { ...base, deviceId: { exact: micId } } : base,
    });
  } catch (error) {
    return { error };
  }
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  try {
    await ctx.resume?.();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    ctx.createMediaStreamSource(stream).connect(analyser);
    const buf = new Float32Array(analyser.fftSize);
    let peak = 0;
    for (let i = 0; i < 12; i++) {
      await wait(100);
      analyser.getFloatTimeDomainData(buf);
      for (const v of buf) peak = Math.max(peak, Math.abs(v));
    }
    return { peak, label: stream.getAudioTracks()[0]?.label || '' };
  } finally {
    stream.getTracks().forEach(t => t.stop());
    ctx.close().catch(() => {});
  }
}

async function runDiagnostics(settings, onUpdate) {
  const results = [];
  const push = r => {
    results.push(r);
    onUpdate([...results]);
  };

  // ---- browser support ----
  const secure = window.isSecureContext;
  push(
    secure
      ? out('secure', 'Secure page (HTTPS or localhost)', 'pass', 'This page is allowed to use the microphone.')
      : out('secure', 'Secure page (HTTPS or localhost)', 'fail', 'This page is not secure, so the browser blocks the microphone.',
          'Open the app over https:// or from localhost.')
  );

  const hasMedia = !!navigator.mediaDevices?.getUserMedia;
  push(
    hasMedia
      ? out('media', 'Microphone access supported', 'pass', 'The browser can capture audio.')
      : out('media', 'Microphone access supported', secure ? 'fail' : 'skip',
          secure ? 'This browser cannot capture audio.' : 'Skipped: needs a secure page.',
          'Use a current version of Chrome, Edge, Firefox or Safari.')
  );

  const mimes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'];
  const mime = typeof MediaRecorder !== 'undefined' ? mimes.find(m => MediaRecorder.isTypeSupported?.(m)) : null;
  push(
    mime
      ? out('recorder', 'Audio recording supported', 'pass', `Recording as ${mime}.`)
      : out('recorder', 'Audio recording supported', 'fail', 'This browser cannot record audio clips.',
          'Use a current version of Chrome, Edge, Firefox or Safari.')
  );

  const hasAudioCtx = !!(window.AudioContext || window.webkitAudioContext) && !!(window.OfflineAudioContext || window.webkitOfflineAudioContext);
  push(
    hasAudioCtx
      ? out('webaudio', 'Audio conversion supported', 'pass', 'Clips can be converted to WAV before they are analysed.')
      : out('webaudio', 'Audio conversion supported', 'fail', 'Web Audio is missing, so clips cannot be converted to WAV.',
          'Use a current version of Chrome, Edge, Firefox or Safari.')
  );

  // ---- microphone ----
  let permission = 'unknown';
  try {
    permission = (await navigator.permissions?.query({ name: 'microphone' }))?.state || 'unknown';
  } catch {
    /* not supported */
  }
  if (permission === 'denied') {
    push(out('permission', 'Microphone permission', 'fail', 'Access is blocked for this site.',
      'Click the padlock or site-settings icon in the address bar, set Microphone to Allow, then reload the page.'));
  } else if (permission === 'prompt') {
    push(out('permission', 'Microphone permission', 'warn', 'The browser has not been given permission yet.',
      'Press "Allow access" in Devices, or start a recording and choose Allow when asked.'));
  } else if (permission === 'granted') {
    push(out('permission', 'Microphone permission', 'pass', 'Access is allowed.'));
  } else {
    push(out('permission', 'Microphone permission', 'skip', 'This browser does not report the permission state. The signal test below shows whether access works.'));
  }

  let mics = [];
  if (navigator.mediaDevices?.enumerateDevices) {
    try {
      mics = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'audioinput');
    } catch {
      /* handled below */
    }
  }
  if (!hasMedia) {
    push(out('devices', 'Microphones found', 'skip', 'Skipped: microphone access is not available.'));
  } else if (mics.length === 0) {
    push(out('devices', 'Microphones found', 'fail', 'No microphone was detected.',
      'Plug in a microphone, check it is enabled in your system sound settings, then press Refresh in Devices.'));
  } else if (settings.micId && !mics.some(d => d.deviceId === settings.micId)) {
    push(out('devices', 'Microphones found', 'warn', `${mics.length} found, but the microphone you chose earlier is not connected.`,
      'Choose another microphone in Devices. The system default is used until then.'));
  } else {
    push(out('devices', 'Microphones found', 'pass', `${mics.length} microphone${mics.length === 1 ? '' : 's'} detected.`));
  }

  if (!hasMedia || mics.length === 0 || permission === 'denied') {
    push(out('signal', 'Microphone signal', 'skip', 'Skipped: no usable microphone.'));
  } else {
    const sig = await measureSignal(settings.micId || '');
    if (sig.error) {
      push(out('signal', 'Microphone signal', 'fail', micErrorMessage(sig.error),
        sig.error.name === 'NotReadableError'
          ? 'Close other apps or tabs that use the microphone (video calls, recorders) and try again.'
          : 'Check the microphone in Devices and in your system sound settings.'));
    } else if (sig.peak < 0.0005) {
      push(out('signal', 'Microphone signal', 'fail', 'The microphone is connected but sends complete silence.',
        'Check the microphone is not muted (look for a mute switch or key), that the right device is selected in Devices, and that its input volume is up in your system settings.'));
    } else if (sig.peak < 0.01) {
      push(out('signal', 'Microphone signal', 'warn', `A signal was detected but it is very quiet (peak ${(sig.peak * 100).toFixed(1)}%).`,
        'Raise the microphone input volume in your system settings or move the microphone closer to the sounds.'));
    } else {
      push(out('signal', 'Microphone signal', 'pass', `Signal detected (peak ${Math.round(sig.peak * 100)}%)${sig.label ? ` on ${sig.label}` : ''}.`));
    }
  }

  // ---- backend ----
  if (settings.demo) {
    push(out('backend', 'Backend connection', 'skip', 'Demo mode is on, so no backend is used.'));
  } else {
    const api = settings.api;
    const mixed = location.protocol === 'https:' && /^http:\/\//i.test(api) && !/^http:\/\/(localhost|127\.0\.0\.1)/i.test(api);
    if (mixed) {
      push(out('backend', 'Backend connection', 'fail', 'This page is https but the backend URL is http.',
        'Browsers block this. Serve the backend over https, or open this app over http on your local network.'));
    } else if (!navigator.onLine) {
      push(out('backend', 'Backend connection', 'fail', 'This device is offline.', 'Reconnect to the network and run the checks again.'));
    } else {
      try {
        const ok = await checkHealth(api);
        push(
          ok
            ? out('backend', 'Backend connection', 'pass', `${api} is reachable.`)
            : out('backend', 'Backend connection', 'warn', `${api} answered, but /health did not return 200.`,
                'If your backend has no /health route this is harmless. Analyse a short upload to confirm it works.')
        );
      } catch {
        push(out('backend', 'Backend connection', 'fail', `Could not reach ${api}.`,
          'Check the URL in Settings, that the server is running, and that it allows requests from this page (CORS).'));
      }
    }
  }

  // ---- alert outputs ----
  if (!('Notification' in window)) {
    push(out('notifications', 'Notifications', 'skip', 'Not supported in this browser.'));
  } else if (Notification.permission === 'denied') {
    push(out('notifications', 'Notifications', settings.notify ? 'warn' : 'skip', 'Notifications are blocked for this site.',
      'Allow notifications in the browser site settings, then switch them on in Settings.'));
  } else if (Notification.permission === 'granted') {
    push(out('notifications', 'Notifications', 'pass', settings.notify ? 'Allowed and switched on.' : 'Allowed, but switched off in Settings.'));
  } else {
    push(
      settings.notify
        ? out('notifications', 'Notifications', 'warn', 'Switched on, but permission was never granted.',
            'Use "Enable" in Devices to be asked for permission.')
        : out('notifications', 'Notifications', 'skip', 'Not enabled.')
    );
  }

  push(
    'vibrate' in navigator
      ? out('vibration', 'Vibration', 'pass', settings.vibrate ? 'Supported and switched on.' : 'Supported, but switched off in Settings.')
      : out('vibration', 'Vibration', 'skip', 'Not available on this device (normal for desktops and iPhones).')
  );

  if (!settings.sound) {
    push(out('sound', 'Alert sound', 'skip', 'Switched off in Settings.'));
  } else {
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      const ctx = new Ctx();
      await ctx.resume?.();
      const running = ctx.state === 'running';
      ctx.close().catch(() => {});
      push(
        running
          ? out('sound', 'Alert sound', 'pass', 'Audio output is ready.')
          : out('sound', 'Alert sound', 'warn', 'Audio output is blocked until you interact with the page.',
              'Press "Play test sound" in Devices, then try again.')
      );
    } catch {
      push(out('sound', 'Alert sound', 'fail', 'Audio output could not be started.', 'Check your system volume and output device.'));
    }
  }

  return results;
}

/** Plain-text report, handy to paste into a support message. */
function reportText(appName, results) {
  const lines = results.map(
    r => `[${r.status.toUpperCase()}] ${r.title}: ${r.detail}${r.fix && r.status !== 'pass' ? ` Fix: ${r.fix}` : ''}`
  );
  return [`${appName} diagnostics`, new Date().toString(), navigator.userAgent, '', ...lines].join('\n');
}

