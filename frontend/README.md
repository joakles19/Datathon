# Break-in Monitor – frontend

React (Vite) web app for the break-in detection system: microphone recording, continuous monitoring, audio uploads, and alerts.

## Run it

```bash
npm install
npm run dev          # http://localhost:5173
npm run build        # production build in dist/
```

The microphone only works on `localhost` or HTTPS.

The app starts in **demo mode** (fake detections) so you can try the whole flow without a backend. Untick it in *Settings* and set the backend URL to go live (or set `VITE_API_URL`, see `.env.example`).

## Backend contract

Defined at the top of `src/lib/api.js` — change `analyzeAudio()` / `isAlert()` if your API differs.

```
POST {api}/analyze   multipart/form-data, file field "audio"
-> { "break_in": true, "label": "glass_breaking", "confidence": 0.93,
     "events": [{ "label": "glass_breaking", "confidence": 0.93, "start": 1.2, "end": 2.0 }] }

GET  {api}/health    -> 200 when up (optional)
```

Notes for the backend:
- Enable CORS for the frontend origin.
- The frontend converts every clip (recorded or uploaded) to **WAV, 16 kHz, mono, 16-bit PCM** before sending, so no server-side conversion is needed. Change `WAV_OPTIONS` in `src/lib/api.js` if the model wants something else (e.g. 44100 Hz, 2 channels).
- Conversion happens with the Web Audio API in the browser, so it works for anything the browser can decode (webm/opus, ogg, m4a, mp3, flac, wav). Files the browser cannot decode show an error in the results list.

## Structure

```
src/
  App.jsx                 state + wiring (analysis, alerts, history)
  lib/api.js              backend calls, demo mode, alert rule, WAV_OPTIONS
  lib/wav.js              decode + resample + mono mixdown + PCM16 WAV encoder
  lib/storage.js          localStorage-backed state hook
  hooks/useRecorder.js    mic capture: single clip + chunked monitoring
  hooks/useAlarm.js       WebAudio alarm
  components/             Recorder, Uploader, Results, AlertLog, AlertBanner, Settings, LevelMeter
```
