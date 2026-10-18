# Break-in Monitor – frontend

React (Vite) web app for the break-in detection system: microphone recording, continuous monitoring, audio uploads, and alerts.

## Run it

```bash
npm install
npm run dev          # http://localhost:5173
npm run build        # production build in dist/
```

The microphone only works on `localhost` or HTTPS.

The app connects to `http://localhost:8000` by default. Start the FastAPI backend, then open the frontend and leave **Demo mode** off to send recordings to the real model. Set `VITE_API_URL` for a different backend or `VITE_DEMO_MODE=true` when you explicitly want simulated results.

## Backend contract

Defined at the top of `src/lib/api.js` — the frontend and backend in this repository share this contract.

Live microphone monitoring enables browser noise suppression and automatic gain control when available. It sends a rolling five-second audio window every second. The backend raises an important-sound alert only when the same class clears its threshold in at least two of the latest three windows. Single recordings and uploads keep their existing one-clip behavior.

```
POST {api}/classify  multipart/form-data, file field "file"
-> { "label": "siren", "predicted_label": "siren", "confidence": 0.86,
     "scores": { "siren": 0.86, "...": 0.02 }, "model_version": "hearhere-rf-v2",
     "action": "alert", "priority": "high", "message": "Siren detected",
     "event_id": 17, "processing_ms": 18.8 }

GET  {api}/health    -> { "status": "ok", "model_version": "hearhere-rf-v2" }
```

Notes for the backend:
- The included backend enables CORS for the frontend origin.
- The frontend converts every clip (recorded or uploaded) to **WAV, 16 kHz, mono, 16-bit PCM** before sending, so no server-side conversion is needed. Change `WAV_OPTIONS` in `src/lib/api.js` if the model wants something else (e.g. 44100 Hz, 2 channels).
- Conversion happens with the Web Audio API in the browser, so it works for anything the browser can decode (webm/opus, ogg, m4a, mp3, flac, wav). Files the browser cannot decode show an error in the results list.

## Structure

```
src/
  App.jsx                 state + wiring (analysis, alerts, history)
  lib/api.js              backend calls, demo mode, alert rule, WAV_OPTIONS
  lib/wav.js              decode + resample + mono mixdown + PCM16 WAV encoder
  lib/storage.js          localStorage-backed state hook
  hooks/useRecorder.js    mic capture: single clip + overlapping live windows
  hooks/useAlarm.js       WebAudio alarm
  components/             Recorder, Uploader, Results, AlertLog, AlertBanner, Settings, LevelMeter
```
