# Sound Watch – frontend

React (Vite) web app that listens for important sounds and shows them visually: alarms, doorbells, timers and appliances, a baby crying, a break-in and more. Built to help deaf and hard-of-hearing people.

## Run it

```bash
npm install
npm run dev          # http://localhost:5173
npm run build        # production build in dist/
```

The microphone only works on `localhost` or HTTPS.

The app starts in **demo mode** (fake detections) so you can try the whole flow without a backend. Untick it in *Settings* and set the backend URL to go live (or set `VITE_API_URL`, see `.env.example`).

## Customising

- **Name:** `src/lib.js` (`APP_NAME`).
- **Logo:** replace `public/logo.svg` (top left), or change `LOGO_FILE` in `src/components.jsx`.
- **Sound categories and how serious they are:** `src/lib.js`. Each `CATALOG` entry has an `id`, `level` (`urgent` or `notice`), a `title`, `examples` (shown in the app and used by demo mode) and optional `aliases` (other labels the backend may send).

## How alerts work

| Level | Categories | What happens |
| --- | --- | --- |
| Urgent | Alarm, Break-in, Screaming or shouting | Full-screen flashing alert, vibration, stays until acknowledged. History entry is kept until you clear it. |
| Everyday | Doorbell or knock, Timers and appliances, Phone or alarm clock, Baby crying, Animal noises, Car horn | Banner for about 12 s and a brief flash. Auto-deleted from results and history after the delay in *Settings*. |
| Clear | speech, footsteps, background noise | Logged only, then auto-deleted. |

Each category can be switched off in *Sounds to detect*. The specific sound the model heard (for example "Smoke alarm") is shown on the alert as a detail line. The confidence threshold applies to everything. Clips are checked for the top label **and** every entry in `events`, so a doorbell inside a clip that is mostly speech still alerts.

## Devices and troubleshooting

- **Devices** lists the microphones the browser can see (it updates when one is plugged in or removed), lets you pick which one to use and connect or disconnect it, shows whether the backend is reachable, and shows the state of the alert outputs (screen flash, vibration, notifications, sound) with test buttons. The chosen microphone is remembered; if it is missing next time, the system default is used.
- **Troubleshooting** runs automatic checks (secure page, browser support, microphone permission, microphones found, a live signal test, backend connection, notifications, vibration, sound) and says how to fix each problem. "Copy report" puts the results on the clipboard so they can be sent to someone for help. Below the checks is a list of common problems and fixes (`FAQ` in `src/components.jsx`).
- Only devices the browser itself can see are listed. Separate sensors or phones that report to the backend would need extra backend support.

## Backend contract

Defined at the top of `src/lib.js`.

```
POST {api}/analyze   multipart/form-data, file field "audio"
-> { "label": "microwave_ding", "confidence": 0.93,
     "events": [{ "label": "doorbell", "confidence": 0.88, "start": 1.2, "end": 2.0 }] }

GET  {api}/health    -> 200 when up (optional)
```

- `label` is matched against a category `id`, `aliases` or `examples` (case and punctuation are ignored), so specific labels like `microwave_ding` and AudioSet-style names like `Smoke detector, smoke alarm` or `Microwave oven` work as they are.
- Unknown labels are ignored, unless the response has `"break_in": true`, which is treated as urgent (older API).
- The backend model must actually be able to recognise the sounds. A model trained on AudioSet classes (for example YAMNet) covers many of them; a break-in-only model will not detect a microwave.
- Enable CORS for the frontend origin.
- Every clip is converted in the browser to **WAV, 16 kHz, mono, 16-bit PCM** before sending. Change `WAV_OPTIONS` in `src/lib.js` if the model wants something else.

## Structure

```
src/
  App.jsx                 state + wiring (analysis, alerts, auto-delete)
  components.jsx          every screen and panel (one section per component)
  hooks.js                persistent state, alert sounds, microphone list, mic capture
  lib.js                  app name, sound categories, WAV conversion, backend calls, diagnostics
```
