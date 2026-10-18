# Current shared interface: POST /classify

The `main` branch includes the frontend, FastAPI service and validated model. The frontend and backend contract was verified locally on 17 October 2026; this does not establish a public or LinuxONE deployment.

## Request

- Method: POST.
- Path: `/classify`.
- Content type: `multipart/form-data`.
- Upload field: **`file`**.
- File content: a WAV recording of at most five seconds.
- Existing upload limit: 10 MiB.
- Optional live-monitor fields: `stream_id` and non-negative `window_index`, supplied together. For live microphone monitoring, the backend confirms the same threshold-clearing important class in at least two of the last three windows before creating an alert event. Other clients can omit both fields and keep single-clip behavior.

With the local backend running, execute from this repository:

```powershell
.\.venv\Scripts\python.exe scripts/classify_api.py examples/1-31482-A-42.wav --base-url http://127.0.0.1:8000 --pretty
```

`http://127.0.0.1:8000` is the default local address, not a public deployment. Supply the base URL without `/classify`. The client appends that path. Stdout contains the server JSON unchanged; stderr reports request round-trip time. Errors use a nonzero exit code and stderr JSON.

### Current team connectivity

`127.0.0.1` and `localhost` refer to the computer making the request. The frontend and backend work together on one computer by default. Teammates on separate computers need a shared backend URL reachable from both; loopback URLs do not provide remote access. LinuxONE deployment remains a separate check.

Equivalent curl request (use `curl.exe` in Windows PowerShell):

```powershell
curl.exe -X POST http://127.0.0.1:8000/classify -F "file=@examples/1-31482-A-42.wav;type=audio/wav"
```

## Response semantics

| Field | Meaning |
|---|---|
| `predicted_label` | A class with the highest model score |
| `label` | Backend-selected important-sound candidate; falls back to top class when none qualifies |
| `confidence` | Uncalibrated score associated with `scores[label]` |
| `scores` | All ten uncalibrated scores from the selected model |
| `action` | Backend result: `alert`, `ignore` or `suppressed` |
| `confirmation` | `pending` while a live-monitor sound awaits a second matching window; `confirmed` after the rule is met; `not_required` for ordinary requests |
| `priority`, `message` | Server-provided display information, possibly null |
| `event_id` | Server-created event identifier, possibly null |
| `model_version` | Identifier for the server's loaded model |

For example, `predicted_label` may be `rain` at 0.40 while `label` is `siren` at 0.31 after backend selection. In that case `confidence` is 0.31. Display and store these meanings correctly; do not assume the two labels are identical.

Class scores sum to one by estimator construction, but they are not calibrated real-world event probabilities. The current Fold 4 selection enables five alert classes and leaves `car_horn` disabled because it did not meet the 85% validation-precision criterion; its score remains in every response. `action` and notification behavior remain server decisions. The client validates response structure and score mapping; it never reselects thresholds, replaces labels, derives an action from score alone, or changes event state itself.

## Python API consumer

```python
from hearhere_audio.api_client import ClassifyClient

client = ClassifyClient("http://127.0.0.1:8000", timeout=60)
result = client.classify_file("example.wav")
print(result["predicted_label"])  # Classifier result
print(result["label"])            # Backend-selected label
print(result["action"])           # Preserve alert/ignore/suppressed
```

The client does not load local model or threshold artifacts. It uses the existing package installation and checks the WAV signature/size before posting; decoding, duration checks and feature extraction happen on the server. It does not open a live microphone.

## Events endpoint

`GET /events?limit=N` returns the newest events, where `N` must be between 1 and 200. The default is 50. Values outside that range return HTTP 422 before querying SQLite.

`POST /events/{event_id}/ack` acknowledges one stored event and returns `{"ok": true}`. A missing event returns HTTP 404.

## Errors and side effects

- Invalid uploaded audio is returned by the inspected server as HTTP 422.
- Uploads above the server's 10 MiB limit return HTTP 413.
- Client connection failures, malformed responses and contract mismatches raise `ClassifyAPIError`.
- HTTP failures raise `ClassifyHTTPError` with `status_code` and `detail`.
- Default timeout is 60 seconds. No automatic retry is made: POST can create an event before a response is lost.
- Repeating a valid sound can return `suppressed`; this is expected server cooldown behavior, not a failed classification.
- A live-monitor response with `confirmation: pending` is intentionally returned as `action: ignore` and creates no event record.

`POST /classify` can create event records. Use an agreed development server for smoke tests, and do not automatically change preferences, snooze settings, cooldowns or acknowledgements to make an alert reappear.

## Remaining deployment checks

- Obtain the real base URL from B/D.
- Send the known WAV and compare `predicted_label`, all ten `scores` and `model_version` with the local reference.
- Verify selected-label/confidence mapping and preserve the returned action.
- Use E's broader test plan for false alerts, unfamiliar sounds and realistic microphone conditions.
- Have D verify the same pipeline on LinuxONE. API-client tests alone do not prove LinuxONE deployment.
