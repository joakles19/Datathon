# Current shared interface: POST /classify

Verified against `backends/main.py` and `hearhere_audio/inference.py` on `joakles19/Datathon`, branch `ai_david`, reported commit `7044188652cdb20f160dd1e3b07dff83275f140e`, on 17 October 2026. Source inspection is separate from confirming a deployed service is running.

## Request

- Method: POST.
- Path: `/classify`.
- Content type: `multipart/form-data`.
- Upload field: **`file`**.
- File content: a WAV recording of at most five seconds.
- Existing upload limit: 10 MiB.

After B/D supplies the running service's base URL, execute from this repository:

```powershell
.\.venv\Scripts\python.exe scripts/classify_api.py examples/1-31482-A-42.wav --base-url http://127.0.0.1:8000 --pretty
```

`http://127.0.0.1:8000` is an example local address, not a confirmed deployment. Supply the base URL without `/classify`. The client appends that path. Stdout contains the server JSON unchanged; stderr reports request round-trip time. Errors use a nonzero exit code and stderr JSON.

### Current team connectivity

A1's service runs on A1's PC at `http://127.0.0.1:8000/classify`. That loopback address cannot be used from another computer. A1 and ZiYi are on different networks, so a private LAN address alone will not connect them either. Use a backend running on ZiYi's machine for local development, or a reachable shared LinuxONE service once D provides it. An agreed VPN/tunnel is another option if access to A1's exact running instance is required.

A temporary local copy of the inspected branch has already passed the HTTP smoke test; it was stopped after testing. This is not evidence of connectivity to A1's PC. Results are in `classify_integration_verification.json`.

For a new clone, use B/D's running service URL or obtain the backend server from the team. This A2 branch provides the client and local prediction tools; it does not supply the backend server.

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
| `scores` | All eight uncalibrated Random Forest scores |
| `action` | Backend result: `alert`, `ignore` or `suppressed` |
| `priority`, `message` | Server-provided display information, possibly null |
| `event_id` | Server-created event identifier, possibly null |
| `model_version` | Identifier for the server's loaded model |

For example, `predicted_label` may be `rain` at 0.40 while `label` is `siren` at 0.31 after backend selection. In that case `confidence` is 0.31. Display and store these meanings correctly; do not assume the two labels are identical.

Class scores sum to one by estimator construction, but they are not calibrated real-world event probabilities. `action` and notification behavior remain server decisions. The client validates response structure and score mapping; it never reselects thresholds, replaces labels, derives an action from score alone, or changes event state itself.

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

## Errors and side effects

- Invalid uploaded audio is returned by the inspected server as HTTP 422.
- Uploads above the server's 10 MiB limit return HTTP 413.
- Client connection failures, malformed responses and contract mismatches raise `ClassifyAPIError`.
- HTTP failures raise `ClassifyHTTPError` with `status_code` and `detail`.
- Default timeout is 60 seconds. No automatic retry is made: POST can create an event before a response is lost.
- Repeating a valid sound can return `suppressed`; this is expected server cooldown behavior, not a failed classification.

`POST /classify` can create event records. Use an agreed development server for smoke tests, and do not automatically change preferences, snooze settings, cooldowns or acknowledgements to make an alert reappear.

## Remaining deployment checks

- Obtain the real base URL from B/D.
- Send the known WAV and compare `predicted_label`, all eight `scores` and `model_version` with the local reference.
- Verify selected-label/confidence mapping and preserve the returned action.
- Use E's broader test plan for false alerts, unfamiliar sounds and realistic microphone conditions.
- Have D verify the same pipeline on LinuxONE. API-client tests alone do not prove LinuxONE deployment.
