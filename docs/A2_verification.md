# A2 local verification

Verified on Windows in the repository's own Python environment, 17 October 2026.

- Branch: `ai_ZiYi`; no commit or push performed.
- All 31 A2 unittest tests passed after the API-consumer update. The published branch runs these with the two explicit test patterns in README.md; inherited backend tests need B's server source.
- CLI smoke test used A1's supplied WAV and real saved model.
- Result: `siren`, uncalibrated score `0.8566666666666667`.
- Both CLI entry points also passed in the staging environment.
- A1 config, feature source, model and metadata remain unchanged.
- A1 sample parity is confirmed. Local HTTP integration against an unchanged copy of ai_david passed. Connectivity to A1's PC and LinuxONE execution remain pending.
- These are integration checks using a supplied fold-1 sample, not accuracy evaluation.

Runtime:

```json
{
  "python": "3.12.14",
  "packages": {
    "numpy": "2.5.4",
    "scipy": "1.18.1",
    "scikit-learn": "1.9.1",
    "joblib": "1.6.0",
    "librosa": "0.11.0",
    "soundfile": "0.14.0",
    "numba": "0.68.0",
    "soxr": "1.1.0"
  }
}
```

Full test output: `.venv/a2-tests.log` (local, Git-ignored).

## API integration update

- The current shared interface is POST /classify, multipart field `file`.
- 11 added API-client tests cover field mapping, selected/predicted label differences, error handling and multipart upload.
- A local HTTP smoke test used the inspected ai_david backend and real model: first alert, repeated suppression, malformed-input HTTP 422 and the API CLI all passed.
- No backend, model or threshold changes were made. No Git commits or pushes were made.
- The temporary test server was stopped. It did not contact A1's PC.
- A1 and ZiYi are on different networks; use a local development backend or the shared LinuxONE URL once available.
- See `classify_api.md`, `classify_integration_verification.json` and `A1_parity_confirmation.json`.
