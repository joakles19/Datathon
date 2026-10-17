# LinuxONE loading and smoke-test guide

Status: these are handoff instructions. LinuxONE execution has not been verified by A2's Windows tests.

## 1. Verify D's actual environment

```bash
uname -m
python3 --version
```

Confirm architecture (normally s390x for LinuxONE), access permissions, Python 3.12 availability, network/package access, disk/memory limits and the organisers' approved runtime. Coordinate any system package installation with D/the organiser.

## 2. Create an isolated environment

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

NumPy, SciPy, scikit-learn, soundfile, soxr, numba and llvmlite may require architecture-specific wheels, native libraries or an organiser-provided environment. Do not assume Windows wheels work on LinuxONE. If the pinned environment cannot install, preserve the error log and ask D/A1 to agree on a supported stack. Do not silently change scikit-learn versions and load the old pickle. A matching retrained/exported artifact and updated metadata may be necessary.

The review used Python 3.12.14; A1 records 3.12.3. A1's six recorded library versions are pinned in `requirements.txt`; numba/soxr pins are the A2-tested values, not verified A1 versions. The complete Windows dependency list is in `a2-tested-environment.txt` for comparison.

## 3. Copy source, matching artifacts and example WAV

Keep `hearhere_audio/`, `scripts/`, `models/`, `tests/` and the agreed WAV available to the host. The integration sample is included at `examples/1-31482-A-42.wav`; see `examples/README.md` for attribution.

Run from the project root:

```bash
.venv/bin/python -m hearhere_audio examples/1-31482-A-42.wav --pretty
.venv/bin/python -m unittest discover -s tests -v
```

The provided sample should match `tests/sample_reference.json`. Its expected label is `siren`; the recorded model score is `0.8566666666666667`. Investigate differences before substituting a new reference. A CLI cold run loads the model each time; use a persistent predictor for service performance measurements.

## 4. Backend lifecycle

Use B's existing server. Instantiate the predictor at startup, complete a warm-up call, then mark the service ready. Supply explicit model paths, a readable artifact directory and a writable temporary-upload directory. The service must enforce HTTP upload limits and clean up request files.

Keep event alerting disabled for warm-up. Use limited concurrent workers appropriate to the allocated CPU/memory. This delivery does not install a system service, configure public ingress or provide HTTPS; D connects it to the provided deployment environment.

## 5. Record evidence

- Actual runtime/platform and package versions.
- Successful model loading and checksum check.
- Real WAV input and JSON/tuple output from the deployed service.
- A1 reference comparison and test results.
- Startup time, warm inference timing, request concurrency and error handling.

LinuxONE-hosted inference is the deployment claim to verify. This implementation does not invoke or demonstrate a specialised hardware AI accelerator.
