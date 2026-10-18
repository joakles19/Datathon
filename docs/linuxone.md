# LinuxONE loading and smoke-test guide

Status: LinuxONE execution has not been verified. The selected PANNs model is not deployable until D runs every gate below on the approved LinuxONE CPU runtime.

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

NumPy, SciPy, scikit-learn, soundfile, soxr, numba, llvmlite, PyTorch and torchlibrosa may require architecture-specific wheels, native libraries or an organiser-provided environment. Standard PyPI PyTorch 2.10.0 has no s390x wheel. Do not assume Windows wheels work on LinuxONE. If the approved runtime cannot install or load PyTorch/PANNs, deployment is blocked: do not silently substitute the Random Forest or another model. Preserve the error log and agree a supported runtime with D/A1 before changing the selected model. Do not silently change scikit-learn versions and load the old pickle. A matching retrained/exported artifact and updated metadata may be necessary.

The PANNs comparison was trained with Python 3.12.3, PyTorch 2.10.0, torchlibrosa 0.1.0, NumPy 2.5.4 and scikit-learn 1.9.1. These local versions are recorded in `models/model_metadata.json`; they are not evidence of LinuxONE compatibility. The complete Windows dependency list is in `a2-tested-environment.txt` for comparison.

## 3. Copy source, matching artifacts and example WAV

Keep `hearhere_audio/`, `scripts/`, `models/`, `tests/` and the agreed WAV available to the host. The integration sample is included at `examples/1-31482-A-42.wav`; see `examples/README.md` for attribution.

Install the environment from the project root, then run from the project root:

```bash
.venv/bin/python -m hearhere_audio examples/1-31482-A-42.wav --pretty
.venv/bin/python -m unittest discover -s tests -v
```

The setup script downloads the official 16 kHz CNN14 checkpoint to `~/.cache/hearhere/panns/` and verifies its 358,668,570-byte size and Zenodo MD5. The source is Zenodo record 3987831 (`Cnn14_16k_mAP=0.438.pth`), licensed CC BY 4.0. Run the inference command and backend WAV contract using the selected model; verify a finite 2048-value embedding and all ten scores. Do not report readiness if model loading fails.

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
- Successful loading of the approved PANNs runtime, verified checkpoint and selected Logistic Regression model on s390x.
- A real five-second WAV using the exact mono 16 kHz preprocessing path, with output parity against the local reference within an agreed numerical tolerance.
- Warm request latency against the website's one-second request cadence.

LinuxONE-hosted inference is a hard deployment gate and remains unverified from the macOS development host. This implementation does not invoke or demonstrate a specialised hardware AI accelerator.
