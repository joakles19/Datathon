# LinuxONE audio compatibility package

The approved s390x runtime has no usable Numba wheel for the project's
librosa release. This package implements only the librosa APIs needed by
HearHere and torchlibrosa: WAV loading, deterministic resampling, Hann
windows, mel filters and padding helpers.

Use it only on the documented LinuxONE runtime by placing this directory
before site-packages in `PYTHONPATH`:

```sh
PYTHONPATH="$PWD/scripts/linuxone_compat:$PWD" \
  /home/linux1/panns39/bin/python -m uvicorn backends.main:app
```

The regular local Python 3.12 environment continues to use the pinned
librosa dependency from `requirements.txt`.
