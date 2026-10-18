"""Create this repository's Python 3.12 environment without global installs."""

from pathlib import Path
import os
import subprocess
import sys
import venv


def main() -> int:
    if sys.version_info[:2] != (3, 12):
        print("Run this setup script with Python 3.12; the current interpreter is " + sys.version.split()[0], file=sys.stderr)
        return 1
    root = Path(__file__).resolve().parents[1]
    environment = root / ".venv"
    executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if environment.exists() and not executable.is_file():
        print("An incomplete .venv already exists. Inspect it before creating a new environment.", file=sys.stderr)
        return 1
    if not environment.exists():
        print(f"Creating isolated environment: {environment}", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run([
        str(executable), "-c",
        "import sys; assert sys.version_info[:2] == (3, 12), 'Existing .venv must use Python 3.12'",
    ], check=True)
    subprocess.run([
        str(executable), "-m", "pip", "install", "--disable-pip-version-check",
        "--retries", "1", "--timeout", "30", "-r", str(root / "requirements.txt"),
    ], check=True)
    subprocess.run([str(executable), "-m", "pip", "check"], check=True)
    subprocess.run([
        str(executable), "-m", "hearhere_audio.panns", "--download-checkpoint",
    ], cwd=root, check=True)
    print(f"Ready. Project Python: {executable}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
