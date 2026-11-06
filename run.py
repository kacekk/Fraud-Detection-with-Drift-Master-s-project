"""One-command launcher for Windows, macOS and Linux.

    python run.py              # set up (first time only) and open the app in your browser
    python run.py notebook     # open the thesis notebook in JupyterLab
    python run.py experiment   # run the full benchmark and save results/ (extra args are passed on)
    python run.py test         # run the automated tests
    python run.py setup        # only create the environment and install packages

The first run creates a private virtual environment in the `.venv` folder and
installs the exact package versions from requirements.txt (a few minutes).
Later runs start immediately. Nothing is installed outside this folder.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
MARKER = VENV / ".installed-requirements"
SUPPORTED = [(3, 13), (3, 12), (3, 14)]          # preference order
IS_WINDOWS = os.name == "nt"


def say(msg: str) -> None:
    print(f"[fraud-drift] {msg}", flush=True)


# ----------------------------------------------------------------------------
# 1. make sure we run on a supported Python (3.12 - 3.14)
# ----------------------------------------------------------------------------
def supported(version) -> bool:
    return tuple(version[:2]) in SUPPORTED


def find_other_python() -> list[str] | None:
    """Look for a supported Python next to the one that started this script."""
    candidates = []
    if IS_WINDOWS and shutil.which("py"):
        candidates += [["py", f"-{a}.{b}"] for a, b in SUPPORTED]
    for a, b in SUPPORTED:
        exe = shutil.which(f"python{a}.{b}")
        if exe:
            candidates.append([exe])
    for cmd in candidates:
        try:
            out = subprocess.run(cmd + ["-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
                                 capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0 and supported(tuple(int(x) for x in out.stdout.split())):
            return cmd
    return None


def ensure_supported_python() -> None:
    if supported(sys.version_info) or os.environ.get("FRAUD_DRIFT_REEXEC"):
        if not supported(sys.version_info):
            sys.exit("Could not find Python 3.12, 3.13 or 3.14.")
        return
    other = find_other_python()
    if other:
        say(f"Python {sys.version_info[0]}.{sys.version_info[1]} is not supported, switching to: {' '.join(other)}")
        env = {**os.environ, "FRAUD_DRIFT_REEXEC": "1"}
        sys.exit(subprocess.call(other + [str(Path(__file__).resolve())] + sys.argv[1:], env=env))
    sys.exit(
        f"\nThis project needs Python 3.12, 3.13 or 3.14 (you have {sys.version.split()[0]}).\n"
        "Download Python 3.13 from https://www.python.org/downloads/ and run this again.\n"
        "Windows: tick 'Add python.exe to PATH' during installation.\n"
    )


# ----------------------------------------------------------------------------
# 2. private virtual environment with the pinned packages
# ----------------------------------------------------------------------------
def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def requirements_hash() -> str:
    data = REQUIREMENTS.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data + sys.version.encode()).hexdigest()


def ensure_environment() -> Path:
    py = venv_python()
    if not py.exists():
        say("Creating a virtual environment in .venv (first run only)…")
        if VENV.exists():
            shutil.rmtree(VENV)
        venv.create(VENV, with_pip=True)
    if MARKER.exists() and MARKER.read_text(encoding="utf-8") == requirements_hash():
        return py
    say("Installing the pinned packages from requirements.txt (a few minutes, first run only)…")
    subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    subprocess.check_call([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    MARKER.write_text(requirements_hash(), encoding="utf-8")
    say("Environment ready.")
    return py


# ----------------------------------------------------------------------------
# 3. commands
# ----------------------------------------------------------------------------
def free_port(preferred: int = 8501) -> int:
    for port in [preferred] + list(range(8502, 8600)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def open_when_ready(url: str) -> None:
    def wait_and_open():
        for _ in range(120):
            try:
                urllib.request.urlopen(url + "/_stcore/health", timeout=1)
                webbrowser.open(url)
                return
            except OSError:
                time.sleep(0.5)
    threading.Thread(target=wait_and_open, daemon=True).start()


def run(cmd: list[str]) -> int:
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONUTF8": "1"}
    try:
        return subprocess.call(cmd, cwd=ROOT, env=env)
    except KeyboardInterrupt:
        return 0


def cmd_app(py: Path, extra: list[str]) -> int:
    port = free_port()
    url = f"http://localhost:{port}"
    say(f"Starting the app at {url}  (press Ctrl+C in this window to stop)")
    open_when_ready(url)
    return run([str(py), "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                "--server.port", str(port), "--browser.gatherUsageStats", "false", *extra])


def cmd_notebook(py: Path, extra: list[str]) -> int:
    say("Opening JupyterLab (press Ctrl+C in this window to stop)")
    return run([str(py), "-m", "jupyter", "lab", "notebooks/fraud_concept_drift.ipynb", *extra])


def cmd_experiment(py: Path, extra: list[str]) -> int:
    return run([str(py), "run_experiment.py", *extra])


def cmd_test(py: Path, extra: list[str]) -> int:
    return run([str(py), "-m", "pytest", "-q", *extra])


COMMANDS = {"app": cmd_app, "notebook": cmd_notebook, "experiment": cmd_experiment, "test": cmd_test}


def main() -> int:
    args = sys.argv[1:]
    command = args[0] if args else "app"
    if command in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if command not in COMMANDS and command != "setup":
        print(f"Unknown command '{command}'. Use one of: app, notebook, experiment, test, setup")
        return 2
    ensure_supported_python()
    py = ensure_environment()
    if command == "setup":
        say("Setup finished. Start the app with:  python run.py")
        return 0
    return COMMANDS[command](py, args[1:])


if __name__ == "__main__":
    sys.exit(main())
