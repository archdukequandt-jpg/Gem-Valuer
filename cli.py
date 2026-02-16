import argparse
import sys
import subprocess
from importlib import resources

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="gem-valuer",
        description="Launch the Industry Standard Gem Valuer Streamlit app."
    )
    parser.add_argument("--port", type=int, default=8501, help="Port to serve on (default: 8501)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no auto-browser)")
    parser.add_argument("--server-address", default="localhost", help="Bind address (default: localhost)")
    args = parser.parse_args(argv)

    # Locate the packaged app.py on disk
    try:
        app_path = resources.files("gem_valuer").joinpath("app.py")
    except Exception:
        print("Could not locate gem_valuer/app.py inside the package.", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.server_address,
        "--browser.gatherUsageStats", "false",
    ]
    if args.headless:
        cmd += ["--server.headless", "true"]

    # Defer all runtime logging/serving to Streamlit
    sys.exit(subprocess.call(cmd))
