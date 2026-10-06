"""Start the dubbing studio: API + built React UI on http://127.0.0.1:8000

    .venv\\Scripts\\python serve.py
    (frontend dev mode: cd web && npm run dev  ->  http://localhost:5173, proxied to this API)
"""
import argparse
import sys

from dubber.config import configure_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dubbing studio web server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_environment()          # model caches -> models/, before any ML import

    import uvicorn

    from server.app import app

    print(f"Dubbing studio running at http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
