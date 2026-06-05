from __future__ import annotations

import argparse
import os

from src.web.config import LOCAL_HOST, LOCAL_PORT


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the ML-STATSSOCCER local web app.")
    parser.add_argument("--port", type=int, default=LOCAL_PORT)
    args = parser.parse_args(argv)
    run_server(port=args.port)


def run_server(port: int = LOCAL_PORT):
    import uvicorn

    os.environ["MLSTATSSOCCER_PORT"] = str(port)
    uvicorn.run("src.web.app:app", host=LOCAL_HOST, port=port, reload=False)


if __name__ == "__main__":
    main()
