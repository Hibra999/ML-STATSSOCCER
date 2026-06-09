import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.web.config import LOCAL_HOST
from src.web.mundial import MUNDIAL_PORT


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the standalone Mundial 2026 local app.")
    parser.add_argument("--port", type=int, default=MUNDIAL_PORT)
    args = parser.parse_args(argv)
    run_server(port=args.port)


def run_server(port: int = MUNDIAL_PORT):
    import uvicorn

    os.environ["MLSTATSSOCCER_MUNDIAL_PORT"] = str(port)
    uvicorn.run("src.web.mundial:app", host=LOCAL_HOST, port=port, reload=False)


if __name__ == "__main__":
    main()
