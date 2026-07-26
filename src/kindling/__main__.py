"""启动入口：uv run python -m kindling"""
from __future__ import annotations

import os
import webbrowser
from threading import Timer

import uvicorn

HOST = os.environ.get("KINDLING_HOST", "127.0.0.1")
PORT = int(os.environ.get("KINDLING_PORT", "8777"))


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    print(f"\n  Kindling — 上下文不是想出来的，是做出来的")
    print(f"  → {url}\n")
    if os.environ.get("KINDLING_NO_BROWSER") != "1":
        Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run("kindling.api:app", host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
