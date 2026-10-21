"""Start the web app: `ddf-web`, or `python -m deepfake_detection.webapp`."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    from .engines import Registry
    from .server import create_app

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load the ensemble at start-up so the first analysis is not slow.",
    )
    arguments = parser.parse_args(argv)
    registry = Registry(device=arguments.device)
    if arguments.preload:
        registry.ensemble()
        registry.familiarity()
    uvicorn.run(create_app(registry), host=arguments.host, port=arguments.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
