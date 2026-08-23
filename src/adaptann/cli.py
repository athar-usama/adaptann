"""Command-line entry point: ``adaptann benchmark``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="adaptann")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("benchmark", help="run the drifting-workload benchmark and produce the charts")
    args = parser.parse_args(argv)

    if args.command == "benchmark":
        from .demos.benchmark import main as run

        run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
