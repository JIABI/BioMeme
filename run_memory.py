#!/usr/bin/env python3
"""Build, use and update BioMeme's kernel memory from prepared measurements."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "frozen"))
from biomeme.community import (ResponseMemory, load_bundle, save_prediction, load_prediction,
                               contribution_rows, ranking_rows, write_csv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("fit", "predict", "update"):
        p = sub.add_parser(command)
        p.add_argument("--data", required=True, type=Path, help="Bundle prefix, without .npz/.json")
        p.add_argument("--output", required=True, type=Path, help="New output directory")
        if command != "fit":
            p.add_argument("--memory", required=True, type=Path)
    p = sub.add_parser("trace")
    p.add_argument("--memory", required=True, type=Path)
    p.add_argument("--prediction", required=True, type=Path)
    p.add_argument("--query-id", required=True)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("rank")
    p.add_argument("--prediction", required=True, type=Path)
    p.add_argument("--by", choices=("uncertainty", "activity"), default="uncertainty")
    p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "fit":
        ResponseMemory.fit(*load_bundle(args.data, "training")).save(args.output)
    elif args.command == "update":
        ResponseMemory.load(args.memory).update(*load_bundle(args.data, "update")).save(args.output)
    elif args.command == "predict":
        memory = ResponseMemory.load(args.memory)
        save_prediction(args.output, memory.predict(*load_bundle(args.data, "query")), memory)
    elif args.command == "trace":
        memory = ResponseMemory.load(args.memory)
        write_csv(args.output, contribution_rows(load_prediction(args.prediction), memory,
                  query_id=args.query_id, top=args.top))
    else:
        write_csv(args.output, ranking_rows(load_prediction(args.prediction), args.by))
    print(f"{args.command}: wrote {args.output}")


if __name__ == "__main__":
    main()
