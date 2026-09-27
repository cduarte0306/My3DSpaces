#!/usr/bin/env python3
"""Milestone 2: extract sharp keyframes from a room walkthrough video."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roommodel.frame_extraction.extract_frames import extract_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to the input video (e.g. data/raw/home-video.mov)")
    parser.add_argument("output_dir", nargs="?", default="data/frames", help="Directory to write frames to")
    parser.add_argument("--fps", type=float, default=2.5, help="Target sampling rate in frames per second")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    frames = extract_frames(args.video, args.output_dir, fps=args.fps)
    print(f"{len(frames)} sharp frames written to {args.output_dir}")


if __name__ == "__main__":
    main()
