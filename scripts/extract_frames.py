from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 23-08-26
# Reason     : Extracts frames from a video in specific ranges to grow the
#              labeling dataset. Dataset tooling, not part of the
#              inference pipeline (src/boxarm).
# -----------------------------------------------------------------------
"""Extracts frames from a video within the given time ranges.

A surveillance video has consecutive frames that are nearly identical:
dumping all of them into the dataset inflates the count without adding
information, and worse, if they land on both sides of the train/val
split, the model validates against images it has practically already
seen. That's why every N frames are sampled, and frames that barely
changed from the last one kept are discarded on top of that.

Usage:
    python scripts/extract_frames.py
    python scripts/extract_frames.py --every 3 --min-diff 0.0
    python scripts/extract_frames.py --ranges 0:5 20:30 --out data/frames
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("frames")

ROOT = Path(__file__).resolve().parent.parent

# Default ranges (start:end in seconds) -- the ones asked to review from
# the pallet video. "end" can be None = until the end of the video.
DEFAULT_RANGES = ((0.0, 5.0), (20.0, 30.0), (60.0, 70.0), (106.0, None))


def parse_range(text: str) -> tuple[float, float | None]:
    """'20:30' -> (20.0, 30.0); '106:' -> (106.0, None)."""
    if ":" not in text:
        raise argparse.ArgumentTypeError(f"range '{text}' must be start:end (seconds)")
    start, end = text.split(":", 1)
    return float(start), (float(end) if end.strip() else None)


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized mean difference (0-1) between two already-gray,
    downscaled frames -- enough to detect 'nothing happened' between frames."""
    return float(np.mean(cv2.absdiff(a, b))) / 255.0


def _fingerprint(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extracts video frames within given ranges")
    ap.add_argument("--video", type=Path, default=ROOT / "videos" / "p2r1.mp4")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "frames_etiquetar")
    ap.add_argument("--ranges", type=parse_range, nargs="*", default=None,
                    help="start:end in seconds, e.g. 0:5 20:30 106:")
    ap.add_argument("--every", type=int, default=5,
                    help="keep 1 out of every N frames within the range (5 = 2 fps if the video is 10 fps)")
    ap.add_argument("--min-diff", type=float, default=0.0,
                    help="discard the frame if it differs less than this from the last one kept (0 = no filtering)")
    ap.add_argument("--per-range", type=int, default=0,
                    help="if >0, keep only the N frames that change the MOST within each range. "
                         "In an almost-static scene, those are the only ones that add anything new")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    if not args.video.is_file():
        logger.error("video not found: %s", args.video)
        sys.exit(1)

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        logger.error("could not read fps from %s", args.video)
        sys.exit(1)
    logger.info("%s: %.2f fps, %d frames, %.1fs", args.video.name, fps, total, total / fps)

    ranges = args.ranges if args.ranges else list(DEFAULT_RANGES)
    args.out.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_near_duplicate = 0
    for start_s, end_s in ranges:
        start_f = max(0, int(round(start_s * fps)))
        end_f = total if end_s is None else min(total, int(round(end_s * fps)))
        if start_f >= end_f:
            logger.warning("range %.1f:%s is empty or out of bounds -- skipping", start_s, end_s)
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        last: np.ndarray | None = None
        candidates: list[tuple[float, int, np.ndarray]] = []  # (change, idx, frame)
        for idx in range(start_f, end_f):
            ok, frame = cap.read()
            if not ok:
                logger.warning("read failed at frame %d -- cutting the range short", idx)
                break
            if (idx - start_f) % args.every:
                continue

            fingerprint = _fingerprint(frame)
            # How much this changed from the last sampled frame. The first
            # frame of a range has nothing to compare against: it gets top
            # priority so it always gets in (it's the range's starting state).
            change = float("inf") if last is None else _diff(fingerprint, last)
            if last is not None and args.min_diff > 0 and change < args.min_diff:
                skipped_near_duplicate += 1
                last = fingerprint
                continue
            last = fingerprint
            candidates.append((change, idx, frame.copy()))

        # In an almost-static scene, the information is where something
        # moved, not in even sampling: --per-range keeps the N with the
        # biggest change instead of N consecutive frames that are the same photo.
        if args.per_range > 0 and len(candidates) > args.per_range:
            chosen = sorted(candidates, key=lambda c: -c[0])[:args.per_range]
            chosen.sort(key=lambda c: c[1])
        else:
            chosen = candidates

        for _change, idx, frame in chosen:
            # Filename carries the exact second: if a weird label needs to
            # be checked later, the moment in the video is found without searching.
            sec = idx / fps
            dest = args.out / f"{args.video.stem}_t{sec:07.2f}_f{idx:05d}.jpg"
            cv2.imwrite(str(dest), frame)
            saved += 1

        logger.info("range %5.1f - %-6s -> %3d frames (out of %d sampled)",
                    start_s, f"{end_s}" if end_s else "end", len(chosen), len(candidates))

    cap.release()
    logger.info("total %d frames in %s (%d skipped as near-duplicates)",
                saved, args.out, skipped_near_duplicate)


if __name__ == "__main__":
    main()
