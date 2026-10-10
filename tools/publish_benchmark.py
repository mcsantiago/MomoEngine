#!/usr/bin/env python3
"""Add a benchmark run to the website's MomoEngine benchmarks data file.

Run the benchmark first (from the build directory):
    MOMO_BENCHMARK=results.json ./application/HyperbolicTimeChamber

Then:
    tools/publish_benchmark.py results.json --label "Baseline"

This appends the run, tagged with the MomoEngine commit it was measured on, to
<site>/_data/momoengine_benchmarks.json. It doesn't commit or push the site:
review the change there first, since pushing publishes it.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SITE = Path.home() / "repos" / "mcsantiago.github.io"
DATA_FILE = Path("_data") / "momoengine_benchmarks.json"
# Screenshots of each view; the latest run's replace the previous ones
IMAGE_DIR = Path("images") / "momoengine" / "benchmarks"


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="JSON written by MOMO_BENCHMARK=<path>")
    parser.add_argument("--label", required=True, help='Short description of the run, e.g. "Baseline" or "Child bounds in parent"')
    parser.add_argument("--site", type=Path, default=DEFAULT_SITE, help=f"Local clone of the website (default: {DEFAULT_SITE})")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Publish even if MomoEngine has uncommitted changes (the run then doesn't match its commit)")
    parser.add_argument("--replace", action="store_true", help="Replace an existing run for the same commit and label")
    args = parser.parse_args()

    # A run is only meaningful if the commit it's tagged with is what was measured
    if git("status", "--porcelain", "--untracked-files=no") and not args.allow_dirty:
        sys.exit("MomoEngine has uncommitted changes, so the results wouldn't match HEAD. "
                 "Commit first, rebuild, rerun the benchmark (or pass --allow-dirty).")

    results = json.loads(args.results.read_text())
    run = {
        "date": results["date"],
        "label": args.label,
        "commit": git("rev-parse", "--short", "HEAD"),
        "commit_full": git("rev-parse", "HEAD"),
        "gpu": results["gpu"],
        "resolution": results["resolution"],
        "bounces": results["bounces"],
        "warmup_frames": results["warmup_frames"],
        "frames_per_view": results["frames_per_view"],
        "bvh": results["bvh"],
        "views": [
            {
                "name": view["name"],
                "dispatch_ms": view["dispatch_ms"],
                "frame_ms": view["frame_ms"],
                # GPU-bound, so frame rate follows from the GPU frame time
                "fps": round(1000.0 / view["frame_ms"]["median"], 1),
                "cpu": {key: view[key + "_ms"] for key in ("cpu_total", "record", "submit", "fence_wait")},
            }
            for view in results["views"]
        ],
    }

    data_path = args.site / DATA_FILE
    if not data_path.exists():
        sys.exit(f"{data_path} not found. Is --site a clone of the website?")
    data = json.loads(data_path.read_text())

    existing = [i for i, r in enumerate(data["runs"]) if r["commit_full"] == run["commit_full"] and r["label"] == run["label"]]
    if existing and not args.replace:
        sys.exit(f"A run for {run['commit']} labelled '{run['label']}' already exists (pass --replace to overwrite it).")
    for i in reversed(existing):
        del data["runs"][i]

    # Screenshots are saved next to the results file
    for view, run_view in zip(results["views"], run["views"]):
        if "screenshot" not in view:
            continue
        source = args.results.parent / view["screenshot"]
        if not source.exists():
            sys.exit(f"Screenshot {source} listed in the results is missing")
        (args.site / IMAGE_DIR).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, args.site / IMAGE_DIR / source.name)
        run_view["screenshot"] = "/" + (IMAGE_DIR / source.name).as_posix()

    data["runs"].append(run)
    data_path.write_text(json.dumps(data, indent=2) + "\n")

    print(f"Added '{run['label']}' ({run['commit']}) to {data_path}:")
    for view in run["views"]:
        print(f"  {view['name']:<10} {view['dispatch_ms']['avg']:>7.2f} ms avg  {view['dispatch_ms']['median']:>7.2f} ms median")
    print(f"Review it, then commit and push from {args.site} to publish.")


if __name__ == "__main__":
    main()
