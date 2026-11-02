#!/usr/bin/env python3
"""
Build a ready-to-run MiroFish simulation from synthetic guests

Reads the guest pipeline's synthetic_guests.jsonl and writes profiles, config
and state into a new simulation folder. No ontology, Zep graph or LLM call is
involved.

Example (from backend/):
    uv run python scripts/build_guest_simulation.py \
        --guests ../mirofish_guest_pipeline_v4.0/out/synthetic_guests.jsonl
"""

import argparse
import os
import sys

_backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _backend_dir)

from app.population.mirofish_adapter import (  # noqa: E402
    ASSUMPTIONS_FILENAME,
    GuestRecordError,
    TimeSettings,
    build_guest_simulation,
    load_guest_records,
    new_simulation_id,
)

DEFAULT_SIMULATIONS_DIR = os.path.join(_backend_dir, "uploads", "simulations")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a MiroFish simulation from synthetic guests")
    parser.add_argument("--guests", required=True, help="Path to synthetic_guests.jsonl")
    parser.add_argument("--out", default=None,
                        help="Simulation folder (default: uploads/simulations/pop_<id>)")
    parser.add_argument("--assumptions", default=None,
                        help=f"Generator {ASSUMPTIONS_FILENAME} (default: next to --guests)")
    parser.add_argument("--hours", type=int, default=24, help="Simulated hours")
    parser.add_argument("--minutes-per-round", type=int, default=60)
    parser.add_argument("--agents-per-hour-min", type=int, default=10)
    parser.add_argument("--agents-per-hour-max", type=int, default=25)
    args = parser.parse_args()

    settings = TimeSettings(
        total_hours=args.hours,
        minutes_per_round=args.minutes_per_round,
        agents_per_hour_min=args.agents_per_hour_min,
        agents_per_hour_max=args.agents_per_hour_max,
    )
    try:
        settings.validate()
        records = load_guest_records(args.guests)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.out:
        simulation_dir = os.path.abspath(args.out)
        simulation_id = os.path.basename(simulation_dir.rstrip(os.sep))
        if os.path.exists(os.path.join(simulation_dir, "state.json")):
            print(f"Error: {simulation_dir} already contains a simulation", file=sys.stderr)
            return 1
    else:
        simulation_id = new_simulation_id()
        simulation_dir = os.path.join(DEFAULT_SIMULATIONS_DIR, simulation_id)

    assumptions = args.assumptions or os.path.join(
        os.path.dirname(os.path.abspath(args.guests)), ASSUMPTIONS_FILENAME
    )
    summary = build_guest_simulation(
        records,
        simulation_dir,
        simulation_id,
        settings=settings,
        assumptions_path=assumptions,
        source_path=args.guests,
    )

    config_path = os.path.join(summary["simulation_dir"], "simulation_config.json")
    print(f"Simulation: {summary['simulation_id']}")
    print(f"Folder: {summary['simulation_dir']}")
    print(f"Agents: {summary['agents']} ({summary['distinct_prompts']} distinct prompts)")
    if not summary["assumptions_copied"]:
        print(f"Warning: {ASSUMPTIONS_FILENAME} not found; generator settings not recorded")
    if summary["distinct_prompts"] < summary["agents"]:
        print("Warning: some guests share an identical prompt")
    print()
    print("Verify (no LLM calls):")
    print(f"  uv run python scripts/verify_population.py {summary['simulation_dir']}")
    print("Smoke run (uses your LLM key):")
    print(f"  uv run python scripts/run_parallel_simulation.py --config {config_path} "
          "--twitter-only --max-rounds 2 --no-wait")
    return 0


if __name__ == "__main__":
    sys.exit(main())
