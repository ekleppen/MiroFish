#!/usr/bin/env python3
"""
Verify that a simulation folder loads as distinct OASIS agents

Builds the OASIS Twitter agent graph from twitter_profiles.csv with CAMEL's
stub model, so no LLM is called and no API key is needed. With --actions it
also summarizes which agents acted in twitter/actions.jsonl after a run.

Example (from backend/):
    uv run python scripts/verify_population.py uploads/simulations/pop_xxx
    uv run python scripts/verify_population.py uploads/simulations/pop_xxx --actions
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter


def load_config(simulation_dir: str) -> dict:
    with open(os.path.join(simulation_dir, "simulation_config.json"), encoding="utf-8") as f:
        return json.load(f)


async def build_agent_graph(simulation_dir: str):
    from camel.models import StubModel
    from camel.types import ModelType
    from oasis import generate_twitter_agent_graph

    # OASIS configures this logger on import, so it can only be quieted afterwards.
    logging.getLogger("social.agent").setLevel(logging.WARNING)

    return await generate_twitter_agent_graph(
        profile_path=os.path.join(simulation_dir, "twitter_profiles.csv"),
        model=StubModel(ModelType.STUB),
    )


def check_agents(simulation_dir: str) -> bool:
    config = load_config(simulation_dir)
    expected_ids = [cfg["agent_id"] for cfg in config.get("agent_configs", [])]
    expected_names = {cfg["agent_id"]: cfg.get("entity_name") for cfg in config.get("agent_configs", [])}

    graph = asyncio.run(build_agent_graph(simulation_dir))
    agents = dict(graph.get_agents())
    prompts = {agent_id: agent.user_info.to_twitter_system_message()
               for agent_id, agent in agents.items()}
    usernames = {agent_id: agent.user_info.name for agent_id, agent in agents.items()}

    problems = []
    if sorted(agents) != sorted(expected_ids):
        problems.append(f"agent ids {sorted(agents)[:5]}... do not match agent_configs "
                        f"{sorted(expected_ids)[:5]}...")
    if len(set(usernames.values())) != len(agents):
        problems.append("usernames are not unique")
    duplicate_prompts = [p for p, n in Counter(prompts.values()).items() if n > 1]

    print(f"Agents loaded by OASIS: {len(agents)} (agent_configs: {len(expected_ids)})")
    print(f"Distinct usernames: {len(set(usernames.values()))}")
    print(f"Distinct system prompts: {len(set(prompts.values()))}")
    if duplicate_prompts:
        print(f"Warning: {len(duplicate_prompts)} prompt(s) shared by more than one agent")
    if agents:
        first_id = min(agents)
        print(f"Example: agent {first_id} = {expected_names.get(first_id)} "
              f"(@{usernames[first_id]})")

    for problem in problems:
        print(f"FAIL: {problem}")
    if not problems:
        print("OK: every guest loaded as an individual agent")
    return not problems


def summarize_actions(simulation_dir: str) -> bool:
    path = os.path.join(simulation_dir, "twitter", "actions.jsonl")
    if not os.path.exists(path):
        print(f"No action log at {path}; run the simulation first")
        return False

    config = load_config(simulation_dir)
    total_agents = len(config.get("agent_configs", []))
    per_agent = Counter()
    action_types = Counter()
    names = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "agent_id" not in entry or "action_type" not in entry:
                continue
            per_agent[entry["agent_id"]] += 1
            action_types[entry["action_type"]] += 1
            names[entry["agent_id"]] = entry.get("agent_name")

    print(f"Actions logged: {sum(per_agent.values())}")
    print(f"Distinct agents that acted: {len(per_agent)} of {total_agents}")
    for action_type, count in action_types.most_common():
        print(f"  {action_type}: {count}")
    for agent_id, count in per_agent.most_common(5):
        print(f"  agent {agent_id} ({names.get(agent_id)}): {count} action(s)")
    return len(per_agent) > 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a synthetic population simulation folder")
    parser.add_argument("simulation_dir", help="Simulation folder from build_guest_simulation.py")
    parser.add_argument("--actions", action="store_true",
                        help="Summarize twitter/actions.jsonl after a run instead of loading agents")
    args = parser.parse_args()

    simulation_dir = os.path.abspath(args.simulation_dir)
    if not os.path.exists(os.path.join(simulation_dir, "simulation_config.json")):
        print(f"Error: no simulation_config.json in {simulation_dir}", file=sys.stderr)
        return 1

    ok = summarize_actions(simulation_dir) if args.actions else check_agents(simulation_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
