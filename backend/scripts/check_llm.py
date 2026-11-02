"""
Preflight check for the configured LLM

Reads the project .env and sends three small requests: plain text, JSON mode,
and a single tool call. Prints pass/fail for each; never prints the API key.

Usage:
    cd backend && uv run python scripts/check_llm.py
    cd backend && LLM_PROVIDER=bedrock LLM_MODEL_NAME=us.anthropic.claude-haiku-5-5 uv run python scripts/check_llm.py
    cd backend && uv run python scripts/check_llm.py --list-models   # Bedrock Claude model IDs
"""

import json
import os
import sys

from dotenv import load_dotenv

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts_dir)

from bedrock_converse import (  # noqa: E402
    extract_json_object,
    is_bedrock_provider,
    make_chat_client,
    resolve_region,
)

_project_root = os.path.abspath(os.path.join(_scripts_dir, "..", ".."))

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _check_text(client, model):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        max_tokens=20,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("empty response")
    return f"reply={text[:40]!r}"


def _check_json(client, model):
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": 'Return a JSON object with keys "name" (string) and "age" (integer) for a fictional person.',
        }],
        response_format={"type": "json_object"},
        max_tokens=100,
    )
    content = response.choices[0].message.content or ""
    data = json.loads(extract_json_object(content))
    if not isinstance(data, dict):
        raise RuntimeError("response is not a JSON object")
    return f"keys={sorted(data)}"


def _check_tool_call(client, model):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[WEATHER_TOOL],
        tool_choice="required",
        max_tokens=200,
    )
    choice = response.choices[0]
    tool_calls = choice.message.tool_calls or []
    if not tool_calls:
        raise RuntimeError(f"no tool call (finish_reason={choice.finish_reason})")
    call = tool_calls[0]
    arguments = json.loads(call.function.arguments or "{}")
    return f"tool={call.function.name} args={arguments}"


def _list_bedrock_models(api_key, region):
    """Print the Anthropic inference profiles usable as LLM_MODEL_NAME."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig

    client = boto3.client("bedrock", region_name=region, config=BotoConfig(signature_version=UNSIGNED))

    def add_bearer(request, **_kwargs):
        request.headers["Authorization"] = f"Bearer {api_key}"

    client.meta.events.register("before-send.bedrock.*", add_bearer)

    profiles = []
    for page in client.get_paginator("list_inference_profiles").paginate():
        profiles.extend(page.get("inferenceProfileSummaries", []))
    anthropic = sorted(
        (p for p in profiles if "anthropic" in p.get("inferenceProfileId", "")),
        key=lambda p: p["inferenceProfileId"],
    )
    if not anthropic:
        print("No Anthropic inference profiles found in this region")
        return 1
    for profile in anthropic:
        print(f"{profile['inferenceProfileId']}  ({profile.get('inferenceProfileName', '')}, {profile.get('status', '')})")
    return 0


def main():
    # Variables already set in the shell win, so a model can be tried before editing .env
    load_dotenv(os.path.join(_project_root, ".env"), override=False)

    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL") or None
    model = os.environ.get("LLM_MODEL_NAME", "")
    provider = os.environ.get("LLM_PROVIDER", "")

    if "--list-models" in sys.argv[1:]:
        if not api_key:
            print("FAIL  LLM_API_KEY must be set in .env")
            return 1
        try:
            return _list_bedrock_models(api_key, resolve_region())
        except Exception as exc:
            print(f"FAIL  list models: {type(exc).__name__}: {str(exc)[:300]}")
            return 1

    if not api_key or not model:
        print("FAIL  LLM_API_KEY and LLM_MODEL_NAME must be set in .env")
        return 1

    if is_bedrock_provider(provider):
        print(f"provider=bedrock model={model} region={resolve_region()}")
    else:
        print(f"provider=openai model={model} base_url={base_url}")

    client = make_chat_client(api_key, base_url, provider=provider, region=resolve_region(), timeout=60)

    failures = 0
    for name, check in (("text", _check_text), ("json", _check_json), ("tool_call", _check_tool_call)):
        try:
            detail = check(client, model)
            print(f"PASS  {name}: {detail}")
        except Exception as exc:
            failures += 1
            status = getattr(exc, "status_code", None)
            status_text = f" (HTTP {status})" if status else ""
            print(f"FAIL  {name}{status_text}: {type(exc).__name__}: {str(exc)[:300]}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
