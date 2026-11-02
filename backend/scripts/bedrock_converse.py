"""
Amazon Bedrock Converse adapter with an OpenAI Chat Completions interface

Bedrock serves Anthropic Claude only through the Messages and Converse APIs,
while MiroFish and CAMEL/OASIS speak OpenAI Chat Completions. BedrockChatClient
exposes `client.chat.completions.create(**kwargs)` and returns an OpenAI
`ChatCompletion`, so existing call sites work unchanged.

Authentication uses a Bedrock API key sent as a bearer token on each request.
The key is attached per client, never through process-wide environment
variables, so the main and boost keys can coexist in one process.

Imported by the simulation scripts and by the backend (as scripts.bedrock_converse).
"""

import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)

logger = logging.getLogger(__name__)

BEDROCK_PROVIDER = "bedrock"
DEFAULT_REGION = "us-east-1"
# Converse returns only after the whole reply is generated; long JSON replies need minutes
DEFAULT_TIMEOUT_SECONDS = 600
# Used when the caller sets no output limit; override with BEDROCK_MAX_OUTPUT_TOKENS
DEFAULT_MAX_OUTPUT_TOKENS = 32000

JSON_MODE_INSTRUCTION = (
    "Respond with a single valid JSON object only. Do not add any text, "
    "explanation, or Markdown code fences before or after the JSON."
)

# Converse rejects blank text blocks
EMPTY_TEXT_PLACEHOLDER = "(empty)"

_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
}

_AUTH_ERROR_STATUS = {
    "UnrecognizedClientException": 401,
    "ExpiredTokenException": 401,
    "InvalidSignatureException": 401,
    "AccessDeniedException": 403,
}

# AWS-generated messages that explain configuration problems; they do not
# echo request content, unlike some provider error bodies.
_ACTIONABLE_ERROR_CODES = {
    "ValidationException",
    "ResourceNotFoundException",
    "AccessDeniedException",
    "UnrecognizedClientException",
    "ExpiredTokenException",
}

_TOOL_ID_PATTERN = re.compile(r"[^a-zA-Z0-9_-]")

_SAMPLING_KEYS = ("temperature", "topP")
# Model IDs that rejected temperature/top_p (newer Claude models deprecate them)
_MODELS_WITHOUT_SAMPLING: set = set()
# Output-token caps reported by Bedrock when a request exceeded them
_MODEL_OUTPUT_LIMITS: Dict[str, int] = {}
_MAX_TOKENS_REJECTION_PATTERN = re.compile(r"max(imum)?[ _]?tokens", re.I)
# Model IDs that rejected output_config.effort; thinking effort then stays at the model default
_MODELS_WITHOUT_EFFORT: set = set()
_EFFORT_REJECTION_PATTERN = re.compile(r"output_config|effort|additionalModelRequestFields", re.I)
_SAMPLING_REJECTION_PATTERN = re.compile(r"\b(temperature|top_p|topP)\b.*\b(deprecated|not supported|unsupported)\b", re.I)


def _is_sampling_rejection(error: ClientError) -> bool:
    details = error.response.get("Error") or {}
    return details.get("Code") == "ValidationException" and bool(
        _SAMPLING_REJECTION_PATTERN.search(str(details.get("Message") or ""))
    )


def _is_effort_rejection(error: ClientError) -> bool:
    details = error.response.get("Error") or {}
    return details.get("Code") == "ValidationException" and bool(
        _EFFORT_REJECTION_PATTERN.search(str(details.get("Message") or ""))
    )


def default_effort() -> Optional[str]:
    """Thinking effort for calls that don't set one; BEDROCK_EFFORT, unset by default."""
    value = os.environ.get("BEDROCK_EFFORT", "").strip().lower()
    return value or None


class BedrockAPIError(Exception):
    """A Bedrock request failure carrying OpenAI-SDK-like attributes."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.request_id = request_id


def normalize_provider(provider: Optional[str]) -> str:
    return (provider or "").strip().lower()


def is_bedrock_provider(provider: Optional[str]) -> bool:
    return normalize_provider(provider) == BEDROCK_PROVIDER


def resolve_region(region: Optional[str] = None) -> str:
    return (
        region
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or DEFAULT_REGION
    )


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------

def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _content_text(content: Any) -> str:
    """Flatten OpenAI message content (string or parts list) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif _get(part, "type") == "text" or _get(part, "text") is not None:
                text = _get(part, "text")
                if isinstance(text, dict):
                    text = text.get("value")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return str(content)


def _tool_use_id(raw_id: Any) -> str:
    cleaned = _TOOL_ID_PATTERN.sub("_", str(raw_id or ""))
    return cleaned or f"tool_{uuid.uuid4().hex[:12]}"


def _parse_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return {"input": arguments}
    return parsed if isinstance(parsed, dict) else {"input": parsed}


def _text_block(text: str) -> Dict[str, Any]:
    return {"text": text if text.strip() else EMPTY_TEXT_PLACEHOLDER}


def _translate_tools(tools: Optional[List[Any]]) -> List[Dict[str, Any]]:
    specs = []
    for tool in tools or []:
        function = _get(tool, "function", tool)
        name = _get(function, "name")
        if not name:
            continue
        spec: Dict[str, Any] = {
            "name": name,
            "inputSchema": {
                "json": _get(function, "parameters")
                or {"type": "object", "properties": {}}
            },
        }
        description = _get(function, "description")
        if description:
            spec["description"] = description
        specs.append({"toolSpec": spec})
    return specs


def _translate_tool_choice(tool_choice: Any) -> Optional[Dict[str, Any]]:
    if tool_choice in (None, "auto", "none"):
        return {"auto": {}}
    if tool_choice == "required":
        return {"any": {}}
    name = _get(_get(tool_choice, "function", {}), "name")
    if name:
        return {"tool": {"name": name}}
    return {"auto": {}}


def translate_messages(
    messages: List[Any],
    *,
    tools_enabled: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert OpenAI messages to Converse (system, messages).

    Without tools in the request, Converse rejects toolUse/toolResult blocks,
    so tool history is rendered as text instead.
    """
    system: List[Dict[str, Any]] = []
    converse: List[Dict[str, Any]] = []

    def append(role: str, blocks: List[Dict[str, Any]]) -> None:
        if not blocks:
            return
        if converse and converse[-1]["role"] == role:
            converse[-1]["content"].extend(blocks)
        else:
            converse.append({"role": role, "content": list(blocks)})

    for message in messages:
        role = _get(message, "role")
        content = _get(message, "content")

        if role in ("system", "developer"):
            text = _content_text(content)
            if text.strip():
                system.append({"text": text})
            continue

        if role == "tool":
            tool_call_id = _tool_use_id(_get(message, "tool_call_id"))
            result_text = _content_text(content)
            if tools_enabled:
                append("user", [{
                    "toolResult": {
                        "toolUseId": tool_call_id,
                        "content": [_text_block(result_text)],
                    }
                }])
            else:
                append("user", [_text_block(f"[Tool result {tool_call_id}]\n{result_text}")])
            continue

        if role == "assistant":
            blocks: List[Dict[str, Any]] = []
            text = _content_text(content)
            if text.strip():
                blocks.append({"text": text})
            for call in _get(message, "tool_calls") or []:
                function = _get(call, "function", {})
                name = _get(function, "name", "")
                arguments = _parse_arguments(_get(function, "arguments"))
                call_id = _tool_use_id(_get(call, "id"))
                if tools_enabled:
                    blocks.append({
                        "toolUse": {"toolUseId": call_id, "name": name, "input": arguments}
                    })
                else:
                    blocks.append({
                        "text": f"[Called tool {name} ({call_id}) with {json.dumps(arguments, ensure_ascii=False)}]"
                    })
            append("assistant", blocks or [_text_block("")])
            continue

        # user and any unknown role
        append("user", [_text_block(_content_text(content))])

    # Converse conversations must start with a user turn
    if not converse or converse[0]["role"] != "user":
        converse.insert(0, {"role": "user", "content": [_text_block("Continue.")]})

    return system, converse


def build_converse_request(
    *,
    model: str,
    messages: List[Any],
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    stop: Any = None,
    tools: Optional[List[Any]] = None,
    tool_choice: Any = None,
    response_format: Optional[Dict[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
    **_ignored: Any,
) -> Dict[str, Any]:
    tool_specs = _translate_tools(tools)
    system, converse_messages = translate_messages(
        messages, tools_enabled=bool(tool_specs)
    )

    json_mode = isinstance(response_format, dict) and response_format.get("type") in (
        "json_object",
        "json_schema",
    )
    if json_mode:
        system.append({"text": JSON_MODE_INSTRUCTION})

    request: Dict[str, Any] = {"modelId": model, "messages": converse_messages}
    if system:
        request["system"] = system

    inference: Dict[str, Any] = {}
    limit = max_completion_tokens or max_tokens
    if limit:
        inference["maxTokens"] = int(limit)
    # Recent Claude models reject temperature and top_p together
    if temperature is not None:
        inference["temperature"] = float(temperature)
    elif top_p is not None:
        inference["topP"] = float(top_p)
    if stop:
        inference["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)
    if inference:
        request["inferenceConfig"] = inference

    # Newer Claude models think adaptively by default, and thinking tokens count
    # toward maxTokens; effort is the documented control for how much they think.
    effort = reasoning_effort or default_effort()
    if effort:
        request["additionalModelRequestFields"] = {"output_config": {"effort": effort}}

    if tool_specs:
        request["toolConfig"] = {
            "tools": tool_specs,
            "toolChoice": _translate_tool_choice(tool_choice),
        }
    return request


# ---------------------------------------------------------------------------
# Response translation
# ---------------------------------------------------------------------------

def extract_json_object(text: str) -> str:
    """Trim model text to its outermost JSON object when one is present."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return text
    candidate = text[start:end + 1]
    try:
        json.loads(candidate)
    except ValueError:
        return text
    return candidate


def build_chat_completion(
    response: Dict[str, Any],
    *,
    model: str,
    json_mode: bool = False,
) -> ChatCompletion:
    message = (response.get("output") or {}).get("message") or {}
    texts: List[str] = []
    tool_calls: List[ChatCompletionMessageFunctionToolCall] = []

    for block in message.get("content") or []:
        if "text" in block:
            texts.append(block["text"])
        elif "toolUse" in block:
            tool_use = block["toolUse"]
            tool_calls.append(ChatCompletionMessageFunctionToolCall(
                id=tool_use.get("toolUseId") or _tool_use_id(None),
                type="function",
                function=Function(
                    name=tool_use.get("name", ""),
                    arguments=json.dumps(tool_use.get("input") or {}, ensure_ascii=False),
                ),
            ))

    text = "".join(texts)
    if json_mode and text:
        text = extract_json_object(text)

    finish_reason = _STOP_REASONS.get(response.get("stopReason"), "stop")
    if tool_calls:
        finish_reason = "tool_calls"

    usage = response.get("usage") or {}
    if finish_reason == "length":
        block_types = [next(iter(block), "?") for block in message.get("content") or []]
        logger.warning(
            "Bedrock output hit the token limit: model=%s outputTokens=%s blocks=%s text_chars=%d",
            model, usage.get("outputTokens"), block_types, len("".join(texts)),
        )
    request_id = (response.get("ResponseMetadata") or {}).get("RequestId")

    return ChatCompletion(
        id=f"bedrock-{request_id or uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[Choice(
            index=0,
            finish_reason=finish_reason,
            message=ChatCompletionMessage(
                role="assistant",
                content=text or None,
                tool_calls=tool_calls or None,
            ),
        )],
        usage=CompletionUsage(
            prompt_tokens=usage.get("inputTokens", 0),
            completion_tokens=usage.get("outputTokens", 0),
            total_tokens=usage.get("totalTokens", 0),
        ),
    )


def translate_client_error(error: ClientError) -> BedrockAPIError:
    details = error.response.get("Error") or {}
    metadata = error.response.get("ResponseMetadata") or {}
    code = details.get("Code") or "BedrockError"
    status = _AUTH_ERROR_STATUS.get(code) or metadata.get("HTTPStatusCode")
    message = f"Bedrock {code} (HTTP {status})"
    if code in _ACTIONABLE_ERROR_CODES and details.get("Message"):
        message += f": {str(details['Message'])[:300]}"
    return BedrockAPIError(
        message,
        status_code=status,
        code=code,
        request_id=metadata.get("RequestId"),
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class _Completions:
    def __init__(self, owner: "BedrockChatClient"):
        self._owner = owner

    def create(self, **kwargs: Any) -> ChatCompletion:
        return self._owner.create_completion(**kwargs)


class _Chat:
    def __init__(self, owner: "BedrockChatClient"):
        self.completions = _Completions(owner)


class BedrockChatClient:
    """OpenAI-client-shaped wrapper around Bedrock Converse."""

    def __init__(
        self,
        api_key: str,
        *,
        region: Optional[str] = None,
        timeout: Optional[float] = None,
        max_attempts: int = 8,
        boto_client: Any = None,
    ):
        if not api_key:
            raise ValueError("A Bedrock API key is required (LLM_API_KEY)")
        self.region = resolve_region(region)
        self._api_key = api_key
        self._client = boto_client or boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            config=BotoConfig(
                signature_version=UNSIGNED,
                retries={"mode": "adaptive", "max_attempts": max_attempts},
                read_timeout=timeout or DEFAULT_TIMEOUT_SECONDS,
                connect_timeout=10,
            ),
        )
        self._client.meta.events.register(
            "before-send.bedrock-runtime.*", self._add_bearer_token
        )
        self.chat = _Chat(self)

    def _add_bearer_token(self, request: Any, **_kwargs: Any) -> None:
        request.headers["Authorization"] = f"Bearer {self._api_key}"

    def create_completion(
        self,
        *,
        model: str,
        messages: List[Any],
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion:
        if stream:
            raise BedrockAPIError("Streaming is not supported by the Bedrock adapter")
        request = build_converse_request(model=model, messages=messages, **kwargs)
        response_format = kwargs.get("response_format")
        json_mode = isinstance(response_format, dict) and response_format.get("type") in (
            "json_object",
            "json_schema",
        )
        if model in _MODELS_WITHOUT_SAMPLING:
            _strip_sampling(request)
        if model in _MODELS_WITHOUT_EFFORT:
            request.pop("additionalModelRequestFields", None)
        # Converse falls back to a small model default when maxTokens is
        # omitted, whereas OpenAI callers expect the model's full output limit
        inference = request.setdefault("inferenceConfig", {})
        inference["maxTokens"] = min(
            inference.get("maxTokens") or default_max_output_tokens(),
            _MODEL_OUTPUT_LIMITS.get(model, float("inf")),
        )

        # At most one sampling, one effort and one output-limit retry
        for _ in range(4):
            try:
                response = self._client.converse(**request)
                break
            except ClientError as error:
                if _is_sampling_rejection(error) and _strip_sampling(request):
                    _MODELS_WITHOUT_SAMPLING.add(model)
                    continue
                if "additionalModelRequestFields" in request and _is_effort_rejection(error):
                    request.pop("additionalModelRequestFields")
                    _MODELS_WITHOUT_EFFORT.add(model)
                    logger.warning("Bedrock rejected output_config.effort for %s; "
                                   "using the model's default thinking effort", model)
                    continue
                limit = _output_limit_from_error(error, request["inferenceConfig"]["maxTokens"])
                if limit:
                    _MODEL_OUTPUT_LIMITS[model] = limit
                    request["inferenceConfig"]["maxTokens"] = limit
                    continue
                raise translate_client_error(error) from None
            except BotoCoreError as error:
                raise BedrockAPIError(f"Bedrock request failed: {type(error).__name__}") from None
        else:
            raise BedrockAPIError("Bedrock rejected the request parameters after retries")
        return build_chat_completion(response, model=model, json_mode=json_mode)


def default_max_output_tokens() -> int:
    try:
        return max(1, int(os.environ.get("BEDROCK_MAX_OUTPUT_TOKENS", "")))
    except ValueError:
        return DEFAULT_MAX_OUTPUT_TOKENS


def _output_limit_from_error(error: ClientError, requested: int) -> Optional[int]:
    """Read the model's output cap from a maxTokens rejection, if that is what this is."""
    details = error.response.get("Error") or {}
    message = str(details.get("Message") or "")
    if details.get("Code") != "ValidationException" or not _MAX_TOKENS_REJECTION_PATTERN.search(message):
        return None
    candidates = [int(n) for n in re.findall(r"\d+", message) if 0 < int(n) < requested]
    return max(candidates) if candidates else None


def _strip_sampling(request: Dict[str, Any]) -> bool:
    """Remove temperature/topP from a Converse request; True if anything was removed."""
    inference = request.get("inferenceConfig")
    if not inference:
        return False
    removed = False
    for key in _SAMPLING_KEYS:
        if key in inference:
            del inference[key]
            removed = True
    if not inference:
        request.pop("inferenceConfig")
    return removed


def make_chat_client(
    api_key: Optional[str],
    base_url: Optional[str] = None,
    *,
    provider: Optional[str] = None,
    region: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Any:
    """Return an OpenAI client, or a Bedrock Converse client for provider=bedrock."""
    if is_bedrock_provider(provider):
        return BedrockChatClient(api_key, region=region, timeout=timeout)

    from openai import OpenAI

    kwargs: Dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    if timeout:
        kwargs["timeout"] = timeout
    return OpenAI(**kwargs)
