import json
import os
import sys
from types import SimpleNamespace

import boto3
import pytest
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig
from botocore.stub import Stubber
from openai import OpenAI

from scripts.bedrock_converse import (
    JSON_MODE_INSTRUCTION,
    BedrockAPIError,
    BedrockChatClient,
    build_chat_completion,
    build_converse_request,
    extract_json_object,
    make_chat_client,
    translate_messages,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

MODEL = "us.anthropic.claude-haiku-5-5"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def _converse_response(content, stop_reason="end_turn"):
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop_reason,
        "usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18},
        "metrics": {"latencyMs": 5},
        "ResponseMetadata": {"RequestId": "req-123", "HTTPStatusCode": 200},
    }


@pytest.fixture(autouse=True)
def _no_default_effort(monkeypatch):
    from scripts import bedrock_converse

    monkeypatch.delenv("BEDROCK_EFFORT", raising=False)
    monkeypatch.setattr(bedrock_converse, "_MODELS_WITHOUT_EFFORT", set())


def _stubbed_client():
    boto_client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        config=BotoConfig(signature_version=UNSIGNED),
    )
    return BedrockChatClient("test-key", boto_client=boto_client), Stubber(boto_client)


def test_translate_messages_moves_system_and_merges_roles():
    system, messages = translate_messages(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "developer", "content": "Use English."},
            {"role": "user", "content": "Hi"},
            {"role": "user", "content": [{"type": "text", "text": "there"}]},
            {"role": "assistant", "content": "Hello"},
        ],
        tools_enabled=False,
    )

    assert system == [{"text": "Be brief."}, {"text": "Use English."}]
    assert messages == [
        {"role": "user", "content": [{"text": "Hi"}, {"text": "there"}]},
        {"role": "assistant", "content": [{"text": "Hello"}]},
    ]


def test_translate_messages_prepends_user_turn():
    _, messages = translate_messages(
        [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}],
        tools_enabled=False,
    )

    assert messages[0]["role"] == "user"
    assert messages[1] == {"role": "assistant", "content": [{"text": "a"}]}


def test_translate_messages_tool_round_trip():
    _, messages = translate_messages(
        [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call:abc.1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call:abc.1", "content": "Sunny"},
        ],
        tools_enabled=True,
    )

    assert messages[1] == {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "call_abc_1", "name": "get_weather", "input": {"city": "Paris"}}}],
    }
    assert messages[2] == {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "call_abc_1", "content": [{"text": "Sunny"}]}}],
    }


def test_translate_messages_renders_tool_history_as_text_without_tools():
    _, messages = translate_messages(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ],
        tools_enabled=False,
    )

    flattened = json.dumps(messages)
    assert "toolUse" not in flattened and "toolResult" not in flattened
    assert "[Tool result c1]\\nresult" in flattened


def test_build_request_sampling_json_mode_and_tools():
    request = build_converse_request(
        model=MODEL,
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}],
        temperature=0.3,
        top_p=0.9,
        max_tokens=256,
        stop="END",
        tools=[WEATHER_TOOL],
        tool_choice="required",
        response_format={"type": "json_object"},
        frequency_penalty=0.5,
    )

    assert request["modelId"] == MODEL
    assert request["system"] == [{"text": "sys"}, {"text": JSON_MODE_INSTRUCTION}]
    assert request["inferenceConfig"] == {"maxTokens": 256, "temperature": 0.3, "stopSequences": ["END"]}
    assert request["toolConfig"]["toolChoice"] == {"any": {}}
    spec = request["toolConfig"]["tools"][0]["toolSpec"]
    assert spec["name"] == "get_weather"
    assert spec["inputSchema"]["json"]["properties"] == {"city": {"type": "string"}}


@pytest.mark.parametrize(
    "tool_choice, expected",
    [
        ("auto", {"auto": {}}),
        ("none", {"auto": {}}),
        ({"type": "function", "function": {"name": "get_weather"}}, {"tool": {"name": "get_weather"}}),
    ],
)
def test_build_request_tool_choice(tool_choice, expected):
    request = build_converse_request(
        model=MODEL,
        messages=[{"role": "user", "content": "u"}],
        tools=[WEATHER_TOOL],
        tool_choice=tool_choice,
    )

    assert request["toolConfig"]["toolChoice"] == expected


def test_build_request_uses_top_p_without_temperature():
    request = build_converse_request(model=MODEL, messages=[{"role": "user", "content": "u"}], top_p=0.8)

    assert request["inferenceConfig"] == {"topP": 0.8}


@pytest.mark.parametrize(
    "stop_reason, finish_reason",
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("content_filtered", "content_filter"),
        ("guardrail_intervened", "content_filter"),
        ("unknown_reason", "stop"),
    ],
)
def test_build_chat_completion_stop_reasons(stop_reason, finish_reason):
    completion = build_chat_completion(_converse_response([{"text": "ok"}], stop_reason), model=MODEL)

    assert completion.choices[0].finish_reason == finish_reason
    assert completion.choices[0].message.content == "ok"
    assert completion.usage.total_tokens == 18
    assert completion.id == "bedrock-req-123"


def test_build_chat_completion_tool_calls():
    completion = build_chat_completion(
        _converse_response(
            [
                {"text": "Checking."},
                {"toolUse": {"toolUseId": "tu_1", "name": "get_weather", "input": {"city": "Paris"}}},
            ],
            "tool_use",
        ),
        model=MODEL,
    )

    choice = completion.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == "Checking."
    call = choice.message.tool_calls[0]
    assert call.id == "tu_1"
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}


def test_json_mode_trims_surrounding_text():
    completion = build_chat_completion(
        _converse_response([{"text": 'Here you go:\n```json\n{"a": 1}\n```'}]),
        model=MODEL,
        json_mode=True,
    )

    assert completion.choices[0].message.content == '{"a": 1}'


def test_extract_json_object_leaves_invalid_text_alone():
    assert extract_json_object("no json here") == "no json here"
    assert extract_json_object("{not json}") == "{not json}"


def test_client_round_trip_with_stubber():
    client, stubber = _stubbed_client()
    stubber.add_response(
        "converse",
        _converse_response([{"text": "pong"}]),
        expected_params={
            "modelId": MODEL,
            "messages": [{"role": "user", "content": [{"text": "ping"}]}],
            "inferenceConfig": {"maxTokens": 20},
        },
    )

    with stubber:
        completion = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "ping"}], max_tokens=20
        )

    assert completion.choices[0].message.content == "pong"
    stubber.assert_no_pending_responses()


@pytest.mark.parametrize(
    "code, http_status, expected_status",
    [
        ("UnrecognizedClientException", 403, 401),
        ("ExpiredTokenException", 400, 401),
        ("AccessDeniedException", 403, 403),
        ("ThrottlingException", 429, 429),
    ],
)
def test_client_error_mapping(code, http_status, expected_status):
    client, stubber = _stubbed_client()
    stubber.add_client_error(
        "converse",
        service_error_code=code,
        service_message="The security token included in the request is invalid.",
        http_status_code=http_status,
    )

    with stubber, pytest.raises(BedrockAPIError) as exc_info:
        client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}])

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.code == code
    assert "test-key" not in str(exc_info.value)


def test_retries_without_temperature_when_model_rejects_it(monkeypatch):
    from scripts import bedrock_converse

    monkeypatch.setattr(bedrock_converse, "_MODELS_WITHOUT_SAMPLING", set())
    monkeypatch.setattr(bedrock_converse, "_MODEL_OUTPUT_LIMITS", {})
    monkeypatch.delenv("BEDROCK_MAX_OUTPUT_TOKENS", raising=False)
    client, stubber = _stubbed_client()
    messages = [{"role": "user", "content": [{"text": "ping"}]}]
    stubber.add_client_error(
        "converse",
        service_error_code="ValidationException",
        service_message="The model returned the following errors: `temperature` is deprecated for this model.",
        http_status_code=400,
        expected_params={
            "modelId": MODEL,
            "messages": messages,
            "inferenceConfig": {"maxTokens": 50, "temperature": 0.3},
        },
    )
    stubber.add_response(
        "converse",
        _converse_response([{"text": "pong"}]),
        expected_params={"modelId": MODEL, "messages": messages, "inferenceConfig": {"maxTokens": 50}},
    )
    stubber.add_response(
        "converse",
        _converse_response([{"text": "pong again"}]),
        expected_params={"modelId": MODEL, "messages": messages, "inferenceConfig": {"maxTokens": 32000}},
    )

    with stubber:
        first = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "ping"}], temperature=0.3, max_tokens=50
        )
        second = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "ping"}], temperature=0.7
        )

    assert first.choices[0].message.content == "pong"
    assert second.choices[0].message.content == "pong again"
    stubber.assert_no_pending_responses()


def test_uncapped_request_gets_default_output_limit_and_learns_model_cap(monkeypatch):
    from scripts import bedrock_converse

    monkeypatch.setattr(bedrock_converse, "_MODELS_WITHOUT_SAMPLING", set())
    monkeypatch.setattr(bedrock_converse, "_MODEL_OUTPUT_LIMITS", {})
    monkeypatch.setenv("BEDROCK_MAX_OUTPUT_TOKENS", "70000")
    client, stubber = _stubbed_client()
    messages = [{"role": "user", "content": [{"text": "ontology"}]}]
    stubber.add_client_error(
        "converse",
        service_error_code="ValidationException",
        service_message="max_tokens: 70000 > 64000, which is the maximum allowed number of output tokens",
        http_status_code=400,
        expected_params={"modelId": MODEL, "messages": messages, "inferenceConfig": {"maxTokens": 70000}},
    )
    for _ in range(2):
        stubber.add_response(
            "converse",
            _converse_response([{"text": "{}"}]),
            expected_params={"modelId": MODEL, "messages": messages, "inferenceConfig": {"maxTokens": 64000}},
        )

    with stubber:
        for _ in range(2):
            client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": "ontology"}], max_tokens=None
            )

    stubber.assert_no_pending_responses()


def test_reasoning_effort_maps_to_output_config(monkeypatch):
    request = build_converse_request(model=MODEL, messages=[{"role": "user", "content": "hi"}],
                                     reasoning_effort="low")
    assert request["additionalModelRequestFields"] == {"output_config": {"effort": "low"}}
    assert "additionalModelRequestFields" not in build_converse_request(
        model=MODEL, messages=[{"role": "user", "content": "hi"}])

    monkeypatch.setenv("BEDROCK_EFFORT", "Medium")
    request = build_converse_request(model=MODEL, messages=[{"role": "user", "content": "hi"}])
    assert request["additionalModelRequestFields"] == {"output_config": {"effort": "medium"}}
    request = build_converse_request(model=MODEL, messages=[{"role": "user", "content": "hi"}],
                                     reasoning_effort="low")
    assert request["additionalModelRequestFields"]["output_config"]["effort"] == "low"


def test_rejected_effort_is_dropped_and_remembered(monkeypatch):
    from scripts import bedrock_converse

    monkeypatch.setattr(bedrock_converse, "_MODELS_WITHOUT_SAMPLING", set())
    monkeypatch.setattr(bedrock_converse, "_MODEL_OUTPUT_LIMITS", {})
    monkeypatch.delenv("BEDROCK_MAX_OUTPUT_TOKENS", raising=False)
    client, stubber = _stubbed_client()
    messages = [{"role": "user", "content": [{"text": "ping"}]}]
    with_effort = {"modelId": MODEL, "messages": messages, "inferenceConfig": {"maxTokens": 50},
                   "additionalModelRequestFields": {"output_config": {"effort": "low"}}}
    without_effort = {"modelId": MODEL, "messages": messages, "inferenceConfig": {"maxTokens": 50}}
    stubber.add_client_error(
        "converse",
        service_error_code="ValidationException",
        service_message="output_config: Extra inputs are not permitted",
        http_status_code=400,
        expected_params=with_effort,
    )
    stubber.add_response("converse", _converse_response([{"text": "pong"}]),
                         expected_params=without_effort)
    stubber.add_response("converse", _converse_response([{"text": "pong"}]),
                         expected_params=without_effort)

    with stubber:
        for _ in range(2):
            response = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": "ping"}], max_tokens=50,
                reasoning_effort="low")
            assert response.choices[0].message.content == "pong"
    stubber.assert_no_pending_responses()


def test_truncation_logs_output_tokens_and_block_types(caplog):
    response = _converse_response(
        [{"reasoningContent": {"reasoningText": {"text": ""}}}, {"text": '{"a": '}],
        stop_reason="max_tokens",
    )
    with caplog.at_level("WARNING", logger="scripts.bedrock_converse"):
        completion = build_chat_completion(response, model=MODEL, json_mode=True)
    assert completion.choices[0].finish_reason == "length"
    assert "outputTokens=7" in caplog.text
    assert "reasoningContent" in caplog.text and "text_chars=6" in caplog.text


def test_other_validation_errors_are_not_retried(monkeypatch):
    from scripts import bedrock_converse

    monkeypatch.setattr(bedrock_converse, "_MODELS_WITHOUT_SAMPLING", set())
    client, stubber = _stubbed_client()
    stubber.add_client_error(
        "converse",
        service_error_code="ValidationException",
        service_message="The provided model identifier is invalid.",
        http_status_code=400,
    )

    with stubber, pytest.raises(BedrockAPIError) as exc_info:
        client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "hi"}], temperature=0.3
        )

    assert "model identifier is invalid" in str(exc_info.value)
    stubber.assert_no_pending_responses()


def test_streaming_is_rejected():
    client, _ = _stubbed_client()

    with pytest.raises(BedrockAPIError):
        client.chat.completions.create(model=MODEL, messages=[], stream=True)


def test_bearer_token_header_is_per_client():
    client, _ = _stubbed_client()
    request = SimpleNamespace(headers={})

    client._add_bearer_token(request)

    assert request.headers["Authorization"] == "Bearer test-key"
    assert "AWS_BEARER_TOKEN_BEDROCK" not in os.environ


def test_make_chat_client_selects_provider():
    assert isinstance(make_chat_client("k", "https://example.com/v1"), OpenAI)
    assert isinstance(make_chat_client("k", "https://example.com/v1", provider="openai"), OpenAI)
    bedrock = make_chat_client("k", None, provider=" Bedrock ", region="us-west-2")
    assert isinstance(bedrock, BedrockChatClient)
    assert bedrock.region == "us-west-2"


class _FakeBedrockClient:
    def __init__(self, completion):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._completion = completion

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._completion


def test_camel_model_returns_tool_calls():
    from camel_bedrock_model import BedrockConverseModel

    completion = build_chat_completion(
        _converse_response(
            [{"toolUse": {"toolUseId": "tu_9", "name": "get_weather", "input": {"city": "Oslo"}}}],
            "tool_use",
        ),
        model=MODEL,
    )
    fake = _FakeBedrockClient(completion)
    model = BedrockConverseModel(model_type=MODEL, api_key="test-key", bedrock_client=fake)

    response = model.run([{"role": "user", "content": "Weather?"}], tools=[WEATHER_TOOL])

    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
    assert fake.calls[0]["model"] == MODEL
    assert fake.calls[0]["tools"] == [WEATHER_TOOL]
    assert "stream" not in fake.calls[0]


def test_camel_model_structured_output_uses_json_mode():
    from pydantic import BaseModel

    from camel_bedrock_model import BedrockConverseModel

    class Person(BaseModel):
        name: str

    fake = _FakeBedrockClient(build_chat_completion(_converse_response([{"text": '{"name": "Ada"}'}]), model=MODEL))
    model = BedrockConverseModel(model_type=MODEL, api_key="test-key", bedrock_client=fake)

    response = model.run([{"role": "user", "content": "Name someone."}], response_format=Person)

    assert fake.calls[0]["response_format"] == {"type": "json_object"}
    assert Person.model_validate_json(response.choices[0].message.content).name == "Ada"


@pytest.mark.asyncio
async def test_camel_model_async_runs_in_thread():
    from camel_bedrock_model import BedrockConverseModel

    fake = _FakeBedrockClient(build_chat_completion(_converse_response([{"text": "hi"}]), model=MODEL))
    model = BedrockConverseModel(model_type=MODEL, api_key="test-key", bedrock_client=fake)

    response = await model.arun([{"role": "user", "content": "Hello"}])

    assert response.choices[0].message.content == "hi"
