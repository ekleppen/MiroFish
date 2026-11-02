"""
CAMEL model backend that runs OASIS agents on Amazon Bedrock Converse

CAMEL's own Bedrock and Anthropic backends expect an OpenAI-compatible
endpoint, which Bedrock does not offer for Claude. This backend keeps CAMEL's
OpenAI-shaped plumbing and swaps the HTTP client for BedrockChatClient.
"""

import asyncio
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from camel.messages import OpenAIMessage
from camel.models._utils import try_modify_message_with_format
from camel.models.openai_compatible_model import OpenAICompatibleModel
from camel.types import ChatCompletion

from bedrock_converse import BedrockChatClient


class BedrockConverseModel(OpenAICompatibleModel):
    """OpenAI-compatible CAMEL backend backed by Bedrock Converse."""

    def __init__(
        self,
        model_type: str,
        api_key: str,
        *,
        region: Optional[str] = None,
        model_config_dict: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        bedrock_client: Optional[BedrockChatClient] = None,
    ) -> None:
        # The parent builds unused OpenAI clients; no request is sent to this URL
        super().__init__(
            model_type=model_type,
            model_config_dict=model_config_dict or {},
            api_key=api_key,
            url="https://bedrock.invalid/v1",
            timeout=timeout,
        )
        self._client = bedrock_client or BedrockChatClient(
            api_key, region=region, timeout=self._timeout
        )
        self._async_client = None

    def _request_config(self, tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        request_config = dict(self.model_config_dict)
        request_config.pop("stream", None)
        if tools:
            request_config["tools"] = tools
        return request_config

    def _request_chat_completion(
        self,
        messages: List[OpenAIMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        return self._client.chat.completions.create(
            messages=messages,
            model=self.model_type,
            **self._request_config(tools),
        )

    async def _arequest_chat_completion(
        self,
        messages: List[OpenAIMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        return await asyncio.to_thread(
            self._request_chat_completion, messages, tools
        )

    def _request_parse(
        self,
        messages: List[OpenAIMessage],
        response_format: Type[BaseModel],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        # Converse has no structured-output parse; ask for JSON instead
        try_modify_message_with_format(messages[-1], response_format)
        request_config = self._request_config(tools)
        request_config["response_format"] = {"type": "json_object"}
        return self._client.chat.completions.create(
            messages=messages,
            model=self.model_type,
            **request_config,
        )

    async def _arequest_parse(
        self,
        messages: List[OpenAIMessage],
        response_format: Type[BaseModel],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        return await asyncio.to_thread(
            self._request_parse, messages, response_format, tools
        )
