"""Google Gemini model provider implementation."""

import logging
from typing import Any
from google import genai
from google.genai import types
from google.genai.errors import APIError

from backend.src.config import settings

logger = logging.getLogger(__name__)

# Gemini 2.5+ models reason before answering, and those thinking tokens are
# charged to the same `max_output_tokens` budget as the answer. That budget is
# tuned here for a model that does not think: measured on one query-generation
# prompt at MODEL_MAX_TOKENS=1500, thinking took 1441 tokens and left 55 for the
# SQL, which came back cut off mid-statement with finish_reason=MAX_TOKENS --
# and a truncated query is worse than no query, because it still parses far
# enough to look like an answer. With thinking off the same prompt answers in
# 491 tokens. KoboldCppProvider solves the identical problem for Qwen3 with its
# /no_think directive, and MODEL_DISABLE_THINKING is the shared switch.
#
# Only some models can turn it off: Flash accepts a zero budget, while Pro
# rejects it outright ("Budget 0 is invalid"), so a refusal is treated as "this
# model always thinks" and the request is retried as the model wants it.
THINKING_DISABLED_BUDGET = 0


class GeminiProvider:
    """Model provider for Gemini via the official google-genai SDK.

    Talks to Google directly by default. Given a `base_url` -- a LiteLLM proxy
    or any other OpenAI-compatible gateway that fronts Gemini -- the same SDK is
    pointed there instead, which is all the switch amounts to: the gateway
    speaks the same protocol, so only the endpoint and the credential change.
    The key then belongs to the gateway (LiteLLM issues `sk-...` virtual keys,
    not Google `AIza...` keys), and the model name has to be whatever the
    gateway publishes, which for LiteLLM carries the provider prefix --
    `gemini/gemini-2.5-flash` rather than `gemini-2.5-flash`. That name is
    passed through untouched rather than prefixed here, because a proxy is free
    to expose its own aliases and guessing at them would break the ones that do.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        client: genai.Client | None = None,
        disable_thinking: bool | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model = model if model is not None else settings.gemini_model
        base = base_url if base_url is not None else settings.gemini_base_url
        # Trailing slashes are stripped because the SDK joins its own paths onto
        # this, and a doubled separator is rejected by some gateways.
        self.base_url = base.strip().rstrip("/") if base and base.strip() else None
        self.default_temperature = (
            temperature if temperature is not None else settings.model_temperature
        )
        self.default_max_tokens = (
            max_tokens if max_tokens is not None else settings.model_max_tokens
        )
        if timeout is not None:
            self.timeout = float(timeout) if timeout > 0 else None
        elif settings.model_timeout_seconds is not None and settings.model_timeout_seconds > 0:
            self.timeout = float(settings.model_timeout_seconds)
        else:
            self.timeout = None

        self.disable_thinking = (
            disable_thinking
            if disable_thinking is not None
            else settings.model_disable_thinking
        )
        # Set once a model has refused a zero thinking budget, so the whole
        # pipeline pays for that discovery once rather than on every call.
        self._thinking_unsupported = False

        self._client = client

        # Validation: do not allow empty credentials or model unless pre-injected client is provided
        if not self._client:
            if not self.api_key or not self.api_key.strip():
                raise ValueError(
                    "GEMINI_API_KEY must be configured in environment/.env to use "
                    + (
                        f"the Gemini gateway at {self.base_url}."
                        if self.base_url
                        else "Gemini provider."
                    )
                )
            if not self.model or not self.model.strip():
                raise ValueError(
                    "GEMINI_MODEL must be configured in environment/.env to use Gemini provider."
                )

    def _get_client(self) -> genai.Client:
        if self._client is not None:
            return self._client
        if not self.base_url:
            return genai.Client(api_key=self.api_key)
        # HttpOptions.timeout is milliseconds, unlike every other timeout in
        # this codebase, which are seconds.
        http_options = types.HttpOptions(base_url=self.base_url)
        if self.timeout is not None:
            http_options.timeout = int(self.timeout * 1000)
        return genai.Client(api_key=self.api_key, http_options=http_options)

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate response from Gemini model.

        Returns raw model response string without any post-processing.
        """
        temp = temperature if temperature is not None else self.default_temperature
        max_toks = max_tokens if max_tokens is not None else self.default_max_tokens

        def build_config(with_thinking_off: bool) -> types.GenerateContentConfig:
            config = types.GenerateContentConfig(
                temperature=temp,
                max_output_tokens=max_toks,
                system_instruction=system_prompt,
            )
            if with_thinking_off:
                config.thinking_config = types.ThinkingConfig(
                    thinking_budget=THINKING_DISABLED_BUDGET
                )
            return config

        client = self._get_client()
        thinking_off = self.disable_thinking and not self._thinking_unsupported

        try:
            try:
                response = await client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=build_config(thinking_off),
                )
            except APIError as err:
                # A model that cannot stop thinking says so with a 400 rather
                # than by ignoring the field, so the request is reissued on the
                # model's terms instead of failing over something optional.
                if not (thinking_off and self._is_thinking_unsupported(err)):
                    raise
                logger.warning(
                    "Model %s does not support disabling thinking; retrying with it on. "
                    "Its reasoning is charged to MODEL_MAX_TOKENS (%s), so raise that "
                    "if answers come back truncated.",
                    self.model,
                    max_toks,
                )
                self._thinking_unsupported = True
                response = await client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=build_config(False),
                )

            self._raise_if_truncated(response, max_toks)
            return response.text or ""

        except APIError as err:
            raise RuntimeError(f"Gemini API error: {err}") from err
        except TimeoutError as err:
            raise TimeoutError(f"Gemini request timed out: {err}") from err
        except Exception as err:
            if not isinstance(err, (ConnectionError, TimeoutError, RuntimeError, ValueError)):
                raise RuntimeError(f"Unexpected error communicating with Gemini: {err}") from err
            raise

    @staticmethod
    def _is_thinking_unsupported(err: APIError) -> bool:
        """Whether this error is the model refusing a zero thinking budget."""
        message = str(err).lower()
        return "budget" in message and "invalid" in message

    def _raise_if_truncated(self, response: Any, max_toks: int) -> None:
        """Reject a reply the model did not finish writing.

        Returning it would hand the pipeline a half-written query, and a
        half-written query is not a soft failure: it survives extraction, so it
        reaches validation looking like the model's answer and is retried
        against the same budget that cut it off. Raising instead puts the real
        reason in the error the caller sees.
        """
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return
        if getattr(candidates[0], "finish_reason", None) != types.FinishReason.MAX_TOKENS:
            return

        usage = getattr(response, "usage_metadata", None)
        thoughts = getattr(usage, "thoughts_token_count", None) or 0
        answer = getattr(usage, "candidates_token_count", None) or 0
        detail = (
            f"Gemini hit the {max_toks}-token output limit before finishing"
            + (
                f": {thoughts} tokens went to reasoning and only {answer} to the answer. "
                "Disable thinking, or raise MODEL_MAX_TOKENS."
                if thoughts
                else f" ({answer} tokens used). Raise MODEL_MAX_TOKENS."
            )
        )
        raise RuntimeError(detail)
