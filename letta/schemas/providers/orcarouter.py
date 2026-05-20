from typing import Literal

from openai import AsyncOpenAI, AuthenticationError, PermissionDeniedError
from pydantic import Field

from letta.errors import (
    ErrorCode,
    LLMAuthenticationError,
    LLMError,
    LLMPermissionDeniedError,
)
from letta.log import get_logger
from letta.schemas.enums import ProviderCategory, ProviderType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.providers.openai import OpenAIProvider

logger = get_logger(__name__)

# Default context window for models not in the API response
DEFAULT_CONTEXT_WINDOW = 128000


class OrcaRouterProvider(OpenAIProvider):
    """
    OrcaRouter provider - https://www.orcarouter.ai/

    OrcaRouter is an OpenAI-compatible API gateway that routes requests across multiple LLM providers (OpenAI, Anthropic, DeepSeek, etc.)
    with a variety of intelligent routing strategies.

    Documentation: https://docs.orcarouter.ai/introduction

    """

    provider_type: Literal[ProviderType.orcarouter] = Field(ProviderType.orcarouter, description="The type of the provider.")
    provider_category: ProviderCategory = Field(ProviderCategory.base, description="The category of the provider (base or byok)")
    api_key: str | None = Field(None, description="API key for the OrcaRouter API.", deprecated=True)
    base_url: str = Field("https://api.orcarouter.ai/v1", description="Base URL for the OrcaRouter API.")

    async def check_api_key(self):
        """Check if the API key is valid by making a test request to the OrcaRouter API."""
        api_key = await self.api_key_enc.get_plaintext_async() if self.api_key_enc else None
        if not api_key:
            raise ValueError("No API key provided")

        try:
            # Use async OpenAI client pointed at OrcaRouter's endpoint
            client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
            # Just list models to verify API key works
            await client.models.list()
        except AuthenticationError as e:
            raise LLMAuthenticationError(
                message=f"Failed to authenticate with OrcaRouter: {e}",
                code=ErrorCode.UNAUTHENTICATED,
            )
        except PermissionDeniedError as e:
            raise LLMPermissionDeniedError(
                message=f"Permission denied by OrcaRouter: {e}",
                code=ErrorCode.PERMISSION_DENIED,
            )
        except AttributeError as e:
            if "_set_private_attributes" in str(e):
                raise LLMError(
                    message=f"OrcaRouter endpoint at {self.base_url} returned an unexpected non-JSON response. Verify the base URL and API key.",
                    code=ErrorCode.INTERNAL_SERVER_ERROR,
                )
            raise LLMError(message=f"{e}", code=ErrorCode.INTERNAL_SERVER_ERROR)
        except Exception as e:
            raise LLMError(message=f"{e}", code=ErrorCode.INTERNAL_SERVER_ERROR)

    def get_model_context_window_size(self, model_name: str) -> int | None:
        """Get the context window size for an OrcaRouter model.

        OrcaRouter models provide context_length in the API response,
        so this is mainly a fallback.
        """
        return DEFAULT_CONTEXT_WINDOW

    async def list_llm_models_async(self) -> list[LLMConfig]:
        """
        Return available OrcaRouter models that support tool calling.

        OrcaRouter provides a models endpoint that supports filtering by supported_parameters.
        We filter for models that support 'tools' to ensure Letta compatibility.
        """
        from letta.llm_api.openai import openai_get_model_list_async

        api_key = await self.api_key_enc.get_plaintext_async() if self.api_key_enc else None

        # OrcaRouter supports filtering models by supported parameters
        extra_params = {"supported_parameters": "tools"}

        response = await openai_get_model_list_async(
            self.base_url,
            api_key=api_key,
            extra_params=extra_params,
        )

        data = response.get("data", response)

        configs = []
        for model in data:
            if "id" not in model:
                logger.warning(f"OrcaRouter model missing 'id' field: {model}")
                continue

            model_name = model["id"]

            # OrcaRouter returns context_length in the model listing
            if model.get("context_length"):
                context_window_size = model["context_length"]
            else:
                context_window_size = self.get_model_context_window_size(model_name)
                logger.debug(f"Model {model_name} missing context_length, using default: {context_window_size}")

            configs.append(
                LLMConfig(
                    model=model_name,
                    model_endpoint_type="orcarouter",
                    model_endpoint=self.base_url,
                    context_window=context_window_size,
                    handle=self.get_handle(model_name),
                    max_tokens=self.get_default_max_output_tokens(model_name),
                    provider_name=self.name,
                    provider_category=self.provider_category,
                )
            )

        return configs
