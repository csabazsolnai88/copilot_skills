"""Reusable async vLLM client for vllm-app skill.

Copy this file into your project and import it:

    from vllm_client import VLLMClient, AppSettings

Usage:

    config = AppSettings(vllm_api_url="http://localhost:8000")
    async with VLLMClient(config) as client:
        result = await client.complete(prompt, response_schema=schema)
"""

import asyncio
import json
import logging
from typing import Any

import aiohttp
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)


class AppSettings(BaseSettings):
    """vLLM app settings with environment variable support.

    All fields can be overridden via environment variables with the
    TERMFORCE_ prefix, e.g. TERMFORCE_VLLM_API_URL.
    """

    model_config = SettingsConfigDict(env_prefix="TERMFORCE_")

    vllm_api_url: str = "http://localhost:8000"
    vllm_api_key: str | None = None
    batch_size: int = 5
    vllm_concurrency: int = 8
    request_timeout: int = 120
    max_retries: int = 3
    retry_backoff: float = 1.0


class VLLMClient:
    """Async client for vLLM OpenAI-compatible API.

    Features:
    - Semaphore-based concurrency control
    - Exponential backoff retry on 429/503/timeout
    - structured_outputs for guided JSON generation
    - Markdown code-block fallback JSON parsing
    """

    def __init__(self, config: AppSettings):
        self.base_url = config.vllm_api_url.rstrip("/")
        self.api_key = config.vllm_api_key
        self.timeout = aiohttp.ClientTimeout(total=config.request_timeout)
        self.max_retries = config.max_retries
        self.retry_backoff = config.retry_backoff
        self.concurrency = config.vllm_concurrency
        self._session: aiohttp.ClientSession | None = None
        self._semaphore: asyncio.Semaphore | None = None

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=self.concurrency)
        self._session = aiohttp.ClientSession(connector=connector, timeout=self.timeout)
        self._semaphore = asyncio.Semaphore(self.concurrency)
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()

    async def complete(
        self,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> dict:
        """Send single completion request with retry logic.

        Args:
            prompt: Text prompt for the LLM.
            response_schema: Optional JSON schema for guided generation.
            temperature: Sampling temperature (0.0 for deterministic).

        Returns:
            Parsed JSON response from the LLM.

        Raises:
            Exception: If all retries are exhausted.
        """
        if not self._session or not self._semaphore:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": 2048,
        }

        if response_schema:
            payload["structured_outputs"] = {"json": response_schema}

        url = f"{self.base_url}/v1/completions"

        for attempt in range(self.max_retries):
            try:
                async with (
                    self._semaphore,
                    self._session.post(url, headers=headers, json=payload) as resp,
                ):
                    if resp.status == 200:
                        data = await resp.json()
                        completion = data["choices"][0]["text"].strip()
                        try:
                            return json.loads(completion)
                        except json.JSONDecodeError:
                            if "```json" in completion:
                                try:
                                    start = completion.find("```json") + 7
                                    end = completion.find("```", start)
                                    extracted = completion[start:end].strip()
                                    result = json.loads(extracted)
                                    logger.debug("Extracted JSON from markdown code block")
                                    return result
                                except (json.JSONDecodeError, ValueError):
                                    logger.debug(f"JSON parsing failed for: {completion[:100]}")
                                    raise
                            else:
                                logger.debug(f"No valid JSON found in: {completion[:100]}")
                                raise

                    elif resp.status in (429, 503):
                        wait_time = self.retry_backoff * (2**attempt)
                        logger.warning(
                            f"Status {resp.status}, retrying in {wait_time}s "
                            f"(attempt {attempt + 1}/{self.max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        error_text = await resp.text()
                        logger.error(f"HTTP {resp.status}: {error_text}")
                        raise Exception(f"HTTP {resp.status}: {error_text}")

            except TimeoutError:
                wait_time = self.retry_backoff * (2**attempt)
                logger.warning(
                    f"Timeout, retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})"
                )
                await asyncio.sleep(wait_time)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"All retries exhausted: {e}")
                    raise
                wait_time = self.retry_backoff * (2**attempt)
                logger.warning(
                    f"Request failed ({type(e).__name__}), retrying in {wait_time}s "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                await asyncio.sleep(wait_time)

        raise Exception("All retries exhausted")

    async def batch_complete(
        self,
        prompts: list[str],
        response_schema: dict[str, Any] | None = None,
    ) -> list[dict | Exception]:
        """Dispatch multiple prompts concurrently.

        Args:
            prompts: List of text prompts.
            response_schema: Optional JSON schema for all prompts.

        Returns:
            List of parsed responses or exceptions.
        """
        tasks = [self.complete(p, response_schema) for p in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)
