"""Concrete providers. All return parsed JSON; none of them are given tools (the grader can only emit data).

mock           deterministic, offline - default; lets the whole pipeline run with zero accounts or cost
gemini         Google Gemini REST API (free tier available)
openai_compat  any OpenAI-compatible /chat/completions endpoint (Groq free tier, OpenRouter, vLLM, ...)
bedrock        AWS Bedrock Converse API; structured output via a forced tool call (the production target)
"""

import hashlib
import json
import threading

import httpx

from app.config import Settings
from app.pipeline.llm.base import (
    LLMAuthError, LLMMalformedResponse, LLMRateLimited, LLMRequest, LLMResponse, LLMTimeout, LLMTokenLimit,
    LLMUnavailable, Provider,
)


def _tokens(*texts: str) -> int:
    return max(1, sum(len(t) for t in texts) // 4)


def _parse_json_text(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise LLMMalformedResponse(f"response was not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMMalformedResponse("response JSON was not an object")
    return data


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return float(resp.headers.get("retry-after", ""))
    except ValueError:
        return None


def _raise_for_http(resp: httpx.Response) -> None:
    code = resp.status_code
    if code < 400:
        return
    body = resp.text[:500].lower()
    if code == 429:
        raise LLMRateLimited("provider rate limit", _retry_after(resp))
    if code in (401, 403):
        raise LLMAuthError(f"provider rejected credentials (HTTP {code})")
    if code in (408, 504):
        raise LLMTimeout(f"provider timeout (HTTP {code})")
    if code >= 500:
        raise LLMUnavailable(f"provider error (HTTP {code})")
    if code in (400, 413) and ("token" in body or "context length" in body or "too large" in body or "too long" in body):
        raise LLMTokenLimit("input exceeds the model's context window")
    raise LLMMalformedResponse(f"provider rejected the request (HTTP {code})")


class MockProvider(Provider):
    """Deterministic stand-in. Scores are a stable function of (criterion, context) so runs are reproducible."""

    name = "mock"

    def __init__(self, settings: Settings):
        self.settings = settings

    def model_for(self, tier: str) -> str:
        return {"cheap": self.settings.llm_model_cheap, "premium": self.settings.llm_model_premium}.get(tier, self.settings.llm_model_standard)

    def complete(self, req: LLMRequest) -> LLMResponse:
        meta = req.meta
        paths = meta.get("known_paths", [])
        if req.schema_name == "file_summaries":
            data = {"items": [{"path": f["path"], "summary": f"{f['path']}: " + (next((ln.strip() for ln in f["head"].splitlines() if ln.strip()), "empty file"))[:120]} for f in meta.get("files", [])]}
        elif req.schema_name == "dir_summary":
            data = {"summary": f"{meta.get('directory')}: " + " ".join(meta.get("summaries", []))[:300]}
        elif req.schema_name == "understanding":
            data = {
                "project_type": meta.get("project_type", "software project"),
                "purpose": "A student software project (mock analysis).",
                "tech_stack": meta.get("frameworks", []),
                "main_features": [p for p in paths[:3]],
                "summary": f"Project with {meta.get('file_count', 0)} analysed files.",
            }
        elif req.schema_name == "analysis":
            data = {
                "summary": f"Mock {meta.get('focus')} analysis.",
                "strengths": ["Reasonable structure"] if paths else [],
                "weaknesses": [] if meta.get("has_tests") else ["No automated tests found"],
                "notable_files": paths[:3],
            }
        elif req.schema_name == "criterion_score":
            digest = hashlib.sha256(f"{meta.get('criterion')}|{meta.get('context_digest')}".encode()).digest()
            ratio = 0.55 + (digest[0] / 255) * 0.37
            max_score = float(meta.get("max_score", 10))
            data = {
                "score": round(ratio * max_score, 1),
                "justification": f"The submission shows a {'solid' if ratio > 0.7 else 'moderate'} level for {meta.get('criterion')} based on the available files. "
                f"Evidence was taken from {', '.join(paths[:2]) or 'the project metadata'}.",
                "evidence": paths[:2],
                "confidence": 0.85,
                "injection_suspected": False,
            }
        else:
            raise LLMMalformedResponse(f"mock provider does not know schema {req.schema_name}")
        return LLMResponse(data, _tokens(req.system, req.context, req.task), _tokens(json.dumps(data)), self.model_for(req.tier))


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        if not settings.llm_api_key:
            raise ValueError("LLM_API_KEY is required for the gemini provider")
        self.settings = settings
        self.base = (settings.llm_base_url or "https://generativelanguage.googleapis.com").rstrip("/")
        self.client = httpx.Client(timeout=settings.llm_timeout_s, transport=transport)

    def model_for(self, tier: str) -> str:
        return {"cheap": self.settings.llm_model_cheap, "premium": self.settings.llm_model_premium}.get(tier, self.settings.llm_model_standard)

    def complete(self, req: LLMRequest) -> LLMResponse:
        model = self.model_for(req.tier)
        body = {
            "systemInstruction": {"parts": [{"text": req.system}]},
            "contents": [{"role": "user", "parts": [{"text": (req.context + "\n\n" if req.context else "") + req.task}]}],
            "generationConfig": {"temperature": req.temperature, "maxOutputTokens": req.max_output_tokens, "responseMimeType": "application/json"},
        }
        try:
            resp = self.client.post(f"{self.base}/v1beta/models/{model}:generateContent", json=body, headers={"x-goog-api-key": self.settings.llm_api_key})
        except httpx.TimeoutException as exc:
            raise LLMTimeout("request timed out") from exc
        except httpx.TransportError as exc:
            raise LLMUnavailable(f"network error: {type(exc).__name__}") from exc
        _raise_for_http(resp)
        payload = resp.json()
        try:
            cand = payload["candidates"][0]
            text = "".join(p.get("text", "") for p in cand["content"]["parts"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMMalformedResponse("response had no candidate text (blocked or empty)") from exc
        if cand.get("finishReason") == "MAX_TOKENS":
            raise LLMMalformedResponse("output was truncated at the token limit")
        usage = payload.get("usageMetadata", {})
        return LLMResponse(
            _parse_json_text(text), int(usage.get("promptTokenCount", 0)), int(usage.get("candidatesTokenCount", 0)), model,
            int(usage.get("cachedContentTokenCount", 0)),
        )


class OpenAICompatProvider(Provider):
    name = "openai_compat"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        if not (settings.llm_api_key and settings.llm_base_url):
            raise ValueError("LLM_API_KEY and LLM_BASE_URL are required for the openai_compat provider")
        self.settings = settings
        self.client = httpx.Client(timeout=settings.llm_timeout_s, transport=transport)

    def model_for(self, tier: str) -> str:
        return {"cheap": self.settings.llm_model_cheap, "premium": self.settings.llm_model_premium}.get(tier, self.settings.llm_model_standard)

    def complete(self, req: LLMRequest) -> LLMResponse:
        model = self.model_for(req.tier)
        user = (req.context + "\n\n" if req.context else "") + req.task
        body = {
            "model": model,
            "messages": [{"role": "system", "content": req.system}, {"role": "user", "content": user}],
            "temperature": req.temperature,
            "max_tokens": req.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = self.client.post(
                self.settings.llm_base_url.rstrip("/") + "/chat/completions", json=body,
                headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeout("request timed out") from exc
        except httpx.TransportError as exc:
            raise LLMUnavailable(f"network error: {type(exc).__name__}") from exc
        _raise_for_http(resp)
        payload = resp.json()
        try:
            choice = payload["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMMalformedResponse("response had no message content") from exc
        if choice.get("finish_reason") == "length":
            raise LLMMalformedResponse("output was truncated at the token limit")
        usage = payload.get("usage", {})
        return LLMResponse(_parse_json_text(text), int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), model)


class BedrockProvider(Provider):
    """AWS Bedrock Converse API. Structured output = a forced tool call whose input schema is our JSON schema.
    The model is never given an executable tool: the 'tool call' is only a typed output channel."""

    name = "bedrock"

    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "bedrock-runtime", region_name=settings.s3_region,
                config=Config(read_timeout=settings.llm_timeout_s, connect_timeout=10, retries={"max_attempts": 1}),
            )
        self.client = client
        self._lock = threading.Lock()

    def model_for(self, tier: str) -> str:
        return {"cheap": self.settings.llm_model_cheap, "premium": self.settings.llm_model_premium}.get(tier, self.settings.llm_model_standard)

    def complete(self, req: LLMRequest) -> LLMResponse:
        from botocore.exceptions import BotoCoreError, ClientError, ReadTimeoutError

        model = self.model_for(req.tier)
        content: list[dict] = []
        if req.context:
            content.append({"text": req.context})
            if self.settings.llm_prompt_cache:
                content.append({"cachePoint": {"type": "default"}})
        content.append({"text": req.task})
        try:
            resp = self.client.converse(
                modelId=model,
                system=[{"text": req.system}],
                messages=[{"role": "user", "content": content}],
                inferenceConfig={"maxTokens": req.max_output_tokens, "temperature": req.temperature},
                toolConfig={
                    "tools": [{"toolSpec": {"name": req.schema_name, "description": "Return the structured result.", "inputSchema": {"json": req.schema}}}],
                    "toolChoice": {"tool": {"name": req.schema_name}},
                },
            )
        except ReadTimeoutError as exc:
            raise LLMTimeout("request timed out") from exc
        except ClientError as exc:
            err = exc.response.get("Error", {})
            code, msg = err.get("Code", ""), err.get("Message", "").lower()
            if code in ("ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException"):
                raise LLMRateLimited("bedrock throttled") from exc
            if code in ("ModelTimeoutException",):
                raise LLMTimeout("bedrock model timeout") from exc
            if code in ("ServiceUnavailableException", "InternalServerException", "ModelNotReadyException"):
                raise LLMUnavailable(f"bedrock unavailable ({code})") from exc
            if code == "ValidationException" and ("too long" in msg or "too many tokens" in msg or "context" in msg):
                raise LLMTokenLimit("input exceeds the model's context window") from exc
            if code in ("AccessDeniedException", "UnrecognizedClientException"):
                raise LLMAuthError(f"bedrock access denied ({code})") from exc
            raise LLMMalformedResponse(f"bedrock rejected the request ({code})") from exc
        except BotoCoreError as exc:
            raise LLMUnavailable(f"network error: {type(exc).__name__}") from exc
        if resp.get("stopReason") == "max_tokens":
            raise LLMMalformedResponse("output was truncated at the token limit")
        data = None
        for block in resp.get("output", {}).get("message", {}).get("content", []):
            if "toolUse" in block:
                data = block["toolUse"]["input"]
                break
        if not isinstance(data, dict):
            raise LLMMalformedResponse("model did not return the structured result")
        usage = resp.get("usage", {})
        return LLMResponse(data, int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0)), model, int(usage.get("cacheReadInputTokens", 0)))


def build_provider(settings: Settings) -> Provider:
    name = settings.llm_provider
    if name == "mock":
        return MockProvider(settings)
    if name == "gemini":
        return GeminiProvider(settings)
    if name == "openai_compat":
        return OpenAICompatProvider(settings)
    if name == "bedrock":
        return BedrockProvider(settings)
    raise ValueError(f"unknown LLM_PROVIDER: {name}")
