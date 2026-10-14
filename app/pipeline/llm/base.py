"""LLM client: one provider-neutral request/response type and a resilient wrapper around any provider.

Failure handling implemented here:
  timeout / rate limit / unavailable -> exponential backoff with jitter (honours Retry-After), bounded retries
  repeated failures                  -> circuit breaker (fail fast instead of hammering a dead provider)
  malformed / invalid JSON           -> validate against a pydantic schema, then bounded repair-retry
  token limit exceeded               -> LLMTokenLimit is raised so the caller can shrink the context and retry
  provider quota                     -> process-wide concurrency semaphore
  runaway cost                       -> per-attempt token budget
"""

import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from pydantic import BaseModel, ValidationError

from app.config import Settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    retryable = True


class LLMRateLimited(LLMError):
    def __init__(self, message: str = "rate limited", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMTimeout(LLMError):
    pass


class LLMUnavailable(LLMError):
    pass


class CircuitOpen(LLMUnavailable):
    pass


class LLMMalformedResponse(LLMError):
    retryable = False


class LLMTokenLimit(LLMError):
    retryable = False


class LLMAuthError(LLMError):
    retryable = False


class BudgetExceeded(Exception):
    pass


class SemanticError(ValueError):
    """Raised by validators when the JSON parses but its content is unacceptable."""


@dataclass
class LLMRequest:
    system: str
    context: str  # shared, stable prefix (cache-friendly): identical across all calls of one evaluation
    task: str
    schema: dict
    schema_name: str
    tier: str = "standard"
    temperature: float = 0.1
    max_output_tokens: int = 2048
    meta: dict = field(default_factory=dict)  # provider-agnostic hints; only the mock provider reads it


@dataclass
class LLMResponse:
    data: dict
    input_tokens: int
    output_tokens: int
    model: str
    cached_tokens: int = 0


class Provider(ABC):
    name = "provider"

    @abstractmethod
    def model_for(self, tier: str) -> str: ...

    @abstractmethod
    def complete(self, req: LLMRequest) -> LLMResponse:
        """One attempt. Raise an LLMError subclass on failure."""


class UsageTracker:
    def __init__(self, settings: Settings, token_budget: int):
        self.settings = settings
        self.token_budget = token_budget
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.calls = 0
        self.cost_usd = 0.0
        self.models: set[str] = set()

    def check(self) -> None:
        if self.input_tokens + self.output_tokens >= self.token_budget:
            raise BudgetExceeded("token budget for this evaluation attempt is exhausted")

    def add(self, resp: LLMResponse) -> None:
        with self._lock:
            self.input_tokens += resp.input_tokens
            self.output_tokens += resp.output_tokens
            self.cached_tokens += resp.cached_tokens
            self.calls += 1
            self.models.add(resp.model)
            pin = self.settings.price_in_per_mtok.get(resp.model, 0.0)
            pout = self.settings.price_out_per_mtok.get(resp.model, 0.0)
            self.cost_usd += resp.input_tokens / 1e6 * pin + resp.output_tokens / 1e6 * pout

    def snapshot(self) -> dict:
        return {
            "calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens, "cost_usd": round(self.cost_usd, 6), "models": sorted(self.models),
        }


class CircuitBreaker:
    def __init__(self, threshold: int, cooldown_s: float, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.cooldown_s, self.clock = threshold, cooldown_s, clock
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def before_call(self) -> None:
        with self._lock:
            if self._opened_at is None:
                return
            if self.clock() - self._opened_at >= self.cooldown_s:
                self._opened_at = None  # half-open: let the next call probe
                self._failures = self.threshold - 1
                return
            raise CircuitOpen("LLM provider circuit breaker is open")

    def success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.threshold:
                self._opened_at = self.clock()


class LLMClient:
    def __init__(
        self,
        provider: Provider,
        settings: Settings,
        tracker: UsageTracker,
        *,
        breaker: CircuitBreaker | None = None,
        semaphore: threading.BoundedSemaphore | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ):
        self.provider, self.settings, self.tracker = provider, settings, tracker
        self.breaker = breaker or CircuitBreaker(settings.llm_breaker_threshold, settings.llm_breaker_cooldown_s)
        self.semaphore = semaphore or threading.BoundedSemaphore(settings.llm_max_concurrency)
        self.sleep, self.rng = sleep, rng

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        s = self.settings
        computed = min(s.llm_backoff_max_s, s.llm_backoff_base_s * (2 ** attempt))
        delay = computed / 2 + self.rng() * computed / 2  # equal jitter
        return max(delay, retry_after or 0.0)

    def _complete(self, req: LLMRequest) -> LLMResponse:
        last: LLMError | None = None
        for attempt in range(self.settings.llm_max_retries + 1):
            self.breaker.before_call()
            self.tracker.check()
            retry_after = None
            try:
                with self.semaphore:
                    resp = self.provider.complete(req)
            except LLMRateLimited as exc:
                last, retry_after = exc, exc.retry_after
            except (LLMTimeout, LLMUnavailable) as exc:
                last = exc
            except LLMError:
                raise  # token limit / auth / malformed: retrying the same call will not help
            else:
                self.breaker.success()
                self.tracker.add(resp)
                return resp
            self.breaker.failure()
            if attempt == self.settings.llm_max_retries:
                break
            delay = self._backoff(attempt, retry_after)
            log.warning("llm call failed (%s), retrying in %.1fs", type(last).__name__, delay, extra={"ctx": {"attempt": attempt + 1}})
            self.sleep(delay)
        assert last is not None
        raise last

    def call(self, req: LLMRequest, model_cls: type[BaseModel], validator: Callable[[BaseModel], None] | None = None) -> BaseModel:
        task = req.task
        for repair in range(self.settings.llm_max_repairs + 1):
            resp = self._complete(replace(req, task=task))
            try:
                obj = model_cls.model_validate(resp.data)
                if validator is not None:
                    validator(obj)
                return obj
            except (ValidationError, ValueError) as exc:
                reason = _short_reason(exc)
                if repair == self.settings.llm_max_repairs:
                    raise LLMMalformedResponse(f"model output invalid after {repair} repair attempts: {reason}") from exc
                log.info("llm output rejected, requesting repair: %s", reason)
                task = f"{req.task}\n\nYour previous answer was rejected: {reason}\nReturn corrected JSON only."
        raise AssertionError("unreachable")


def _short_reason(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:4])
    return str(exc)[:300]
