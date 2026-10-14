"""LLM evaluation: understanding, three parallel analyses, then one structured scoring call per rubric criterion,
re-sampled when confidence is low. The final score is computed in aggregate.py, never by the model.
"""

import json
import logging
import math
import statistics
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from app.config import Settings
from app.pipeline.checkpoints import Checkpoints
from app.pipeline.context.pack import digest, render_context
from app.pipeline.llm import prompts
from app.pipeline.llm.base import LLMClient, LLMMalformedResponse, LLMRequest, LLMTokenLimit, SemanticError
from app.pipeline.llm.schemas import AnalysisOut, CriterionScoreOut, UnderstandingOut

log = logging.getLogger(__name__)
ANALYSIS_KINDS = ("architecture", "code_quality", "documentation")


def _evidence_base(ref: str) -> str:
    return ref.strip().removeprefix("./").split("#")[0].split(":")[0]


def make_score_validator(criterion: dict, known_paths: set[str]) -> Callable[[BaseModel], None]:
    def validate(out: BaseModel) -> None:
        assert isinstance(out, CriterionScoreOut)
        if math.isnan(out.score) or not (0 <= out.score <= criterion["max_score"]):
            raise SemanticError(f"score must be between 0 and {criterion['max_score']}, got {out.score}")
        if len(out.justification.strip()) < 20:
            raise SemanticError("justification must be at least a couple of sentences")
        if known_paths:
            valid = [e for e in out.evidence if _evidence_base(e) in known_paths]
            if not valid:
                raise SemanticError("evidence must cite file paths that appear in the context (e.g. src0_zip/app/main.py)")
            out.evidence = valid

    return validate


def combine_samples(samples: list[CriterionScoreOut], max_score: float, tolerance: float) -> dict:
    scores = [s.score for s in samples]
    median = statistics.median(scores)
    agreement = sum(1 for sc in scores if abs(sc - median) <= tolerance * max_score) / len(scores)
    representative = min(samples, key=lambda s: abs(s.score - median))
    mean_conf = sum(s.confidence for s in samples) / len(samples)
    return {
        "score": round(median, 2),
        "justification": representative.justification,
        "evidence": representative.evidence,
        "confidence": round(mean_conf * agreement, 3),
        "agreement": round(agreement, 3),
        "samples": [round(s, 2) for s in scores],
        "injection_suspected": any(s.injection_suspected for s in samples),
    }


class Evaluator:
    def __init__(
        self, llm: LLMClient, settings: Settings, ckpt: Checkpoints, boundary: str, pack: dict,
        criteria: list[dict], retrieved: dict[str, list[dict]], check_lease: Callable[[], None],
    ):
        self.llm, self.settings, self.ckpt, self.boundary, self.pack = llm, settings, ckpt, boundary, pack
        self.criteria, self.retrieved, self.check_lease = criteria, retrieved, check_lease
        self.known_paths = set(pack["manifest"]) | set(pack.get("assets", []))
        ranked = pack.get("ranked_files", [])
        readme = [pack["readme"]["path"]] if pack.get("readme") else []
        self.meta_base = {
            "known_paths": (readme + ranked + pack["manifest"])[:20],
            "file_count": pack["metadata"]["file_count"],
            "frameworks": pack["metadata"]["frameworks"],
            "project_type": ", ".join(pack["metadata"]["frameworks"]) or "software project",
            "has_tests": pack["signals"]["structure"]["has_tests"],
            "context_digest": digest(json.dumps(pack["repo_map"]) + json.dumps(pack["file_tree"]) + json.dumps(pack.get("readme"))),
        }

    def _call(self, build: Callable[[str], LLMRequest], model_cls: type[BaseModel], validator=None) -> BaseModel:
        budget = self.settings.context_max_tokens
        for _ in range(4):
            try:
                return self.llm.call(build(render_context(self.pack, budget)), model_cls, validator)
            except LLMTokenLimit:
                budget //= 2
                log.warning("context too large for the model, shrinking to %d tokens", budget)
        raise LLMTokenLimit("context still too large after shrinking")

    def understanding(self) -> dict:
        def compute():
            self.check_lease()
            return self._call(lambda ctx: prompts.understanding_request(self.boundary, ctx, self.meta_base), UnderstandingOut).model_dump()

        return self.ckpt.get_or_compute("understanding", compute)

    def analyses(self, understanding: dict) -> dict[str, dict]:
        def one(kind: str) -> dict:
            def compute():
                self.check_lease()
                return self._call(lambda ctx: prompts.analysis_request(self.boundary, kind, ctx, understanding, self.meta_base), AnalysisOut).model_dump()

            return self.ckpt.get_or_compute(f"analysis_{kind}", compute)

        with ThreadPoolExecutor(max_workers=len(ANALYSIS_KINDS)) as ex:
            futures = {k: ex.submit(one, k) for k in ANALYSIS_KINDS}
        return {k: f.result() for k, f in futures.items()}

    def _sample(self, criterion: dict, understanding: dict, analyses: dict, temperature: float) -> CriterionScoreOut:
        retrieved = self.retrieved.get(criterion["id"], [])
        validator = make_score_validator(criterion, self.known_paths)
        meta = {**self.meta_base}
        return self._call(
            lambda ctx: prompts.criterion_request(self.boundary, ctx, criterion, understanding, analyses, retrieved, meta, temperature),
            CriterionScoreOut,
            validator,
        )

    def _score_criterion(self, criterion: dict, understanding: dict, analyses: dict) -> dict:
        self.check_lease()
        s = self.settings
        samples = [self._sample(criterion, understanding, analyses, 0.1)]
        first = samples[0]
        ratio = first.score / criterion["max_score"]
        if first.confidence < s.sc_confidence_threshold or s.sc_borderline_low <= ratio <= s.sc_borderline_high:
            for _ in range(s.sc_extra_samples):
                samples.append(self._sample(criterion, understanding, analyses, 0.4))
        out = combine_samples(samples, criterion["max_score"], s.sc_agreement_tolerance)
        return {"criterion_id": criterion["id"], "name": criterion["name"], "max_score": criterion["max_score"], "weight": criterion["weight"], "status": "ok", **out}

    def score_criteria(self, understanding: dict, analyses: dict) -> list[dict]:
        def one(criterion: dict) -> dict:
            return self.ckpt.get_or_compute(f"criterion_{criterion['id']}", lambda: self._score_criterion(criterion, understanding, analyses))

        with ThreadPoolExecutor(max_workers=max(1, min(len(self.criteria), self.settings.llm_max_concurrency))) as ex:
            futures = [(c, ex.submit(one, c)) for c in self.criteria]
        results: list[dict] = []
        hard_error: Exception | None = None
        for criterion, future in futures:
            try:
                results.append(future.result())
            except LLMMalformedResponse as exc:
                log.warning("criterion %s failed validation permanently: %s", criterion["name"], exc)
                results.append({
                    "criterion_id": criterion["id"], "name": criterion["name"], "max_score": criterion["max_score"],
                    "weight": criterion["weight"], "status": "failed", "score": None, "confidence": 0.0, "agreement": 0.0,
                    "justification": "", "evidence": [], "samples": [], "injection_suspected": False,
                    "error": "the model could not produce a valid score for this criterion",
                })
            except Exception as exc:  # LeaseLost, rate limits, outages: surface after siblings checkpointed their work
                hard_error = hard_error or exc
        if hard_error is not None:
            raise hard_error
        return results
