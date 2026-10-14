"""Deterministic aggregation. The LLM proposes per-criterion scores; the final number is computed here, from
weights stored in the database, so it is reproducible and auditable."""

from app.config import Settings


def aggregate(results: list[dict]) -> tuple[float | None, float | None]:
    """results: dicts with score|None, max_score, weight, confidence. Returns (overall 0-100, confidence 0-1)."""
    if not results or any(r.get("score") is None for r in results):
        return None, None
    total_w = sum(r["weight"] for r in results)
    overall = sum(r["weight"] / total_w * (r["score"] / r["max_score"]) for r in results) * 100
    confidence = sum(r["weight"] / total_w * r.get("confidence", 0.0) for r in results)
    return round(overall, 2), round(confidence, 3)


def review_reasons(results: list[dict], overall: float | None, confidence: float | None, pack: dict, settings: Settings) -> list[str]:
    reasons: list[str] = []
    for r in results:
        if r.get("score") is None:
            reasons.append(f"criterion_failed:{r['name']}")
        elif r.get("agreement", 1.0) < settings.review_agreement_threshold:
            reasons.append(f"low_agreement:{r['name']}")
    if confidence is not None and confidence < settings.review_confidence_threshold:
        reasons.append("low_confidence")
    if overall is not None and (overall <= 10 or overall >= 98):
        reasons.append("extreme_score")
    if pack.get("injection_flags") or any(r.get("injection_suspected") for r in results):
        reasons.append("possible_prompt_injection")
    if any(s.get("status") == "failed" for s in pack.get("sources", [])):
        reasons.append("partial_submission")
    return reasons
