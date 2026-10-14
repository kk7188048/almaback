"""Heuristic prompt-injection detection over student content. A hit never changes a score by itself;
it routes the evaluation to a human (the grader is also tool-less and data-marked, see llm/prompts.py)."""

import re

_PATTERNS = [
    re.compile(r"(?i)ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)"),
    re.compile(r"(?i)disregard\s+(?:the\s+|all\s+|any\s+)?(?:rubric|instructions?|grading|criteria)"),
    re.compile(r"(?i)(?:give|award|assign|grant)\s+(?:me\s+|this\s+(?:project|submission)\s+)?(?:full|maximum|max|perfect|top)\s+(?:marks?|score|points?|grade)"),
    re.compile(r"(?i)you\s+are\s+now\s+(?:a|an|the|in)\b"),
    re.compile(r"(?i)(?:reveal|print|show|output)\s+(?:your\s+|the\s+)?(?:system\s+prompt|hidden\s+instructions?)"),
    re.compile(r"(?i)\b(?:score|grade)\s*(?:of|=|:)\s*(?:100|10/10|full)\b.{0,40}\b(?:evaluator|grader|llm|ai)\b"),
    re.compile(r"(?i)note\s+to\s+(?:the\s+)?(?:ai|llm|grader|evaluator|model)\b"),
    re.compile(r"(?i)<\s*/?\s*(?:system|assistant)\s*>"),
]


def find_injection(text: str) -> str | None:
    """Returns the matched snippet (truncated) or None."""
    for pattern in _PATTERNS:
        m = pattern.search(text)
        if m:
            return text[max(0, m.start() - 20): m.end() + 20].replace("\n", " ")[:160]
    return None
