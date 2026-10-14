from pydantic import BaseModel, Field


class FileSummary(BaseModel):
    path: str
    summary: str


class FileSummariesOut(BaseModel):
    items: list[FileSummary]


class DirSummaryOut(BaseModel):
    summary: str


class UnderstandingOut(BaseModel):
    project_type: str
    purpose: str
    tech_stack: list[str]
    main_features: list[str]
    summary: str


class AnalysisOut(BaseModel):
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    notable_files: list[str]


class CriterionScoreOut(BaseModel):
    score: float
    justification: str
    evidence: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    injection_suspected: bool = False


def inline_refs(schema: dict) -> dict:
    """Resolve $ref/$defs so providers with a restricted JSON-schema dialect accept the schema."""
    defs = schema.get("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def schema_for(model_cls: type[BaseModel]) -> dict:
    return inline_refs(model_cls.model_json_schema())
