"""Per-criterion retrieval over the embedded chunks of one submission (pgvector on Postgres, in-process otherwise)."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Embedding
from app.pipeline.context.embeddings import HashEmbedder, cosine


def store_chunks(db: Session, submission_id: str, job_id: str, chunks: list[tuple[str, str, list[float]]]) -> None:
    db.query(Embedding).filter(Embedding.submission_id == submission_id).delete()
    db.add_all([Embedding(submission_id=submission_id, job_id=job_id, chunk_ref=ref, text=text, vector=vec) for ref, text, vec in chunks])


def retrieve(db: Session, submission_id: str, query: str, settings: Settings, k: int | None = None) -> list[dict]:
    k = k or settings.retrieval_top_k
    qvec = HashEmbedder(settings.embedding_dim).embed(query)
    if db.get_bind().dialect.name == "postgresql" and settings.use_pgvector:
        rows = db.scalars(
            select(Embedding).where(Embedding.submission_id == submission_id).order_by(Embedding.vector.cosine_distance(qvec)).limit(k)
        ).all()
        return [{"ref": r.chunk_ref, "text": r.text[:1500]} for r in rows]
    rows = db.scalars(select(Embedding).where(Embedding.submission_id == submission_id)).all()
    scored = sorted(rows, key=lambda r: cosine(qvec, r.vector), reverse=True)[:k]
    return [{"ref": r.chunk_ref, "text": r.text[:1500]} for r in scored]
