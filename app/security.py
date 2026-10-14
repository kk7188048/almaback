import hashlib
import json
import secrets
import uuid
from datetime import timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from app.config import get_settings
from app.db import utcnow
from app.redis_client import get_redis

_hasher = PasswordHasher()
ALGORITHM = "HS256"


class TokenError(Exception):
    pass


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def create_access_token(user_id: str, role: str) -> str:
    s = get_settings()
    now = utcnow()
    payload = {
        "sub": user_id,
        "role": role,
        "type": "access",
        "jti": uuid.uuid4().hex,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=s.access_token_ttl_s)).timestamp()),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, get_settings().jwt_secret, algorithms=[ALGORITHM], options={"require": ["exp", "sub"]})
    except jwt.PyJWTError as exc:
        raise TokenError(str(exc)) from exc
    if payload.get("type") != "access":
        raise TokenError("wrong token type")
    return payload


def _rt_key(token: str) -> str:
    return "rt:" + hashlib.sha256(token.encode()).hexdigest()


def issue_refresh_token(user_id: str) -> str:
    """Opaque, server-side, revocable. Only the hash is stored."""
    token = secrets.token_urlsafe(48)
    get_redis().set(_rt_key(token), json.dumps({"user_id": user_id}), ex=get_settings().refresh_token_ttl_s)
    return token


def consume_refresh_token(token: str) -> str:
    """Rotation: a refresh token is single-use. Returns the user id."""
    raw = get_redis().getdel(_rt_key(token))
    if raw is None:
        raise TokenError("refresh token invalid or revoked")
    return json.loads(raw)["user_id"]


def revoke_refresh_token(token: str) -> None:
    get_redis().delete(_rt_key(token))
