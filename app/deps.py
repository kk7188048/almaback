import re

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import User
from app.ratelimit import take_token
from app.security import TokenError, decode_access_token

_bearer = HTTPBearer(auto_error=False)
_IDEMPOTENCY = re.compile(r"^[A-Za-z0-9_\-:.]{8,100}$")


def _unauthorized() -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, "not authenticated", headers={"WWW-Authenticate": "Bearer"})


def get_current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer), db: Session = Depends(get_db)) -> User:
    if creds is None:
        raise _unauthorized()
    try:
        claims = decode_access_token(creds.credentials)
    except TokenError:
        raise _unauthorized() from None
    user = db.get(User, claims["sub"])
    if user is None or not user.is_active:
        raise _unauthorized()
    return user


def require_roles(*roles: str):
    def dep(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "insufficient permissions")
        return user

    return dep


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    """Token bucket per authenticated user, else per client IP."""
    s = get_settings()
    ident = "ip:" + _client_ip(request)
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        try:
            ident = "u:" + decode_access_token(auth[7:])["sub"]
        except TokenError:
            pass
    ok, retry_after = take_token(ident, s.rate_limit_capacity, s.rate_limit_per_s)
    if not ok:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded", headers={"Retry-After": str(int(retry_after) + 1)})


def login_rate_limit(request: Request) -> None:
    s = get_settings()
    ok, retry_after = take_token("login:" + _client_ip(request), s.login_rate_capacity, s.login_rate_per_s)
    if not ok:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "too many login attempts", headers={"Retry-After": str(int(retry_after) + 1)})


def idempotency_key(key: str | None = Header(default=None, alias="Idempotency-Key")) -> str:
    if not key or not _IDEMPOTENCY.match(key):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Idempotency-Key header is required (8-100 chars: letters, digits, _ - : .)")
    return key
