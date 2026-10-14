from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.deps import get_current_user, login_rate_limit
from app.models import User
from app.schemas import LoginIn, RefreshIn, UserOut
from app.security import (
    TokenError, consume_refresh_token, create_access_token, hash_password, issue_refresh_token, revoke_refresh_token,
    verify_password,
)

router = APIRouter(prefix="/api/v1", tags=["auth"])
_DUMMY_HASH = hash_password("not-a-real-password")  # equalises timing for unknown emails


def _tokens(user: User) -> dict:
    return {
        "access_token": create_access_token(user.id, user.role),
        "token_type": "bearer",
        "expires_in": get_settings().access_token_ttl_s,
        "refresh_token": issue_refresh_token(user.id),
    }


@router.post("/auth/login", dependencies=[Depends(login_rate_limit)])
def login(body: LoginIn, db: Session = Depends(get_db)) -> dict:
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    ok = verify_password(user.password_hash if user else _DUMMY_HASH, body.password)
    if user is None or not ok or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
    return _tokens(user)


@router.post("/auth/refresh", dependencies=[Depends(login_rate_limit)])
def refresh(body: RefreshIn, db: Session = Depends(get_db)) -> dict:
    try:
        user_id = consume_refresh_token(body.refresh_token)
    except TokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid refresh token") from None
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid refresh token")
    return _tokens(user)


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(body: RefreshIn, _user: User = Depends(get_current_user)) -> Response:
    revoke_refresh_token(body.refresh_token)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> User:
    return user
