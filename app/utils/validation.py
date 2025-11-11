import re
from fastapi import HTTPException, status

from app.core.config import settings


_user_id_regex = re.compile(settings.USER_ID_PATTERN)


def validate_user_id(user_id: str) -> str:
    """Validate inbound user identifiers against policy constraints."""
    if not user_id:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="user_id is required")

    if len(user_id) > settings.USER_ID_MAX_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"user_id exceeds maximum length of {settings.USER_ID_MAX_LENGTH} characters",
        )

    if not _user_id_regex.match(user_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="user_id must contain only letters, numbers, underscores, or hyphens",
        )

    return user_id
