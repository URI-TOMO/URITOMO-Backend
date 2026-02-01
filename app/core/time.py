from datetime import datetime, timedelta, timezone
from typing import Optional

JST = timezone(timedelta(hours=9))


def to_jst_iso(value: Optional[datetime]) -> Optional[str]:
    if not value:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(JST).isoformat()
