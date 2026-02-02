from datetime import datetime
import zoneinfo

def to_jst_iso(dt: datetime = None) -> str:
    """
    UTCのdatetimeを日本時間(JST)のISOフォーマット文字列に変換する。
    dtがNoneの場合は現在の日本時間を返す。
    """
    jst = zoneinfo.ZoneInfo("Asia/Tokyo")
    if dt is None:
        dt = datetime.now(jst)
    elif dt.tzinfo is None:
        # タイムゾーンがない場合はUTCとみなしてJSTに変換
        dt = dt.replace(tzinfo=zoneinfo.ZoneInfo("UTC")).astimezone(jst)
    else:
        dt = dt.astimezone(jst)
        
    return dt.isoformat()