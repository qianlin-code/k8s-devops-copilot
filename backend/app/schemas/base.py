from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, field_serializer


class StrictBaseModel(BaseModel):
    """全部 Request/Response 模型的基类。

    extra="forbid" 让未声明字段直接报错，避免前后端字段漂移悄悄通过。
    """

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class UTCTimestampMixin(BaseModel):
    """把 datetime 统一序列化成 UTC ISO 字符串，前端不用猜时区。"""

    @field_serializer("*", when_used="json", check_fields=False)
    def _serialize_datetimes(self, value: object) -> object:
        if isinstance(value, datetime):
            aware = (
                value.replace(tzinfo=timezone.utc)
                if value.tzinfo is None
                else value.astimezone(timezone.utc)
            )
            return aware.isoformat().replace("+00:00", "Z")
        return value


def to_utc_iso(value: datetime) -> str:
    aware = (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )
    return aware.isoformat().replace("+00:00", "Z")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
