"""身份解析：前端 token -> user_id（记忆隔离的隔离键来源）。

两种模式（MEMORY_AUTH_MODE）：
  - passthrough（默认）：token 即 user_id（内网/测试；等价于调用方自报身份）
  - jwt：PyJWT 验签 HS256，sub 即 user_id（真正的令牌，防伪造他人身份）

fail-closed：token 缺失 / 解析失败 -> 返回 None，调用方据此跳过记忆（主链照跑，不留 degraded）。
"""
from __future__ import annotations

import logging

from .config import get_settings

log = logging.getLogger("retrieval.auth")


def resolve_user_id(token: str | None) -> str | None:
    """把前端 token 解析成 user_id；无法解析返回 None。"""
    if not token:
        return None
    s = get_settings()
    try:
        if s.memory_auth_mode == "jwt":
            return _resolve_jwt(token.strip(), s.memory_jwt_secret)
        # passthrough：token 即 user_id
        return token.strip() or None
    except Exception as e:  # fail-closed：任何解析异常都不阻断主链
        log.warning("resolve_user_id failed (记忆本轮跳过): %s", e)
        return None


def _resolve_jwt(token: str, secret: str) -> str | None:
    """验签 JWT，取 sub 作为 user_id；无 secret / 验签失败返回 None。"""
    if not secret:
        log.warning("MEMORY_AUTH_MODE=jwt 但未配置 MEMORY_JWT_SECRET，无法解析身份")
        return None
    import jwt  # PyJWT，仅 jwt 模式才 import（passthrough 无此依赖）

    payload = jwt.decode(token, secret, algorithms=["HS256"])
    sub = payload.get("sub")
    return str(sub) if sub else None
