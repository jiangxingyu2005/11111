"""身份解析测试：前端 token -> user_id（passthrough / jwt 两模式 + 无效 token 返回 None）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app import auth


class _FakeSettings:
    def __init__(self, mode: str = "passthrough", secret: str = ""):
        self.memory_auth_mode = mode
        self.memory_jwt_secret = secret


def test_passthrough_token_is_user_id(monkeypatch):
    monkeypatch.setattr(auth, "get_settings", lambda: _FakeSettings())
    assert auth.resolve_user_id("user-123") == "user-123"
    assert auth.resolve_user_id("  alice  ") == "alice"     # strip 空白


def test_missing_or_blank_token_returns_none(monkeypatch):
    monkeypatch.setattr(auth, "get_settings", lambda: _FakeSettings())
    assert auth.resolve_user_id(None) is None
    assert auth.resolve_user_id("") is None
    assert auth.resolve_user_id("   ") is None               # 全空白 = 无身份


def test_jwt_sub_is_user_id(monkeypatch):
    import jwt
    secret = "test-secret-0123456789-0123456789-0123456789"   # ≥32 字节，避开 InsecureKeyLengthWarning
    token = jwt.encode({"sub": "u-42"}, secret, algorithm="HS256")
    monkeypatch.setattr(auth, "get_settings", lambda: _FakeSettings("jwt", secret))
    assert auth.resolve_user_id(token) == "u-42"


def test_jwt_without_secret_returns_none(monkeypatch):
    """jwt 模式但未配密钥：不能验签，直接判无身份（fail-closed，不冒认）。"""
    monkeypatch.setattr(auth, "get_settings", lambda: _FakeSettings("jwt", ""))
    assert auth.resolve_user_id("any-token") is None


def test_invalid_jwt_returns_none(monkeypatch):
    """密钥不匹配 / 篡改 token：验签失败 -> None（不得伪造他人身份）。"""
    import jwt
    token = jwt.encode({"sub": "u-42"}, "other-secret-0123456789-0123456789-0123456789",
                       algorithm="HS256")
    monkeypatch.setattr(auth, "get_settings",
                        lambda: _FakeSettings("jwt", "right-secret-0123456789-0123456789-0123456789"))
    assert auth.resolve_user_id(token) is None
