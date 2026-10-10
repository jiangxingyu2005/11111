"""Nacos 配置中心轻量客户端（httpx，零新依赖）。

只依赖 Nacos HTTP OpenAPI 三个端点：
  - GET  {base}/nacos/v1/cs/configs?dataId=..&group=..&tenant=..   读配置
  - POST {base}/nacos/v1/cs/configs/listener                        长轮询监听（最长 30s）
  - POST {base}/nacos/v1/cs/configs                                  发布（仅播种脚本用）

线程模型：调用方自起 daemon 线程跑 watch()；listen 阻塞最长 ~30s 只在子线程里，
不阻塞 asyncio 事件循环。不引 nacos-sdk-python（同步、维护弱、长轮询/tenant 边角坑）。

变更检测策略：listen_once 只负责给「有变更」信号；watch 收到信号后重新拉全量比对
md5，md5 变了才回调 on_change —— 不依赖 Nacos 各版本间有差异的变更体格式。
"""
from __future__ import annotations

import hashlib
import logging
import threading

log = logging.getLogger("common.nacos")

SEP_FIELD = "\x02"
SEP_ITEM = "\x01"


class NacosError(RuntimeError):
    pass


class NacosConfigClient:
    def __init__(self, server_addr: str, namespace: str = "",
                 group: str = "DEFAULT_GROUP", username: str = "",
                 password: str = "", timeout: float = 10.0,
                 listen_timeout_ms: int = 30000):
        self.base = server_addr.rstrip("/")
        self.namespace = namespace
        self.group = group
        self.username = username
        self.password = password
        self.timeout = timeout
        self.listen_timeout_ms = listen_timeout_ms
        self._md5s: dict[str, str] = {}

    # ---- 读配置 ----
    def get_config(self, data_id: str, group: str | None = None) -> str:
        import httpx
        params = {"dataId": data_id, "group": group or self.group}
        if self.namespace:
            params["tenant"] = self.namespace
        with httpx.Client(timeout=self.timeout) as cli:
            r = cli.get(f"{self.base}/nacos/v1/cs/configs", params=params)
        if r.status_code == 200:
            return r.text
        raise NacosError(f"get_config {data_id} -> HTTP {r.status_code}: {r.text[:200]}")

    # ---- 发布（播种脚本用） ----
    def publish(self, data_id: str, content: str, group: str | None = None,
                type_: str = "properties") -> bool:
        import httpx
        data = {"dataId": data_id, "group": group or self.group,
                "content": content, "type": type_}
        if self.namespace:
            data["tenant"] = self.namespace
        with httpx.Client(timeout=self.timeout) as cli:
            r = cli.post(f"{self.base}/nacos/v1/cs/configs", data=data)
        if r.status_code == 200:
            log.info("published dataId=%s group=%s", data_id, group or self.group)
            return True
        raise NacosError(f"publish {data_id} -> HTTP {r.status_code}: {r.text[:200]}")

    # ---- 长轮询监听 ----
    def _listening_header(self) -> str:
        parts = []
        for data_id, md5 in self._md5s.items():
            parts.append(f"{data_id}{SEP_FIELD}{self.group}{SEP_FIELD}{md5}{SEP_FIELD}{self.namespace}")
        return SEP_ITEM.join(parts) + (SEP_ITEM if parts else "")

    def listen_once(self) -> list[str]:
        """单次长轮询。返回非空列表 = 有变更信号；空 = 无变更。

        Nacos 契约：`Listening-Configs` 是 query/form 参数（非 header），`Long-Pulling-Timeout`
        才走 header；变更时响应体为 URL 编码的 `dataId\\x02group(\\x01dataId\\x02group)*`。
        """
        import httpx
        from urllib.parse import unquote
        headers = {"Long-Pulling-Timeout": str(self.listen_timeout_ms)}
        params = {"Listening-Configs": self._listening_header()}
        read_timeout = self.listen_timeout_ms / 1000.0 + 5.0
        with httpx.Client(timeout=(self.timeout, read_timeout)) as cli:
            r = cli.post(f"{self.base}/nacos/v1/cs/configs/listener",
                         params=params, headers=headers)
        if r.status_code != 200:
            raise NacosError(f"listen -> HTTP {r.status_code}: {r.text[:200]}")
        body = unquote((r.text or "").strip())  # Nacos 对变更 key 做了 URL 编码
        if not body:
            return []
        return [item.split(SEP_FIELD)[0] for item in body.split(SEP_ITEM) if item]

    # ---- 监听循环（丢进 daemon 线程） ----
    def watch(self, data_ids: list[str], on_change, stop_event: threading.Event) -> None:
        # 初始 md5 置空：首次 listen 即收到「全量变更」，_sync 拉真实 md5 并幂等重放一次
        self._md5s = {d: "" for d in data_ids}
        while not stop_event.is_set():
            try:
                changed = self.listen_once()
            except Exception as e:
                log.warning("nacos listen error: %s", e)
                stop_event.wait(1.0)
                continue
            if changed:
                self._sync(data_ids, on_change)

    def _sync(self, data_ids: list[str], on_change) -> None:
        for data_id in data_ids:
            try:
                content = self.get_config(data_id)
            except Exception as e:
                log.warning("nacos sync get_config %s failed: %s", data_id, e)
                continue
            new_md5 = hashlib.md5(content.encode("utf-8")).hexdigest()
            if new_md5 != self._md5s.get(data_id):
                self._md5s[data_id] = new_md5
                try:
                    on_change(data_id, content)
                except Exception as e:
                    log.warning("nacos on_change %s failed: %s", data_id, e)


def parse_properties(content: str) -> dict[str, str]:
    """解析 properties 文本：去空行/整行注释，按首个 '=' 拆 k/v。

    不剥行内 '#' —— 值可能含 '#'（口令/URL fragment）；行内注释在播种时剥。
    """
    out: dict[str, str] = {}
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if not k:
            continue
        out[k] = v.strip()
    return out
