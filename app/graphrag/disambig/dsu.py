"""并查集:全局消歧求 canonical 连通分量(别名/指纹/向量命中两两 union)。"""

from __future__ import annotations


class DSU:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self._parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # 路径压缩
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for x in self._parent:
            groups.setdefault(self.find(x), []).append(x)
        return groups
