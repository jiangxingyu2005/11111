"""内存三库（fake 后端）——dev/契约测试用，零外部依赖跑通全链路。

设计纪律（与项目降级思想一致，禁止"两套代码路径"）：
  - fake 与 real 共用【同一份接口签名】，route 层只认接口不认实现
  - fake 数据来自 scripts/gen_seed.py 产出的 data/seed/*.json，与真库同构
  - 故障注入：RETRIEVAL_FAULT_INJECT=vector,graph,pg,llm,rerank,bm25,community
    -> 对应方法抛异常，用于真跑降级矩阵（G-QC-03 要求"行为+留痕"双验）
  - 生产禁用：RETRIEVAL_BACKEND 必须为 real，启动时校验（见 config.py）
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from .hash_embed import cosine, lex_overlap

log = logging.getLogger("retrieval.store.fake")

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SEED_DIR = ROOT / "data" / "seed"


class FaultInjected(RuntimeError):
    """故障注入专用异常（测试用，生产不会出现）。"""


class FakeStore:
    """内存版 PG(parents) + Milvus(chunks/entities) + Neo4j(edges)。"""

    def __init__(self, seed_dir: Path | str = DEFAULT_SEED_DIR, faults: Iterable[str] = ()) -> None:
        d = Path(seed_dir)
        self.faults = {f.strip() for f in faults if f and f.strip()}
        self.chunks: dict[str, dict] = {}
        self.parents: dict[str, dict] = {}
        self.entities: dict[str, dict] = {}
        self.edges: list[dict] = []
        self.communities: dict[str, dict] = {}     # community_id -> {scene_vector, summary, member_entity_ids, ...}
        self._adj: dict[str, list[tuple[str, str, float]]] = {}
        self._binds: dict[str, list[str]] = {}      # entity_id -> [child_id]
        self._load(d)
        log.info("fake store loaded: chunks=%d parents=%d entities=%d edges=%d communities=%d faults=%s",
                 len(self.chunks), len(self.parents), len(self.entities),
                 len(self.edges), len(self.communities), sorted(self.faults) or "-")

    # ─────────────────────────── 装载 ───────────────────────────
    def _load(self, d: Path) -> None:
        self.parents = {r["parent_id"]: r for r in self._read(d / "parents.json")}
        self.chunks = {r["child_id"]: r for r in self._read(d / "chunks.json")}
        # JSON 往返把 sparse 的 int 键转成了字符串；归一化回 int，与 hash_sparse 产出的
        # {token_id: weight} 对齐（否则 lex_overlap 的 int/str 键对不上，稀疏打分恒为 0）
        for c in self.chunks.values():
            sp = c.get("sparse")
            if isinstance(sp, dict):
                c["sparse"] = {int(k): float(v) for k, v in sp.items()}
        self.entities = {r["entity_id"]: r for r in self._read(d / "entities.json")}
        self.edges = self._read(d / "edges.json")
        # 社区数据（③-B 路）：旧 seed 可能还没有此文件，graceful 降为空（社区路按无起点处理）
        cp = d / "communities.json"
        if cp.exists():
            self.communities = {r["community_id"]: r for r in self._read(cp)}
        for e in self.edges:
            if e.get("rel") == "BINDS":
                # 段落绑定边单独存放，不进实体邻接表（图谱页只画实体图）
                self._binds.setdefault(e["src"], []).append(e["dst"])
                continue
            self._adj.setdefault(e["src"], []).append((e["dst"], e.get("rel", "RELATED"), float(e.get("weight", 1.0))))
            self._adj.setdefault(e["dst"], []).append((e["src"], e.get("rel", "RELATED"), float(e.get("weight", 1.0))))

    @staticmethod
    def _read(p: Path) -> list[dict]:
        if not p.exists():
            raise FileNotFoundError(
                f"seed 数据缺失: {p} —— 请先运行 `python scripts/gen_seed.py`（fake 后端的唯一数据来源）"
            )
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    # ─────────────────────────── Milvus 侧 ───────────────────────────
    def _check(self, key: str) -> None:
        if key in self.faults:
            raise FaultInjected(f"fault injected: {key}")

    def _scope_ok(self, row: dict, scope: list[str] | None) -> bool:
        if not scope:
            return True
        return row.get("kb_id") in set(scope)

    def search_chunks(
        self,
        routes: dict[str, dict],
        *,
        top_k: int = 50,
        scope: list[str] | None = None,
    ) -> list[dict]:
        """返回 [{"id","distance","entity":{...}}]，形状对齐 pymilvus MilvusClient.search。

        routes: {"dense": {"vector": [...], "tau": τ_a},
                 "dense_v2": {"vector": [...], "tau": τ_b},
                 "sparse": {"vector": {token_id: weight}, "tau": 0.0}}
        单路 -> 单路 top_k（卡各自 τ）；多路 -> 内层 N 路 RRF（与 Milvus RRFRanker 语义一致）。
        """
        self._check("vector")
        chunks = [c for c in self.chunks.values() if self._scope_ok(c, scope)]

        if len(routes) == 1:
            name, r = next(iter(routes.items()))
            tau = float(r.get("tau", 0.0))
            scored = [(self._route_score(name, r["vector"], c), c) for c in chunks]
            scored = [(s, c) for s, c in scored if s >= tau]
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {"id": c["child_id"], "distance": s, "entity": self._row_entity(c)}
                for s, c in scored[:top_k]
            ]

        scored = [
            ({name: self._route_score(name, r["vector"], c) for name, r in routes.items()}, c)
            for c in chunks
        ]
        return self._rrf_rows(scored, tuple(routes), top_k)

    @staticmethod
    def _route_score(name: str, vector: object, chunk: dict) -> float:
        """按子路名称对 chunk 打分。sparse 走 lex_overlap，其余走余弦。"""
        if name == "sparse":
            return lex_overlap(vector, chunk.get("sparse") or {})
        if name == "dense_v2":
            return cosine(vector, chunk.get("embedding_v2") or [])
        return cosine(vector, chunk.get("embedding") or [])

    @staticmethod
    def _row_entity(c: dict) -> dict:
        """行内实体：剔除向量列，保留 PG 熔断兜底所需字段。"""
        return {k: v for k, v in c.items()
                if k not in ("embedding", "embedding_v2", "sparse")}

    @staticmethod
    def _rrf_rows(scored: list[tuple[dict, dict]], route_names: tuple[str, ...],
                  top_k: int, k: int = 60) -> list[dict]:
        """内层 N 路 RRF：每个子路排名序列各投一票，Σ 1/(k+rank)。"""
        ranks = {
            name: {c["child_id"]: i + 1 for i, (_, c) in enumerate(
                sorted(scored, key=lambda x: x[0][name], reverse=True))}
            for name in route_names
        }
        agg: dict[str, dict] = {}
        for scores, c in scored:
            r = sum(1.0 / (k + ranks[name][c["child_id"]]) for name in route_names)
            cur = agg.get(c["child_id"])
            if cur is None or r > cur["score"]:
                agg[c["child_id"]] = {
                    "score": r,
                    "row": {"id": c["child_id"], "distance": round(r, 6),
                            "entity": FakeStore._row_entity(c)},
                }
        out = sorted(agg.values(), key=lambda x: x["score"], reverse=True)
        return [x["row"] for x in out[:top_k]]

    def knn_entities(self, dense: list[float], *, top_k: int = 10,
                     scope: list[str] | None = None, tau: float = 0.0) -> list[dict]:
        self._check("vector")
        rows = []
        for e in self.entities.values():
            if not self._scope_ok(e, scope):
                continue
            s = cosine(dense, e.get("embedding", []))
            if s < tau:
                continue
            rows.append((s, e))
        rows.sort(key=lambda x: x[0], reverse=True)
        return [{"id": e["entity_id"], "distance": s, "entity": e} for s, e in rows[:top_k]]

    def bm25_search(self, query: str, *, top_k: int = 100,
                    scope: list[str] | None = None) -> list[dict]:
        """② 路外置 BM25 的 fake 等价：稀疏内积（lex_overlap）对内层 chunk 排序。

        语义对齐真实 BM25 的 lexical 排序效果；仅作链路验证，不评估召回率（M-D5）。
        """
        self._check("bm25")
        from .hash_embed import hash_sparse
        qs = hash_sparse(query)
        scored = []
        for c in self.chunks.values():
            if not self._scope_ok(c, scope):
                continue
            ls = lex_overlap(qs, c.get("sparse") or {})
            if ls <= 0.0:
                continue
            scored.append((ls, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"child_id": c["child_id"], "parent_id": c.get("parent_id", ""),
                 "score": round(ls, 6), "content": c.get("content", ""),
                 "kb_id": c.get("kb_id", ""), "kg_entity_ids": c.get("kg_entity_ids", []),
                 "span": c.get("span")}
                for ls, c in scored[:top_k]]

    def community_knn(self, dense: list[float], *, top_k: int = 5,
                      scope: list[str] | None = None, tau: float = 0.0) -> list[dict]:
        """③-B Scene KNN：query 向量 -> community_scene 向量最近邻（对齐 real Milvus）。"""
        self._check("community")
        rows = []
        for c in self.communities.values():
            if not self._scope_ok(c, scope):
                continue
            s = cosine(dense, c.get("scene_vector", []))
            if s < tau:
                continue
            rows.append((s, c))
        rows.sort(key=lambda x: x[0], reverse=True)
        return [{"id": c["community_id"], "distance": s, "entity": c} for s, c in rows[:top_k]]

    # ─────────────────────────── Neo4j 侧 ───────────────────────────
    def expand(self, entity_ids: list[str], scope: list[str] | None = None,
               *, hops: int = 2, max_degree: int = 50, limit: int = 50) -> list[dict]:
        """1~N 跳邻域 -> BINDS 回捞子块（对齐 graph_route._expand_and_backfill 的 Cypher 语义）。"""
        self._check("graph")
        seen = set(entity_ids)
        frontier = list(entity_ids)
        for _ in range(max(1, hops)):
            nxt: list[str] = []
            for eid in frontier:
                for dst, _rel, _w in self._adj.get(eid, []):
                    if dst not in seen:
                        seen.add(dst)
                        nxt.append(dst)
            frontier = nxt
            if not frontier:
                break
        # 度数截断（高度数实体按 confidence 截断，防爆炸）
        kept = [e for e in seen if len(self._adj.get(e, [])) <= max_degree]
        out: dict[str, dict] = {}
        for eid in kept:
            for cid in self._binds.get(eid, []):
                c = self.chunks.get(cid)
                if not c or not self._scope_ok(c, scope):
                    continue
                out[cid] = {"child_id": cid, "parent_id": c.get("parent_id", ""),
                            "content": c.get("content", ""), "kb_id": c.get("kb_id", ""),
                            "kg_entity_ids": c.get("kg_entity_ids", []),
                            "span": c.get("span"),
                            "via_entity": eid}
                if len(out) >= limit:
                    return list(out.values())
        return list(out.values())

    def subgraph(self, focus_entities: list[str], scope: list[str] | None = None,
                 *, hops: int = 1, node_limit: int = 50, edge_limit: int = 200,
                 relation_types: list[str] | None = None) -> tuple[list[dict], list[dict], bool]:
        """返回 (nodes, edges, pruned)。节点带 hops_from_focus，边带 evidence（G-API-02）。"""
        self._check("graph")
        hop_of = {e: 0 for e in focus_entities}
        frontier = list(focus_entities)
        for d in range(1, max(1, hops) + 1):
            nxt = []
            for eid in frontier:
                for dst, _rel, _w in self._adj.get(eid, []):
                    if dst not in hop_of:
                        hop_of[dst] = d
                        nxt.append(dst)
            frontier = nxt
            if not frontier:
                break
        pruned = len(hop_of) > node_limit
        keep_eids = list(hop_of)[:node_limit]

        nodes = []
        for eid in keep_eids:
            e = self.entities.get(eid, {"entity_id": eid, "name": eid, "type": "Unknown", "kb_id": ""})
            if not self._scope_ok(e, scope):
                continue
            ev = None
            for cid in self._binds.get(eid, [])[:1]:
                c = self.chunks.get(cid)
                if c:
                    ev = {"child_id": cid, "span": c.get("span"), "parent_id": c.get("parent_id")}
            nodes.append({"id": eid, "name": e.get("name", eid),
                          "type": e.get("type", "Unknown"),
                          "kb_id": e.get("kb_id", ""),
                          "degree": len(self._adj.get(eid, [])),
                          "hops_from_focus": hop_of[eid],
                          "is_focus": eid in focus_entities,
                          "evidence": ev})
        keep = {n["id"] for n in nodes}
        edges = []
        for e in self.edges:
            if e["src"] in keep and e["dst"] in keep:
                if relation_types and e.get("rel") not in relation_types:
                    continue
                ev = e.get("evidence")
                if not ev:   # 兜底：无证据的边不得渲染为可点击
                    cid = e.get("dst") if e.get("rel") == "BINDS" else None
                    c = self.chunks.get(cid) if cid else None
                    ev = ({"child_id": cid, "span": c.get("span"), "parent_id": c.get("parent_id")}
                          if c else {"child_id": "", "span": [0, 0], "parent_id": None})
                edges.append({"source": e["src"], "target": e["dst"],
                              "rel": e.get("rel", "RELATED"),
                              "confidence": min(1.0, float(e.get("weight", 1.0))),
                              "evidence": ev})
            if len(edges) >= edge_limit:
                pruned = True
                break
        return nodes, edges, pruned

    # ─────────────────────────── PG 侧 ───────────────────────────
    def parents_batch(self, parent_ids: list[str]) -> dict[str, dict]:
        self._check("pg")
        return {
            pid: {"parent_id": pid,
                  "content": p.get("content", ""),
                  "content_version": p.get("content_version", 1),
                  "summary": p.get("summary", ""),
                  "keywords": p.get("keywords", []),
                  "kg_confidence": p.get("kg_confidence")}
            for pid, p in ((pid, self.parents.get(pid)) for pid in parent_ids) if p
        }


def guard(key: str) -> None:
    """非 FakeStore 组件（rerank / generate / bm25）的故障注入检查。

    与 FakeStore._check 同一套语义：配了就抛，抛了就必须有人记 degraded。
    """
    from ..config import get_settings
    s = get_settings()
    if s.retrieval_backend == "fake" and key in (s.retrieval_fault_inject or ""):
        raise FaultInjected(f"fault injected: {key}")


_store: FakeStore | None = None


def get_fake_store() -> FakeStore:
    """进程级单例（懒加载；seed 缺失时抛 FileNotFoundError 并给出修复命令）。"""
    global _store
    if _store is None:
        from ..config import get_settings
        s = get_settings()
        _store = FakeStore(s.retrieval_seed_dir, (s.retrieval_fault_inject or "").split(","))
    return _store


def reset_fake_store() -> None:
    """测试用：清空单例（改了 env / seed 后重新装载）。"""
    global _store
    _store = None
