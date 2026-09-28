"""M-Q4 RRF 纯算法单测（绿档，无外部依赖）。"""
from app.fuse.rrf import Hit, assert_one_source_one_vote, merge_within_source, rrf_fuse


def _h(cid: str, score: float, src: str) -> Hit:
    return Hit(child_id=cid, rank_score=score, source=src)


def test_rrf_one_source_one_vote():
    vec = [_h("a", 0.9, "vector_hybrid"), _h("b", 0.8, "vector_hybrid")]
    graph = [_h("b", 1.0, "graph_local"), _h("c", 0.9, "graph_local")]
    fused = rrf_fuse({"vector_hybrid": vec, "graph_local": graph}, k=60)
    # b 两路命中应排第一（两票），a/c 各一票
    assert fused[0].child_id == "b"
    assert set(fused[0].meta["voted_by"]) == {"vector_hybrid", "graph_local"}


def test_rrf_route_stripped_keeps_order():
    vec = [_h("a", 0.9, "vector_hybrid"), _h("b", 0.8, "vector_hybrid")]
    full = rrf_fuse({"vector_hybrid": vec, "graph_local": [_h("x", 1.0, "graph_local")]}, k=60)
    stripped = rrf_fuse({"vector_hybrid": vec}, k=60)  # ③ 路被剥离
    va = [h.child_id for h in full if h.source == "vector_hybrid"]
    sa = [h.child_id for h in stripped]
    assert va == sa  # 单路剥离不影响他路相对序


def test_merge_within_source_dedup():
    hits = [_h("a", 0.7, "vector_hybrid"), _h("a", 0.9, "vector_hybrid")]
    merged = merge_within_source(hits)
    assert len(merged) == 1 and merged[0].rank_score == 0.9
    assert_one_source_one_vote({"vector_hybrid": merged})
