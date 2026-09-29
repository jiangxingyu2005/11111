"""POST /api/retrieval —— SSE 主端点（M-C1）。

SSE 只是传输方式，契约对象 = 所有事件聚合后的一份响应（见 retrieval_response.v1.schema.json）：
  {contract_version, trace_id, meta, results, assets, artifacts, graph_ref, degraded, timings}
事件序列：meta -> progress* -> results -> assets -> token* -> done / error

主链：QRY-02(L1+L3) -> ①③并行召回 -> 外层 RRF -> rerank -> QRY-06 父块回捞 -> 生成 -> done
预算看门狗（QRY-07）在此真正驱动砍路（阈值见 config/retrieval.yaml 的 degradation.cutoffs，
  当前 8/10/12/15s）：>8s 砍重排 / >10s 砍图路 / >12s 砍稀疏与 NL2Query；每砍必留 degraded（禁静默降级）
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..auth import resolve_user_id
from ..config import effective_cfg, get_settings
from ..degrade.codes import DegradedCode
from ..degrade.orchestrator import BudgetWatchdog, DegradedCollector
from ..fuse.parent_backfill import backfill, check_version
from ..fuse.rerank import rerank
from ..fuse.rrf import rrf_fuse
from ..fuse.rules import evaluate_rules, load_rules
from ..generate.report import stream_report
from ..graph.expand import focus_entities
from ..memory import service as memory_service
from ..recall import bm25_route, community_route, graph_route, vector_route
from ..trace.spans import TraceRecorder, bind_trace, new_trace_id
from ..understand.query_parser import parse, parse_l2

log = logging.getLogger("retrieval.api")
router = APIRouter()
CONTRACT_VERSION = "1.0"


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1)
    scope: list[str] = Field(min_length=1)   # R4：必填无默认值（权限从第一行代码就是硬约束）
    session_id: str | None = None            # QRY-00 会话（记忆管理按此隔离会话）
    eval_mode: bool = False                  # EV-1/EV-5：评测模式（不写缓存/不计被引）
    token: str | None = None                 # 记忆身份令牌：resolve_user_id 解析出 user_id（缺省则跳过记忆）


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/api/retrieval")
async def retrieval(req: RetrievalRequest) -> StreamingResponse:
    async def event_stream():
        trace_id = new_trace_id()
        collector = DegradedCollector()
        tracer = TraceRecorder(trace_id, eval_mode=req.eval_mode)
        watchdog = BudgetWatchdog()
        t_start = time.monotonic()
        timings: dict[str, int] = {}
        first_token_at: float | None = None

        # ── 记忆身份：前端 token -> user_id（fail-closed，解析失败=本轮跳过记忆）──
        user_id = resolve_user_id(req.token)

        yield _sse("meta", {"contract_version": CONTRACT_VERSION, "trace_id": trace_id,
                            "eval_mode": req.eval_mode})
        try:
            # ── QRY-02 查询理解（L1 词典正则 + L3 校验器）──
            span = tracer.start("understand")
            pq = parse(req.query, collector)
            span.hit_count = len(pq.conditions)
            span.degraded_codes = collector.codes()
            span.close()
            timings["parse_ms"] = int(span.duration_ms)
            yield _sse("progress", {"stage": "understand", "intent": pq.intent,
                                    "conditions": [c.to_dict() for c in pq.conditions]})

            # ── L2 结构化抽取：不在召回前起跑，改为重排后起跑（串行化）──
            # CPU 机上 ollama(L2/生成) 与 model_stub(BGE-M3 嵌入 + reranker 交叉编码器)
            # 争同一批核。并行会互相拖慢：首查召回返空、重排 16s 超时。L2 结果只影响
            # 响应 meta.conditions，从不影响召回（召回在合并前跑），故可安全延后到重排后，
            # 与嵌入/重排完全串行。真正起跑见下方 rerank 之后的 fire 块。
            l2_task = None
            t_l2_fire = 0.0

            # ── QRY-07 预算决策（砍路，不是砍语义）──
            cfg = effective_cfg()
            level = watchdog.current_level()
            run_graph = get_settings().retrieval_graph_route_enabled
            run_community = get_settings().retrieval_community_enabled
            if level.value in ("L2", "L1", "L0"):        # >10s：先砍社区摘要，再砍图路保主链
                if run_community:
                    collector.record(DegradedCode.BUDGET_CUT_COMMUNITY,
                                     f"budget level={level.value}, elapsed={watchdog.elapsed_s():.2f}s",
                                     "graph")
                    run_community = False
                if run_graph:
                    collector.record(DegradedCode.BUDGET_CUT_ROUTE3,
                                     f"budget level={level.value}, elapsed={watchdog.elapsed_s():.2f}s",
                                     "graph")
                run_graph = False

            # ── 三路并行召回（② 默认关；③-B 社区默认关）──
            span = tracer.start("recall")
            t0 = time.monotonic()
            vec_hits, bm25_hits, graph_hits, community_hits = await asyncio.gather(
                vector_route.recall(req.query, scope=req.scope, collector=collector,
                                    conditions=pq.conditions),
                bm25_route.recall(req.query, scope=req.scope, collector=collector),
                graph_route.recall(req.query, scope=req.scope, collector=collector)
                if run_graph else asyncio.sleep(0, result=[]),
                community_route.recall(req.query, scope=req.scope, collector=collector)
                if run_community else asyncio.sleep(0, result=[]),
            )
            timings["vector_ms"] = int((time.monotonic() - t0) * 1000)
            span.hit_count = len(vec_hits) + len(bm25_hits) + len(graph_hits) + len(community_hits)
            span.degraded_codes = collector.codes()
            span.close()
            yield _sse("progress", {"stage": "recall",
                                    "vector": len(vec_hits), "bm25": len(bm25_hits),
                                    "graph": len(graph_hits), "community": len(community_hits),
                                    "budget_level": level.value})

            # ── 外层 RRF（1 source 1 vote）──
            span = tracer.start("fuse")
            fused = rrf_fuse({"vector_hybrid": vec_hits, "bm25": bm25_hits,
                              "graph_local": graph_hits, "graph_global": community_hits},
                             k=int(cfg.get("fusion", {}).get("k", 60)),
                             top_k=int(cfg.get("fusion", {}).get("top_k", 50)))
            span.hit_count = len(fused)
            span.close()

            # ── Rerank（预算 L3 砍精排；超时/失败直出 RRF 序）──
            span = tracer.start("rerank")
            t0 = time.monotonic()
            level = watchdog.current_level()
            if level.value in ("L3", "L2", "L1", "L0"):   # >8s：砍重排
                collector.record(DegradedCode.RERANK_SKIPPED,
                                 f"budget level={level.value}", "rerank")
                top = fused[:int(cfg.get("reranker", {}).get("top_k_out", 5))]
            else:
                top = await rerank(req.query, fused, collector)
            timings["rerank_ms"] = int((time.monotonic() - t0) * 1000)
            span.hit_count = len(top)
            span.hit_parent_ids = [h.parent_id for h in top]
            span.degraded_codes = collector.codes()
            span.close()

            # ── L2 结构化抽取（重排后起跑：与嵌入/重排串行，避免 CPU 争用）──
            # 放这里之后，后续 backfill(PG IO)/results 组装都是毫秒级且不占模型算力，
            # ollama 跑 L2 时不会再与 BGE-M3 或 reranker 抢核；生成阶段自然排队在 L2 后。
            if pq.l2_needed and get_settings().graphrag_llm_base_url:
                t_l2_fire = time.monotonic()
                l2_task = asyncio.create_task(parse_l2(req.query, collector))

            # ── QRY-06 父块回捞（PG 熔断 -> 行内 content 兜底）──
            span = tracer.start("backfill")
            t0 = time.monotonic()
            parents = await backfill(top, collector)
            timings["pg_fetch_ms"] = int((time.monotonic() - t0) * 1000)
            span.hit_count = len(parents)
            span.close()

            # ── 规则求值（需求(7)预设推理规则）：先算后写，只标记不改召回序 ──
            # 规则现查 rules 表（增删改即时生效）；fake 后端无 PG，规则为空（无规则 = 稳态）。
            span = tracer.start("rules")
            rules = await load_rules(collector)
            rule_hits = evaluate_rules(rules, top, parents, req.scope)
            span.hit_count = len(rule_hits)
            span.close()

            # ── 合并 L2 结果（生成前；按 L2 自身起跑时间算预算，不被生成时长吃掉）──
            if l2_task is not None:
                budget_ms = int(cfg.get("timeout", {}).get("nl2query_ms", 20000))
                elapsed_l2_ms = (time.monotonic() - t_l2_fire) * 1000
                grace = max(0.0, (budget_ms - elapsed_l2_ms) / 1000.0)
                if grace > 0:
                    try:
                        l2pq = await asyncio.wait_for(l2_task, timeout=grace)
                        if l2pq is not None:
                            pq = l2pq
                    except asyncio.TimeoutError:
                        collector.record(DegradedCode.NL2QUERY_LLM_TIMEOUT,
                                         f"L2 exceeded {budget_ms}ms budget", "query_parse")
                else:
                    # L2 自身已跑超预算，不阻塞主链；结果丢弃（后台任务仍在跑，下次请求受益）
                    collector.record(DegradedCode.NL2QUERY_LLM_TIMEOUT,
                                     f"L2 skipped: elapsed {elapsed_l2_ms:.0f}ms >= budget {budget_ms}ms",
                                     "query_parse")

            # ── results（证据锚点 child_id + span；IF-2 未就位则 span=null 并留痕）──
            results = []
            for i, h in enumerate(top, start=1):
                p = parents.get(h.parent_id, {})
                if p and not check_version(h, p):
                    collector.record(DegradedCode.PARENT_FETCH_FAILED,
                                     f"content_version drift: {h.child_id}", "pg_fetch")
                sp = h.meta.get("span")
                if not sp and not collector.has(DegradedCode.EVIDENCE_SPAN_MISSING):
                    collector.record(DegradedCode.EVIDENCE_SPAN_MISSING,
                                     "IF-2 未就位：子块无 span，前端不得渲染回原文高亮", "vector")
                results.append({
                    "parent_id": h.parent_id,
                    "child_id": h.child_id,
                    "title": h.meta.get("section") or None,
                    "content": p.get("content") or h.content,
                    "metadata": {"kb_id": h.meta.get("kb_id", ""),
                                 "voted_by": h.meta.get("voted_by", []),
                                 "rank": i,
                                 "via_entity": h.meta.get("via_entity"),
                                 "via_community": h.meta.get("via_community")},
                    "evidence": {"child_id": h.child_id, "span": sp},
                    "score": {"rrf": h.meta.get("rrf_score"),
                              "rerank": h.meta.get("rerank_score")},
                })
            yield _sse("results", {"hits": results, "degraded": collector.as_list()})

            # ── rules：规则触发详情（需求(8)；空 = 无规则命中，不算降级）──
            yield _sse("rules", {"hits": rule_hits})

            # ── assets / artifacts（IF-5 未就位：空表，不算降级）──
            yield _sse("assets", {"items": []})

            # ── graph_ref：focus 取自重排 Top-N 命中实体（不取 query 里的实体）──
            focus = focus_entities(top, top_n=int(cfg.get("graph_ref", {}).get("top_n", 5)))
            bind_trace(trace_id, focus, req.scope, [h.child_id for h in top])
            sg = cfg.get("subgraph", {})
            graph_ref = {
                "available": bool(focus),
                "focus_entities": focus,
                "endpoint": "/api/graph/subgraph",
                "binding": "trace_id",
                "default_hops": int(sg.get("default_hops", 1)),
                "node_limit": int(sg.get("node_limit", 50)),
                "edge_limit": int(sg.get("edge_limit", 100)),
            }

            # ── RPT 流式生成（LLM 熔断 -> 无 token，done 带降级）──
            # 记忆读侧：装载三层上下文（fail-closed，失败返回空不影响主链）
            ctx = await memory_service.load_context(
                user_id or "", req.session_id or "", req.query)
            span = tracer.start("generate")
            answer_parts: list[str] = []
            async for tok in stream_report(req.query, top, parents, collector, rule_hits,
                                           ctx["memories"], ctx["summary"], ctx["window"]):
                if tok:
                    answer_parts.append(tok)
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                        timings["first_token_ms"] = int((first_token_at - t_start) * 1000)
                    yield _sse("token", {"text": tok})
            span.degraded_codes = collector.codes()
            span.close()

            # ── 记忆写侧（fire-and-forget：append 本轮 -> 压缩，不阻塞 done）──
            if user_id:
                memory_service.spawn_background(memory_service.commit_turn(
                    user_id, req.session_id or "", req.query, "".join(answer_parts)))

            timings["total_ms"] = int((time.monotonic() - t_start) * 1000)
            # budget_level 必须取【检索段】的最终档位，不能在生成后再 current_level()——
            # 生成(35s+)会把 elapsed 顶到 >15s，恒报 L0，误导成「已退化为 PG 兜底」。
            # 看门狗最后一次真正驱动砍路是在重排决策点(见 rerank 段)，此处 level 即该档位。
            yield _sse("done", {
                "contract_version": CONTRACT_VERSION,
                "trace_id": trace_id,
                "meta": {"intent": pq.intent, "query_rewritten": None,
                         "conditions": [c.to_dict() for c in pq.conditions]},
                "graph_ref": graph_ref,
                "degraded": collector.as_list(),
                "timings": timings,
                "budget_level": level.value,
                "rule_hits": rule_hits,
            })
        except Exception as e:  # 兜底：主链任何未捕获异常 -> error 事件，绝不裸 500
            log.exception("retrieval pipeline failed: %s", e)
            yield _sse("error", {"trace_id": trace_id, "message": str(e)[:300],
                                 "degraded": collector.as_list()})
        finally:
            tracer.export_async()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
