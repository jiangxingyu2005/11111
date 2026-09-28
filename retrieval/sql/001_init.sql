-- ============================================================
-- retrieval 依赖的 PG 最小子集（001_init.sql）
--
-- 定位说明（重要）：
--   生产库由 qingxi 离线链路写入，已有多张表。本脚本只建【在线检索侧真正会读】的表，
--   用于独立起环境/CI 验证；若目标库已由 qingxi 建过同名表，本脚本会跳过（IF NOT EXISTS）。
--   在线检索只读这些表，不写任何一张（IF-9 只读账号）。
--
-- 表职责：
--   parent_document  父块全文（QRY-06 唯一回捞来源；small-to-big 的 big）
--   child_chunk      子块留档（在线主查 Milvus，此表用于版本核对与兜底排查）
--   scalar_extract   ④ 路标量抽取结果（当前版本不参与在线过滤，仅供 probe 与后续启用）
--   entity_registry  实体注册表（实体链接的兜底名称匹配）
-- ============================================================

CREATE TABLE IF NOT EXISTS parent_document (
    parent_id        TEXT PRIMARY KEY,
    doc_id           TEXT NOT NULL,
    title            TEXT,
    content          TEXT NOT NULL,
    summary          TEXT,
    keywords         JSONB DEFAULT '[]'::jsonb,
    content_version  INTEGER NOT NULL DEFAULT 1,
    kb_id            TEXT NOT NULL,
    source_type      TEXT,
    toa_text         TEXT,
    created_at       TIMESTAMPTZ DEFAULT now(),
    updated_at       TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_parent_kb ON parent_document (kb_id);
-- QRY-06 回捞按 parent_id IN (...) 批量取，主键索引已覆盖；此索引服务按库统计
CREATE INDEX IF NOT EXISTS idx_parent_version ON parent_document (content_version);

CREATE TABLE IF NOT EXISTS child_chunk (
    child_id         TEXT PRIMARY KEY,
    parent_id        TEXT NOT NULL REFERENCES parent_document (parent_id) ON DELETE CASCADE,
    kb_id            TEXT NOT NULL,
    content          TEXT NOT NULL,
    section          TEXT,
    span_start       INTEGER,
    span_end         INTEGER,
    content_version  INTEGER NOT NULL DEFAULT 1,
    kg_entity_ids    TEXT[] DEFAULT '{}',
    scalar           JSONB DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chunk_parent ON child_chunk (parent_id);
CREATE INDEX IF NOT EXISTS idx_chunk_kb ON child_chunk (kb_id);

CREATE TABLE IF NOT EXISTS scalar_extract (
    child_id         TEXT PRIMARY KEY,
    parent_id        TEXT NOT NULL,
    rf_ghz           DOUBLE PRECISION,
    prf_hz           DOUBLE PRECISION,
    pw_us            DOUBLE PRECISION,
    doa_deg          DOUBLE PRECISION,
    snr_db           DOUBLE PRECISION,
    coverage         JSONB DEFAULT '{}'::jsonb,   -- probe 覆盖率：{rf: 0.83, pw: 0.41, ...}
    extracted_at     TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_scalar_rf ON scalar_extract (rf_ghz);
CREATE INDEX IF NOT EXISTS idx_scalar_prf ON scalar_extract (prf_hz);

CREATE TABLE IF NOT EXISTS entity_registry (
    entity_id        TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    type             TEXT NOT NULL,
    aliases          TEXT[] DEFAULT '{}',
    kb_id            TEXT,
    updated_at       TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_entity_name ON entity_registry (name);

-- ─────────────────────────────────────────────────────────────
-- 只读账号（IF-9）：在线检索一律用 retrieval_ro，写操作全拒为验收条件
-- 生产口令由运维下发，此处仅为 dev 值
-- ─────────────────────────────────────────────────────────────
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'retrieval_ro') THEN
        CREATE ROLE retrieval_ro LOGIN PASSWORD 'dev_only';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE qingxi TO retrieval_ro;
GRANT USAGE ON SCHEMA public TO retrieval_ro;
GRANT SELECT ON parent_document, child_chunk, scalar_extract, entity_registry TO retrieval_ro;
-- 显式不授予 INSERT/UPDATE/DELETE/TRUNCATE：写操作在数据库层就被拒（不是靠应用层自觉）
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA public FROM retrieval_ro;
