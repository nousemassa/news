"""Схема новостного продукта: черновики, дедуп публикаций.

База принадлежит этому продукту: у него свой экземпляр postgres, поэтому накат новостей не зависит от
готовности других продуктов и наоборот.
"""
from __future__ import annotations

SCHEMA_CHECK_SQL = """
select
    (select count(*) from information_schema.tables
       where table_schema='public' and table_name='news_drafts')
  + (select count(*) from information_schema.tables
       where table_schema='public' and table_name='news_seen')
"""

SCHEMA_CHECK_EXPECTED = 2

SCHEMA_CHECK_HINT = "news_drafts, news_seen"

DDL = """
CREATE TABLE IF NOT EXISTS news_drafts (
    id             bigserial PRIMARY KEY,
    author_chat_id text NOT NULL,
    title          text NOT NULL DEFAULT '',
    text           text NOT NULL,
    original_text  text,
    url            text NOT NULL DEFAULT '',
    source         text NOT NULL DEFAULT '',
    status         text NOT NULL DEFAULT 'draft',   -- draft | editing | publishing | published
    message_id     bigint,
    run_id         text,
    channel_id     text,
    published_at   timestamptz,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_news_author ON news_drafts (author_chat_id, status);

-- Дедуп между прогонами: «новости за неделю» возвращаются, повторно их не показываем.
CREATE TABLE IF NOT EXISTS news_seen (
    url     text PRIMARY KEY,
    seen_at timestamptz NOT NULL DEFAULT now()
);
"""

DEFAULT_SETTINGS: dict = {}
