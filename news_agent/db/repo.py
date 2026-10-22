"""Доступ к данным новостного продукта: черновики, дедуп, публикация, гейт прогона.

Наследует общий `Db` и добавляет только свои операции. Схема — в `news_agent/db/schema.py`.
"""
from __future__ import annotations

import time
from collections.abc import Iterable

from agent_core.db.registry import CORE_COMPONENTS, Component
from agent_core.db.store import Db
from news_agent.db import schema

REVISION = 2

NEWS_COMPONENT = Component(
    name="news",
    ddl=schema.DDL,
    revision=REVISION,
    check_sql=schema.SCHEMA_CHECK_SQL,
    expected=schema.SCHEMA_CHECK_EXPECTED,
    settings=schema.DEFAULT_SETTINGS,
    hint=schema.SCHEMA_CHECK_HINT,
)

NEWS_COMPONENTS = (NEWS_COMPONENT,)

#: Таблицы, которые обязан иметь на месте новостной процесс: схему создаёт накат.
NEWS_TABLES = ("news_drafts", "news_seen")


class NewsDb(Db):
    """Общий `Db` плюс операции новостного контура. По умолчанию накатывает и требует оба компонента."""

    def __init__(self, *args, components: tuple[Component, ...] | None = None, **kwargs) -> None:
        super().__init__(*args, components=components or (CORE_COMPONENTS + NEWS_COMPONENTS), **kwargs)

    async def log_news_publish(self, *, draft_id: int, channel_id: str, actor: str,
                               cost_usd: float = 0.0) -> None:
        """Аудит публикации в общий журнал + расход прогона."""
        await self.log_audit(
            actor=actor, action="news_publish", result=channel_id, meta={"draft_id": draft_id})
        if cost_usd:
            await self.bump_metric(metric="news_cost_usd", delta=cost_usd)

    async def create_news_draft(self, *, author_chat_id: str, title: str, text: str, url: str = "",
                                source: str = "", run_id: str = "") -> dict:
        """Черновик новости; попутно ретенция 90 дней — таблица не растёт бесконечно."""
        row = await self.pool.fetchrow(
            """INSERT INTO news_drafts (author_chat_id, title, text, original_text, url, source, run_id)
               VALUES ($1,$2,$3,$3,$4,$5,$6)
               RETURNING id, title, text, url, source, status, run_id, created_at""",
            author_chat_id, title, text, url, source, run_id)
        await self.pool.execute("DELETE FROM news_drafts WHERE created_at < now() - interval '90 days'")
        return dict(row)

    async def attach_news_message(self, *, draft_id: int, message_id: int) -> None:
        """Запомнить сообщение-карточку: по нему потом снимаем кнопки при переотправке/публикации."""
        await self.pool.execute(
            "UPDATE news_drafts SET message_id=$2, updated_at=now() WHERE id=$1", draft_id, message_id)

    async def news_draft(self, *, draft_id: int) -> dict | None:
        row = await self.pool.fetchrow("SELECT * FROM news_drafts WHERE id=$1", draft_id)
        return dict(row) if row else None

    async def set_news_editing(self, *, draft_id: int, author_chat_id: str) -> None:
        """Ровно один черновик в правке на автора: прежний возвращаем в draft, чтобы не залипал."""
        await self.pool.execute(
            """UPDATE news_drafts SET status='draft', updated_at=now()
               WHERE author_chat_id=$1 AND status='editing' AND id<>$2""", author_chat_id, draft_id)
        await self.pool.execute(
            "UPDATE news_drafts SET status='editing', updated_at=now() WHERE id=$1 AND status='draft'", draft_id)

    async def news_editing_for(self, *, author_chat_id: str, ttl_s: int) -> dict | None:
        """Активный режим правки с ЛЕНИВЫМ истечением TTL: старое состояние возвращаем в draft."""
        row = await self.pool.fetchrow(
            """SELECT * FROM news_drafts
               WHERE author_chat_id=$1 AND status='editing'
                 AND updated_at > now() - ($2::int * interval '1 second')
               ORDER BY updated_at DESC LIMIT 1""", author_chat_id, ttl_s)
        if row:
            return dict(row)
        await self.pool.execute(
            "UPDATE news_drafts SET status='draft', updated_at=now() WHERE author_chat_id=$1 AND status='editing'",
            author_chat_id)
        return None

    async def set_news_text(self, *, draft_id: int, text: str) -> dict | None:
        """Новая редакция от владельца: только для черновика в режиме правки, статус → draft."""
        row = await self.pool.fetchrow(
            """UPDATE news_drafts SET text=$2, status='draft', updated_at=now()
               WHERE id=$1 AND status='editing'
               RETURNING id, title, text, url, source, status""", draft_id, text)
        return dict(row) if row else None

    async def cancel_news_editing(self, *, author_chat_id: str) -> int:
        result = await self.pool.execute(
            "UPDATE news_drafts SET status='draft', updated_at=now() WHERE author_chat_id=$1 AND status='editing'",
            author_chat_id)
        return int(result.split()[-1]) if result else 0

    async def claim_news_publish(self, *, draft_id: int) -> dict | None:
        """CAS draft→publishing: два тапа не дадут двух постов в канал."""
        row = await self.pool.fetchrow(
            """UPDATE news_drafts SET status='publishing', updated_at=now()
               WHERE id=$1 AND status='draft'
               RETURNING id, title, text, url, source, created_at""", draft_id)
        return dict(row) if row else None

    async def confirm_news_published(self, *, draft_id: int, channel_id: str) -> None:
        await self.pool.execute(
            """UPDATE news_drafts SET status='published', channel_id=$2, published_at=now(), updated_at=now()
               WHERE id=$1""", draft_id, channel_id)

    async def release_news_publish(self, *, draft_id: int) -> None:
        """Ошибка отправки: возвращаем черновик в draft — новость не теряется."""
        await self.pool.execute(
            "UPDATE news_drafts SET status='draft', updated_at=now() WHERE id=$1 AND status='publishing'", draft_id)

    async def release_stale_publishing(self, *, ttl_s: int = 300) -> int:
        """Залипшие `publishing` (упал процесс между CAS и отправкой) — обратно в draft."""
        result = await self.pool.execute(
            """UPDATE news_drafts SET status='draft', updated_at=now()
               WHERE status='publishing' AND updated_at < now() - ($1::int * interval '1 second')""", ttl_s)
        return int(result.split()[-1]) if result else 0

    async def news_seen_filter(self, *, urls: Iterable[str]) -> set[str]:
        """Какие ссылки уже показывали (дедуп между прогонами: «новость за неделю» вернётся снова)."""
        items = [item for item in urls if item]
        if not items:
            return set()
        rows = await self.pool.fetch("SELECT url FROM news_seen WHERE url = ANY($1::text[])", items)
        return {str(row["url"]) for row in rows}

    async def news_mark_seen(self, *, urls: Iterable[str]) -> None:
        rows = [(item,) for item in urls if item]
        if rows:
            await self.pool.executemany(
                "INSERT INTO news_seen(url) VALUES($1) ON CONFLICT (url) DO UPDATE SET seen_at=now()", rows)

    async def news_month_cost(self) -> float:
        """Сколько потрачено на новости в текущем месяце: гейт по деньгам, а не по числу прогонов."""
        value = await self.pool.fetchval(
            """SELECT COALESCE(SUM(value),0) FROM meta_metrics
               WHERE metric='news_cost_usd' AND day >= date_trunc('month', now()::date)""")
        return float(value or 0)

    async def news_run_gate(self, *, chat_id: str, cooldown_s: int, daily_cap: int, month_budget_usd: float = 0.0,
                            inflight_ttl_s: int = 600) -> tuple[bool, str]:
        """Общий гейт кнопки и ручки: in-flight, cooldown, дневной потолок и месячный бюджет живут в БД."""
        now = time.time()
        inflight = float(await self.setting(f"news_inflight:{chat_id}", 0) or 0)
        if inflight and now - inflight < inflight_ttl_s:
            return False, "run_in_progress"
        last = float(await self.setting(f"news_last_run:{chat_id}", 0) or 0)
        if cooldown_s and last and now - last < cooldown_s:
            return False, f"cooldown:{int(cooldown_s - (now - last))}"
        used = await self.pool.fetchval(
            "SELECT COALESCE(value,0) FROM meta_metrics WHERE day = now()::date AND metric = 'news_runs'")
        if daily_cap and float(used or 0) >= daily_cap:
            return False, "daily_cap"
        if month_budget_usd:
            spent = await self.news_month_cost()
            if spent >= month_budget_usd:
                return False, f"month_budget:{spent:.2f}"
        return True, ""

    async def news_run_started(self, *, chat_id: str) -> None:
        now = time.time()
        await self.set_setting(f"news_inflight:{chat_id}", now)
        await self.set_setting(f"news_last_run:{chat_id}", now)
        await self.bump_metric(metric="news_runs")

    async def news_run_finished(self, *, chat_id: str) -> None:
        await self.set_setting(f"news_inflight:{chat_id}", 0)
