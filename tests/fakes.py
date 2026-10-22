"""Двойники НОВОСТЕЙ. Общие имена РЕЭКСПОРТИРУЮТСЯ: их зовут и через `fakes.<имя>`.

❗ Реэкспорт обязателен: в новостях есть сайты `import fakes` — они обращаются к
`fakes.make_news_bot/.FakeDb/.make_settings`, и без реэкспорта упадут.
"""
from __future__ import annotations

import pytest

from agent_core.testing import (  # noqa: F401 — реэкспорт общих двойников
    OWNER,
    SECRET,
    TEST_SESSIONS_DIR,
    FakeCallback,
    FakeMsg,
    FakeSender,
    ScriptedChannels,
    StrictChannels,
    _trace_cost,
    fake_client,
)
from agent_core.testing import FakeDb as CoreFakeDb
from news_agent.settings import Settings


class FakeDb(CoreFakeDb):
    """Двойник базы новостного контура: черновики, дедуп, публикация, гейт прогона."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        import datetime as dt

        self.news_drafts: dict[int, dict] = {}
        self.news_seen: set[str] = set()
        self.news_inflight = False
        self.news_runs = 0
        self.news_month_cost_usd = 0.0
        self.news_editing_age_s = 0
        self.news_published_log: list[tuple[int, str, str]] = []
        self._news_seq = 0
        self._news_now = dt.datetime.now(dt.UTC)

    async def create_news_draft(self, *, author_chat_id, title, text, url="", source="", run_id=""):
        self._news_seq += 1
        draft = {"id": self._news_seq, "author_chat_id": author_chat_id, "title": title, "text": text,
                 "original_text": text, "url": url, "source": source, "status": "draft", "message_id": None,
                 "run_id": run_id, "channel_id": None, "published_at": None, "created_at": self._news_now}
        self.news_drafts[draft["id"]] = draft
        return dict(draft)

    async def attach_news_message(self, *, draft_id, message_id) -> None:
        if draft_id in self.news_drafts:
            self.news_drafts[draft_id]["message_id"] = message_id

    async def news_draft(self, *, draft_id):
        draft = self.news_drafts.get(draft_id)
        return dict(draft) if draft else None

    async def set_news_editing(self, *, draft_id, author_chat_id) -> None:
        for other in self.news_drafts.values():                     # ровно один в правке на автора
            if other["author_chat_id"] == author_chat_id and other["status"] == "editing" and other["id"] != draft_id:
                other["status"] = "draft"
        draft = self.news_drafts.get(draft_id)
        if draft and draft["status"] == "draft":
            draft["status"] = "editing"

    async def news_editing_for(self, *, author_chat_id, ttl_s):
        for draft in self.news_drafts.values():
            if draft["author_chat_id"] == author_chat_id and draft["status"] == "editing":
                if self.news_editing_age_s > ttl_s:                 # ленивое истечение TTL
                    draft["status"] = "draft"
                    return None
                return dict(draft)
        return None

    async def set_news_text(self, *, draft_id, text):
        draft = self.news_drafts.get(draft_id)
        if not draft or draft["status"] != "editing":
            return None
        draft["text"] = text
        draft["status"] = "draft"
        return {key: draft[key] for key in ("id", "title", "text", "url", "source", "status")}

    async def cancel_news_editing(self, *, author_chat_id) -> int:
        count = 0
        for draft in self.news_drafts.values():
            if draft["author_chat_id"] == author_chat_id and draft["status"] == "editing":
                draft["status"] = "draft"
                count += 1
        return count

    async def claim_news_publish(self, *, draft_id):
        draft = self.news_drafts.get(draft_id)
        if not draft or draft["status"] != "draft":
            return None
        draft["status"] = "publishing"
        return {key: draft[key] for key in ("id", "title", "text", "url", "source", "created_at")}

    async def confirm_news_published(self, *, draft_id, channel_id) -> None:
        draft = self.news_drafts.get(draft_id)
        if draft:
            draft["status"] = "published"
            draft["channel_id"] = channel_id
            draft["published_at"] = self._news_now

    async def release_news_publish(self, *, draft_id) -> None:
        draft = self.news_drafts.get(draft_id)
        if draft and draft["status"] == "publishing":
            draft["status"] = "draft"

    async def release_stale_publishing(self, *, ttl_s: int = 300) -> int:
        return 0

    async def news_seen_filter(self, *, urls) -> set[str]:
        return {url for url in urls if url in self.news_seen}

    async def news_mark_seen(self, *, urls) -> None:
        self.news_seen.update(url for url in urls if url)

    async def news_run_gate(self, *, chat_id, cooldown_s, daily_cap, month_budget_usd: float = 0.0,
                            inflight_ttl_s: int = 600):
        if self.news_inflight:
            return False, "run_in_progress"
        if daily_cap and self.news_runs >= daily_cap:
            return False, "daily_cap"
        if month_budget_usd and self.news_month_cost_usd >= month_budget_usd:
            return False, f"month_budget:{self.news_month_cost_usd:.2f}"
        return True, ""

    async def news_month_cost(self) -> float:
        return self.news_month_cost_usd

    async def news_run_started(self, *, chat_id) -> None:
        self.news_inflight = True
        self.news_runs += 1

    async def news_run_finished(self, *, chat_id) -> None:
        self.news_inflight = False

    async def log_news_publish(self, *, draft_id, channel_id, actor, cost_usd: float = 0.0) -> None:
        self.news_published_log.append((draft_id, channel_id, actor))


def make_settings(**overrides) -> Settings:
    base = {
        "sessions_dir": str(TEST_SESSIONS_DIR),
        "ui_secret": SECRET,
        "bridge_api_key": "k" * 32,
        "telegram_owner_chat_id": OWNER,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture
def db() -> FakeDb:
    return FakeDb()


def make_news_bot(db: FakeDb | None = None, *, settings: Settings | None = None, channels=None,
                  retriever=None, gate_ask=None, gate_threshold: float = 0.75, team_ask=None):
    """Новостной бот на фейках: сеть не нужна — `start()` не вызываем.

    `retriever`, `gate_ask`, `team_ask` — контур команды ролей. Без них бот идёт одиночным путём,
    а с ними проверяется боевой путь.
    """
    from news_agent.tg.news_bot import NewsBot

    return NewsBot(settings=settings or make_settings(), db=db or FakeDb(),
                   channels=channels or ScriptedChannels(), retriever=retriever,
                   gate_ask=gate_ask, gate_threshold=gate_threshold, team_ask=team_ask)
