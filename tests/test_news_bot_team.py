"""Боевой путь новостей на команде ролей.

**Зачем отдельный файл.** `test_news_bot.py` проверяет поведение бота на ОДИНОЧНОМ пути. Здесь
проверяется то же поведение на пути КОМАНДЫ: поиск и переписывание идут графом команды, а девять
свойств боевого прогона (гейт, `finally`, дедуп, CAS, алерты, карточки, отказ при потере фактов,
учёт стоимости, ручка прогона) остаются в силе. Плюс главное свойство пути: **фолбэк** на
одиночный агент при недоступности команды — он и есть страховка.

Сеть не нужна: модель — двойник-сценарий, Telegram и БД — существующие двойники `tests/fakes.py`.
"""
from __future__ import annotations

import json

import fakes
import pytest

from news_agent import news
from news_agent.news_team.errors import TeamUnavailable
from news_agent.news_team.graph import Answer
from news_agent.tg import news_bot as module

OWNER = fakes.OWNER
CHANNEL = -1001234567890


@pytest.fixture(autouse=True)
def _team_searches(monkeypatch):
    """Этот файл — про ПУТЬ КОМАНДЫ, а по умолчанию поиск идёт одиночным агентом (решение владельца).

    Почему так, а не иначе. Замер: команда не берёт выдачу нашего поиска (роль отвечает `NEWS_NONE` при
    десяти результатах и `SEARCH_FAILED` при «только главные страницы и списки»), а тот же агент ту же
    выдачу превращает в карточки. Поэтому `SEARCH_VIA` по умолчанию `agent`, а ветвь команды остаётся
    рабочей и покрытой — здесь её и включаем. Значение по умолчанию закреплено отдельным тестом
    (`test_news_bot.py::test_search_is_done_by_the_agent_by_default`).
    """
    monkeypatch.setattr(module, "SEARCH_VIA", "team")


def press(kind: str, arg: str):
    """Нажатие кнопки от владельца: подпись и чат согласованы, иначе `verify` отклонит тап."""
    from agent_core.ui import tokens

    return fakes.FakeCallback(
        tokens.sign(fakes.SECRET, chat_id=OWNER, kind=kind, arg=arg, token=""), chat_id=OWNER)


def news_lines(*urls: str) -> str:
    return "\n".join(
        news.NEWS_PREFIX + json.dumps({"title": f"Новость {index}", "text": "Т" * 60, "url": url,
                                       "source": "источник"}, ensure_ascii=False)
        for index, url in enumerate(urls, start=1)
    )


def verdict_line(text: str, *, lost: list[str] | None = None) -> str:
    return "VERDICT|" + json.dumps({"text": text, "lost": lost or [], "show": True})


class TeamScript:
    """Двойник команды: что ответит каждая роль и падает ли шаг.

    Интерфейс — тот же шов, что у графа: `(role_key, prompt) -> Answer`. Цена задаётся сценарием,
    потому что от неё зависит учёт стоимости, а это одно из девяти свойств.
    """

    def __init__(self, *, found: str = "", rewrite: str = "", cost: float | None = 0.02,
                 fail: str | None = None) -> None:
        self.found = found
        self.rewrite = rewrite
        self.cost = cost
        self.fail = fail
        self.roles: list[str] = []

    async def __call__(self, role_key: str, prompt: str) -> Answer:
        self.roles.append(role_key)
        if self.fail == role_key:
            raise TeamUnavailable(f"шаг {role_key} упал: проверка сценария")
        if role_key == "researcher":
            return Answer(text=self.found, cost_usd=self.cost, tokens=11)
        if role_key == "author":
            return Answer(text=self.rewrite, cost_usd=self.cost, tokens=22)
        return Answer(text=verdict_line(self.rewrite), cost_usd=self.cost, tokens=33)


def bot_with_team(*, db=None, team: TeamScript | None = None, single: str = "",
                  extra_settings: dict | None = None):
    """Бот с контуром команды. `single` — ответ одиночного пути (для проверки фолбэка)."""
    settings = fakes.make_settings(news_max_items=5, **(extra_settings or {}))
    channels = fakes.ScriptedChannels(single)
    bot = fakes.make_news_bot(db or fakes.FakeDb(), settings=settings, channels=channels,
                              team_ask=team)
    client = fakes.fake_client()
    client.connected = True
    bot._client = client
    return bot, bot._db, client, channels


# ---------- девять свойств боевого прогона на новом пути ----------


@pytest.mark.asyncio
async def test_team_search_drives_cards_and_the_single_path_is_not_called():
    """Путь команды — боевой: карточки собираются её находками, одиночный агент не зовётся."""
    team = TeamScript(found=news_lines("https://a.invalid/1", "https://a.invalid/2"))
    bot, db, client, channels = bot_with_team(team=team)

    result = await bot.run_once(chat_id=OWNER)

    assert result == {"status": "ok", "cards": 2, "dropped_seen": 0}
    assert team.roles == ["researcher"], "поиск идёт ролью исследователя"
    assert channels.calls == 0, "одиночный агент на новом пути не вызывается"
    cards = [item for item in client.sent if "📰 Новость" in item[1]]
    assert len(cards) == 2
    for _, _text, buttons in cards:
        labels = [button.text for row in buttons for button in row]
        assert labels == ["✏️ Редактировать", "📤 Опубликовать в канале"]
    assert len(db.news_drafts) == 2 and db.news_seen == {"https://a.invalid/1", "https://a.invalid/2"}


@pytest.mark.asyncio
async def test_gate_closed_stops_the_run_before_any_model_call():
    team = TeamScript(found=news_lines("https://a.invalid/1"))
    db = fakes.FakeDb()
    db.news_inflight = True
    bot, _, client, channels = bot_with_team(db=db, team=team)

    assert (await bot.run_once(chat_id=OWNER))["status"] == "run_in_progress"
    assert team.roles == [] and channels.calls == 0, "закрытый гейт не тратит ни команду, ни одиночный путь"
    assert any("Прогон уже идёт" in text for _, text, _ in client.sent)


@pytest.mark.asyncio
async def test_month_budget_gate_speaks_about_money_on_the_team_path():
    db = fakes.FakeDb()
    db.news_month_cost_usd = 5.12
    bot, _, client, _ = bot_with_team(db=db, team=TeamScript(found=news_lines("https://a.invalid/1")),
                                      extra_settings={"news_month_budget_usd": 5.0})
    assert (await bot.run_once(chat_id=OWNER))["status"] == "month_budget:5.12"
    assert any("Месячный лимит" in text for _, text, _ in client.sent)


@pytest.mark.asyncio
async def test_run_finished_is_called_even_when_the_team_and_the_single_path_fail():
    """`finally` боевого прогона: незавершённый прогон иначе держал бы гейт in-flight навсегда."""
    team = TeamScript(fail="researcher")
    bot, db, _, channels = bot_with_team(team=team, single="")
    channels.done = False                                # одиночный путь тоже ничего не даёт
    result = await bot.run_once(chat_id=OWNER)

    assert result["status"] == "timeout"
    assert db.news_inflight is False, "гейт обязан быть отпущен"


@pytest.mark.asyncio
async def test_seen_is_marked_after_the_cards_are_sent():
    """Пометка «показывали» идёт ПОСЛЕ отправки: сбой на карточке не должен терять новость."""
    bot, db, client, _ = bot_with_team(team=TeamScript(found=news_lines("https://a.invalid/1")))
    sent_at_mark: list[int] = []
    original = db.news_mark_seen

    async def spy(*, urls):
        sent_at_mark.append(len([item for item in client.sent if "📰 Новость" in item[1]]))
        await original(urls=urls)

    db.news_mark_seen = spy
    await bot.run_once(chat_id=OWNER)

    assert sent_at_mark == [1], "на момент пометки карточка уже отправлена"


@pytest.mark.asyncio
async def test_publish_on_the_team_path_keeps_the_cas_triple():
    """Публикация идёт тройкой: захват → подтверждение; повтор не публикует второй раз."""
    bot, db, client, _ = bot_with_team(team=TeamScript(found=news_lines("https://a.invalid/1")),
                                       extra_settings={"telegram_news_test_channel_id": str(CHANNEL)})
    await bot.run_once(chat_id=OWNER)
    draft_id = next(iter(db.news_drafts))

    await bot._on_callback(press(module.KIND_NEWS_PUBLISH, str(draft_id)))

    assert db.news_drafts[draft_id]["status"] == "published"
    assert db.news_drafts[draft_id]["channel_id"] == str(CHANNEL)
    assert db.news_published_log == [(draft_id, str(CHANNEL), OWNER)]
    posts = [item for item in client.sent if item[0] == CHANNEL]
    assert len(posts) == 1
    await bot._on_callback(press(module.KIND_NEWS_PUBLISH, str(draft_id)))
    assert len([item for item in client.sent if item[0] == CHANNEL]) == 1, "повтор не публикует"


@pytest.mark.asyncio
async def test_team_prose_is_rejected_and_the_single_path_decides():
    """Проза команды — ОТКАЗ ПУТИ ПОИСКА с НАЗВАННОЙ причиной, после чего решает одиночный путь.

    Текст команды, доезжающий до бота как есть, даёт владельцу «агент не отдал карточек» —
    сообщение называет не ту причину. Ответ роли проверяет
    `check_researcher_answer` — тот же механизм, что у графа выпуска, — поэтому проза становится
    названным отказом команды, а новости пытается достать страховка: одиночный путь.
    """
    team = TeamScript(found="Просто проза без служебных строк")
    bot, _, client, _ = bot_with_team(team=team, single="Тоже проза, без служебных строк")

    assert (await bot.run_once(chat_id=OWNER))["status"] == "format"
    assert team.roles == ["researcher"], "команду позвали и её ответ проверили"
    text, buttons = client.sent[-1][1], client.sent[-1][2]
    assert "не тем форматом" in text, "причина называется, а не подменяется общей строкой"
    assert [button.text for row in buttons for button in row] == ["🔁 Ещё раз", "🤖 Переписать по подсказке"]


# ---------- путь правки: автор и проверяющий команды ----------


@pytest.mark.asyncio
async def test_rewrite_applies_clean_text_and_drops_old_buttons():
    bot, db, client, _ = bot_with_team(team=TeamScript(found=news_lines("https://a.invalid/1")))
    await bot.run_once(chat_id=OWNER)
    draft_id = next(iter(db.news_drafts))
    card_message_id = db.news_drafts[draft_id]["message_id"]
    client.sent.clear()

    rewritten = news.NEWS_PREFIX.replace("NEWS|", "") + "Т" * 60
    bot._team_ask = TeamScript(rewrite=rewritten)
    await bot._on_callback(press(module.KIND_NEWS_REWRITE, str(draft_id)))
    await bot._on_message(fakes.FakeMsg("сделай короче", chat_id=OWNER))

    assert db.news_drafts[draft_id]["text"].startswith(rewritten[:20])
    assert client.edited == [(int(OWNER), card_message_id, None)], "у старой карточки кнопки сняты"


@pytest.mark.asyncio
async def test_rewrite_is_refused_when_facts_are_lost_even_after_the_verifier():
    """Отказ применять текст остаётся в силе: проверяющий правит текст, но последнее слово — за счётом."""
    bot, db, client, _ = bot_with_team(team=TeamScript(found=news_lines("https://a.invalid/1")))
    original = "Цена выросла на 12 процентов, дом 5"
    await bot.run_once(chat_id=OWNER)
    draft_id = next(iter(db.news_drafts))
    db.news_drafts[draft_id]["text"] = original
    client.sent.clear()

    # Автор отдаёт текст без чисел, проверяющий его не спасает: потери видит НАШ счёт.
    bot._team_ask = TeamScript(
        rewrite="новость без всяких цифр и адресов вовсе, но достаточно длинная для порога")
    await bot._on_callback(press(module.KIND_NEWS_REWRITE, str(draft_id)))
    await bot._on_message(fakes.FakeMsg("сделай короче", chat_id=OWNER))

    assert db.news_drafts[draft_id]["text"] == original, "потерянный факт дороже формулировки"
    assert any("потерял факты" in text or "потери" in text for _, text, _ in client.sent)


# ---------- стоимость ----------


@pytest.mark.asyncio
async def test_cost_metric_grows_when_the_team_price_is_known():
    bot, db, _client, _ = bot_with_team(
        team=TeamScript(found=news_lines("https://a.invalid/1"), cost=0.07))
    await bot.run_once(chat_id=OWNER)
    assert db.metrics.get("news_cost_usd") == pytest.approx(0.07)


@pytest.mark.asyncio
async def test_cost_metric_does_not_grow_when_the_price_is_unknown():
    """Ноль вместо «неизвестно» выдал бы бесплатный прогон — поэтому метрику не трогаем."""
    bot, db, _client, _ = bot_with_team(
        team=TeamScript(found=news_lines("https://a.invalid/1"), cost=None))
    await bot.run_once(chat_id=OWNER)
    assert "news_cost_usd" not in db.metrics


# ---------- фолбэк: страховка до срока боя ----------


@pytest.mark.asyncio
async def test_unavailable_team_falls_back_to_the_single_agent():
    """Команда недоступна ⇒ вызов канала и обычные карточки: владелец новостей не теряет."""
    team = TeamScript(fail="researcher")
    bot, _db, _client, channels = bot_with_team(
        team=team, single=news_lines("https://a.invalid/1"))

    result = await bot.run_once(chat_id=OWNER)

    assert team.roles == ["researcher"], "команда была позвана и отказала"
    assert channels.calls == 1 and channels.strict_users == [True], "одиночный путь спросил свою сессию"
    assert result == {"status": "ok", "cards": 1, "dropped_seen": 0}
    assert bot.stats["last_error"].startswith("team:"), "сбой команды виден в статистике процесса"


@pytest.mark.asyncio
async def test_rewrite_falls_back_to_the_single_agent_when_the_team_is_unavailable():
    bot, db, _client, channels = bot_with_team(
        team=TeamScript(found=news_lines("https://a.invalid/1")))
    await bot.run_once(chat_id=OWNER)
    draft_id = next(iter(db.news_drafts))
    channels.text = "Совсем другой текст новости про рынок недвижимости"
    bot._team_ask = TeamScript(fail="author")

    await bot._on_callback(press(module.KIND_NEWS_REWRITE, str(draft_id)))
    await bot._on_message(fakes.FakeMsg("перепиши", chat_id=OWNER))

    assert channels.calls == 1, "поиск шёл командой, значит канал вызван ровно один раз — на фолбэке"
    assert db.news_drafts[draft_id]["text"].startswith("Совсем другой текст")


# ---------- ручка ручного прогона ----------


@pytest.mark.asyncio
async def test_ops_news_handle_delegates_to_the_run(monkeypatch):
    """Ручка остаётся рабочей: она зовёт тот же прогон, а не свою копию логики.

    `settings` здесь — граница процесса (ключ и чат владельца приходят из окружения), поэтому в тесте
    подменяется только она; сама ручка и прогон исполняются настоящие.
    """
    from news_agent import news_main

    class Boundary:
        bridge_api_key = "k" * 32
        news_owner_chat_id = OWNER

    monkeypatch.setattr(news_main, "settings", Boundary())
    bot, _, _, _ = bot_with_team(team=TeamScript(found=news_lines("https://a.invalid/1")))

    class Request:
        app = type("App", (), {"state": type("State", (), {"news": bot})()})()

    result = await news_main.news_run(news_main.NewsRunBody(query="ставка"),
                                     Request(), x_api_key="k" * 32)

    assert result["status"] == "ok" and result["cards"] == 1
    assert bot.stats["runs"] == 1
