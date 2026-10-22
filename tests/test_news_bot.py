"""Бот новостей: овнер-гейт, карточки, дедуп, правка, публикация.

Сеть не нужна: клиент и канал — фейки (`fakes.fake_client`, `fakes.ScriptedChannels`),
БД — `FakeDb` с новостными методами.
"""
from __future__ import annotations

import asyncio
import json
import logging

import fakes
from telethon.errors import RPCError

from agent_core.ui import tokens
from news_agent import news
from news_agent.tg import news_bot as module


def _agent_reply(*urls: str) -> str:
    return "\n".join(
        news.NEWS_PREFIX + json.dumps({"title": f"Новость {index}", "text": "Т" * 60, "url": url,
                                       "source": "источник"}, ensure_ascii=False)
        for index, url in enumerate(urls, start=1)
    )


def _bot(*, agent_text: str = "", extra_settings: dict | None = None, db=None):
    settings = fakes.make_settings(news_max_items=5, **(extra_settings or {}))
    channels = fakes.ScriptedChannels(agent_text)
    bot = fakes.make_news_bot(db or fakes.FakeDb(), settings=settings, channels=channels)
    client = fakes.fake_client()
    client.connected = True       # бот «подключён»: состояние соединения, а не отдельный флаг
    bot._client = client          # start() не вызываем: сеть не нужна
    return bot, bot._db, client, channels


def _signed(kind: str, arg: str, *, chat_id: str = fakes.OWNER) -> str:
    return tokens.sign(fakes.SECRET, chat_id=chat_id, kind=kind, arg=arg, token="")


def _cb(kind: str, arg: str, *, chat_id: str = fakes.OWNER) -> fakes.FakeCallback:
    """Нажатие кнопки от владельца: подпись и чат согласованы (иначе verify отклонит)."""
    return fakes.FakeCallback(_signed(kind, arg, chat_id=chat_id), chat_id=chat_id)


async def test_foreign_chat_is_ignored():
    bot, _, client, channels = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    message = fakes.FakeMsg("📰 Найти новости", chat_id="999999")
    await bot._on_message(message)
    assert client.sent == [] and channels.calls == 0, "чужой чат не должен ничего запускать"

    callback = fakes.FakeCallback(_signed(module.KIND_NEWS_RETRY, ""), chat_id="999999")
    await bot._on_callback(callback)
    assert channels.calls == 0


async def test_run_sends_card_per_news_with_two_buttons():
    bot, db, client, channels = _bot(agent_text=_agent_reply("https://a.invalid/1", "https://a.invalid/2"))
    result = await bot.run_once(chat_id=fakes.OWNER)
    assert result == {"status": "ok", "cards": 2, "dropped_seen": 0}
    cards = [item for item in client.sent if "📰 Новость" in item[1]]
    assert len(cards) == 2, "каждая новость — отдельное сообщение"
    for _, text, buttons in cards:
        labels = [button.text for row in buttons for button in row]
        assert labels == ["✏️ Редактировать", "📤 Опубликовать в канале"], "ровно две кнопки под карточкой"
    assert channels.strict_users == [True], "служебный прогон обязан спрашивать только свою сессию"
    assert len(db.news_drafts) == 2 and db.news_seen == {"https://a.invalid/1", "https://a.invalid/2"}
    assert all(draft["message_id"] for draft in db.news_drafts.values()), "карточкам запомнили сообщения"


async def test_dedup_between_runs():
    bot, db, client, _ = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    await bot.run_once(chat_id=fakes.OWNER)
    db.news_inflight = False      # гейт in-flight проверяется отдельно
    client.sent.clear()
    result = await bot.run_once(chat_id=fakes.OWNER)
    assert result == {"status": "no_new"}
    assert any("Нового нет" in text for _, text, _ in client.sent)


def test_search_is_done_by_the_agent_by_default():
    """Новости ИЩЕТ одиночный агент, а не команда ролей.

    Замер на одной выдаче: команда отвечает `NEWS_NONE` при десяти результатах поиска и `SEARCH_FAILED` при
    «только главные страницы и списки, не статьи», а тот же агент ТУ ЖЕ выдачу превращает в карточки
    (два живых прогона: 2 и 5 карточек). Команда остаётся на пути ПРАВКИ и остаётся рабочей ветвью —
    её покрывают тесты файла про команду, а смена решения — правка одной строки.
    """
    assert module.SEARCH_VIA == "agent", \
        "значение по умолчанию названо и обязано быть видно тестом, а не подразумеваться"


async def test_agent_is_used_by_default_and_the_team_is_not_called():
    """По умолчанию команду НЕ зовут вовсе — ищем агентом.

    Тест важен именно СЧЁТОМ обращений: без него «агент основной» осталось бы намерением в комментарии —
    ветвь команды жива, и её вызов возвращал бы `None` незаметно для результата.
    """
    calls: list[str] = []

    class Team:
        async def __call__(self, role_key: str, prompt: str):
            calls.append(role_key)
            raise AssertionError("команду не должны звать при SEARCH_VIA=agent")

    settings = fakes.make_settings(news_max_items=5)
    channels = fakes.ScriptedChannels(_agent_reply("https://a.invalid/1"))
    bot = fakes.make_news_bot(fakes.FakeDb(), settings=settings, channels=channels, team_ask=Team())
    client = fakes.fake_client()
    client.connected = True
    bot._client = client

    result = await bot.run_once(chat_id=fakes.OWNER)
    assert result["cards"] == 1, "карточку обязан дать агент"
    assert calls == [], "команда при SEARCH_VIA=agent не вызывается вовсе"


async def test_dead_run_offers_retry_and_rewrite():
    bot, _, client, _ = _bot(agent_text="Просто проза без строк")
    result = await bot.run_once(chat_id=fakes.OWNER)
    # Статус называет ПРИЧИНУ, а не «карточек нет»: проза одиночного агента — нарушенный формат.
    assert result == {"status": "format"}
    text, buttons = client.sent[-1][1], client.sent[-1][2]
    assert "не тем форматом" in text
    labels = [button.text for row in buttons for button in row]
    assert labels == ["🔁 Ещё раз", "🤖 Переписать по подсказке"]


async def test_inflight_and_daily_cap_are_reported():
    bot, db, _, _ = _bot()
    db.news_inflight = True
    assert (await bot.run_once(chat_id=fakes.OWNER))["status"] == "run_in_progress"
    db.news_inflight = False
    db.news_runs = 99
    assert (await bot.run_once(chat_id=fakes.OWNER))["status"] == "daily_cap"


async def test_external_entry_and_button_share_one_lock():
    """Ручка `/api/ops/news` и кнопка — ОДИН вход.

    Гейт в базе отбивает второй прогон, но у него есть окно (чтение `inflight` → запись `started`), а лок
    чата закрывает окно целиком. Занят — отдаём ТОТ ЖЕ статус, что гейт, и не ждём: ожидание съело бы
    таймаут вызывающего.
    """
    bot, _, _, channels = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    started, release = asyncio.Event(), asyncio.Event()
    runs = bot.run_once

    async def slow_run_once(*, chat_id: str, topic: str | None = None) -> dict:
        started.set()
        await release.wait()
        return await runs(chat_id=chat_id, topic=topic)

    bot.run_once = slow_run_once
    task = asyncio.create_task(bot.run_once_locked(chat_id=fakes.OWNER))
    await started.wait()

    busy = await bot.run_once_locked(chat_id=fakes.OWNER)
    release.set()
    done = await task

    assert busy == {"status": "run_in_progress"}
    assert done["status"] == "ok"
    assert channels.calls == 1, "второй вход не должен доходить до агента"


async def test_edit_flow_replaces_text_and_resends_card():
    bot, db, client, _ = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    await bot.run_once(chat_id=fakes.OWNER)
    draft_id = next(iter(db.news_drafts))
    card_message_id = db.news_drafts[draft_id]["message_id"]
    client.sent.clear()

    await bot._on_callback(_cb(module.KIND_NEWS_EDIT, str(draft_id)))
    assert db.news_drafts[draft_id]["status"] == "editing"
    assert "Пришлите новый текст" in client.sent[-1][1]

    await bot._on_message(fakes.FakeMsg("Совсем новый текст новости про рынок", chat_id=fakes.OWNER))
    draft = db.news_drafts[draft_id]
    assert draft["text"].startswith("Совсем новый") and draft["status"] == "draft"
    assert any("Совсем новый" in text for _, text, _ in client.sent), "карточка переотправлена с новым текстом"
    assert client.edited == [(int(fakes.OWNER), card_message_id, None)], "у старого сообщения кнопки сняты"


async def test_non_text_in_edit_mode_keeps_text():
    bot, db, client, _ = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    await bot.run_once(chat_id=fakes.OWNER)
    draft_id = next(iter(db.news_drafts))
    await bot._on_callback(_cb(module.KIND_NEWS_EDIT, str(draft_id)))
    before = db.news_drafts[draft_id]["text"]
    client.sent.clear()

    await bot._on_message(fakes.FakeMsg("", chat_id=fakes.OWNER))            # фото/пустой текст
    assert db.news_drafts[draft_id]["text"] == before
    assert "Пришлите новый текст" in client.sent[-1][1]


async def test_edit_ttl_expires_lazily():
    bot, db, _, _ = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    await bot.run_once(chat_id=fakes.OWNER)
    draft_id = next(iter(db.news_drafts))
    await bot._on_callback(_cb(module.KIND_NEWS_EDIT, str(draft_id)))
    db.news_editing_age_s = 10000          # правка «залипла» дольше TTL
    assert await db.news_editing_for(author_chat_id=fakes.OWNER, ttl_s=600) is None
    assert db.news_drafts[draft_id]["status"] == "draft", "истёкшую правку снимаем, а не держим вечно"


async def test_cancel_edit_button_clears_state():
    bot, db, client, _ = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    await bot.run_once(chat_id=fakes.OWNER)
    draft_id = next(iter(db.news_drafts))
    await bot._on_callback(_cb(module.KIND_NEWS_EDIT, str(draft_id)))
    await bot._on_callback(_cb(module.KIND_NEWS_CANCEL, str(draft_id)))
    assert db.news_drafts[draft_id]["status"] == "draft"
    assert "Правка отменена" in client.sent[-1][1]


async def test_publish_sends_to_channel_and_is_idempotent():
    bot, db, client, _ = _bot(agent_text=_agent_reply("https://a.invalid/1"),
                              extra_settings={"telegram_news_test_channel_id": "-1001234567890",
                                              "news_stale_h": 999})
    await bot.run_once(chat_id=fakes.OWNER)
    draft_id = next(iter(db.news_drafts))
    await bot._on_callback(_cb(module.KIND_NEWS_PUBLISH, str(draft_id)))

    channel_posts = [item for item in client.sent if item[0] == -1001234567890]
    assert len(channel_posts) == 1, "ровно один пост в канал"
    assert channel_posts[0][1].endswith("Источник: https://a.invalid/1")
    assert db.news_drafts[draft_id]["status"] == "published"
    assert db.news_published_log == [(draft_id, "-1001234567890", fakes.OWNER)]

    await bot._on_callback(_cb(module.KIND_NEWS_PUBLISH, str(draft_id)))
    assert len([item for item in client.sent if item[0] == -1001234567890]) == 1, "повтор не публикует второй раз"


class ChatWriteForbiddenError(RPCError):        # имя класса важно: по нему выбирается текст владельцу
    def __init__(self) -> None:
        super().__init__(request=None, message="forbidden")


async def test_forbidden_publish_returns_draft_and_explains():
    bot, db, client, _ = _bot(agent_text=_agent_reply("https://a.invalid/1"),
                              extra_settings={"telegram_news_test_channel_id": "-1001234567890",
                                              "news_stale_h": 999})
    await bot.run_once(chat_id=fakes.OWNER)
    draft_id = next(iter(db.news_drafts))
    client.send_raises_for = -1001234567890          # падает только публикация в канал
    client.send_raises = ChatWriteForbiddenError()
    await bot._on_callback(_cb(module.KIND_NEWS_PUBLISH, str(draft_id)))
    assert db.news_drafts[draft_id]["status"] == "draft", "ошибка канала не теряет новость"
    assert any("администратором" in text for _, text, _ in client.sent)


async def test_stale_news_asks_confirmation_then_publishes_on_confirm():
    bot, db, client, _ = _bot(agent_text=_agent_reply("https://a.invalid/1"),
                              extra_settings={"telegram_news_test_channel_id": "-1001234567890",
                                              "news_stale_h": 0})
    await bot.run_once(chat_id=fakes.OWNER)
    draft_id = next(iter(db.news_drafts))
    client.sent.clear()

    await bot._on_callback(_cb(module.KIND_NEWS_PUBLISH, str(draft_id)))
    assert not [item for item in client.sent if item[0] == -1001234567890], "без подтверждения не публикуем"
    assert "Публикуем?" in client.sent[-1][1]

    await bot._on_callback(_cb(module.KIND_NEWS_CONFIRM, str(draft_id)))
    assert len([item for item in client.sent if item[0] == -1001234567890]) == 1
    assert db.news_drafts[draft_id]["status"] == "published"


async def test_dry_run_does_not_publish():
    bot, db, client, _ = _bot(agent_text=_agent_reply("https://a.invalid/1"),
                              extra_settings={"telegram_news_test_channel_id": "-1001234567890",
                                              "news_stale_h": 999, "news_publish_dry_run": True})
    await bot.run_once(chat_id=fakes.OWNER)
    draft_id = next(iter(db.news_drafts))
    await bot._on_callback(_cb(module.KIND_NEWS_PUBLISH, str(draft_id)))
    assert not [item for item in client.sent if item[0] == -1001234567890]
    assert db.news_drafts[draft_id]["status"] == "draft", "dry-run возвращает черновик в работу"
    assert any("Dry-run" in text for _, text, _ in client.sent)


async def test_rewrite_prompt_replaces_text_via_agent():
    bot, db, client, channels = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    await bot.run_once(chat_id=fakes.OWNER)
    draft_id = next(iter(db.news_drafts))
    channels.text = "Переписанный агентом текст новости целиком"
    client.sent.clear()

    await bot._on_callback(_cb(module.KIND_NEWS_REWRITE, str(draft_id)))
    assert "подсказку" in client.sent[-1][1]
    await bot._on_message(fakes.FakeMsg("сделай короче", chat_id=fakes.OWNER))
    assert db.news_drafts[draft_id]["text"].startswith("Переписанный")
    assert any("Переписанный" in text for _, text, _ in client.sent)


async def test_reason_of_an_empty_answer_is_named_not_a_generic_lie():
    """Владельцу называется ПРИЧИНА: «поиск не сработал» и «не тот формат» — разные состояния.

    Одна строка «агент не отдал карточек» на все состояния называет не то, что случилось: при
    работающем поиске, вернувшем три десятка результатов, владелец прочитает про агента, и причина
    отказа останется неизвестной.
    """
    failed, _, client, _ = _bot(agent_text=f"{news.SEARCH_FAILED}: searxng_web_search не сработал")
    await failed._on_message(fakes.FakeMsg(module.MENU_LABEL, chat_id=fakes.OWNER))
    said = " ".join(text for _, text, _ in client.sent)
    assert "поиск не сработал" in said, "отказ ИСТОЧНИКА обязан называться отказом источника"
    assert "не отдал карточек" not in said

    prose, _, prose_client, _ = _bot(agent_text="Проза без служебных строк, которой парсер не место")
    await prose._on_message(fakes.FakeMsg(module.MENU_LABEL, chat_id=fakes.OWNER))
    said_prose = " ".join(text for _, text, _ in prose_client.sent)
    assert "не тем форматом" in said_prose, "нарушенный контракт роли — это НЕ «новостей нет»"


async def test_menu_button_and_unknown_text():
    bot, _, client, _ = _bot()
    await bot._on_message(fakes.FakeMsg("/start", chat_id=fakes.OWNER))
    assert "Бот новостей" in client.sent[-1][1]
    keyboard = client.sent[-1][2]
    assert keyboard[0][0].button.text == module.MENU_LABEL, "reply-клавиатура с единственным пунктом меню"
    await bot._on_message(fakes.FakeMsg("привет", chat_id=fakes.OWNER))
    assert "Я умею одно" in client.sent[-1][1]


async def test_menu_label_and_news_command_start_a_run():
    """Метка reply-клавиатуры и `/news` ЗАПУСКАЮТ прогон.

    Дефект, который этот тест и закрывает. Пункт reply-клавиатуры присылает ТЕКСТ метки, поэтому «нажатие
    кнопки» и «набранный текст» для бота неразличимы, а один общий набор команд уводил ОБА текста в показ
    меню — то есть нажатие «📰 Найти новости» показывало меню СНОВА, и первый прогон запустить было нечем
    (`_run_locked` звался только кнопкой «🔁 Ещё раз» ПОСЛЕ прогона). Теста на нажатие метки не было.
    """
    bot, _, client, channels = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    await bot._on_message(fakes.FakeMsg(module.MENU_LABEL, chat_id=fakes.OWNER))
    assert channels.calls, "метка кнопки обязана запускать прогон, а не показывать меню"
    assert any("Ищу новости" in text for _, text, _ in client.sent), "прогон обязан сразу отозваться"

    other, _, other_client, other_channels = _bot(agent_text=_agent_reply("https://a.invalid/2"))
    await other._on_message(fakes.FakeMsg("/news", chat_id=fakes.OWNER))
    assert other_channels.calls, "`/news` — та же команда прогона"
    assert any("Ищу новости" in text for _, text, _ in other_client.sent)


async def test_start_shows_the_menu_without_spending_a_run():
    """`/start` остаётся показом меню: одна команда — одно дело, и прогон он не тратит."""
    bot, _, client, channels = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    await bot._on_message(fakes.FakeMsg("/start", chat_id=fakes.OWNER))
    assert "Бот новостей" in client.sent[-1][1]
    assert not channels.calls, "показ меню не должен запускать прогон"


def test_callback_kinds_are_all_named_for_the_log():
    """Каждый вид нажатия назван словом: иначе в журнале появится код, и разбор снова ослепнет.

    Строка в журнале — единственный след нажатия: callback приходит своим обработчиком, и без неё
    «не нажимали» и «нажали, а подпись не сошлась» неотличимы, и разбор упирается в догадку.
    """
    declared = {value for name, value in vars(module).items() if name.startswith("KIND_NEWS_")}
    assert declared <= set(module._CALLBACK_KINDS), \
        f"виды без имени для журнала: {declared - set(module._CALLBACK_KINDS)}"


async def test_unsigned_callback_is_rejected():
    bot, db, _, _ = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    await bot.run_once(chat_id=fakes.OWNER)
    draft_id = next(iter(db.news_drafts))
    callback = fakes.FakeCallback("c:npb:1:deadbeef", chat_id=fakes.OWNER)
    await bot._on_callback(callback)
    assert callback.answers and "устарела" in (callback.answers[-1] or ""), "битая подпись — отказ"
    assert db.news_drafts[draft_id]["status"] == "draft"


async def test_health_reports_state():
    bot, _, _, _ = _bot()
    health = bot.health()
    assert health["status"] == "ok" and health["configured"] is False and health["telegram"] is True


async def test_health_follows_connection_not_a_flag():
    """`/health.telegram` отражает ЖИВОЕ соединение, а не факт «первого подключения».

    Отдельный флаг, ставящийся только на старте, оставляет бота, подключившегося в фоне после обрыва
    (после перезагрузки хоста это штатный путь), неподключённым в `/health`, и сторож шлёт ложную
    тревогу при работающем боте. Проверяются оба направления: подключён → true, оборван → false.
    """
    bot, _, _, _ = _bot()

    class C:
        def __init__(self) -> None:
            self.connected = False

        def is_connected(self) -> bool:
            return self.connected

    client = C()
    bot._connection.client = client

    client.connected = True
    assert bot.health()["telegram"] is True, "подключён — здоровье обязано быть зелёным"
    client.connected = False
    assert bot.health()["telegram"] is False, "обрыв — здоровье обязано быть красным"


class _FakeStartClient:
    """Клиент для проверки СТАРТА бота: регистрация обработчиков, `running`, отключение (без сети).

    Контракт повторяет то, чем пользуется ОБЩИЙ механизм подключения (`agent_core.tg.connection`):
    `connect`, `is_connected`, `is_user_authorized`. Без них двойник не похож на настоящий клиент,
    и проверка старта падала бы на отсутствии метода, а не на поведении бота.
    """

    def __init__(self) -> None:
        self.handlers: list[tuple] = []
        self.disconnected = False
        self.connected = False

    def add_event_handler(self, *args) -> None:
        self.handlers.append(args)

    async def connect(self) -> None:
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected

    async def is_user_authorized(self) -> bool:
        return True

    async def start(self, **kwargs) -> None:
        self.connected = True

    async def get_me(self):
        from types import SimpleNamespace

        return SimpleNamespace(username="news_test_bot")

    async def get_entity(self, target):
        raise ValueError("нет канала")

    async def disconnect(self) -> None:
        self.disconnected = True


async def test_start_registers_handlers_and_marks_running(monkeypatch):
    """Старт не падает из-за сборки клиента: клиент приходит из общего механизма, а не собирается здесь."""
    fake = _FakeStartClient()
    # Сборка клиента живёт в ОБЩЕМ механизме (`agent_core.tg.connection`) — второй вход обращения
    # запрещён архитектурной проверкой, поэтому и подменяем там же.
    import agent_core.tg.connection as connection_module

    monkeypatch.setattr(connection_module, "build_client", lambda settings, **kwargs: fake)
    bot, _, client, _ = _bot()
    client.connected = False                                  # проверяем именно эффект start()
    bot._settings.telegram_news_bot_token = "123456:TEST"     # без токена start() выходит мягко
    await bot.start()
    assert bot.running is True, "бот должен подниматься на живом клиенте"
    assert len(fake.handlers) == 2, "оба обработчика зарегистрированы"
    await bot.stop()
    assert fake.disconnected is True


async def test_channel_message_is_ignored_but_logged(caplog):
    """Канал/группа: не отвечаем, но пишем id в лог — так владелец узнаёт -100… для .env."""
    import logging

    bot, _, client, channels = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    with caplog.at_level(logging.INFO, logger="news_agent.tg.news_bot"):
        await bot._on_message(fakes.FakeMsg("пост в канале", chat_id="-1001234567890"))
    assert client.sent == [] and channels.calls == 0, "на сообщения из канала бот не реагирует"
    assert any("-1001234567890" in record.message for record in caplog.records), "id канала попадает в лог"


# ---------- лимиты расхода (один прогон ≈ $0.95) ----------


def test_default_limits_are_money_first():
    from news_agent.settings import Settings

    assert Settings.model_fields["news_max_per_day"].default == 5, "потолок 30 прогонов ≈ $28/сутки"
    assert Settings.model_fields["news_month_budget_usd"].default == 5.0


async def test_month_budget_gate_stops_run_with_money_message():
    db = fakes.FakeDb()
    db.news_month_cost_usd = 5.12          # перерасход: последний прогон вывел месяц за лимит
    bot, _, client, channels = _bot(db=db, extra_settings={"news_month_budget_usd": 5.0})
    assert bot._db is db, "гейт проверяем на подготовленной базе"
    result = await bot.run_once(chat_id=fakes.OWNER)
    assert result["status"] == "month_budget:5.12"
    assert channels.calls == 0, "при исчерпанном бюджете агент не вызывается"
    assert "Месячный лимит расходов на новости исчерпан: $5.12 из $5.00" in client.sent[-1][1]


# ---------- кнопки-подсказки под «Напишите подсказку…» ----------


async def test_rewrite_prompt_offers_hint_buttons():
    bot, db, client, _ = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    await bot.run_once(chat_id=fakes.OWNER)
    draft_id = next(iter(db.news_drafts))
    client.sent.clear()

    await bot._on_callback(_cb(module.KIND_NEWS_REWRITE, str(draft_id)))
    text, buttons = client.sent[-1][1], client.sent[-1][2]
    labels = [button.text for row in buttons for button in row]
    assert labels == [label for label, _ in news.PRESET_HINTS] + ["✖️ Отмена"]
    assert "выберите готовую" in text


async def test_each_hint_button_sends_its_preset_prompt():
    """Тап по подсказке == набор той же подсказки руками: агенту уходит ровно текст из PRESET_HINTS."""
    for index, (label, preset) in enumerate(news.PRESET_HINTS):
        bot, db, _, channels = _bot(agent_text=_agent_reply("https://a.invalid/1"))
        await bot.run_once(chat_id=fakes.OWNER)
        draft_id = next(iter(db.news_drafts))
        channels.text = f"Переписанный текст новости по подсказке {label}"
        channels.prompts.clear()

        await bot._on_callback(_cb(module.KIND_NEWS_REWRITE, str(draft_id)))
        await bot._on_callback(_cb(module.KIND_NEWS_HINT, str(index)))
        assert preset in channels.prompts[-1], f"подсказка {label} ушла агенту дословно"
        assert db.news_drafts[draft_id]["text"].startswith("Переписанный")


async def test_details_hint_demands_keeping_facts():
    """«Подробнее» обязано требовать сохранения фактов."""
    label, prompt = news.PRESET_HINTS[0]
    assert "Подробнее" in label and "Сохрани ВСЕ факты" in prompt


def test_hint_callbacks_fit_telegram_limit():
    for index in range(len(news.PRESET_HINTS)):
        assert len(_signed(module.KIND_NEWS_HINT, str(index)).encode()) <= tokens.MAX_CALLBACK_BYTES


async def test_hint_without_active_rewrite_mode_is_rejected():
    """Тап «из прошлого» (состояние истекло): честный текст и НИ ОДНОГО вызова агента."""
    bot, _, client, channels = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    await bot._on_callback(_cb(module.KIND_NEWS_HINT, "0"))
    assert channels.calls == 0
    assert "устарела" in client.sent[-1][1]


# ---------- защита от потери фактов ----------

_FACTS_TEXT = "Цена выросла на 18,5% и составила 12 400 000 ₽ за 76 м²."


async def test_rewrite_rejected_when_agent_loses_facts():
    bot, db, client, channels = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    await bot.run_once(chat_id=fakes.OWNER)
    draft_id = next(iter(db.news_drafts))
    db.news_drafts[draft_id]["text"] = _FACTS_TEXT
    channels.text = "Цены заметно выросли, спрос остаётся высоким на рынке недвижимости."
    client.sent.clear()

    await bot._on_callback(_cb(module.KIND_NEWS_REWRITE, str(draft_id)))
    await bot._on_callback(_cb(module.KIND_NEWS_HINT, "0"))

    assert db.news_drafts[draft_id]["text"] == _FACTS_TEXT, "текст с потерянными фактами не применяем"
    assert "потерял факты" in client.sent[-1][1] and "18,5%" in client.sent[-1][1]


async def test_hint_works_again_after_rejection():
    """После отказа владелец жмёт другую подсказку БЕЗ повторного «Переписать» — режим остаётся живым."""
    bot, db, client, channels = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    await bot.run_once(chat_id=fakes.OWNER)
    draft_id = next(iter(db.news_drafts))
    db.news_drafts[draft_id]["text"] = _FACTS_TEXT
    await bot._on_callback(_cb(module.KIND_NEWS_REWRITE, str(draft_id)))

    channels.text = "Короткий текст без единой цифры, чтобы сработал гейт фактов."
    await bot._on_callback(_cb(module.KIND_NEWS_HINT, "1"))
    assert "потерял факты" in client.sent[-1][1]

    channels.text = "Цена выросла на 18,5%, это 12 400 000 ₽ за 76 м²."
    await bot._on_callback(_cb(module.KIND_NEWS_HINT, "0"))
    assert db.news_drafts[draft_id]["text"].startswith("Цена выросла"), "второй тап дошёл до агента"


async def test_rewrite_applied_when_all_facts_kept():
    bot, db, _, channels = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    await bot.run_once(chat_id=fakes.OWNER)
    draft_id = next(iter(db.news_drafts))
    db.news_drafts[draft_id]["text"] = _FACTS_TEXT
    channels.text = ("Подробнее: цена выросла на 18,5% и составила 12 400 000 ₽ за 76 м², "
                     "спрос на такие лоты сохраняется.")
    await bot._on_callback(_cb(module.KIND_NEWS_REWRITE, str(draft_id)))
    await bot._on_callback(_cb(module.KIND_NEWS_HINT, "0"))
    assert db.news_drafts[draft_id]["text"].startswith("Подробнее"), "факты на месте — правка применяется"
    assert db.news_drafts[draft_id]["status"] == "draft"


async def test_rewrite_turn_carries_fact_rule_and_url():
    bot, db, _, channels = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    await bot.run_once(chat_id=fakes.OWNER)
    draft_id = next(iter(db.news_drafts))
    channels.text = "Короткий переписанный текст новости про рынок недвижимости"
    channels.prompts.clear()
    await bot._on_callback(_cb(module.KIND_NEWS_REWRITE, str(draft_id)))
    await bot._on_message(fakes.FakeMsg("короче", chat_id=fakes.OWNER))
    turn = channels.prompts[-1]
    assert "ЖЁСТКОЕ ПРАВИЛО О ФАКТАХ" in turn, "инструкция должна запрещать выбрасывать факты"
    assert db.news_drafts[draft_id]["url"] in turn, "ссылку агенту передаём (подсказка «С источником»)"


async def test_rewrite_outcome_is_named_in_the_log(caplog):
    """Исход переписывания ОБЯЗАН быть назван в журнале — иначе успех неотличим от «не дошло».

    Без записи в журнале нет НИ ОДНОЙ
    строки — исход приходится восстанавливать по отметке времени в базе (`news_drafts.updated_at`).
    Тест держит обе стороны, как логирование входящего и нажатий: успех («применено», с размерами
    «до → после» и ИСТОЧНИКОМ — команда это или одиночный агент) и отказ («отброшено», с причиной).
    """
    bot, db, _, channels = _bot(agent_text=_agent_reply("https://a.invalid/1"))
    await bot.run_once(chat_id=fakes.OWNER)
    draft_id = next(iter(db.news_drafts))
    db.news_drafts[draft_id]["text"] = _FACTS_TEXT
    await bot._on_callback(_cb(module.KIND_NEWS_REWRITE, str(draft_id)))
    channels.text = ("Подробнее: цена выросла на 18,5% и составила 12 400 000 ₽ за 76 м², "
                     "спрос на такие лоты сохраняется.")
    with caplog.at_level(logging.INFO, logger=module.log.name):
        await bot._on_callback(_cb(module.KIND_NEWS_HINT, "0"))
    said = [record.getMessage() for record in caplog.records]
    applied = [line for line in said if "переписывание применено" in line]
    assert applied, "успешное переписывание не оставило следа в журнале"
    assert "источник: агент" in applied[-1], "путь не назван: команда это или одиночный агент"
    assert "→" in applied[-1], "размеры «до → после» не записаны"

    caplog.clear()
    await bot._on_callback(_cb(module.KIND_NEWS_REWRITE, str(draft_id)))
    channels.text = "Мало текста."
    with caplog.at_level(logging.WARNING, logger=module.log.name):
        await bot._on_callback(_cb(module.KIND_NEWS_HINT, "1"))
    said = [record.getMessage() for record in caplog.records]
    dropped = [line for line in said if "переписывание отброшено" in line]
    assert dropped, "отказ переписывания не оставил следа в журнале"
    assert "короткий ответ" in dropped[-1], "отказ по длине не назвал причину"


# --- ГРОМКАЯ ГРАНИЦА ОТКАЗОВ (до́лг 9) -------------------------------------------------------------
#
# **Что было.** Исключение внутри обработки входящего уносило обработчик: бот продолжал работать, но на
# сообщения владельца больше не отвечал. След оставался только в журнале контейнера, которого никто не
# читает ⇒ снаружи это «бот молчит», то есть тихая деградация.
#
# **Что доказывает тест.** `_on_message` не выпускает исключение НАРУЖУ и сообщает владельцу двумя путями:
# обычным чатом и ручкой алертов моста. На коде ДО правки первая же проверка падала бы: там `_on_message`
# был самим обработчиком, и исключение уходило в Telethon.

async def test_handler_failure_is_loud_not_silent(monkeypatch, caplog):
    bot, _db, _client, _channels = _bot()
    sent: list[str] = []
    alerted: list[str] = []

    async def boom(_event) -> None:
        raise RuntimeError("поломка внутри обработки")

    async def record_send(chat_id: str, text: str, **_kwargs) -> None:
        sent.append(text)

    async def record_alert(text: str) -> None:
        alerted.append(text)

    monkeypatch.setattr(bot, "_handle_message", boom)
    monkeypatch.setattr(bot, "_send", record_send)
    monkeypatch.setattr(bot, "_alert_bridge", record_alert)

    event = fakes.FakeMsg(module.MENU_LABEL, chat_id=fakes.OWNER)
    with caplog.at_level(logging.ERROR):
        await bot._on_message(event)          # НЕ должно бросить: граница обязана поймать

    assert any("ОШИБКА обработки" in record.message for record in caplog.records), \
        "отказ обязан быть виден в журнале (иначе разбор идёт вслепую)"
    assert sent and "Внутренняя ошибка" in sent[0], "владельцу обязано уйти сообщение словами"
    assert alerted, "последний рубеж (ручка алертов моста) обязан быть вызван"


async def test_handler_failure_alerts_bridge_when_chat_is_dead(monkeypatch):
    """Если и чат не отвечает, рубеж всё равно должен сработать — иначе отказ снова тихий."""
    bot, _db, _client, _channels = _bot()
    alerted: list[str] = []

    async def boom(_event) -> None:
        raise RuntimeError("поломка")

    async def dead_send(*_args, **_kwargs) -> None:
        raise RuntimeError("канал тоже мёртв")

    async def record_alert(text: str) -> None:
        alerted.append(text)

    monkeypatch.setattr(bot, "_handle_message", boom)
    monkeypatch.setattr(bot, "_send", dead_send)
    monkeypatch.setattr(bot, "_alert_bridge", record_alert)

    await bot._on_message(fakes.FakeMsg("привет", chat_id=fakes.OWNER))
    assert alerted, "отказ чата не отменяет алерт через мост"
