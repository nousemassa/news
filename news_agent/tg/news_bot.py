"""Telegram-бот новостей: кнопка → агент → карточки с «Редактировать»/«Опубликовать».

Почему отдельный лёгкий транспорт, а не `TelegramTransport`: тот держит клиентский путь (согласие,
брони, меню менеджеров) и при старте поднимает петли напоминаний/календаря — в новостном процессе
это дало бы дубли уведомлений. Здесь только овнер и новости; общие слои (`ui/*`, `tg/mtproto`,
`channels/client`, `db/*`, `news.py`) переиспользуются как есть.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from datetime import UTC, datetime

from telethon import events
from telethon.errors import RPCError

from agent_core.decisions.ladder import DEFAULT_THRESHOLD
from agent_core.tg.connection import TelegramConnection
from agent_core.tg.render import ButtonPolicy, to_buttons
from agent_core.ui import tokens
from agent_core.ui.reply import INLINE, REPLY, Action, Reply
from news_agent import news
from news_agent.news_team.errors import TeamUnavailable
from news_agent.news_team.run import Search as TeamSearch
from news_agent.news_team.run import run_once as team_run_once
from news_agent.news_team.run import search_news

log = logging.getLogger(__name__)

MENU_LABEL = "📰 Найти новости"
#: Показ меню и ЗАПУСК прогона — РАЗНЫЕ действия, и один общий набор их смешивал: пункт reply-клавиатуры
#: присылает ТЕКСТ метки, а обработчик на этот текст показывал меню СНОВА. Из-за этого первый прогон
#: запустить было нечем — `_run_locked` звался только кнопкой «🔁 Ещё раз» ПОСЛЕ прогона. Документация
#: требует обратного («📰 Найти новости» → карточки), поэтому наборы разведены.
MENU_SHOW = {"/start", "/menu"}
MENU_RUN = {MENU_LABEL, "/news"}
DEFAULT_TOPIC = "свежие новости рынка недвижимости Москвы"
TEXT_MAX = 3800              # предел текста правки (точный лимит Telegram считаем при отправке)
PROMPT_TTL_S = 900           # сколько ждём подсказку для «Переписать»
SEND_PAUSE_S = 0.3           # пауза между карточками — лимиты Telegram
NETWORK_ERRORS = (RPCError, ConnectionError, OSError, asyncio.TimeoutError)

#: Кто ИЩЕТ новости: `agent` — одиночный агент платформы, `team` — команда ролей.
#:
#: **Ищет АГЕНТ.** На замере команда не берёт выдачу нашего поиска
#: (роль отвечает `NEWS_NONE` при десяти результатах и `SEARCH_FAILED` с объяснением «только главные
#: страницы и списки, не статьи»), а тот же агент ТУ ЖЕ выдачу превращает в карточки (два живых прогона:
#: 2 и 5). Команда остаётся на пути ПРАВКИ (`_rewrite_via_team`) — там она отрабатывает.
#:
#: **Почему строкой, а не настройкой.** Настройка живёт в общем слое (`agent_core`), и её добавление
#: тянет за собой колесо, пины и пересборку образов — цена несоразмерна одному значению. Смена решения
#: — правка этой строки; тесты команды выставляют её себе сами.
SEARCH_VIA = "agent"

# Виды действий новостного бота живут здесь, а не в общем `ui/reply.py`.
#: Виды кнопок, допустимые в reply-раскладке новостного бота: пункт меню.
MENU_KIND = "m"
NEWS_BUTTON_POLICY = ButtonPolicy(reply_kinds=frozenset({MENU_KIND}))

KIND_NEWS_EDIT = "ned"       # arg = draft_id: прислать новый текст
KIND_NEWS_PUBLISH = "npb"    # arg = draft_id: опубликовать в канал
KIND_NEWS_CONFIRM = "npc"    # arg = draft_id: подтвердить публикацию устаревшей новости
KIND_NEWS_CANCEL = "nca"     # arg = draft_id: выйти из режима правки
KIND_NEWS_RETRY = "nrt"      # arg = "": повторить прогон
KIND_NEWS_REWRITE = "nrw"    # arg = draft_id|"0": переписать по подсказке
KIND_NEWS_HINT = "nhp"       # arg = индекс в news.PRESET_HINTS: готовая подсказка

#: Виды нажатий СЛОВАМИ — для журнала. Нажатие кнопки иначе не видно вообще: callback приходит в
#: свой обработчик, и отличить «не нажимали» от «нажали, а подпись не сошлась» было нечем (разбор
#: дефекта «кнопка показывает меню» шёл вслепую). Тест сверяет, что каждый `KIND_NEWS_*` назван.
_CALLBACK_KINDS = {
    KIND_NEWS_EDIT: "правка",
    KIND_NEWS_PUBLISH: "публикация",
    KIND_NEWS_CONFIRM: "подтверждение публикации",
    KIND_NEWS_CANCEL: "отмена",
    KIND_NEWS_RETRY: "прогон",
    KIND_NEWS_REWRITE: "переписать по подсказке",
    KIND_NEWS_HINT: "готовая подсказка",
}

#: Причина отказа прогона СЛОВАМИ — ОДНО место: статус парсера (`news.parse_news_blocks`) → текст
#: владельцу. Прежде все состояния давали одну строку «агент не отдал карточек», и сообщение называло
#: НЕ то, что случилось: «поиск не сработал (отказ источника)», «поиск сработал, новостей нет» и
#: «ответ не тем форматом» лечатся по-разному (два состояния — два маркера).
DEAD_RUN_REASONS = {
    "search_failed": "поиск не сработал (отказ источника) — это не «новостей нет»",
    "none": "поиск сработал, новостей не нашлось",
    "format": "агент ответил не тем форматом (нужны строки NEWS|)",
    "empty": "агент не ответил",
}

# Подключение к Telegram — ОБЩИЙ механизм (`agent_core.tg.connection`): массив прокси, файл-сессия,
# ожидание готовности и сторож соединения. Своя сборка клиента здесь — второй вход обращения к
# тому же механизму, и он расходится с мостом: однократная попытка не переживает перезагрузку хоста,
# и бот не публикует, пока его не поднимут руками.

WRITE_ERRORS = ("ChatWriteForbiddenError", "ChatAdminRequiredError")
PEER_ERRORS = ("ChannelPrivateError", "ChannelInvalidError", "PeerIdInvalidError", "UsernameInvalidError")


class NewsBot:
    def __init__(self, *, settings, db, channels, retriever=None, gate_ask=None,
                 gate_threshold: float = DEFAULT_THRESHOLD, team_ask=None) -> None:
        self._settings = settings
        self._db = db
        self._channels = channels
        # Контур команды ролей: ретривер для поиска по базе знаний, шов гейта новости
        # и, в тестах, шов модели. Без них бот работает одиночным путём.
        self._retriever = retriever
        self._gate_ask = gate_ask
        self._gate_threshold = gate_threshold
        self._team_ask = team_ask
        self._connection = TelegramConnection(
            settings=settings, name="новости", session_name="news",
            token=getattr(settings, "telegram_news_bot_token", "") or "",
        )
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.stats = {"runs": 0, "cards": 0, "published": 0, "last_error": ""}

    @property
    def running(self) -> bool:
        """Бот принимает сообщения прямо сейчас: ЖИВОЕ состояние соединения.

        Не флаг «первого подключения»: отдельный флаг, ставящийся только на старте, оставлял бота,
        подключившийся в фоне после обрыва, в `/health` оставался неподключённым, и сторож слал ложную
        тревогу при работающем боте. Обратная ошибка того же класса: при обрыве флаг не сбрасывался, и
        здоровье врало в другую сторону. Один источник истины: механизм соединения.
        """
        return self._connection.running

    # ---------- жизнь ----------

    @property
    def _client(self):
        return self._connection.client

    @_client.setter
    def _client(self, value) -> None:
        self._connection.client = value

    async def start(self) -> None:
        # Залипшие `publishing` (процесс упал между «взял в работу» и подтверждением) возвращаем в
        # черновики ДО приёма сообщений: иначе новость навсегда остаётся «Публикуется…», и ни один
        # повтор её уже не возьмёт.
        released = await self._db.release_stale_publishing()
        if released:
            log.warning("news: возвращено в черновики залипших публикаций: %s", released)
        token = self._settings.telegram_news_bot_token
        if not token:
            log.warning("news: TELEGRAM_NEWS_BOT_TOKEN не задан — бот выключен (процесс жив, /health отвечает)")
            return
        # Отдельная ветка для невалидного токена остаётся: «сеть моргает» и «токен не принят» — разные
        # причины, и вторая не лечится повторами. Механизм ждёт готовности маршрута и держит соединение.
        try:
            await self._connection.start(self._on_connected)
        except ValueError as error:
            log.error("news: Telegram не принял токен (%s) — бот не подключён", type(error).__name__)
            self.stats["last_error"] = f"start:{type(error).__name__}"

    async def _on_connected(self, first: bool) -> None:
        """Что делаем после (пере)подключения: обработчики ставим один раз, канал проверяем всегда."""
        client = self._connection.client
        if client is None:
            return
        if not first:
            log.warning("news: соединение восстановлено, публикация снова доступна")
            await self._check_channel()
            return
        client.add_event_handler(self._on_message, events.NewMessage(incoming=True))
        client.add_event_handler(self._on_callback, events.CallbackQuery())
        log.info("news: бот @%s запущен через MTProto",
                 getattr(self._connection.me, "username", "?"))
        await self._check_channel()

    async def stop(self) -> None:
        await self._connection.stop()

    def health(self) -> dict:
        return {
            "status": "ok",
            "telegram": self.running,
            "transport": "mtproto",
            "configured": bool(self._settings.telegram_news_bot_token),
            "channel": bool(self._settings.telegram_news_channel_id or self._settings.telegram_news_test_channel_id),
            **self.stats,
        }

    async def _check_channel(self) -> None:
        """Предупредить владельца заранее, если канал недоступен (иначе узнает при первой публикации)."""
        channel = self._channel_id()
        if not self._client or not channel:
            return
        try:
            entity = await self._client.get_entity(int(channel))
        except (*NETWORK_ERRORS, ValueError) as error:
            log.warning("news: канал %s недоступен (%s)", channel[-4:], type(error).__name__)
            await self._send(self._owner(), f"⚠️ Канал недоступен ({type(error).__name__}): проверьте "
                                           "TELEGRAM_NEWS_CHANNEL_ID и права бота.")
            return
        log.info("news: канал на месте (%s)", type(entity).__name__)

    def _channel_id(self) -> str:
        return self._settings.telegram_news_test_channel_id or self._settings.telegram_news_channel_id

    def _owner(self) -> str:
        return self._settings.news_owner_chat_id

    def _is_owner(self, chat_id: str) -> bool:
        """Свой чат владельца (может отличаться от основного бота) — `resolve_role` тут не нужен."""
        owner = self._owner()
        return bool(owner) and str(chat_id) == str(owner)

    def _topic(self) -> str:
        return (self._settings.news_topic or "").strip() or DEFAULT_TOPIC

    # ---------- отправка ----------

    async def _send(self, chat_id: str, text: str, *, rows=None, kind: str = REPLY):
        if not self._client:
            return None
        buttons = None
        if rows:
            try:
                buttons = to_buttons(Reply(text=text, rows=rows, kind=kind), secret=self._settings.ui_secret,
                                policy=NEWS_BUTTON_POLICY,
                                     chat_id=str(chat_id), token="")
            except ValueError as error:  # подпись >64 байт: карточку отправляем БЕЗ кнопок
                log.error("news: кнопки не подписаны (%s) — отправляю без них", error)
                buttons = None
        try:
            return await self._client.send_message(int(chat_id), text, buttons=buttons)
        except NETWORK_ERRORS as error:
            log.warning("news: сообщение не отправлено (%s)", type(error).__name__)
            return None

    async def _menu(self, chat_id: str) -> None:
        rows = [[Action(MENU_LABEL, "m", "news")]]
        await self._send(chat_id, "📰 Бот новостей. Нажмите кнопку — соберу свежие новости и покажу карточками: "
                                  "каждую можно отредактировать или опубликовать в канале.", rows=rows, kind=REPLY)

    # ---------- входящие ----------

    async def _on_message(self, event) -> None:
        """Обработчик входящего — и ГРОМКАЯ граница отказов.

        **Зачем обёртка.** Исключение внутри обработки уносит обработчик целиком: бот
        ПРОДОЛЖАЕТ работать, но на сообщения владельца больше не отвечает, а след остаётся только в
        журнале контейнера (который никто не читает). Снаружи это выглядит как «бот молчит» — то есть
        как тихая деградация, самая дорогая разновидность отказа.
        """
        try:
            await self._handle_message(event)
        except Exception as error:
            log.exception("news: ОШИБКА обработки сообщения владельца — бот не молчит")
            await self._report_failure(event, error)

    async def _report_failure(self, event, error: Exception) -> None:
        """Сказать владельцу СЛОВАМИ. Два пути: обычный чат и мост (он же канал алертов сторожей)."""
        chat_id = str(getattr(event, "chat_id", "") or "")
        text = (f"Внутренняя ошибка бота новостей: {type(error).__name__}. "
                f"Я продолжаю работать; подробности — в журнале контейнера.")
        if chat_id:
            try:
                await self._send(chat_id, text)
            except Exception:
                log.exception("news: не смог сообщить владельцу через чат")
        await self._alert_bridge(text)

    async def _alert_bridge(self, text: str) -> None:
        """Последний рубеж: ручка алертов моста — тот же канал, куда шлют сторожа воркспейса.

        ❗ Адрес задан ЯВНО (`BRIDGE_ALERT_URL`, имя службы в общей сети), а не выведен из другого
        адреса: внутри контейнера `127.0.0.1` — это он сам, и «удобное умолчание» здесь уже давало
        молчаливый отказ.
        """
        key = self._settings.bridge_api_key
        url = os.environ.get("BRIDGE_ALERT_URL", "")
        if not key or not url:
            log.error("news: алерт не отправлен — нет BRIDGE_ALERT_URL или BRIDGE_API_KEY")
            return
        try:
            import httpx

            async with httpx.AsyncClient(timeout=8.0) as client:
                await client.post(f"{url.rstrip('/')}/api/ops/alert",
                                  headers={"X-API-Key": key}, json={"text": text})
        except Exception:
            log.exception("news: алерт через мост не ушёл")

    async def _handle_message(self, event) -> None:
        chat_id = str(event.chat_id)
        if not self._is_owner(chat_id):
            # Канал/группа, где бот админ: печатаем id один раз — так владелец узнаёт `-100…` для .env.
            if chat_id.startswith("-"):
                log.info("news: сообщение из канала/группы %s проигнорировано (не чат владельца) — этот id годится для .env",
                         chat_id)
            return  # чужой чат — молчание
        text = (event.raw_text or "").strip()
        #: Входящее от владельца ВИДНО в журнале. Без этой строки «бот молчит» и «обработчик вернулся
        #: молча» неотличимы, и разбор дефекта «кнопка показывает меню» шёл вслепую. Сырого текста не
        #: пишем: в нём бывают данные клиентов; вид (меню/прогон/текст) для разбора достаточно.
        log.info("news: входящее от владельца %s: %d симв., вид=%s", chat_id[-4:], len(text),
                 "меню" if text in MENU_SHOW else "прогон" if text in MENU_RUN else "текст")

        prompt = await self._get_prompt(chat_id)
        if prompt:
            await self._apply_prompt(chat_id, prompt, text)
            return

        editing = await self._db.news_editing_for(author_chat_id=chat_id, ttl_s=self._settings.news_edit_ttl_s)
        if editing:
            await self._apply_edit(chat_id, editing, text)
            return

        if text in MENU_SHOW:
            await self._menu(chat_id)
            return
        if text in MENU_RUN:
            #: Тот же путь, что у «🔁 Ещё раз»: прогон один, ветка одна — иначе гейт и статистика разошлись бы.
            await self._run_locked(chat_id, topic=None)
            return
        await self._send(chat_id, "Я умею одно: собирать новости. Нажмите «📰 Найти новости».")

    async def _on_callback(self, event) -> None:
        chat_id = str(event.chat_id)
        raw = event.data.decode() if isinstance(event.data, bytes) else str(event.data or "")
        if not self._is_owner(chat_id):
            await event.answer()
            return
        parsed = tokens.verify(self._settings.ui_secret, raw, chat_id=chat_id, token="")
        #: Нажатие ВИДНО в журнале — по той же причине, что и входящее сообщение: без строки «не
        #: нажимали» и «нажали, но подпись не сошлась» неотличимы. Аргумент нажатия не пишем: он
        #: содержит id черновика, а для разбора достаточно вида.
        log.info("news: нажатие от владельца %s: вид=%s, подпись=%s", chat_id[-4:],
                 _CALLBACK_KINDS.get(parsed[0], parsed[0]) if parsed else "не разобрана",
                 "ок" if parsed else "ОТКАЗ")
        if not parsed:
            await event.answer("Кнопка устарела или не подписана")
            return
        kind, arg = parsed
        await event.answer()  # подтверждаем тап ДО тяжёлых операций
        if kind == KIND_NEWS_EDIT:
            await self._begin_edit(chat_id, arg)
        elif kind == KIND_NEWS_PUBLISH:
            await self._publish(chat_id, arg)
        elif kind == KIND_NEWS_CONFIRM:
            await self._publish(chat_id, arg, force=True)
        elif kind == KIND_NEWS_CANCEL:
            await self._db.cancel_news_editing(author_chat_id=chat_id)
            await self._clear_prompt(chat_id)
            await self._send(chat_id, "Правка отменена.")
        elif kind == KIND_NEWS_RETRY:
            await self._run_locked(chat_id, topic=None)
        elif kind == KIND_NEWS_REWRITE:
            await self._begin_rewrite(chat_id, arg)
        elif kind == KIND_NEWS_HINT:
            await self._apply_hint(chat_id, arg)

    # ---------- прогон ----------

    async def _run_locked(self, chat_id: str, *, topic: str | None) -> None:
        async with self._locks[chat_id]:
            await self.run_once(chat_id=chat_id, topic=topic)

    async def run_once_locked(self, *, chat_id: str, topic: str | None = None) -> dict:
        """Тот же прогон, что у кнопки, но для ВНЕШНЕГО входа (ручка `/api/ops/news`).

        ❗ Вход мимо лока чата позволил бы прогону сторожа и нажатию кнопки идти одновременно.
        Занят — **говорим об этом** и не ждём: ожидание съело бы таймаут вызывающего.
        Тот же статус (`run_in_progress`), что отдаёт гейт в базе, — один словарь, а не второй язык.
        """
        lock = self._locks[chat_id]
        if lock.locked():
            return {"status": "run_in_progress"}
        async with lock:
            return await self.run_once(chat_id=chat_id, topic=topic)

    # ---------- команда ролей ----------

    async def _search_via_team(self, topic: str) -> TeamSearch | None:
        """Поиск командой. `None` — команда недоступна, и вызывающий идёт одиночным путём.

        **Почему фолбэк обязателен.** Команда — боевой путь, но одиночный агент остаётся страховкой.
        Показать владельцу техническую ошибку вместо новостей нельзя, а молча
        подменить сбой команды своим текстом — тем более: неисправность должна быть видна в журнале
        и в статистике процесса.
        """
        try:
            return await search_news(topic=topic, max_items=self._settings.news_max_items,
                                     settings=self._settings, ask_role=self._team_ask,
                                     retriever=self._retriever)
        except TeamUnavailable as error:
            log.info("news: команда недоступна (%s) — иду одиночным путём", error)
            self.stats["last_error"] = f"team:{type(error).__name__}"
            return None

    async def _rewrite_via_team(self, *, draft: dict, hint: str):
        """Переписывание командой: автор пишет, проверяющий выправляет текст под сохранность фактов.

        `None` — команда недоступна: вызывающий делает одиночный вызов. Потери фактов при
        этом считает **наш** код (см. `Run.lost_facts`), и отказ применять текст остаётся в силе.
        """
        try:
            return await team_run_once(topic=str(draft.get("title") or ""), max_items=1,
                                       settings=self._settings, ask_role=self._team_ask,
                                       material=str(draft.get("text") or ""), hint=hint,
                                       url=str(draft.get("url") or ""),
                                       gate_ask=self._gate_ask,
                                       gate_threshold=self._gate_threshold,
                                       retriever=self._retriever)
        except TeamUnavailable as error:
            log.info("news: команда недоступна на переписывании (%s) — одиночный путь", error)
            return None

    async def run_once(self, *, chat_id: str, topic: str | None = None) -> dict:
        """Один прогон: гейт → агент → разбор → дедуп → карточки. Возвращает сводку (для логов и ручки)."""
        allowed, why = await self._db.news_run_gate(
            chat_id=chat_id, cooldown_s=self._settings.news_cooldown_s, daily_cap=self._settings.news_max_per_day,
            month_budget_usd=self._settings.news_month_budget_usd)
        if not allowed:
            await self._send(chat_id, self._gate_text(why))
            return {"status": why}
        await self._db.news_run_started(chat_id=chat_id)
        try:
            await self._send(chat_id, "🔎 Ищу новости…")
            # Кто ищет — решением владельца; по умолчанию АГЕНТ (см. `SEARCH_VIA`). Команда остаётся
            # рабочей ветвью: её гоняют тесты, и вернуть её — правка одной строки.
            team_search = await self._search_via_team(topic or self._topic()) if SEARCH_VIA == "team" else None
            if team_search is not None:
                agent_text = team_search.text
            else:
                reply = await self._channels.ask(
                    user=f"news-{int(time.time())}",
                    text=news.build_news_turn(topic=topic or self._topic(), max_items=self._settings.news_max_items),
                    wait_s=240.0,
                    strict_user=True,  # служебный прогон не должен прочитать чужую сессию канала
                )
                if not reply.done or not reply.text:
                    await self._dead_run(chat_id, "агент не ответил")
                    return {"status": "timeout"}
                agent_text = reply.text
            items, status = news.parse_news_blocks(agent_text, max_items=self._settings.news_max_items)
            if not items:
                # Причину НАЗЫВАЕМ словами и ОСТАВЛЯЕМ ответ в журнале: иначе восстановить, что именно
                # ответил агент, нечем — и разбор идёт вслепую (так и вышло, когда команда впервые
                # отработала, а её текст не разобрался: в журнале не было ни строки об ответе).
                log.warning("news: ответ без карточек (статус %s, %d симв.): %r",
                            status, len(agent_text), agent_text[:200])
                await self._dead_run(chat_id, DEAD_RUN_REASONS.get(status, "ответ не разобран"))
                return {"status": status}
            fresh_urls = [item.url for item in items if item.url]
            seen = await self._db.news_seen_filter(urls=fresh_urls)
            fresh = [item for item in items if not (item.url and item.url in seen)]
            if not fresh:
                await self._send(chat_id, "Нового нет — эти новости уже присылались ранее.")
                return {"status": "no_new"}
            now = time.time()
            for item in fresh:
                draft = await self._db.create_news_draft(
                    author_chat_id=chat_id, title=item.title, text=item.text, url=item.url,
                    source=item.source, run_id=f"news-{int(now)}")
                card = news.format_card(item, tz=self._settings.tz, found_at=_now())
                rows = [[Action("✏️ Редактировать", KIND_NEWS_EDIT, str(draft["id"])),
                         Action("📤 Опубликовать в канале", KIND_NEWS_PUBLISH, str(draft["id"]))]]
                message = await self._send(chat_id, card, rows=rows, kind=INLINE)
                if message is not None:
                    await self._db.attach_news_message(draft_id=draft["id"], message_id=message.id)
                    self.stats["cards"] += 1
                await asyncio.sleep(SEND_PAUSE_S)
            await self._db.news_mark_seen(urls=fresh_urls)
            self.stats["runs"] += 1
            if team_search is not None:
                # Новый путь не ходит через канал, поэтому цену даёт НАШ клиент по нашим ценам.
                # При незаданных ценах она честно неизвестна — тогда метрику не трогаем и пишем в
                # журнал: ноль выдал бы бесплатный прогон.
                if team_search.cost_usd:
                    await self._db.bump_metric(metric="news_cost_usd", delta=team_search.cost_usd)
                else:
                    log.warning("news: цена прогона команды неизвестна (цены не заданы) — метрику не меняю")
            else:
                usage = getattr(reply, "usage", None)
                cost = float((usage or {}).get("cost") or 0) if isinstance(usage, dict) else 0.0
                if cost:
                    await self._db.bump_metric(metric="news_cost_usd", delta=cost)
            log.info("news: прогон ok — карточек %d, дедуп отбросил %d", len(fresh), len(items) - len(fresh))
            return {"status": "ok", "cards": len(fresh), "dropped_seen": len(items) - len(fresh)}
        finally:
            await self._db.news_run_finished(chat_id=chat_id)

    def _gate_text(self, why: str) -> str:
        """Почему прогон не пошёл — человеческим языком: лимит по деньгам, а не только по числу."""
        if why == "run_in_progress":
            return "Прогон уже идёт — подождите."
        if why == "daily_cap":
            return f"Дневной лимит прогонов исчерпан ({self._settings.news_max_per_day} в день)."
        if why.startswith("month_budget:"):
            return (f"Месячный лимит расходов на новости исчерпан: ${why.split(':', 1)[1]} из "
                    f"${self._settings.news_month_budget_usd:.2f}. Прогоны возобновятся в новом месяце.")
        return f"Слишком часто: подождите {why.split(':')[-1]} с."

    async def _dead_run(self, chat_id: str, reason: str) -> None:
        rows = [[Action("🔁 Ещё раз", KIND_NEWS_RETRY, ""),
                 Action("🤖 Переписать по подсказке", KIND_NEWS_REWRITE, "0")]]
        await self._send(chat_id, f"Не получилось собрать новости: {reason}. Попробуем иначе?",
                         rows=rows, kind=INLINE)

    # ---------- правка ----------

    async def _begin_edit(self, chat_id: str, arg: str) -> None:
        draft = await self._db.news_draft(draft_id=int(arg))
        if not draft:
            await self._send(chat_id, "Новость не найдена.")
            return
        await self._db.set_news_editing(draft_id=draft["id"], author_chat_id=chat_id)
        rows = [[Action("✖️ Отмена", KIND_NEWS_CANCEL, str(draft["id"])),
                 Action("🤖 Переписать по подсказке", KIND_NEWS_REWRITE, str(draft["id"]))]]
        await self._send(chat_id, f"✏️ Пришлите новый текст новости одним сообщением.\n\nТекущий текст:\n{draft['text']}",
                         rows=rows, kind=INLINE)

    async def _apply_edit(self, chat_id: str, editing: dict, text: str) -> None:
        if not text:
            await self._send(chat_id, "Пришлите новый текст одним сообщением (фото и пересланные посты не годятся).")
            return
        clean = news.neutralize_news_blocks(text)[:TEXT_MAX]
        updated = await self._db.set_news_text(draft_id=editing["id"], text=clean)
        if not updated:
            await self._send(chat_id, "Правка уже неактуальна — откройте карточку заново.")
            return
        await self._drop_card_buttons(chat_id, editing.get("message_id"))
        item = news.NewsItem(title=updated["title"], text=updated["text"], url=updated["url"], source=updated["source"])
        rows = [[Action("✏️ Редактировать", KIND_NEWS_EDIT, str(updated["id"])),
                 Action("📤 Опубликовать в канале", KIND_NEWS_PUBLISH, str(updated["id"]))]]
        message = await self._send(chat_id, news.format_card(item, tz=self._settings.tz), rows=rows, kind=INLINE)
        if message is not None:
            await self._db.attach_news_message(draft_id=updated["id"], message_id=message.id)

    # ---------- переписывание по подсказке ----------

    async def _begin_rewrite(self, chat_id: str, arg: str) -> None:
        await self._set_prompt(chat_id, mode="rewrite", draft_id=arg)
        await self._send(chat_id, "🤖 Напишите подсказку одной строкой — например: «короче», «добавь цифры», "
                                  "«убери канцелярит». Или выберите готовую:",
                         rows=self._hint_rows(arg), kind=INLINE)

    @staticmethod
    def _hint_rows(draft_id: str) -> list[list[Action]]:
        """Кнопки-подсказки: тап отправляет агенту тот же текст, что и набранный руками."""
        buttons = [Action(label, KIND_NEWS_HINT, str(index))
                   for index, (label, _) in enumerate(news.PRESET_HINTS)]
        rows = [buttons[index:index + 2] for index in range(0, len(buttons), 2)]
        rows.append([Action("✖️ Отмена", KIND_NEWS_CANCEL, draft_id)])
        return rows

    async def _apply_hint(self, chat_id: str, arg: str) -> None:
        """Тап по готовой подсказке: черновик берём из состояния режима правки, а не из `arg`."""
        prompt = await self._get_prompt(chat_id)
        if not prompt:
            await self._send(chat_id, "Подсказка устарела — откройте карточку заново.")
            return
        if not arg.isdigit() or int(arg) >= len(news.PRESET_HINTS):
            return
        await self._apply_prompt(chat_id, prompt, news.PRESET_HINTS[int(arg)][1])

    async def _apply_prompt(self, chat_id: str, prompt: dict, text: str) -> None:
        await self._clear_prompt(chat_id)
        if not text:
            log.info("news: переписывание отброшено — пустая подсказка")
            await self._send(chat_id, "Подсказка пустая — попробуйте ещё раз.")
            return
        if prompt.get("mode") == "topic":
            await self._run_locked(chat_id, topic=news.neutralize_news_blocks(text)[:200])
            return
        draft = await self._db.news_draft(draft_id=int(prompt.get("draft_id") or 0))
        if not draft:
            log.warning("news: переписывание отброшено — черновик %s не найден", prompt.get("draft_id"))
            await self._send(chat_id, "Новость не найдена.")
            return
        await self._send(chat_id, "🤖 Переписываю…")
        team = await self._rewrite_via_team(draft=draft, hint=text)
        if team is not None:
            new_text = team.text.strip()
            # Потери считает НАШ код: проверяющий правил текст, но последнее слово за счётом —
            # факты дороже формулировки.
            lost = list(team.lost_facts)
        else:
            try:
                reply = await self._channels.ask(
                    user=f"news-rewrite-{int(time.time())}",
                    text=news.build_rewrite_turn(draft_text=draft["text"], hint=text, url=draft.get("url") or ""),
                    wait_s=120.0,
                    strict_user=True,
                )
            except NETWORK_ERRORS as error:
                log.warning("news: переписывание не удалось (%s)", type(error).__name__)
                await self._send(chat_id, "Не удалось переписать — попробуйте позже.")
                return
            new_text = news.strip_news_blocks(reply.text or "").strip()
            if not reply.done:
                log.warning("news: переписывание отброшено — агент не ответил (%d симв.)", len(new_text))
                await self._send(chat_id, "Агент не дал внятного текста — оставляю прежний.")
                return
            lost = news.facts_kept(draft["text"], new_text)
        if len(new_text) < news.MIN_TEXT:
            log.warning("news: переписывание отброшено — короткий ответ (%d < %d симв.)",
                        len(new_text), news.MIN_TEXT)
            await self._send(chat_id, "Агент не дал внятного текста — оставляю прежний.")
            return
        if lost:
            # Факты дороже формулировки: текст НЕ применяем, черновик остаётся с прежним текстом.
            log.info("news: переписывание отклонено — потеряны факты: %d", len(lost))
            await self._set_prompt(chat_id, mode="rewrite", draft_id=str(draft["id"]))
            await self._send(
                chat_id,
                "⚠️ Агент потерял факты из исходного текста: " + ", ".join(lost[:6])
                + "\nОставляю прежний текст. Попробуйте другую подсказку.",
                rows=self._hint_rows(str(draft["id"])), kind=INLINE)
            return
        #: УСПЕШНОЕ переписывание обязано оставлять след. Без записи в журнале нет НИ ОДНОЙ строки, и
        #: успех неотличим от «до переписывания не дошло» — исход пришлось бы восстанавливать по
        #: отметке времени в базе. Пишем размеры «до → после» и ИСТОЧНИК
        #: (команда или одиночный агент — это разные пути с разной ценой). Текст подсказки не пишем,
        #: по той же причине, что и входящее: в нём бывают слова владельца.
        log.info("news: переписывание применено — %d → %d симв., источник: %s, подсказка %d симв.",
                 len(draft["text"]), len(new_text), "команда" if team is not None else "агент", len(text))
        await self._db.set_news_editing(draft_id=draft["id"], author_chat_id=chat_id)
        updated = await self._db.set_news_text(draft_id=draft["id"], text=new_text[:TEXT_MAX])
        if not updated:
            log.warning("news: переписывание не сохранилось — черновик %s исчез", draft["id"])
            return
        await self._drop_card_buttons(chat_id, draft.get("message_id"))
        item = news.NewsItem(title=updated["title"], text=updated["text"], url=updated["url"], source=updated["source"])
        rows = [[Action("✏️ Редактировать", KIND_NEWS_EDIT, str(updated["id"])),
                 Action("📤 Опубликовать в канале", KIND_NEWS_PUBLISH, str(updated["id"]))]]
        message = await self._send(chat_id, news.format_card(item, tz=self._settings.tz), rows=rows, kind=INLINE)
        if message is not None:
            await self._db.attach_news_message(draft_id=updated["id"], message_id=message.id)

    # ---------- состояние подсказки (в БД: рестарт не сбрасывает) ----------

    async def _set_prompt(self, chat_id: str, *, mode: str, draft_id: str) -> None:
        await self._db.set_setting(f"news_prompt:{chat_id}", {"mode": mode, "draft_id": draft_id, "at": time.time()})

    async def _get_prompt(self, chat_id: str) -> dict | None:
        value = await self._db.setting(f"news_prompt:{chat_id}", None)
        if not isinstance(value, dict):
            return None
        if time.time() - float(value.get("at") or 0) > PROMPT_TTL_S:
            await self._clear_prompt(chat_id)
            return None
        return value

    async def _clear_prompt(self, chat_id: str) -> None:
        await self._db.set_setting(f"news_prompt:{chat_id}", {})

    # ---------- публикация ----------

    async def _publish(self, chat_id: str, arg: str, *, force: bool = False) -> None:
        draft = await self._db.news_draft(draft_id=int(arg))
        if not draft:
            await self._send(chat_id, "Новость не найдена.")
            return
        if draft["status"] == "published":
            await self._send(chat_id, "Уже опубликовано.")
            return
        if draft["status"] == "publishing":
            await self._send(chat_id, "Публикуется…")
            return
        channel = self._channel_id()
        if not channel:
            await self._send(chat_id, "Канал не задан: впишите TELEGRAM_NEWS_CHANNEL_ID в .env и перезапустите бота.")
            return
        if not force and self._is_stale(draft):
            rows = [[Action("✅ Да, публикуем", KIND_NEWS_CONFIRM, str(draft["id"]))]]
            await self._send(chat_id, f"🕒 Новость найдена {self._found_label(draft)}. Публикуем?",
                             rows=rows, kind=INLINE)
            return
        claimed = await self._db.claim_news_publish(draft_id=draft["id"])
        if not claimed:
            await self._send(chat_id, "Уже публикуется или опубликовано.")
            return
        text = news.publish_text(news.NewsItem(title=claimed["title"], text=claimed["text"],
                                               url=claimed["url"], source=claimed["source"]))
        if self._settings.news_publish_dry_run:
            await self._db.release_news_publish(draft_id=draft["id"])
            await self._send(chat_id, "🧪 Dry-run: публикация не выполнена (NEWS_PUBLISH_DRY_RUN=1).")
            return
        try:
            for chunk in news.chunk_for_telegram(text):
                await self._client.send_message(int(channel), chunk, link_preview=False)
        except (*NETWORK_ERRORS, ValueError) as error:
            await self._db.release_news_publish(draft_id=draft["id"])   # черновик не теряем
            await self._send(chat_id, self._publish_error_text(error))
            log.warning("news: публикация не удалась (%s)", type(error).__name__)
            return
        await self._db.confirm_news_published(draft_id=draft["id"], channel_id=channel)
        await self._db.log_news_publish(draft_id=draft["id"], channel_id=channel, actor=chat_id)
        await self._drop_card_buttons(chat_id, draft.get("message_id"))
        self.stats["published"] += 1
        await self._send(chat_id, "📤 Опубликовано в канале.")

    def _is_stale(self, draft: dict) -> bool:
        created = draft.get("created_at")
        if not created:
            return False
        age_h = (_now().timestamp() - created.timestamp()) / 3600
        return age_h >= self._settings.news_stale_h

    @staticmethod
    def _found_label(draft: dict) -> str:
        created = draft.get("created_at")
        return f"{created:%d.%m %H:%M}" if created else "ранее"

    @staticmethod
    def _publish_error_text(error: Exception) -> str:
        name = type(error).__name__
        if name in WRITE_ERRORS:
            return ("Не могу писать в канал: добавьте бота администратором с правом публикации "
                    "(и проверьте, что id канала верный). Черновик сохранён.")
        if name in PEER_ERRORS:
            return ("Канал недоступен: проверьте TELEGRAM_NEWS_CHANNEL_ID — бот должен быть участником "
                    "или админом канала. Черновик сохранён.")
        if name == "FloodWaitError":
            seconds = getattr(error, "seconds", None)
            return f"Telegram просит подождать {seconds or '?'} с — повторите позже. Черновик сохранён."
        return f"Публикация не удалась ({name}). Черновик сохранён — можно повторить."

    async def _drop_card_buttons(self, chat_id: str, message_id: int | None) -> None:
        """Снять кнопки у СТАРОГО сообщения-карточки (переотправка/публикация)."""
        if not message_id or not self._client:
            return
        try:
            await self._client.edit_message(int(chat_id), int(message_id), buttons=None)
        except NETWORK_ERRORS as error:
            log.warning("news: кнопки у карточки не снялись (%s)", type(error).__name__)


def _now() -> datetime:
    """Текущее время UTC (отдельная функция — тесты подменяют её, не трогая модуль `datetime`)."""
    return datetime.now(UTC)
