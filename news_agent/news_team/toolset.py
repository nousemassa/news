"""Инструменты роли: свои и приходящие по MCP.

**Зачем отдельный модуль.** Раздача инструментов — это описание роли на языке возможностей: что у
неё есть и что это делает. Вызов модели (`llm.py`) не должен знать, откуда инструмент взялся — из
нашего кода или из внешнего MCP-сервера: он получает словарь готовых инструментов и умеет только
звать их по имени.

**Почему инструменты не завернуты в агентный фреймворк.** Готовый фреймворк ролей (CrewAI) с его
`@tool`/`mcps=` требует второго окружения. Своя обёртка тонкая: имя,
описание, схема аргументов и функция вызова — ровно то, что требует OpenAI-совместимый API.

**Гейт новости инструментом роли НЕ является:** решение о показе принимает наш код,
а не роль, поэтому `news_gate` здесь не раздаётся.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Self

from news_agent.news import SEARCH_FAILED
from news_agent.news_team.roles import Role
from news_agent.news_team.tools import facts as facts_tool
from news_agent.news_team.tools import mcp as mcp_tools
from news_agent.news_team.tools import retrieval as retrieval_tool

log = logging.getLogger(__name__)

#: Сколько ждём ответа внешнего MCP-сервера на один вызов инструмента.
MCP_CALL_TIMEOUT_S = 30.0

#: Сколько попыток у вызова инструмента и сколько ждём между ними.
#:
#: **Зачем повтор.** Без повтора временный отказ чужого сервера (лимит запросов, таймаут, обрыв) сразу
#: превращается в текст «инструмент не сработал», роль честно отвечает «ничего не нашёл», и выпуск
#: отказывал **по причине, которой не было**. Повтор с растущей паузой стоит секунды, а отличает
#: «источник моргнул» от «источника нет».
MCP_RETRY_ATTEMPTS = 3
MCP_RETRY_PAUSE_S = 1.5

#: Сколько символов ответа инструмента отдаём модели (см. `_mcp_result_text`): прочитанная страница
#: приходит целиком, а роль читает подряд несколько — контекст и цена круга растут с каждой.
MCP_RESULT_LIMIT_CHARS = 2000

#: Слова, по которым отказ считается ВРЕМЕННЫМ. Только по ним и повторяем: неизвестный инструмент
#: или неверные аргументы не пройдут и на второй раз, и повтор лишь тянет время.
_TRANSIENT_SIGNS = ("too many requests", "429", "timeout", "timed out", "temporarily", "temporar")
#: Слова, по которым видно, что СЕССИЯ мертва: её надо переоткрыть, иначе повтор бессмыслен.
_BROKEN_SESSION_SIGNS = ("closed", "not connected", "connection", "broken", "reset", "eof")

#: Поля, которые сторонняя обёртка поиска (`isokoliuk/mcp-searxng`) ПРЕДЛАГАЕТ модели, а наш поиск
#: исполнить не может. Замер: единственный движок с поддержкой фильтра времени (`google cse`) висит
#: `suspended`, а `bing` объявляет `time_range_support: False` ⇒ один и тот же запрос даёт **10**
#: результатов без параметров, **0** — с `time_range=month`, **0** — с `language=ru`.
#:
#: **Почему вырезаем из СХЕМЫ, а не просим модель.** Просьба в промпте — не гарантия: модель, попросив
#: «свежие новости», выбирает фильтр сама, получает пустой ответ, повторяет запросы и выжигает лимит
#: кругов (`llm.MAX_TOOL_ROUNDS`) — прогон падал в одиночный путь, потратив 17–25 тыс. токенов впустую.
#: Вырезанное поле модель не видит вовсе, поэтому обнулить поиск не может.
UNSUPPORTED_SEARCH_FIELDS = ("time_range", "language")

#: Что говорим модели вместо вырезанных полей — иначе она будет просить их текстом запроса.
SEARCH_FIELDS_NOTE = ("Фильтры по времени и языку не поддерживаются: задавай только `query` "
                      "(свежесть задаётся формулировкой).")


def _declared_schema(schema: dict | None) -> tuple[dict, bool]:
    """Схема инструмента без полей, которых поиск не умеет исполнить, и признак «что-то убрали».

    Убираем только НАЛИЧНЫЕ поля и возвращаем признак: вызывающий обязан сказать об этом в журнале.
    Сужение без записи в журнал неотличимо от отказа инструмента и скрывает причину.
    """
    clean = dict(schema or {})
    properties = dict(clean.get("properties") or {})
    removed = [name for name in UNSUPPORTED_SEARCH_FIELDS if name in properties]
    for name in removed:
        properties.pop(name, None)
    if properties:
        clean["properties"] = properties
    else:
        clean.pop("properties", None)
    required = [name for name in (clean.get("required") or []) if name not in UNSUPPORTED_SEARCH_FIELDS]
    if required:
        clean["required"] = required
    else:
        clean.pop("required", None)
    return clean, bool(removed)


def _clean_args(args: dict | None) -> dict:
    """Аргументы вызова без полей, которых поиск не умеет (защита в глубину: схему модель видит, но
    не гарантия — аргументы могли прийти из истории или из подсказки)."""
    return {key: value for key, value in (args or {}).items() if key not in UNSUPPORTED_SEARCH_FIELDS}


@dataclass(frozen=True)
class Tool:
    """Инструмент, каким его видит вызов модели: имя, описание, схема и вызов."""

    name: str
    description: str
    parameters: dict
    call: Callable[[dict], Awaitable[str]]
    #: Откуда инструмент: `local` или адрес MCP-сервера. Нужен журналу и разбору прогона,
    #: а не модели: она видит только имя и описание.
    source: str = "local"


def local_tools(*, retriever) -> dict[str, Tool]:
    """Свои инструменты команды: поиск по базе знаний и проверка фактов.

    Обе функции — существующий код проекта (`retrieval.search` через ретривер и
    `facts.missing_facts` вокруг `facts_kept`): второй реализации не заводится, иначе команда и
    одиночный путь судили бы о потерях фактов по разным критериям.
    """

    async def search(args: dict) -> str:
        return await retrieval_tool.search(
            retriever,
            str(args.get("query") or ""),
            limit=int(args.get("limit") or 4),
        )

    async def lost_facts(args: dict) -> str:
        lost = facts_tool.missing_facts(
            str(args.get("original") or ""),
            str(args.get("new_text") or ""),
        )
        if not lost:
            return "потерь нет"
        return "потеряны факты: " + "; ".join(lost)

    return {
        retrieval_tool.NAME: Tool(
            name=retrieval_tool.NAME,
            description=retrieval_tool.DESCRIPTION,
            parameters=retrieval_tool.SCHEMA,
            call=search,
        ),
        facts_tool.NAME: Tool(
            name=facts_tool.NAME,
            description=facts_tool.DESCRIPTION,
            parameters=facts_tool.SCHEMA,
            call=lost_facts,
        ),
    }


class _StaticBox:
    """Инструменты, подставленные снаружи (тесты) или отсутствующие: сессии открывать нечего."""

    def __init__(self, tools: dict[str, Tool] | None = None) -> None:
        self._tools = dict(tools or {})

    def tools(self) -> dict[str, Tool]:
        return dict(self._tools)


class McpToolbox:
    """Инструменты внешних MCP-серверов с **ОДНОЙ сессией на выпуск**.

    **Почему сессия на выпуск, а не на вызов — это не оптимизация.** Клиент, открытый на КАЖДЫЙ
    вызов инструмента, даёт `1 + N` инициализаций, а обёртка поиска (`mcp-searxng`) считает не
    запросы, а **инициализации сессий** — по умолчанию 20 в минуту. Три выпуска подряд упираются в
    лимит, и поиск отвечает «Too many requests», хотя сам SearXNG возвращает `200` и
    **30 результатов**: отказ выглядит как отказ источника, будучи нашим. Инициализация одна на
    выпуск.

    **Довод «сессия на вызов спасает от мёртвого соединения» учтён иначе:** мёртвую сессию
    переоткрываем при обрыве и повторяем вызов (см. `_call`), а живую не трогаем.

    Ошибка наружу идёт **текстом с маркером** `SEARCH_FAILED`, а не исключением: роль обязана
    узнать, что поиск не сработал, и отличить это от «новостей нет» — иначе отказ источника
    превращается в «материала нет», и причина исчезает из отчёта.
    """

    def __init__(self, servers: tuple[mcp_tools.McpServer, ...],
                 *, timeout: float = MCP_CALL_TIMEOUT_S) -> None:
        self._servers = tuple(servers)
        self._timeout = timeout
        self._clients: dict[str, object] = {}
        self._tools: dict[str, Tool] = {}

    async def __aenter__(self) -> Self:
        for server in self._servers:
            if mcp_tools.address_problem(server.url):
                log.warning("команда: адрес MCP-сервера негоден, пропускаю (%s)", server.url)
                continue
            client = await self._open(server.url)
            if client is None:
                continue
            try:
                declared = await client.list_tools()
            except Exception as error:  # noqa: BLE001 — недоступный источник не роняет прогон
                log.warning("команда: MCP-сервер %s не объявил инструменты (%s: %s)",
                            server.url, type(error).__name__, error)
                continue
            for declared_tool in declared:
                name = mcp_tools.final_tool_name(server.url, declared_tool.name)
                schema, stripped = _declared_schema(declared_tool.inputSchema)
                description = str(declared_tool.description or "")
                if stripped:
                    # Названо в журнале, а не молча: мы сузили то, что объявил ЧУЖОЙ сервер.
                    log.info("команда: у инструмента %s убраны поля %s (наш поиск их не исполняет), "
                             "модели сказано: %s", name, ", ".join(UNSUPPORTED_SEARCH_FIELDS),
                             SEARCH_FIELDS_NOTE)
                    description = f"{description}\n{SEARCH_FIELDS_NOTE}".strip()
                self._tools[name] = Tool(
                    name=name,
                    description=description,
                    parameters=schema,
                    call=self._call(server.url, declared_tool.name),
                    source=server.url,
                )
        return self

    async def __aexit__(self, *exc_info) -> bool:
        for url in list(self._clients):
            await self._close(url)
        self._tools.clear()
        return False

    def tools(self) -> dict[str, Tool]:
        """Инструменты как их видит модель. Пусто — штатно: роль работает своими."""
        return dict(self._tools)

    async def _open(self, url: str):
        """Открыть сессию. `None` — сервер недоступен или клиента MCP нет в окружении."""
        try:
            from fastmcp import Client
        except ImportError as error:
            log.warning("команда: клиент MCP недоступен (%s) — работаю своими инструментами", error)
            return None
        client = Client(url, timeout=self._timeout)
        try:
            await client.__aenter__()
        except Exception as error:  # noqa: BLE001 — недоступный источник не роняет прогон
            log.warning("команда: MCP-сервер %s не ответил (%s: %s)",
                        url, type(error).__name__, error)
            return None
        self._clients[url] = client
        # Строка намеренно на уровне INFO: «одна сессия на выпуск» — это утверждение, ради которого
        # пункт и делался, и в живом журнале оно должно быть видно числом строк, а не только тестом.
        log.info("команда: сессия MCP открыта (%s)", url)
        return client

    async def _close(self, url: str) -> None:
        client = self._clients.pop(url, None)
        if client is None:
            return
        try:
            await client.__aexit__(None, None, None)
        except Exception as error:  # noqa: BLE001 — ошибка закрытия сессии не влияет на выпуск
            log.debug("команда: сессия %s закрылась с ошибкой (%s: %s)", url, type(error).__name__, error)

    async def _reopen(self, url: str):
        await self._close(url)
        return await self._open(url)

    def _call(self, url: str, tool: str) -> Callable[[dict], Awaitable[str]]:
        """Вызов инструмента в общей сессии выпуска: повтор при временном отказе, переоткрытие при обрыве."""

        async def call(args: dict) -> str:
            problem = "неизвестная ошибка"
            for attempt in range(1, MCP_RETRY_ATTEMPTS + 1):
                client = self._clients.get(url) or await self._open(url)
                if client is None:
                    problem = f"сервер инструментов недоступен ({url})"
                    break
                try:
                    result = await client.call_tool(tool, _clean_args(args))
                except Exception as error:  # noqa: BLE001 — отказ инструмента не роняет прогон
                    problem = f"{type(error).__name__}: {error}"
                    log.warning("команда: вызов MCP-инструмента %s не удался, попытка %d/%d (%s)",
                                tool, attempt, MCP_RETRY_ATTEMPTS, problem)
                    # Повторяем ДВА класса отказов: временный (лимит, таймаут) и обрыв сессии.
                    # Разница в подготовке: мёртвую сессию надо сначала переоткрыть, иначе повтор
                    # уйдёт в тот же мёртвый сокет. Всё остальное (неизвестный инструмент, неверные
                    # аргументы) не пройдёт и на третий раз — повторять его значит тянуть время.
                    retryable = _transient(error) or _looks_broken(error)
                    if not retryable:
                        break
                    if _looks_broken(error):
                        await self._reopen(url)
                    if attempt < MCP_RETRY_ATTEMPTS:
                        await asyncio.sleep(MCP_RETRY_PAUSE_S * attempt)
                    continue
                return _mcp_result_text(result)
            return f"{SEARCH_FAILED}: {tool} не сработал ({problem})"

        return call


@asynccontextmanager
async def mcp_toolbox(*, settings=None, mcp: dict | None = None):
    """Инструменты ролей на время выпуска: своя сессия MCP живёт выпуск, а не один вызов.

    Три случая, и все — штатные: инструменты подставлены снаружи (тесты) ⇒ сеть не трогаем; настроек
    или адресов нет ⇒ роль работает своими инструментами; адреса есть ⇒ поднимаем сессию выпуска.
    """
    if mcp is not None:
        yield _StaticBox(mcp)
        return
    servers = mcp_tools.parse_servers(getattr(settings, "team_mcp_servers", "") or "") if settings else ()
    if not servers:
        yield _StaticBox()
        return
    async with McpToolbox(servers) as box:
        yield box


def _transient(error: Exception) -> bool:
    """Отказ ПОХОЖ на временный (лимит, таймаут) — только такие имеет смысл повторять."""
    text = f"{type(error).__name__}: {error}".lower()
    return any(sign in text for sign in _TRANSIENT_SIGNS)


def _looks_broken(error: Exception) -> bool:
    """По отказу видно, что сессия мертва ⇒ её надо переоткрыть, иначе повтор бессмыслен."""
    text = f"{type(error).__name__}: {error}".lower()
    return any(sign in text for sign in _BROKEN_SESSION_SIGNS)


def _mcp_result_text(result) -> str:
    """Текст результата MCP-вызова: части ответа склеиваются, пустое остаётся пустым.

    **Почему ответ ОБРЕЗАЕТСЯ.** Прочитанная страница приходит целиком (живой замер: по 3000 символов на
    статью), а роль читает несколько статей подряд — каждый круг отправляет ВСЮ историю, поэтому контекст
    и цена растут с каждой страницей. Замер живого прогона: шаг исследователя потратил 45 995
    токенов и упёрся в лимит кругов, так и не отдав строк `NEWS|`. Роли нужна суть, а не вся страница.

    Обрезка помечается ЯВНО: молчаливое усечение выглядело бы как «источник короче, чем он есть», и роль
    приняла бы обрывок за полный текст.

    ❗ **JSON режется по СТРУКТУРЕ, а не по символам.** Срез строки после сериализации даёт обрывок,
    который не разбирается ничем: у модели нет парсера (`llm._tool_answer` кладёт текст в `content` как
    есть), поэтому половина ответа тихо превращается в мусор. Поэтому список теряет ХВОСТОВЫЕ элементы, а
    причина остаётся видимой полем `_truncated` — форма ответа сохраняется. Проза режется по символам: у
    неё структурной альтернативы нет.
    """
    parts: list[str] = []
    for item in getattr(result, "content", None) or ():
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))
    whole = "\n".join(parts)
    if len(whole) <= MCP_RESULT_LIMIT_CHARS:
        return whole
    structured = _shorten_json(whole)
    if structured is not None:
        return structured
    return whole[:MCP_RESULT_LIMIT_CHARS] + "\n…[ответ обрезан: показана часть, остальное отброшено]"


def _shorten_json(whole: str) -> str | None:
    """Укоротить JSON, СОХРАНИВ его разбор: `None` — ответ не JSON, режь как прозу.

    Бюджет считаем по ИТОГОВОЙ строке: усечённый ответ обязан остаться в лимите, иначе смысл обрезки
    теряется. Отсюда две тонкости, обе найдены замером: (1) пометка `_truncated` тоже занимает место,
    поэтому набор данных идёт ПОСЛЕ её появления, а не до; (2) набор повторяется из ОРИГИНАЛОВ — уже
    усечённый список сам не вырастет, даже если сброс крупного ключа освободил место.

    **Названная граница.** Режется СПИСОК (это «статьи») и отбрасываются крупные НЕ-списочные ключи.
    Ответ вида «одна огромная строка» или «вложенный словарь» отдаст только пометку с перечнем
    отброшенного: показать его обрывок значило бы вернуть ту самую невалидную строку, от которой уходим.
    Модель узнаёт, что данные были и что они не влезли, — это правда, в отличие от мусора.
    """
    import json

    try:
        data = json.loads(whole)
    except ValueError:
        return None

    def fits(payload) -> str | None:
        text = json.dumps(payload, ensure_ascii=False)
        return text if len(text) <= MCP_RESULT_LIMIT_CHARS else None

    if isinstance(data, list):
        total = len(data)
        kept: list = []
        for item in data:
            trial = {"results": [*kept, item], "_truncated": {"shown": len(kept) + 1, "total": total}}
            if fits(trial) is None:
                break
            kept.append(item)
        payload = {"results": kept, "_truncated": {"shown": len(kept), "total": total}}
        return fits(payload) or fits({"results": [], "_truncated": {"shown": 0, "total": total}})

    if not isinstance(data, dict):
        return None

    payload = dict(data)
    originals = {key: list(value) for key, value in payload.items() if isinstance(value, list)}
    dropped: list[str] = []

    def fill_lists() -> None:
        """Набрать элементы списков (из ОРИГИНАЛА), пока влезает ВМЕСТЕ с пометкой."""
        for key, full in originals.items():
            kept = list(full)
            while kept and fits({**payload, key: kept}) is None:
                kept.pop()
            payload[key] = kept

    # Списки первыми: это и есть «статьи», которые растят ответ.
    fill_lists()
    # Затем — самые крупные ключи: лучше отдать меньше полей, чем обрывок.
    for key in sorted(payload, key=lambda k: len(json.dumps(payload[k], ensure_ascii=False)), reverse=True):
        if fits(payload) is not None or key in originals:
            continue
        payload.pop(key)
        dropped.append(key)
    if dropped:
        payload["_dropped_keys"] = dropped

    def marker() -> dict:
        """Пометка по ФАКТИЧЕСКОМУ состоянию: сколько показано и какие ключи отброшены."""
        lists = {key: {"shown": len(payload[key]), "total": len(full)}
                 for key, full in originals.items() if len(payload[key]) < len(full)}
        return {**({"lists": lists} if lists else {}), **({"dropped_keys": dropped} if dropped else {})}

    # Пометка и данные делят ОДИН бюджет: её байты тоже входят в лимит, а её длина зависит от чисел,
    # которые меняются вместе с набором. Поэтому сходимся циклом, а не одной попыткой.
    payload["_truncated"] = {}
    fill_lists()
    while True:
        payload["_truncated"] = marker()
        if fits(payload) is not None:
            break
        for value in payload.values():
            if isinstance(value, list) and value:
                value.pop()
                break
        else:
            break
    if not payload["_truncated"]:
        payload.pop("_truncated")
    return fits(payload) or json.dumps({"_truncated": marker()}, ensure_ascii=False)


def tools_for_role(
    *,
    definition: Role,
    retriever=None,
    mcp: dict[str, Tool] | None = None,
) -> dict[str, Tool]:
    """Инструменты конкретной роли — по её списку имён из `roles.py` плюс внешние по MCP.

    **Почему внешние инструменты не перечисляются поимённо в роли.** Их состав известен только в
    момент запуска: MCP-сервер объявляет свои инструменты сам, и записывать их в описание роли
    значило бы держать в коде список, который разойдётся с сервером. Роль говорит «пользуюсь
    внешними инструментами» (`uses_mcp`), а конкретные имена приходят из объявления сервера.

    Своими инструментами роль распоряжается точно: лишнего не даём — расхождение между «что роль
    умеет» и «что ей дали» проверяется тестом без сети и без модели.
    """
    available: dict[str, Tool] = dict(local_tools(retriever=retriever))
    missing = [name for name in definition.tools if name not in available]
    if missing:
        log.warning("команда: роль %s просит инструменты, которых нет: %s",
                    definition.key, ", ".join(missing))
    chosen = {name: available[name] for name in definition.tools if name in available}
    if definition.uses_mcp and mcp:
        chosen.update(mcp)
    return chosen


@dataclass
class ToolTrace:
    """След вызовов инструментов за шаг — для журнала и разбора: имя, откуда, длина ответа."""

    calls: list[tuple[str, str, int]] = field(default_factory=list)

    def add(self, tool: Tool, answer: str) -> None:
        self.calls.append((tool.name, tool.source, len(answer)))
